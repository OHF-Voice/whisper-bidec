import argparse
import csv
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import wave
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import kenlm
import numpy as np
import torch
from torch.nn.functional import log_softmax, pad
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

_LOGGER = logging.getLogger(__name__)
_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent

DEFAULT_MODEL = "openai/whisper-tiny.en"
FEATURES_LENGTH = 3000  # 30 seconds
SAMPLE_RATE = 16000
TOOLS_DIR = _PROGRAM_DIR / "local"
TOOLS_URL = "https://huggingface.co/datasets/rhasspy/rhasspy-speech/resolve/main/tools/rhasspy-speech_{arch}.tar.gz?download=true"
EPS = "<eps>"
UNK = "<unk>"
UNK_LOGPROB = -5
NGRAM_ORDER = 5


class KenLMLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        lm: kenlm.Model,
        vocab,
        processor,
        bias: float = 0.5,
    ) -> None:
        self.lm = lm
        self.bias = min(1, max(0, bias))
        self.processor = processor

        # token string -> token id
        self.vocab_to_id: dict[str, int] = {
            t_str: int(t_str.split("_", maxsplit=1)[1]) for t_str in sorted(vocab)
        }
        self.vocab_to_id["</s>"] = processor.tokenizer.eos_token_id  # <|endoftext|>

        self.vocab_ids: set[int] = set(self.vocab_to_id.values())
        self.vocab_to_id_sorted = sorted(self.vocab_to_id.items(), key=lambda kv: kv[1])

        self.vocab_mask: Optional[torch.Tensor] = None
        self.special_token_ids = set(processor.tokenizer.all_special_ids)
        self.unknown_token_mask: Optional[torch.Tensor] = None
        self.id_to_token: dict[int, str] = {}

    def __call__(self, input_ids, scores):
        # Convert to log probabilities
        scores = log_softmax(scores, dim=-1)

        if self.unknown_token_mask is None:
            self.vocab_mask = torch.tensor(
                [(t_id in self.vocab_ids) for t_id in range(scores.shape[-1])]
            )
            self.unknown_token_mask = torch.tensor(
                [
                    (t_id not in self.special_token_ids)
                    and (t_id not in self.vocab_ids)
                    for t_id in range(scores.shape[-1])
                ]
            )

        for batch_idx in range(len(input_ids)):
            current_token_ids = tuple(
                t_id
                for t_id in input_ids[batch_idx].tolist()
                if t_id not in self.special_token_ids
            )

            # TODO: cache
            state = kenlm.State()
            prob = 0.0
            self.lm.BeginSentenceWrite(state)
            for token_id in current_token_ids:
                token_str = self.id_to_token.get(token_id)
                if token_str is None:
                    token_str = f"w_{token_id}"
                    if token_str in self.vocab_to_id:
                        self.id_to_token[token_id] = token_str
                    else:
                        token_str = UNK
                        self.id_to_token[token_id] = UNK

                next_state = kenlm.State()
                prob += self.lm.BaseScore(state, token_str, next_state)
                state = next_state

            # Get probability of next state being unknown
            next_unk_state = kenlm.State()
            next_unk_prob = prob + self.lm.BaseScore(state, UNK, next_unk_state)
            scores[batch_idx, self.unknown_token_mask] = (
                scores[batch_idx, self.unknown_token_mask] * (1 - self.bias)
            ) + (next_unk_prob * self.bias)

            vocab_probs = []
            # for token_str, token_id in self.vocab_to_id.items():
            for token_str, token_id in self.vocab_to_id_sorted:
                next_state = kenlm.State()
                next_prob = prob + self.lm.BaseScore(state, token_str, next_state)
                combined_score = (scores[batch_idx, token_id] * (1.0 - self.bias)) + (
                    next_prob * self.bias
                )
                vocab_probs.append(combined_score)

            scores[batch_idx, self.vocab_mask] = torch.FloatTensor(
                vocab_probs, device=scores.device
            )

        return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Name of whisper model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--processor", help="Name of pretrained Whisper processor (defaults to model)"
    )
    parser.add_argument(
        "--text", action="append", help="Text file with custom sentences"
    )
    #
    parser.add_argument("--bias", type=float, default=0.5)
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search width")
    #
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Don't checking HuggingFace hub for updates every time",
    )
    #
    parser.add_argument(
        "--tools-dir", default=TOOLS_DIR, help="Directory to download tools into"
    )
    parser.add_argument(
        "--tools-arch",
        choices=("amd64", "arm64"),
        help="CPU architecture for tools (default: auto detect)",
    )
    parser.add_argument("--ngram-order", type=int, default=NGRAM_ORDER)
    parser.add_argument("--unk-logprob", type=float, default=UNK_LOGPROB)
    #
    parser.add_argument("wav_file", nargs="+", help="Path to WAV file(s)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if platform.system() != "Linux":
        _LOGGER.warning("Biased decoding only works on Linux")

    if not args.tools_arch:
        machine = platform.machine().lower()
        if machine == "x86_64":
            args.tools_arch = "amd64"
        elif machine in ("aarch64", "arm64"):
            args.tools_arch = "arm64"
        else:
            raise ValueError("Unable to detect CPU architecture. Use --tools-arch")

    tools_dir = Path(args.tools_dir)
    ngram_count_path = tools_dir / "opengrm" / "bin" / "ngramcount"
    if not ngram_count_path.exists():
        tools_dir.mkdir(parents=True, exist_ok=True)
        url = TOOLS_URL.format(arch=args.tools_arch)
        _LOGGER.debug("Download tools from '%s' to '%s'", url, tools_dir)
        with (
            urlopen(url) as response,
            tempfile.NamedTemporaryFile("wb+", suffix=".tar.gz") as tools_temp_file,
        ):
            if response.status != 200:
                raise ValueError(f"Bad status for URL '{url}': {response.status}")

            shutil.copyfileobj(response, tools_temp_file)
            tools_temp_file.seek(0)

            with tarfile.open(tools_temp_file.name, "r:gz") as tools_tar_file:
                tools_tar_file.extractall(tools_dir)

        _LOGGER.info("Downloaded tools")

    model_name = args.model
    processor_name = args.processor or model_name

    _LOGGER.debug("Loading processor %s", processor_name)
    processor = WhisperProcessor.from_pretrained(
        processor_name, local_files_only=args.local_files_only
    )
    _LOGGER.info("Loaded processor: %s", processor_name)

    logits_processor: Optional[LogitsProcessorList] = None
    if args.text:
        vocab: set[str] = set()
        extended_env = os.environ.copy()
        bin_dirs = [tools_dir / "opengrm" / "bin", tools_dir / "openfst" / "bin"]
        if current_path := extended_env.get("PATH"):
            bin_dirs.append(current_path)

        lib_dirs = [tools_dir / "opengrm" / "lib", tools_dir / "openfst" / "lib"]
        if current_lib_path := extended_env.get("LD_LIBRARY_PATH"):
            lib_dirs.append(current_lib_path)

        extended_env["PATH"] = os.pathsep.join(str(p) for p in bin_dirs)
        extended_env["LD_LIBRARY_PATH"] = os.pathsep.join(str(p) for p in lib_dirs)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            sentence_tokens_path = temp_dir / "sentence_tokens.txt"
            symbols_path = temp_dir / "symbols.txt"
            raw_arpa_path = temp_dir / "sentence_tokens_raw.arpa"
            arpa_path = temp_dir / "sentence_tokens.arpa"

            with open(
                sentence_tokens_path, "w", encoding="utf-8"
            ) as sentence_tokens_file:
                for text_path in args.text:
                    _LOGGER.debug("Processing %s", text_path)
                    with open(text_path, "r", encoding="utf-8") as text_file:
                        for line in text_file:
                            line = line.strip()
                            if not line:
                                continue

                            # NOTE: Space in front is critical for first word
                            line_token_ids = processor.tokenizer.encode(
                                f" {line}", add_special_tokens=False
                            )
                            line_tokens = [f"w_{t_id}" for t_id in line_token_ids]
                            print(*line_tokens, file=sentence_tokens_file)
                            vocab.update(line_tokens)

            symbols: list[str] = [EPS] + list(sorted(vocab))
            with open(symbols_path, "w", encoding="utf-8") as symbols_file:
                for i, symbol in enumerate(symbols):
                    print(symbol, i, file=symbols_file)

            _LOGGER.debug("Compiling Whisper token strings into language model")
            command_str = (
                " | ".join(
                    shlex.join(command)
                    for command in [
                        [
                            "farcompilestrings",
                            "--keep_symbols",
                            shlex.quote(f"--symbols={symbols_path}"),
                            shlex.quote(str(sentence_tokens_path)),
                        ],
                        ["ngramcount", f"--order={args.ngram_order}"],
                        ["ngrammake", "--method=witten_bell"],
                        ["ngramprint", "--ARPA"],
                    ]
                )
                + " > "
                + shlex.quote(str(raw_arpa_path))
            )

            _LOGGER.debug(command_str)
            subprocess.check_call(
                command_str,
                shell=True,
                env=extended_env,
            )

            # Patch <unk> into ARPA
            ngram_1_pattern = re.compile(r"^ngram 1=(\d+)")
            with (
                open(raw_arpa_path, "r", encoding="utf-8") as raw_arpa_file,
                open(arpa_path, "w", encoding="utf-8") as arpa_file,
            ):
                for line in raw_arpa_file:
                    line = line.strip()
                    if ngram_1_match := ngram_1_pattern.search(line):
                        ngram_1_count = int(ngram_1_match.group(1))
                        print(f"ngram 1={ngram_1_count+1}", file=arpa_file)
                    else:
                        print(line, file=arpa_file)

                        if line.startswith("\\1-grams:"):
                            print(args.unk_logprob, UNK, sep="\t", file=arpa_file)

            _LOGGER.debug("Loading ARPA language model: %s", arpa_path)
            lm = kenlm.Model(str(arpa_path))
            logits_processor = LogitsProcessorList(
                [
                    KenLMLogitsProcessor(
                        lm,
                        vocab,
                        processor=processor,
                        bias=args.bias,
                    )
                ]
            )
            _LOGGER.info("Loaded custom logits processor")

    _LOGGER.debug("Loading model %s", model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, local_files_only=args.local_files_only
    )
    _LOGGER.info("Loaded model: %s", model_name)

    writer = csv.writer(sys.stdout, delimiter="|")
    for wav_path in args.wav_file:
        _LOGGER.debug("Processing %s", wav_path)
        with wave.open(wav_path, "rb") as wav_file:
            if wav_file.getframerate() != SAMPLE_RATE:
                raise ValueError(f"Sample rate of {wav_path} is not {SAMPLE_RATE}")

            if wav_file.getsampwidth() != 2:
                raise ValueError(f"Sample width of {wav_path} is not 2 bytes (16 bits)")

            if wav_file.getnchannels() != 1:
                raise ValueError(f"{wav_path} is not mono")

            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        audio = torch.FloatTensor(audio)
        _LOGGER.debug("Loaded %s sample(s) for %s", len(audio), wav_path)

        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_features = inputs["input_features"]
        _LOGGER.debug(
            "Generated %s feature(s) for %s", input_features.shape[-1], wav_path
        )
        input_features = pad(
            input_features, (0, FEATURES_LENGTH - input_features.shape[-1])
        )

        attention_mask = inputs["attention_mask"]

        predicted_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            num_beams=args.num_beams,
            repetition_penalty=1.2,
        )
        text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
        text = text.strip()

        writer.writerow((wav_path, text))


if __name__ == "__main__":
    main()

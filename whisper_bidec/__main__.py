"""Demo script."""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

from transformers import (
    LogitsProcessorList,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .const import DEFAULT_NGRAM_ORDER, DEFAULT_UNK_LOGPROB
from .corpus import get_logits_processor, load_corpus_from_files
from .decode import decode_wav

_LOGGER = logging.getLogger(__name__)
_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent

DEFAULT_MODEL = "openai/whisper-tiny.en"


def main() -> None:
    """Main entry point."""
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
    parser.add_argument("--bias-towards-lm", type=float, default=0.5)
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search width")
    #
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Don't checking HuggingFace hub for updates every time",
    )
    #
    parser.add_argument("--ngram-order", type=int, default=DEFAULT_NGRAM_ORDER)
    parser.add_argument("--unk-logprob", type=float, default=DEFAULT_UNK_LOGPROB)
    #
    parser.add_argument("wav_file", nargs="+", help="Path to WAV file(s)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    model_name = args.model
    processor_name = args.processor or model_name

    _LOGGER.debug("Loading processor %s", processor_name)
    processor = WhisperProcessor.from_pretrained(
        processor_name, local_files_only=args.local_files_only
    )
    _LOGGER.info("Loaded processor: %s", processor_name)

    logits_processor: Optional[LogitsProcessorList] = None
    if args.text:
        corpus = load_corpus_from_files(args.text, processor)
        logits_processor = LogitsProcessorList(
            [
                get_logits_processor(
                    corpus,
                    processor,
                    args.bias_towards_lm,
                    ngram_order=args.ngram_order,
                    unk_logprob=args.unk_logprob,
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
        text_without_bias = decode_wav(
            model,
            processor,
            wav_path,
            logits_processor=None,
            num_beams=args.num_beams,
        )
        text_with_bias = decode_wav(
            model,
            processor,
            wav_path,
            logits_processor=logits_processor,
            num_beams=args.num_beams,
        )

        writer.writerow((wav_path, text_without_bias, text_with_bias))


if __name__ == "__main__":
    main()

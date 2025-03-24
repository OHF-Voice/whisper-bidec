"""Decoding methods for audio with Whisper."""

import logging
import wave
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.nn.functional import pad
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

FEATURES_LENGTH = 3000  # 30 seconds
SAMPLE_RATE = 16000

_LOGGER = logging.getLogger(__name__)


def decode_wav(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    wav_file: Union[str, Path, wave.Wave_read],
    logits_processor: Optional[Union[LogitsProcessor, LogitsProcessorList]] = None,
    num_beams: int = 1,
) -> str:
    """Decode WAV audio and returns transcript."""
    if not isinstance(wav_file, wave.Wave_read):
        wav_file = wave.open(str(wav_file), "rb")

    with wav_file:
        if wav_file.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Sample rate of WAV file is not {SAMPLE_RATE}")

        if wav_file.getsampwidth() != 2:
            raise ValueError("Sample width of WAV file is not 2 bytes (16 bits)")

        if wav_file.getnchannels() != 1:
            raise ValueError("WAV file is not mono")

        audio_bytes = wav_file.readframes(wav_file.getnframes())

    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    audio_tensor = torch.tensor(audio_float, device=model.device)

    return decode_audio(
        model,
        processor,
        audio_tensor,
        logits_processor=logits_processor,
        num_beams=num_beams,
    )


def decode_audio(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio: torch.Tensor,
    logits_processor: Optional[Union[LogitsProcessor, LogitsProcessorList]] = None,
    num_beams: int = 1,
) -> str:
    """Decode audio as a float tensor and returns transcript.

    Assumes a batch size of 1.
    """
    if isinstance(logits_processor, LogitsProcessor):
        logits_processor = LogitsProcessorList([logits_processor])

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_features = inputs["input_features"]
    _LOGGER.debug("Generated %s feature(s)", input_features.shape[-1])

    # Pad out to 30 seconds
    input_features = pad(
        input_features, (0, FEATURES_LENGTH - input_features.shape[-1])
    )

    attention_mask = inputs["attention_mask"]

    predicted_ids = model.generate(
        input_features=input_features,
        attention_mask=attention_mask,
        logits_processor=logits_processor,
        num_beams=num_beams,
    )
    text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
    text = text.strip()

    return text

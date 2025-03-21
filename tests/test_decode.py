"""Tests for biased decoding."""

from pathlib import Path

import pytest
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_bidec import decode_wav, get_logits_processor, load_corpus_from_sentences

_TESTS_DIR = Path(__file__).parent
_WAV_DIR = _TESTS_DIR / "wav"

TEST_MODEL = "openai/whisper-tiny.en"


@pytest.fixture
def whisper_model() -> WhisperForConditionalGeneration:
    """Load Whisper model."""
    return WhisperForConditionalGeneration.from_pretrained(
        TEST_MODEL, local_files_only=True
    )


@pytest.fixture
def whisper_processor() -> WhisperForConditionalGeneration:
    """Load Whisper processor/tokenizer."""
    return WhisperProcessor.from_pretrained(TEST_MODEL, local_files_only=True)


@pytest.mark.parametrize(
    ("wav_name", "text"),
    (
        ("play the Beatles", "Play the Beatles."),
        (
            "what's the temperature of the EcoBee",
            "What's the temperature of the incubi?",
        ),
    ),
)
def test_unbiased(
    whisper_model: WhisperForConditionalGeneration,  # pylint: disable=redefined-outer-name
    whisper_processor: WhisperProcessor,  # pylint: disable=redefined-outer-name
    wav_name: str,
    text: str,
) -> None:
    """Test decoding without bias."""
    wav_path = _WAV_DIR / f"{wav_name}.wav"
    actual_text = decode_wav(whisper_model, whisper_processor, wav_path)
    assert text == actual_text


@pytest.mark.parametrize(
    ("wav_name", "text"),
    (
        ("play the Beatles", "Play the Beatles."),
        (
            "what's the temperature of the EcoBee",
            "What's the temperature of the EcoBee?",
        ),
    ),
)
def test_biased(
    whisper_model: WhisperForConditionalGeneration,  # pylint: disable=redefined-outer-name
    whisper_processor: WhisperProcessor,  # pylint: disable=redefined-outer-name
    wav_name: str,
    text: str,
) -> None:
    """Test decoding with bias."""
    corpus = load_corpus_from_sentences(
        [
            "What is the temperature of the EcoBee?",
            "What's the temperature of the EcoBee?",
            "Play the White Album.",
            "Play Pink Floyd.",
        ],
        whisper_processor,
    )

    wav_path = _WAV_DIR / f"{wav_name}.wav"
    actual_text = decode_wav(
        whisper_model,
        whisper_processor,
        wav_path,
        logits_processor=get_logits_processor(
            corpus, whisper_processor, bias_towards_lm=0.6
        ),
    )
    assert text == actual_text

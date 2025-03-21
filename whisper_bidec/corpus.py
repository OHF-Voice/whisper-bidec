"""Utilities for loading and tokenizing a text corpus."""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import kenlm
from transformers import WhisperProcessor

from .const import DEFAULT_NGRAM_ORDER, DEFAULT_UNK_LOGPROB
from .logits_processor import KenLMLogitsProcessor
from .ngram import WittenBellNgram

_LOGGER = logging.getLogger(__name__)


@dataclass
class Corpus:
    """Tokenized corpus."""

    sentences: list[list[str]] = field(default_factory=list)
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)


def load_corpus_from_files(
    text_paths: list[Union[str, Path]],
    processor: WhisperProcessor,
) -> Corpus:
    """Load and tokenize a list of text files with example sentences.

    Tokens are "w_{n}" where {n} is the Whisper token id.
    """
    corpus = Corpus()
    for text_path in text_paths:
        _LOGGER.debug("Processing %s", text_path)
        with open(text_path, "r", encoding="utf-8") as text_file:
            for line in text_file:
                line = line.strip()
                if not line:
                    continue

                tokenize_sentence(line, corpus, processor)

    return corpus


def load_corpus_from_sentences(
    sentences: list[str],
    processor: WhisperProcessor,
) -> Corpus:
    """Tokenize a list of example sentences.

    Tokens are "w_{n}" where {n} is the Whisper token id.
    """
    corpus = Corpus()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        tokenize_sentence(sentence, corpus, processor)

    return corpus


def tokenize_sentence(
    sentence: str, corpus: Corpus, processor: WhisperProcessor
) -> None:
    """Tokenize a single sentence and add it to the corpus."""
    # NOTE: Space in front is critical for first word
    token_ids = processor.tokenizer.encode(f" {sentence}", add_special_tokens=False)
    tokens: list[str] = []
    for token_id in token_ids:
        token_str = corpus.id_to_token.get(token_id)
        if token_str is None:
            token_str = f"w_{token_id}"
            corpus.id_to_token[token_id] = token_str
            corpus.token_to_id[token_str] = token_id

        tokens.append(token_str)

    corpus.sentences.append(tokens)


def corpus_to_lm(corpus: Corpus, order: int, unk_logprob: float) -> kenlm.Model:
    """Convert corpus to a KenLM language model using Witten-Bell smoothing."""
    ngram = WittenBellNgram(corpus.sentences, order, unk_logprob)
    with tempfile.NamedTemporaryFile(
        "w+", encoding="utf-8", suffix=".arpa"
    ) as arpa_file:
        ngram.to_arpa(arpa_file)
        arpa_file.seek(0)
        return kenlm.Model(arpa_file.name)


def get_logits_processor(
    corpus: Corpus,
    processor: WhisperProcessor,
    bias_towards_lm: float,
    ngram_order: int = DEFAULT_NGRAM_ORDER,
    unk_logprob: float = DEFAULT_UNK_LOGPROB,
) -> KenLMLogitsProcessor:
    """Get custom logits processor."""
    return KenLMLogitsProcessor(
        corpus_to_lm(corpus, ngram_order, unk_logprob),
        corpus.token_to_id,
        processor=processor,
        bias_towards_lm=bias_towards_lm,
    )

"""Tests for Witten Bell ngram calculations."""

import tempfile
from pathlib import Path

import kenlm

from whisper_bidec.ngram import WittenBellNgram

_TESTS_DIR = Path(__file__).parent


def test_ngram() -> None:
    """Test Witten-Bell smoothing and KenLM loading."""
    corpus = []
    with open(_TESTS_DIR / "ngram_corpus.txt", "r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            line = line.strip()
            if not line:
                continue

            corpus.append(line.split())

    ngram = WittenBellNgram(corpus, 3)

    # Ensure the ARPA can be loaded by KenLM
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as arpa_file:
        ngram.to_arpa(arpa_file)
        arpa_file.seek(0)
        lm = kenlm.Model(arpa_file.name)

    # Sanity check on probabilities.
    # See ngram_corpus.txt
    start_state = kenlm.State()
    lm.BeginSentenceWrite(start_state)

    # the is more likely to start a sentence
    the_state = kenlm.State()
    cat_state = kenlm.State()
    assert lm.BaseScore(start_state, "the", the_state) > lm.BaseScore(
        start_state, "cat", cat_state
    )

    # cat sat is more likely than cat barked
    lm.BaseScore(the_state, "cat", cat_state)

    sat_state = kenlm.State()
    barked_state = kenlm.State()
    assert lm.BaseScore(cat_state, "sat", sat_state) > lm.BaseScore(
        cat_state, "barked", barked_state
    )

    # dog barked is more likely than dog sat
    dog_state = kenlm.State()
    lm.BaseScore(the_state, "dog", dog_state)

    assert lm.BaseScore(dog_state, "barked", barked_state) > lm.BaseScore(
        dog_state, "sat", sat_state
    )

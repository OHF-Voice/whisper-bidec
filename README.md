# Whisper BiDec

**Bi**ased **Dec**oding for Whisper speech-to-text.

This repository implements a custom `LogitsProcessor` for [HuggingFace's implementation][hf_whisper] of the [OpenAI Whisper model][openai_whisper] which uses [KenLM][kenlm] for word probabilities.


## Install

Use `script/setup` or `pip3 install -e .` to install.

If you want to run the tests and linting scripts, use `script/setup --dev` or `pip3 install -e .[dev]` instead.


## Running

Run `python3 -m whisper_bidec --text <text_file> <wav_file> [<wav_file>...]` to transcribe WAV files and get CSV output like:

``` csv
path_to_1.wav|text 1 without bias|text 1 with bias
path_to_2.wav|text 2 without bias|text 2 with bias
...
```

The text file should contain a list of sentences that you want to bias Whisper towards. These need to have the correct casing and punctuation. You can add multiple `--text` files.

Increase `--bias-towards-lm` to get transcripts more like the example sentences (default: 0.5).

Increase `--unk-logprob` to allow more words outside of the example sentences (default: -5, must be less than 0) or decrease it to restrict words to example sentences (e.g., -10).


## Method

To bias Whisper, an [n-gram language model][ngram] is trained on the example sentences. Because Whisper uses **tokens** internally (partial and whole words), the n-gram model is trained on the **tokenized sentences**. By default, this is a 5th order n-gram model (change with `--ngram-order`) with Witten-Bell smoothing.

During Whisper's beam search, the scores for the next tokens are computed using both the Whisper model's and the language model's probabilities. Setting `--bias-towards-lm 0.5` (the default) means that the probabilities from the language model will be halved before being added to Whisper's probabilities.

When computing the language model probabilities, tokens outside of the example sentences are replaced with an "unknown" token (`<unk>`) whose (log) probability is set by `--unk-logprob`. A lower value will penalize sentences that use "unknown" tokens, while a value closer to 0 will allow them.

Some thoughts on the method:

* Witten-Bell smoothing seems to do the best, but there may be better methods. Additionally, the implementation of it was mostly written by ChatGPT so it hasn't been fully verified.
* Casing is a problem right now since Whisper wants to output case-sensitive sentences. It may be worth considering tokens that only differ in case as identical.

[hf_whisper]: https://huggingface.co/docs/transformers/model_doc/whisper
[openai_whisper]: https://github.com/openai/whisper
[kenlm]: https://github.com/kpu/kenlm
[ngram]: https://en.wikipedia.org/wiki/Word_n-gram_language_model

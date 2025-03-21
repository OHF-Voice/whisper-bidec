ngram_corpus.arpa: ngram_corpus.txt ngram_symbols.txt
	farcompilestrings --keep_symbols --symbols=ngram_symbols.txt $< | \
        ngramcount --order=3 | \
        ngrammake --method=witten_bell | \
        ngramprint --ARPA > $@

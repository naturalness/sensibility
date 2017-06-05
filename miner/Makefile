# Putting the n in n-gram:
ORDER = 4

# Augment path wih applications in ./bin/
PATH := $(PWD)/bin:$(PATH)

# Automatically figure out the shuf(1) utility, even on macOS!
SHUF := $(shell which shuf || which gshuf)

# KenLM
KENLMBIN = $(HOME)/.kenlm/bin
ESTIMATENGRAM = $(KENLMBIN)/lmplz
BUILDBINARY = $(KENLMBIN)/build_binary

all:
	@echo "Targets:"
	@echo "\t" "parse"
	@echo "\t" "lm"

parse: unparsed.list
	parallel --pipepart --round-robin parse-and-insert-all :::: $<

lm: corpus.binary

################################################################################

# Create a binary n-gram model with modkn backoffs, for querying.
%.binary: %.arpa
	$(BUILDBINARY) $< $@

# Estimate an n-gram langauge model from sentences
%.arpa: %.sentences
	$(ESTIMATENGRAM) -o $(ORDER) <$< >$@

corpus.sentences: corpus.list
	parallel --pipepart --line-buffer --round-robin vocabularize :::: $< > $@

unparsed.list:
	list-unparsed-sources > $@

vocabulary.py:
	list-elligible-sources | discover-vocabulary > $@

corpus.list:
	list-elligible-sources | $(SHUF) | head -n10000 > $@


.PHONY: all parse lm
.INTERMEDIATE: unparsed

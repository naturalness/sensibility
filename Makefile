# Augment path wih applications in ./bin/
PATH := $(PWD)/bin:$(PATH)

# Automatically figure out the shuf(1) utility, even on macOS!
SHUF := $(shell which shuf || which gshuf)

ESTIMATENGRAM = false

all:
	@echo "Use the other targets."

parse: unparsed.list
	parallel --pipepart --round-robin parse-and-insert-all :::: $<

lm: corpus.arpa

################################################################################

# Estimate an n-gram langauge model from sentences
%.arpa: %.sentences
	$(ESTIMATENGRAM) < $< -o $@

corpus.sentences: corpus.list
	parallel run-pipeline :::: $< > $@

unparsed.list:
	list-unparsed-sources > $@

vocabulary.py:
	list-elligible-sources | discover-vocabulary > $@

corpus.list:
	list-elligible-sources | $(SHUF) | head -n10000 > $@


.PHONY: all parse lm
.INTERMEDIATE: unparsed

# Copyright 2017 Eddie Antonio Santos <easantos@ualberta.ca>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Putting the n in n-gram:
ORDER = 4

# How many training folds?
FOLDS = 5

# Augment path wih applications in ./bin/
PATH := $(PWD)/bin:$(PATH)

# KenLM stuff. Assumes KenLM executables are installed in ~/.kenlm/bin
KENLMBIN = $(HOME)/.kenlm/bin
ESTIMATENGRAM = $(KENLMBIN)/lmplz
BUILDBINARY = $(KENLMBIN)/build_binary

# Old training stuff!

# 75 million, which is a semi-arbitrarily chosen number.
# My last training phase used 9 folds of around 7.5 million tokens each; hencem
# this value is 10 times that (so the same amount of tokens, plus a few more).
TOKENS_PER_FOLD_TRAINING = 75000000
TOKENS_PER_FOLD_VALIDATION = 25000000

HIDDEN_LAYERS = 300,300
CONTEXT = 20

DATA_DIR = data
MODEL_DIR = models

CORPUS = javascript
SOURCES = $(DATA_DIR)/$(CORPUS)-sources.sqlite3
VECTORS = $(DATA_DIR)/$(CORPUS)-vectors.sqlite3
TEST_SET = $(DATA_DIR)/test_set_hashes

# Always use the GNU versions of shuf(1) and split(1)
# shuf(1) isn't installed as `shuf` on all systems (e.g., macOS...)
SHUF = $(shell which shuf || which gshuf)
SPLIT = $(shell which gsplit || which split)

# Make settings
# See: https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
.PHONY: all
.SECONDARY:

all: results

parse: unparsed.list
	parallel --pipepart --round-robin parse-and-insert-all :::: $<

lm: corpus.binary

# This will include a LOT of rules to make models, mutations, and results.
include extra-rules.mk
# So many in fact, I've written a script to generate all the rules.
%.mk: %.pl
	perl $< > $@

.PHONY: results
results: results.csv
results.%.csv:
	bin/evaluate $*

.PHONY: test-sets
test-sets: $(TEST_SET).0 $(TEST_SET).1 $(TEST_SET).2 $(TEST_SET).3 $(TEST_SET).4
# Create the entire test set.
$(TEST_SET): $(VECTORS) $(SOURCES)
	bin/print-test-set $(VECTORS) | $(SHUF) > $@

# Split the giant test set into several files.
# The pattern rule is a hack that allows one recipe to make several targets,
# but does not run the recipe several times.
# From: http://stackoverflow.com/a/3077254/6626414
$(TEST_SET)%0 $(TEST_SET)%1 $(TEST_SET)%2 $(TEST_SET)%3 $(TEST_SET)%4: $(TEST_SET)
	$(SPLIT) --number=l/$(FOLDS) -d --suffix-length=1 $(TEST_SET) $(TEST_SET).

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

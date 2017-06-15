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

# Augment path wih applications in ./bin/
PATH := $(PWD)/bin:$(PATH)

# Putting the n in n-gram:
ORDER = 4

HIDDEN_LAYERS = 300,300
CONTEXT = 20
LANGUAGE := $(shell bin/language-id)

# KenLM stuff. Assumes KenLM executables are installed in ~/.kenlm/bin
KENLMBIN = $(HOME)/.kenlm/bin
ESTIMATENGRAM = $(KENLMBIN)/lmplz
BUILDBINARY = $(KENLMBIN)/build_binary

# Always use the GNU versions of shuf(1) and split(1)
# shuf(1) isn't installed as `shuf` on all systems (e.g., macOS...)
SHUF = $(shell which shuf || which gshuf)

# Make settings
# See: https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
.PHONY: all
.SECONDARY:

all: models

parse: unparsed.list
	parallel --pipepart --round-robin parse-and-insert-all :::: $<

lm: corpus.binary

# This will include a LOT of rules to make models, mutations, and results.
include extra-rules.mk
# So many in fact, I've written a script to generate all the rules.
%.mk: %.pl
	perl $< > $@

################################################################################

# Create a binary n-gram model with modkn backoffs, for querying.
%.binary: %.arpa
	$(BUILDBINARY) $< $@

# Estimate an n-gram langauge model from sentences
%.arpa: %.sentences
	$(ESTIMATENGRAM) -o $(ORDER) <$< >$@

# Create a list of sentences from the corpus.
corpus.sentences: corpus.list
	parallel --pipepart --line-buffer --round-robin vocabularize :::: $< > $@

# Create a list of unparsed sources.
unparsed.list:
	list-unparsed-sources > $@

# Create the vocabulary.
vocabulary.py:
	list-elligible-sources | discover-vocabulary > $@

# Create a list of file hashes.
corpus.list:
	list-elligible-sources | $(SHUF) | head -n10000 > $@


.PHONY: all parse lm
.INTERMEDIATE: unparsed

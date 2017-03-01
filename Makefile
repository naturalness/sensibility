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

# A directory on a fast filesystem (e.g., a ramdisk)
FAST_DIR = /dev/shm

# How many training folds?
FOLDS = 5

# 75 million, which is a semi-arbitrarily chosen number.
# My last training phase used 9 folds of around 7.5 million tokens each; hencem
# this value is 10 times that (so the same amount of tokens, plus a few more).
TOKENS_PER_FOLD_TRAIN = 75000000
TOKENS_PER_FOLD_VALID = 25000000

CORPUS = javascript
SOURCES = $(CORPUS)-sources.sqlite3
VECTORS = $(CORPUS).sqlite3
ASSIGNED_VECTORS = $(FAST_DIR)/$(VECTORS)
TEST_SET = test_set_hashes

# Always use the GNU versions of shuf(1) and split(1)
# shuf(1) isn't installed as `shuf` on all systems (e.g., macOS...)
SHUF = $(shell which shuf || which gshuf)
SPLIT = $(shell which gsplit || which split)

# Make settings
# See: https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
.PHONY: all
.SECONDARY:

all: results

.PHONY: results rm-results
results: results.csv

results.csv: results.1.csv results.2.csv results.3.csv results.4.csv results.5.csv results.6.csv results.7.csv results.8.csv
	./concat-results.sh $^ > $@

results.%.csv:
	./evaluate.py $*

rm-results:
	$(RM) $(wildcard results.*.csv)

# This will include a LOT of rules to make models.
include all-models.mk
# So many in fact, I've written a script to generate all the rules.
all-models.mk: all-models.pl
	perl $< > $@

# Assign files to folds. Create a vectorized corpus suitable for training.
$(ASSIGNED_VECTORS): $(VECTORS)
	cp $(VECTORS) $(ASSIGNED_VECTORS)
	chmod u+w $(ASSIGNED_VECTORS)
	./place_into_folds.py --overwrite --folds 10 --min-tokens $(TOKENS_PER_FOLD) $(ASSIGNED_VECTORS)

$(TEST_SET): $(ASSIGNED_VECTORS) $(SOURCES)
	./print_test_set.py $(ASSIGNED_VECTORS) $(SOURCES) | $(SHUF) > $@

# The pattern rule is a hack that allows one recipe to make several targets,
# but does not run the recipe several times.
# From: http://stackoverflow.com/a/3077254/6626414
$(TEST_SET)%0 $(TEST_SET)%1 $(TEST_SET)%2 $(TEST_SET)%3 $(TEST_SET)%4: $(TEST_SET)
	$(SPLIT) --number=l/$(FOLDS) -d --suffix-length=1 $(TEST_SET) $(TEST_SET).

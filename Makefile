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

# tokens per fold = total number of tokens * 5% / number of folds 
# 1,578,048,815 * 5% / 10 = 7,890,244
TOKENS_PER_FOLD = 7890244

CORPUS = javascript
VECTORS = $(CORPUS).sqlite3
ASSIGNED_VECTORS = $(FAST_DIR)/$(VECTORS)

all: models

# This will include a LOT of rules to make models.
include all-models.mk

# So many in fact, I've written a script to generate all the rules.
all-models.mk: all-models.pl
	perl $< > $@

# Assign files to folds. Create a vectorized corpus suitable for training.
$(ASSIGNED_VECTORS): $(VECTORS)
	cp $(VECTORS) $(ASSIGNED_VECTORS).tmp
	chmod u+w $(ASSIGNED_VECTORS).tmp
	./place_into_folds.py --overwrite --folds 10 --min-tokens $(TOKENS_PER_FOLD) $(ASSIGNED_VECTORS).tmp
	mv -f $(ASSIGNED_VECTORS).tmp $@

.PHONY: all
# Do not automatically delete ANY prerequisites!
.SECONDARY:

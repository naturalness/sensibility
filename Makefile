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

# Make settings
# See: https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
.PHONY: all
.SECONDARY:

all: experiments

# This adds the 'experiments' rule, and all associated rules
# (there's a lot of them!)
include experiments.mk
experiments.mk: libexec/experiments
	$< > $@

################################################################################

# When a language is active (see bin/shell),
# allow the generation of the active language's vocabulary using:
#
#  	make vocabulary
#
ifdef SENSIBILITY_LANGUAGE
VOCABULARY := sensibility/language/$(shell language-id)/vocabulary.txt
# GNU parallel's --pipepart requires a seekable file, so dump the elligible
# sources in this temporary file:
HASHES_FILE := $(shell mktemp -u hashes.XXXXX)
$(VOCABULARY):
	sensibility sources list-eligible > $(HASHES_FILE)
	parallel --pipepart --round-robin -a $(HASHES_FILE)\
		sensibility sources discover-vocabulary | sort -u > $@
vocabulary: $(VOCABULARY)
.PHONY: vocabulary
endif

################################################################################

# Tools for generating data in a Josh-approved manner.

# Josh is a phony
.PHONY: joshua
joshua: training.squashfs mistakes.squashfs

%.squashfs: %
	mksquashfs $< $@ -comp xz

training: $(addprefix training/,java python javascript)

training/%: %-sources.sqlite3
	sensibility joshcompat export-sources $@ $<

mistakes:
	sensibility joshcompat create-mistakes-dir $@

mistakes.txt: mistakes
	sensibility joshcompat list-mistakes-path $< > $@


# Generates a list of paths.
ifdef SENSIBILITY_LANGUAGE
language-id := $(shell language-id)
# Also allow the creation of partitions
partition-paths:
	for i in {0..4} ; do \
		sensibility sources path --prefix="$(language-id)/" \
		<evaluation/$(language-id)/partitions/$$i/training\
		>training/$(language-id)-$$i.txt ;\
	done
.PHONY: partition-paths
endif

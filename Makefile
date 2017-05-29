parse: PATH:=$(PWD)/bin:$(PATH)
parse: unparsed
	parallel --pipepart -a $< --round-robin parse-and-insert-all

unparsed:
	list-unparsed-sources > $@

vocabulary.py:
	bin/list-elligible-sources | bin/discover-vocabulary > $@

.PHONY: parse
.INTERMEDIATE: unparsed

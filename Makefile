parse: PATH:=$(PWD)/bin:$(PATH)
parse: unparsed
	parallel --pipepart -a $< --round-robin parse-and-insert-all

unparsed:
	list-unparsed-sources > $@

.PHONY: parse
.INTERMEDIATE: unparsed

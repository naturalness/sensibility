parse: PATH:=$(PWD)/bin:$(PATH)
parse:
	list-unparsed-sources | xargs -n1 parse-and-insert

.PHONY: parse

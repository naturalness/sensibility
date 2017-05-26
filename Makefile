parse: PATH:=$(PWD)/bin:$(PATH)
parse:
	list-unparsed-sources | parallel --pipe --eta parse-and-insert-all

.PHONY: parse

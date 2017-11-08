dependencies.png:

%.png: %.dot
	dot -Tpng $< > $@
	-optipng $@

%.pdf: %.dot
	dot -Tpdf $< > $@

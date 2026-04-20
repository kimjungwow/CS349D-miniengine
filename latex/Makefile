PDF=report.pdf
TEX=main.tex

.PHONY: all clean

all:
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	mv -f main.pdf $(PDF)

clean:
	find . -maxdepth 1 -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.pdf' \) -delete

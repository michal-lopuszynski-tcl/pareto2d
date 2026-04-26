SHELL=bash

README.md: README.ipynb
	rm -rf README_files
	quarto render README.ipynb --to gfm

README.ipynb: README.py
	jupytext --to ipynb README.py
	quarto render README.ipynb --execute

clean:
	rm -rf README_files README.ipynb README.md

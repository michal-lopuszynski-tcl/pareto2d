SHELL=bash

README.md: README.ipynb
	rm -rf README_files
	quarto render README.ipynb --to gfm

README.ipynb: README.py
	jupytext --execute --to ipynb README.py

clean:
	rm -rf README_files README.ipynb README.md

SHELL=bash

README.md: README.ipynb
	quarto render README.ipynb --to gfm

README.ipynb: README.py
	jupytext --execute --to ipynb README.py

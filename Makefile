MODULE_NAME=pareto2d

PY_BLACK_ISORT=src/pareto2d aux tests setup.py

PY_MYPY_FLAKE8=src/pareto2d aux tests setup.py

FILES_TO_CLEAN=src/pareto2d.egg-info dist


VER_FILE=src/$(MODULE_NAME)/_version.py

F=set_it_to_reformated_file_from_cli

all: build

build:
	rm -rf dist
	python3 -mbuild


check: -check_black -check_isort -check_flake8 -check_mypy

check_fast: -check_black -check_isort -check_flake8

check_black:
	@black --color --check $(PY_BLACK_ISORT) > make.black.log 2>&1 && \
		echo "PASSED black" || \
		(echo "FAILED black"; cat make.black.log; exit 1)

check_isort:
	@isort --check --profile=black $(ISORT_FLAGS) $(PY_BLACK_ISORT) && echo "PASSED isort"


check_flake8:
	@(flake8  --color=always --ignore=E203,W503 --max-line-length 88 $(PY_MYPY_FLAKE8) > make.flake8.log 2>&1) && \
		echo "PASSED flake8" || \
		(echo "FAILED flake8"; cat make.flake8.log; exit 1)

check_mypy:
	@MYPY_FORCE_COLOR=1 mypy --color-output --disallow-incomplete-defs --disallow-untyped-defs $(PY_MYPY_FLAKE8) > make.mypy.log 2>&1 && \
		echo "PASSED mypy" || \
		(echo "FAILED mypy"; cat make.mypy.log; exit 1)

test:
	pytest -vv tests


clean:
	rm -rf make.black.log make.flake8.log make.mypy.log dist $(FILES_TO_CLEAN)

fix:
	ISORT_FLAGS="$(ISORT_FLAGS)" ./aux/fix.sh $(F)

fixall:
	ISORT_FLAGS="$(ISORT_FLAGS)" ./aux/fix.sh $(PY_MYPY_FLAKE8)

commit:
	./aux/block_empty_commit.sh
	git commit -e

show_version:
	@./aux/get_version.py --version-file=$(VER_FILE)

vercommit:
	git restore --staged $(VER_FILE)
	git checkout $(VER_FILE)
	./aux/block_empty_commit.sh
	./aux/update_version.py --version-segment=2 --version-file=$(VER_FILE)
	git add $(VER_FILE)
	-VER=$$(./aux/get_version.py --version-file=$(VER_FILE)) && git commit -m "$${VER}" -e
	git restore --staged  $(VER_FILE) && git checkout $(VER_FILE) # In case of failed commit

manualvercommit:
	./aux/block_empty_commit.sh
	git add $(VER_FILE)
	VER=$$(./aux/get_version.py --version-file=$(VER_FILE)) && git commit -m "$${VER}" -e

-%:
	-@$(MAKE) --no-print-directory $*

.PHONY: all build test clean fix fixall show_version\
	check  check_black check_isort check_flake8 check_mypy \
	commit vercommit manualvercommit


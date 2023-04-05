NB = $(sort $(wildcard *.ipynb))
.PHONY: help clean lint test doc dist release

help:
	@echo "clean    remove non-source files and clean source files"
	@echo "test     run tests and check coverage"
	@echo "dist     package (source & wheel)"
	@echo "release  package and upload to PyPI"

clean:
	git clean -Xdf
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)


test:
	pytest tifresi

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	ls -lh dist/*
	twine check dist/*

release: dist
	twine upload dist/*
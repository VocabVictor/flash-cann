.PHONY: clean clean_dist build install test format lint help

help:
	@echo "Flash-CANN Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  help         - Show this help message"
	@echo "  clean        - Clean build artifacts"
	@echo "  clean_dist   - Clean distribution files"
	@echo "  build        - Build the project"
	@echo "  install      - Install the package"
	@echo "  test         - Run tests"
	@echo "  format       - Format code with black"
	@echo "  lint         - Run linting checks"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.so' -delete
	find . -type f -name '*.o' -delete

clean_dist:
	rm -rf dist/*

build:
	python setup.py build

install:
	pip install -e .

test:
	pytest tests/ -v

format:
	black python/ tests/

lint:
	flake8 python/ tests/
	pylint python/

create_dist: clean_dist
	python setup.py sdist bdist_wheel

upload_package: create_dist
	twine upload dist/*

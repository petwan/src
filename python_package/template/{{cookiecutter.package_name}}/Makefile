# Makefile for {{ cookiecutter.package_name }}
.PHONY: install test lint check format clean

# Install dependencies
install:
	poetry install

# Run tests
test:
	poetry run pytest

# Lint the code with Ruff
lint:
	poetry run ruff check .

# Run type checking
check:
	poetry run mypy .

# Format the code with Ruff
format:
	poetry run ruff format .

# Clean up cache files
clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -type f -name '*.py[co]' -delete
	find . -type f -name '*~' -delete
	find . -type f -name '.coverage' -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Run all checks (lint, type check, tests)
ci: lint check test
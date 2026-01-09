# {{ cookiecutter.package_name }}

{{ cookiecutter.package_description }}

## Features

- Modern Python packaging with Poetry
- CLI entrypoint with Typer (modern alternative to Click with auto-generated help and type hints)
- Pre-configured logging with Loguru (better support for multi-threading and multi-processing)
- Pre-configured constants for common paths (ROOT_PATH, DEFAULT_CONFIG_FILE, etc.)
- Pre-configured testing with pytest
- Fast code linting and formatting with Ruff (replaces Black and Flake8)
- Type checking with MyPy
- Pre-commit hooks with local configuration (faster and more reliable)
- GitHub Actions CI/CD
- Docker and Docker Compose support
- Standard project structure

## Prerequisites

- Python {{ cookiecutter.python_version }}+
- [Poetry](https://python-poetry.org/): `pip install poetry`

## Installation

To install this package, you need to have Poetry installed:

```bash
pip install poetry
```

Then clone the repository and install the dependencies:

```bash
git clone https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.package_name }}.git
cd {{ cookiecutter.package_name }}
poetry install
```

## Quickstart

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Run the example to verify everything works:
   ```bash
   poetry run python example.py
   ```

3. Run the CLI:
   ```bash
   poetry run {{ cookiecutter.package_name }}
   ```

4. Check the version:
   ```bash
   poetry run {{ cookiecutter.package_name }} version
   ```

5. Run tests:
   ```bash
   make test
   # or
   poetry run pytest
   ```

## Pre-commit Hooks

This project uses pre-commit hooks to enforce code quality before commits. To install the pre-commit hooks:

```bash
poetry run pre-commit install
```

The hooks will run automatically on each commit, checking for code formatting, linting issues, and type errors.

## CLI Commands

This project uses Typer to provide a command-line interface with automatic help generation:

- `{{ cookiecutter.package_name }}` - Run the main application
- `{{ cookiecutter.package_name }} version` - Show the application version
- `{{ cookiecutter.package_name }} --help` - Show available commands

## Code Formatting and Linting with Ruff

This project uses Ruff for fast code linting and formatting. To format and lint your code:

```bash
# Format all files
poetry run ruff format .

# Check linting issues
poetry run ruff check .

# Auto-fix linting issues
poetry run ruff check --fix .
```

## Common Paths Constants

This project provides common path constants that can be used throughout the application:

```python
from {{ cookiecutter.package_name }}.utils import (
    ROOT_PATH,
    DEFAULT_CONFIG_FILE,
    DEFAULT_TMP_OUTPUT_PATH,
    DEFAULT_LOG_PATH
)

print(f"Root path: {ROOT_PATH}")
print(f"Config file: {DEFAULT_CONFIG_FILE}")
print(f"Tmp output path: {DEFAULT_TMP_OUTPUT_PATH}")
print(f"Log path: {DEFAULT_LOG_PATH}")
```

## Logging with Loguru

This project uses Loguru for logging, which provides better support for multi-threading and multi-processing than the standard library logging module. To use logging in your code:

```python
from {{ cookiecutter.package_name }}.utils import get_logger

logger = get_logger()
logger.info("This is an info message")
logger.error("This is an error message")
logger.debug("This is a debug message")
```

## Development Commands

The project includes a Makefile with helpful commands:

- `make install`: Install dependencies
- `make test`: Run tests
- `make lint`: Lint the code with Ruff
- `make check`: Run type checking
- `make format`: Format the code with Ruff
- `make clean`: Clean up cache files
- `make ci`: Run all checks (lint, type check, tests)

## Project Structure

```
{{ cookiecutter.package_name }}/
├── {{ cookiecutter.package_name }}/          # Main package
│   ├── __init__.py              # Package init
│   ├── cli/                     # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py              # CLI entry point with Typer
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── constants.py          # Common path constants
│       ├── exceptions.py         # Custom exceptions
│       ├── logging_config.py     # Logging configuration with Loguru
│       └── type_utils.py         # Type definitions
├── tests/                       # Test directory
├── pyproject.toml               # Poetry configuration with Ruff settings
├── README.md                    # This file
├── Makefile                     # Common commands
├── .gitignore                   # Git ignore rules
├── .pre-commit-config.yaml      # Pre-commit hooks with local configuration
└── example.py                   # Example usage
```

## Adding Dependencies

To add new dependencies:

```bash
# Add a runtime dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name
```

## Docker Support

Build and run with Docker:

```bash
docker build -t {{ cookiecutter.package_name }} .
docker run {{ cookiecutter.package_name }}
```

Or with Docker Compose:

```bash
docker-compose up
```

## License

This project is licensed under the {{ cookiecutter.license }} license.
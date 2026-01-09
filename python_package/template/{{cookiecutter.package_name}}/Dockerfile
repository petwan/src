FROM python:{{ cookiecutter.python_version }}-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock README.md ./
COPY {{ cookiecutter.package_name }}/ ./{{ cookiecutter.package_name }}/
COPY tests/ ./tests/

# Install dependencies
RUN poetry install --no-dev

# Install the package in production mode
RUN poetry install
RUN poetry build
RUN pip install dist/*.whl

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

CMD ["{{ cookiecutter.package_name }}"]
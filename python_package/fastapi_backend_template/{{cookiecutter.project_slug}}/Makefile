.PHONY: help install dev test lint format clean migrate-up migrate-down docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies with Poetry"
	@echo "  make dev          - Run development server"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean generated files"
	@echo "  make migrate-up   - Run database migrations"
	@echo "  make migrate-down - Rollback database migration"
	@echo "  make docker-up    - Start Docker containers"
	@echo "  make docker-down  - Stop Docker containers"

install:
	poetry install

dev:
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest -v --cov=app --cov-report=html

lint:
	poetry run ruff check .
	poetry run black --check .
	poetry run mypy app

format:
	poetry run ruff check --fix .
	poetry run black .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

migrate-up:
	alembic upgrade head

migrate-down:
	alembic downgrade -1

create-migration:
	@read -p "Migration name: " name; \
	alembic revision --autogenerate -m "$$name"

{% if cookiecutter.include_celery -%}
celery-worker:
	poetry run celery -A app.worker worker --loglevel=info

celery-beat:
	poetry run celery -A app.worker beat --loglevel=info
{%- endif %}

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

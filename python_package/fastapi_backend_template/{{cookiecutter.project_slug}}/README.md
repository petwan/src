# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## Features

- FastAPI with async support
- PostgreSQL with async SQLAlchemy
- Redis for caching
- Celery for background tasks
- Alembic for database migrations
- JWT authentication
- Docker support
- CI/CD with GitHub Actions
- Comprehensive tests with pytest
- Type hints with mypy
- Code formatting with Black
- Linting with Ruff

## Requirements

- Python {{ cookiecutter.python_version }}+
- Poetry
- Docker (optional)

## Installation

### Using Poetry

```bash
# Install dependencies
poetry install

# Copy environment file
cp .env.example .env

# Edit .env with your configuration
nano .env

# Run database migrations
poetry run alembic upgrade head

# Start development server
poetry run dev
# or
make dev
```

### Using Docker

```bash
# Start services
docker-compose up -d

# Run migrations
docker-compose exec web alembic upgrade head
```

## Project Structure

```
.
├── app/
│   ├── api/              # API endpoints
│   │   └── v1/
│   │       ├── api.py    # API router
│   │       └── endpoints/ # Route handlers
│   ├── core/             # Core configuration
│   │   ├── config.py     # Settings
│   │   ├── security.py   # Security utilities
│   │   ├── database.py   # Database session
│   │   └── logging.py    # Logging config
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   ├── utils/            # Utility functions
│   ├── middleware/       # Custom middleware
│   ├── main.py           # FastAPI application
│   └── worker.py         # Celery worker
├── alembic/              # Database migrations
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── docs/                 # Documentation
└── pyproject.toml        # Poetry dependencies
```

## API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000{{ cookiecutter.api_prefix }}/docs
- ReDoc: http://localhost:8000{{ cookiecutter.api_prefix }}/redoc
- OpenAPI JSON: http://localhost:8000{{ cookiecutter.openapi_url }}

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_users.py
```

### Code Quality

```bash
# Format code
poetry run black .
# or
make format

# Run linters
poetry run ruff check .
poetry run mypy app
# or
make lint
```

### Database Migrations

```bash
# Create new migration
poetry run alembic revision --autogenerate -m "description"

# Apply migrations
poetry run alembic upgrade head

# Rollback migration
poetry run alembic downgrade -1
```

## Environment Variables

See `.env.example` for available configuration options.

## Deployment

### Docker

```bash
# Build image
docker build -t {{ cookiecutter.project_slug }} .

# Run container
docker run -p 8000:8000 {{ cookiecutter.project_slug }}
```

### Production Considerations

- Set `APP_ENV=production` in environment
- Use strong `SECRET_KEY`
- Configure proper database pooling
- Set up proper CORS origins
- Enable HTTPS
- Configure logging for production
- Set up monitoring and alerts

## License

MIT

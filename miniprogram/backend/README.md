# Astrapi

Enterprise-grade FastAPI application built with modern Python best practices.

## Features

- 🚀 FastAPI for high-performance async API
- 🗄️ PostgreSQL with SQLAlchemy 2.0 async support
- 🔐 JWT authentication and authorization
- 📊 Alembic database migrations
- 🎯 Modular architecture with clear separation of concerns
- 📝 Comprehensive logging with Loguru
- 🐳 Docker support for containerization
- 🔒 RBAC-based permission system
- 🧪 Pytest for testing

## Project Structure

```
astrapi/
├── app/
│   ├── alembic/              # Database migrations
│   ├── api/                  # API routes
│   │   └── v1/
│   │       └── system/       # System modules
│   │           ├── auth/     # Authentication module
│   │           └── users/    # Users module
│   ├── core/                 # Core functionality
│   │   ├── config.py         # Configuration
│   │   ├── dependencies.py   # Dependency injection
│   │   ├── exception.py      # Exception handling
│   │   ├── logging.py        # Logging configuration
│   │   ├── permission.py     # Permission management
│   │   ├── response.py       # Response models
│   │   └── security.py       # Security utilities
│   ├── database/             # Database layer
│   │   ├── crud.py           # Base CRUD operations
│   │   ├── model.py          # Base model
│   │   └── session.py        # Database session
│   ├── modules/              # Business modules
│   └── utils/                # Utility functions
├── tests/                    # Test files
├── alembic.ini               # Alembic configuration
├── docker-compose.yml        # Docker compose configuration
├── Dockerfile                # Docker image definition
├── main.py                   # Application entry point
└── pyproject.toml            # Poetry dependencies
```

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- Poetry (optional)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd astrapi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# or with poetry
poetry install
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run database migrations:
```bash
alembic upgrade head
```

5. Start the development server:
```bash
python main.py
```

### Docker Setup

```bash
docker-compose up -d
```

## API Documentation

Once the server is running, access:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

See `.env.example` for all available configuration options.

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
ruff format .
ruff check .
```

### Type Checking

```bash
mypy app/
```

## License

MIT

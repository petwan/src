# Deployment Guide

## Prerequisites

- Python {{ cookiecutter.python_version }}+
- PostgreSQL 15+
- Redis 7+
- Docker and Docker Compose (optional)

## Environment Configuration

Create a production `.env` file:

```bash
APP_ENV=production
DEBUG=false
SECRET_KEY=<your-super-secret-key-at-least-32-characters>

# Database
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/dbname
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://host:6379/0

# CORS
CORS_ORIGINS=["https://yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true
```

## Database Setup

```bash
# Install alembic
poetry install

# Run migrations
poetry run alembic upgrade head
```

## Running with Uvicorn

### Using gunicorn (recommended for production)

```bash
pip install gunicorn uvicorn

gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

### Using uvicorn directly

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

## Docker Deployment

### Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Swarm

```bash
# Deploy stack
docker stack deploy -f docker-compose.yml {{ cookiecutter.project_slug }}

# Scale services
docker service scale {{ cookiecutter.project_slug }}_web=3

# View services
docker service ls
```

## Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ cookiecutter.project_slug }}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {{ cookiecutter.project_slug }}
  template:
    metadata:
      labels:
        app: {{ cookiecutter.project_slug }}
    spec:
      containers:
      - name: web
        image: {{ cookiecutter.project_slug }}:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: url
```

Deploy:

```bash
kubectl apply -f k8s/
```

## Monitoring

### Health Checks

- HTTP health endpoint: `/health`
- Database connection
- Redis connection

### Logging

Configure structured logging in production:

```python
# app/core/logging.py
LOG_FORMAT=json
LOG_LEVEL=INFO
```

### Metrics

Consider adding Prometheus metrics:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

## Security

- Use HTTPS in production
- Set strong `SECRET_KEY`
- Configure CORS properly
- Enable rate limiting
- Use environment variables for secrets
- Regular security updates

## Backup Strategy

### Database

```bash
# Backup
pg_dump -U postgres dbname > backup.sql

# Restore
psql -U postgres dbname < backup.sql
```

### Redis

```bash
# Backup
redis-cli BGSAVE

# Backup file location
/var/lib/redis/dump.rdb
```

## Scaling

### Horizontal Scaling

- Use load balancer (nginx, traefik)
- Run multiple instances
- Use Redis for session storage

### Vertical Scaling

- Increase database pool size
- Add more workers
- Use caching with Redis

## CI/CD

See `.github/workflows/ci.yml` for GitHub Actions configuration.

## Troubleshooting

### Database Connection Issues

```bash
# Check database connection
psql -U postgres -h localhost -d dbname

# Check migration status
poetry run alembic current
poetry run alembic history
```

### Redis Connection Issues

```bash
# Check Redis
redis-cli ping

# Check Redis logs
docker-compose logs redis
```

### Performance Issues

- Check database query performance
- Review logs for errors
- Monitor resource usage
- Add caching for slow queries

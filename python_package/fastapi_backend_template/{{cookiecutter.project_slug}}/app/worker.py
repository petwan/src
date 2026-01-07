"""Celery worker configuration."""
from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "{{ cookiecutter.project_slug }}",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Example task
@celery_app.task
def example_task(name: str) -> str:
    """Example Celery task."""
    return f"Hello, {name}!"

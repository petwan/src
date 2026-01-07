"""Application configuration."""
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "{{ cookiecutter.project_name }}"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    PROJECT_DESCRIPTION: str = "{{ cookiecutter.project_description }}"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "{{ cookiecutter.api_prefix }}"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    ALLOWED_HOSTS: list[str] = ["*"]

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/{{ cookiecutter.project_slug }}_db",
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    ECHO_SQL: bool = False
    AUTO_CREATE_TABLES: bool = False

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-this-in-production-min-32-chars",
        min_length=32,
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    CORS_ALLOW_CREDENTIALS: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    # OpenAPI
    OPENAPI_URL: str = "{{ cookiecutter.openapi_url }}"
    OPENAPI_DOCS_URL: str = "/docs"
    OPENAPI_REDOC_URL: str = "/redoc"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.APP_ENV == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

"""Main application entry point."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.database import Base, engine
from app.core.logging import get_logger, setup_logging
from app.core.redis import close_redis, get_redis
from app.middleware.request_logging import RequestLoggingMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    setup_logging()
    logger.info("Starting application...")
    
    # Test database connection
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
    
    # Auto-create tables in development (production should use Alembic migrations)
    if settings.AUTO_CREATE_TABLES and settings.is_development:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created automatically")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            if not settings.DEBUG:
                raise
            logger.warning("Continuing despite table creation failure (DEBUG mode)")
    elif settings.is_development:
        logger.info("AUTO_CREATE_TABLES is disabled. Run 'alembic upgrade head' to create tables.")
    
    # Test Redis connection
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
    
    logger.info("Application started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Close Redis connection
    try:
        await close_redis()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")
    
    # Dispose database engine
    try:
        await engine.dispose()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")
    
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    openapi_url=settings.OPENAPI_URL,
    docs_url=settings.OPENAPI_DOCS_URL,
    redoc_url=settings.OPENAPI_REDOC_URL,
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    RequestLoggingMiddleware,
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)

if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Welcome to {{ cookiecutter.project_name }}",
        "docs": f"{settings.API_V1_PREFIX}/docs",
        "version": settings.VERSION,
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

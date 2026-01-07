"""Database connection and session management."""
from collections.abc import AsyncGenerator
from typing import AsyncSession

from sqlalchemy.ext.asyncio import (
    AsyncSession as AsyncSessionType,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.ECHO_SQL,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSessionType,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all database models."""
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)


async def get_db() -> AsyncGenerator[AsyncSessionType, None]:
    """Get database session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

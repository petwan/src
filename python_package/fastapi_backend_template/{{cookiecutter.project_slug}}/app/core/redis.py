"""Redis connection management."""
import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings

redis_client: Redis | None = None


async def get_redis() -> Redis:
    """Get Redis client instance."""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return redis_client


async def close_redis() -> None:
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None

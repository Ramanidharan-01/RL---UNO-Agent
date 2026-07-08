"""
app/core/redis_client.py
────────────────────────
A module-level async Redis client.  We store raw bytes (decode_responses=False)
because game state is serialised as binary pickle, not text.

Usage
-----
    redis = await get_redis()
    await redis.setex("key", 60, b"value")
"""
from __future__ import annotations

import redis.asyncio as aioredis

from app.core.config import get_settings

# Module-level singleton – initialised on first call to get_redis().
_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return (and lazily create) the shared Redis client."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            # We manage our own encoding; store raw bytes.
            decode_responses=False,
            # Maintain a small pool for concurrent WebSocket handlers.
            max_connections=20,
        )
    return _redis_client


async def close_redis() -> None:
    """Gracefully close the connection pool (called in app shutdown handler)."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


async def redis_ping() -> bool:
    """Health-check helper used by the /health endpoint."""
    try:
        client = await get_redis()
        return await client.ping()
    except Exception:
        return False

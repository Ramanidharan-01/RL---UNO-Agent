"""
app/db/base.py
──────────────
Async SQLAlchemy engine and session factory.

Usage
-----
    async with get_db_session() as session:
        result = await session.execute(select(User))
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


_engine = None
_session_factory = None


def _get_engine():
    """Lazily create the async engine singleton."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
    return _engine


def _get_session_factory():
    """Lazily create the session factory singleton."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a transactional scope around a series of operations.

    Commits on clean exit, rolls back on exception.
    """
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def close_db() -> None:
    """Dispose of the engine connection pool (called in app shutdown)."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


async def create_tables() -> None:
    """Create all tables (for development / initial setup only)."""
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

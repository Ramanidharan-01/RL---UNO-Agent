"""
app/api/dependencies.py
───────────────────────
FastAPI dependency injection helpers.

Provides:
- ``get_current_user``:   Extract and validate the JWT bearer token.
- ``get_db``:             Yield an async database session.
- ``get_wrapper``:        Retrieve the shared UNOGameWrapper from app state.
"""
from __future__ import annotations

import uuid
from typing import AsyncGenerator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_token
from app.db.base import get_db_session
from app.db.models import User
from app.db.repository import get_user_by_id
from app.game.wrapper import UNOGameWrapper

# ─────────────────────────────────────────────────────────────────────────────
# Security scheme
# ─────────────────────────────────────────────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)


# ─────────────────────────────────────────────────────────────────────────────
# Database session dependency
# ─────────────────────────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session that auto-commits/rollbacks."""
    async with get_db_session() as session:
        yield session


# ─────────────────────────────────────────────────────────────────────────────
# Auth dependencies
# ─────────────────────────────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Validate the Authorization header and return the authenticated User.

    Raises 401 if the token is missing, expired, or references a deleted user.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID in token",
        )

    user = await get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )

    return user


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Same as get_current_user but returns None instead of raising 401."""
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Game wrapper dependency
# ─────────────────────────────────────────────────────────────────────────────

def get_wrapper(request: Request) -> UNOGameWrapper:
    """Retrieve the shared UNOGameWrapper from application state."""
    return request.app.state.game_wrapper

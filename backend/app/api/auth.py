"""
app/api/auth.py
───────────────
Authentication router — registration, login, token refresh, profile.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiry,
    hash_password,
    verify_password,
)
from app.db.models import User
from app.db.repository import (
    create_session,
    create_user,
    get_session_by_token,
    get_user_by_email,
    get_user_by_username,
    revoke_all_user_sessions,
    revoke_session,
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    avatar_url: str | None
    elo_rating: float
    is_verified: bool
    created_at: str

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account",
)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user and return access + refresh tokens."""
    # Check for existing users
    if await get_user_by_email(db, body.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    if await get_user_by_username(db, body.username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken",
        )

    # Create user
    hashed = hash_password(body.password)
    user = await create_user(db, body.email, body.username, hashed)
    await db.flush()

    # Generate tokens
    access = create_access_token(str(user.id), user.username)
    refresh = create_refresh_token(str(user.id))

    # Store refresh token in DB
    expiry = get_token_expiry(refresh)
    if expiry:
        await create_session(db, user.id, refresh, expiry)

    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in with email and password",
)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate and return access + refresh tokens."""
    user = await get_user_by_email(db, body.email)
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    access = create_access_token(str(user.id), user.username)
    refresh = create_refresh_token(str(user.id))

    expiry = get_token_expiry(refresh)
    if expiry:
        await create_session(db, user.id, refresh, expiry)

    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh an access token",
)
async def refresh_token(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Exchange a valid refresh token for a new access + refresh token pair."""
    payload = decode_token(body.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Verify the refresh token hasn't been revoked
    db_session = await get_session_by_token(db, body.refresh_token)
    if db_session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked",
        )

    # Check expiry
    if db_session.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired",
        )

    # Revoke the old refresh token (token rotation)
    await revoke_session(db, body.refresh_token)

    # Issue new tokens
    user_id = payload["sub"]
    # We need the username for the access token
    from app.db.repository import get_user_by_id
    import uuid as _uuid
    user = await get_user_by_id(db, _uuid.UUID(user_id))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    access = create_access_token(user_id, user.username)
    new_refresh = create_refresh_token(user_id)

    expiry = get_token_expiry(new_refresh)
    if expiry:
        await create_session(db, user.id, new_refresh, expiry)

    return TokenResponse(access_token=access, refresh_token=new_refresh)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
async def get_me(user: User = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return UserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        avatar_url=user.avatar_url,
        elo_rating=user.elo_rating,
        is_verified=user.is_verified,
        created_at=user.created_at.isoformat() if user.created_at else "",
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Revoke all refresh tokens",
)
async def logout(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke all refresh tokens for the current user."""
    await revoke_all_user_sessions(db, user.id)
    return MessageResponse(message="Successfully logged out")


@router.post(
    "/password-reset",
    response_model=MessageResponse,
    summary="Request a password reset (stub)",
)
async def password_reset(email: EmailStr, db: AsyncSession = Depends(get_db)):
    """
    Request a password reset email.

    NOTE: This is a stub — email sending is not implemented yet.
    In production, integrate with an email service (SendGrid, SES, etc.).
    """
    user = await get_user_by_email(db, email)
    # Always return success to avoid email enumeration
    return MessageResponse(
        message="If that email is registered, a reset link has been sent."
    )

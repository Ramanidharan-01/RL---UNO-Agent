"""
app/core/security.py
────────────────────
JWT token creation/validation and password hashing utilities.

Uses python-jose for JWT and passlib[bcrypt] for password hashing.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

# ─────────────────────────────────────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ─────────────────────────────────────────────────────────────────────────────
# JWT tokens
# ─────────────────────────────────────────────────────────────────────────────

def create_access_token(
    user_id: str,
    username: str,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a JWT access token.

    Claims include:
        sub: user_id (UUID string)
        username: display name
        type: "access"
        exp: expiration timestamp
        iat: issued-at timestamp
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=settings.access_token_expire_minutes)

    payload = {
        "sub": str(user_id),
        "username": username,
        "type": "access",
        "exp": expires,
        "iat": now,
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(user_id: str) -> str:
    """
    Create a JWT refresh token with a longer expiry.

    Refresh tokens are also stored in the database so they can be revoked.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=settings.refresh_token_expire_days)

    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "jti": str(uuid.uuid4()),  # unique token ID for revocation
        "exp": expires,
        "iat": now,
    }

    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.

    Returns the payload dict if valid, None if expired or malformed.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        return payload
    except JWTError:
        return None


def get_token_expiry(token: str) -> Optional[datetime]:
    """Extract the expiration datetime from a token without full validation."""
    payload = decode_token(token)
    if payload and "exp" in payload:
        return datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ELO Rating
# ─────────────────────────────────────────────────────────────────────────────

def calculate_elo(
    player_elo: float,
    opponent_elo: float,
    won: bool,
    k_factor: float = 32.0,
) -> float:
    """
    Calculate new ELO rating after a game.

    Uses the standard ELO formula with configurable K-factor.
    """
    expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))
    actual = 1.0 if won else 0.0
    return player_elo + k_factor * (actual - expected)

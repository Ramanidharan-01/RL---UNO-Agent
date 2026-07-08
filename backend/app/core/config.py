"""
app/core/config.py
──────────────────
Centralised settings loaded from environment variables (or .env file).
All other modules import `get_settings()` rather than reading env vars directly
so the config is easily overridable in tests.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "UNO Arena API"
    debug: bool = False

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    frontend_url: str = "http://localhost:5173"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    # TTL for a live human-vs-AI game (seconds).
    game_ttl_seconds: int = 7_200   # 2 hours
    # TTL for a server-side simulation session (shorter; auto-cleaned).
    sim_ttl_seconds: int = 600      # 10 minutes

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://uno:uno@localhost:5432/uno_arena"
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    # Path to the pickled checkpoint written by `save_checkpoint()` in uno_jax.
    checkpoint_path: str = "checkpoints/latest.pkl"
    # When True the agent always picks argmax(logits); False → sample.
    agent_deterministic: bool = False

    # ── JWT ───────────────────────────────────────────────────────────────────
    secret_key: str = "CHANGE_ME_IN_PRODUCTION"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 h
    refresh_token_expire_days: int = 30

    # ── WebSocket ─────────────────────────────────────────────────────────────
    ws_heartbeat_interval: int = 30  # seconds

    # ── Simulation defaults ───────────────────────────────────────────────────
    sim_default_speed: float = 1.0   # 1x realtime
    sim_step_delay_ms: int = 1000    # ms between sim steps at 1x

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton – safe to call anywhere."""
    return Settings()

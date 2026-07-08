"""
app/db/models.py
────────────────
SQLAlchemy ORM models for the UNO Arena platform.

Tables are designed for millions of stored games with proper indexes,
foreign keys, and constraints.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


# ─────────────────────────────────────────────────────────────────────────────
# Users & Authentication
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    avatar_url = Column(String(512), nullable=True)
    elo_rating = Column(Float, default=1000.0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    match_players = relationship("MatchPlayer", back_populates="user")
    stats = relationship("UserStats", back_populates="user", uselist=False)

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.email})>"


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    refresh_token = Column(String(512), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False)

    # Relationships
    user = relationship("User", back_populates="sessions")


# ─────────────────────────────────────────────────────────────────────────────
# Matches & Game Data
# ─────────────────────────────────────────────────────────────────────────────

class Match(Base):
    __tablename__ = "matches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mode = Column(String(30), nullable=False, index=True)  # human_vs_ai, agent_vs_random, etc.
    seed = Column(Integer, nullable=True)
    status = Column(
        String(20),
        nullable=False,
        default="in_progress",
        index=True,
    )  # in_progress, completed, abandoned, timeout
    winner_seat = Column(Integer, nullable=True)  # 0-3 or null
    total_turns = Column(Integer, nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    replay_data = Column(JSON, nullable=True)  # Full replay JSON for compact storage

    # Relationships
    players = relationship("MatchPlayer", back_populates="match", cascade="all, delete-orphan")
    turns = relationship("Turn", back_populates="match", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_matches_mode_status", "mode", "status"),
        Index("ix_matches_started_at", "started_at"),
    )


class MatchPlayer(Base):
    __tablename__ = "match_players"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    seat = Column(Integer, nullable=False)  # 0-3
    player_type = Column(String(20), nullable=False)  # human, ai, random, greedy
    display_name = Column(String(50), nullable=True)
    final_hand_size = Column(Integer, nullable=True)
    is_winner = Column(Boolean, default=False, nullable=False)

    # Relationships
    match = relationship("Match", back_populates="players")
    user = relationship("User", back_populates="match_players")

    __table_args__ = (
        UniqueConstraint("match_id", "seat", name="uq_match_seat"),
        Index("ix_match_players_user_id", "user_id"),
    )


class Turn(Base):
    __tablename__ = "turns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"), nullable=False)
    step = Column(Integer, nullable=False)
    player_seat = Column(Integer, nullable=False)
    player_type = Column(String(20), nullable=False)
    action_idx = Column(Integer, nullable=False)
    action_name = Column(String(50), nullable=False)
    ai_value_estimate = Column(Float, nullable=True)
    ai_top_actions = Column(JSON, nullable=True)  # top-5 actions with probabilities
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    match = relationship("Match", back_populates="turns")

    __table_args__ = (
        Index("ix_turns_match_step", "match_id", "step"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Statistics & Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

class UserStats(Base):
    __tablename__ = "user_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    games_played = Column(Integer, default=0, nullable=False)
    wins = Column(Integer, default=0, nullable=False)
    losses = Column(Integer, default=0, nullable=False)
    draws = Column(Integer, default=0, nullable=False)
    win_rate = Column(Float, default=0.0, nullable=False)
    avg_game_length = Column(Float, default=0.0, nullable=False)
    total_cards_played = Column(Integer, default=0, nullable=False)
    total_draw_actions = Column(Integer, default=0, nullable=False)
    wild_cards_played = Column(Integer, default=0, nullable=False)
    current_streak = Column(Integer, default=0, nullable=False)
    best_streak = Column(Integer, default=0, nullable=False)
    fastest_win_turns = Column(Integer, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="stats")


class LeaderboardEntry(Base):
    __tablename__ = "leaderboard_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    elo = Column(Float, default=1000.0, nullable=False, index=True)
    rank = Column(Integer, nullable=True, index=True)
    season = Column(String(20), default="season_1", nullable=False, index=True)
    games_played = Column(Integer, default=0, nullable=False)
    wins = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "season", name="uq_user_season"),
        Index("ix_leaderboard_season_elo", "season", "elo"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multiplayer Lobbies
# ─────────────────────────────────────────────────────────────────────────────

class Lobby(Base):
    __tablename__ = "lobbies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    host_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    max_players = Column(Integer, default=4, nullable=False)
    current_players = Column(Integer, default=1, nullable=False)
    status = Column(String(20), default="waiting", nullable=False, index=True)  # waiting, in_game, closed
    match_id = Column(UUID(as_uuid=True), nullable=True)
    ai_fill = Column(Boolean, default=True, nullable=False)  # fill empty seats with AI
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_lobbies_status_created", "status", "created_at"),
    )

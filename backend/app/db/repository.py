"""
app/db/repository.py
────────────────────
Async repository pattern — CRUD operations for all database models.
All database access goes through this module so the rest of the codebase
never imports SQLAlchemy directly.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    LeaderboardEntry,
    Lobby,
    Match,
    MatchPlayer,
    Session,
    Turn,
    User,
    UserStats,
)


# ─────────────────────────────────────────────────────────────────────────────
# User CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def create_user(
    session: AsyncSession,
    email: str,
    username: str,
    hashed_password: str,
) -> User:
    """Create a new user and initialise their stats row."""
    user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
    )
    session.add(user)
    await session.flush()  # generate user.id

    stats = UserStats(user_id=user.id)
    session.add(stats)

    return user


async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_username(session: AsyncSession, username: str) -> Optional[User]:
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def update_user_elo(session: AsyncSession, user_id: uuid.UUID, new_elo: float) -> None:
    await session.execute(
        update(User).where(User.id == user_id).values(elo_rating=new_elo)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session / Refresh Token CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def create_session(
    session: AsyncSession,
    user_id: uuid.UUID,
    refresh_token: str,
    expires_at: datetime,
) -> Session:
    db_session = Session(
        user_id=user_id,
        refresh_token=refresh_token,
        expires_at=expires_at,
    )
    session.add(db_session)
    return db_session


async def get_session_by_token(
    session: AsyncSession,
    refresh_token: str,
) -> Optional[Session]:
    result = await session.execute(
        select(Session).where(
            Session.refresh_token == refresh_token,
            Session.is_revoked == False,
        )
    )
    return result.scalar_one_or_none()


async def revoke_session(session: AsyncSession, refresh_token: str) -> None:
    await session.execute(
        update(Session)
        .where(Session.refresh_token == refresh_token)
        .values(is_revoked=True)
    )


async def revoke_all_user_sessions(session: AsyncSession, user_id: uuid.UUID) -> None:
    await session.execute(
        update(Session)
        .where(Session.user_id == user_id)
        .values(is_revoked=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Match CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def create_match(
    session: AsyncSession,
    match_id: str,
    mode: str,
    seed: Optional[int] = None,
    players: Optional[List[Dict[str, Any]]] = None,
) -> Match:
    """
    Persist a new match to the database.

    Parameters
    ----------
    players : list of dicts
        Each dict: {"seat": int, "player_type": str, "user_id": UUID|None, "display_name": str}
    """
    match = Match(
        id=uuid.UUID(match_id),
        mode=mode,
        seed=seed,
    )
    session.add(match)
    await session.flush()

    if players:
        for p in players:
            mp = MatchPlayer(
                match_id=match.id,
                user_id=p.get("user_id"),
                seat=p["seat"],
                player_type=p["player_type"],
                display_name=p.get("display_name", f"Player {p['seat']}"),
            )
            session.add(mp)

    return match


async def complete_match(
    session: AsyncSession,
    match_id: str,
    winner_seat: Optional[int],
    total_turns: int,
    replay_data: Optional[Dict] = None,
) -> None:
    """Mark a match as completed with results."""
    now = datetime.now(timezone.utc)
    match_uuid = uuid.UUID(match_id)

    # Get match to compute duration
    result = await session.execute(select(Match).where(Match.id == match_uuid))
    match = result.scalar_one_or_none()
    duration = None
    if match and match.started_at:
        duration = (now - match.started_at.replace(tzinfo=timezone.utc)).total_seconds()

    await session.execute(
        update(Match)
        .where(Match.id == match_uuid)
        .values(
            status="completed",
            winner_seat=winner_seat,
            total_turns=total_turns,
            ended_at=now,
            duration_seconds=duration,
            replay_data=replay_data,
        )
    )

    # Mark the winner in match_players
    if winner_seat is not None:
        await session.execute(
            update(MatchPlayer)
            .where(
                MatchPlayer.match_id == match_uuid,
                MatchPlayer.seat == winner_seat,
            )
            .values(is_winner=True)
        )


async def get_match_by_id(session: AsyncSession, match_id: str) -> Optional[Match]:
    result = await session.execute(
        select(Match).where(Match.id == uuid.UUID(match_id))
    )
    return result.scalar_one_or_none()


async def get_user_matches(
    session: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 20,
    offset: int = 0,
) -> Sequence[Match]:
    """Get recent matches for a user, ordered by most recent first."""
    result = await session.execute(
        select(Match)
        .join(MatchPlayer, Match.id == MatchPlayer.match_id)
        .where(MatchPlayer.user_id == user_id)
        .order_by(Match.started_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


# ─────────────────────────────────────────────────────────────────────────────
# Turn CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def create_turn(
    session: AsyncSession,
    match_id: str,
    step: int,
    player_seat: int,
    player_type: str,
    action_idx: int,
    action_name: str,
    ai_value_estimate: Optional[float] = None,
    ai_top_actions: Optional[List[Dict]] = None,
) -> Turn:
    turn = Turn(
        match_id=uuid.UUID(match_id),
        step=step,
        player_seat=player_seat,
        player_type=player_type,
        action_idx=action_idx,
        action_name=action_name,
        ai_value_estimate=ai_value_estimate,
        ai_top_actions=ai_top_actions,
    )
    session.add(turn)
    return turn


async def create_turns_batch(
    session: AsyncSession,
    match_id: str,
    events: List[Dict[str, Any]],
    start_step: int = 0,
) -> None:
    """Batch insert multiple turns from MoveEvent dicts."""
    for i, event in enumerate(events):
        turn = Turn(
            match_id=uuid.UUID(match_id),
            step=start_step + i,
            player_seat=event["player"],
            player_type=event["player_type"],
            action_idx=event["action_idx"],
            action_name=event["action_name"],
            ai_value_estimate=event.get("value_estimate"),
            ai_top_actions=event.get("top_actions"),
        )
        session.add(turn)


async def get_match_turns(
    session: AsyncSession,
    match_id: str,
) -> Sequence[Turn]:
    result = await session.execute(
        select(Turn)
        .where(Turn.match_id == uuid.UUID(match_id))
        .order_by(Turn.step)
    )
    return result.scalars().all()


# ─────────────────────────────────────────────────────────────────────────────
# Stats CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def get_user_stats(session: AsyncSession, user_id: uuid.UUID) -> Optional[UserStats]:
    result = await session.execute(
        select(UserStats).where(UserStats.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def update_user_stats_after_game(
    session: AsyncSession,
    user_id: uuid.UUID,
    won: bool,
    game_length: int,
    cards_played: int = 0,
    draw_actions: int = 0,
    wilds_played: int = 0,
) -> None:
    """Update user stats after a completed game."""
    stats = await get_user_stats(session, user_id)
    if stats is None:
        stats = UserStats(user_id=user_id)
        session.add(stats)
        await session.flush()

    stats.games_played += 1
    if won:
        stats.wins += 1
        stats.current_streak += 1
        stats.best_streak = max(stats.best_streak, stats.current_streak)
        if stats.fastest_win_turns is None or game_length < stats.fastest_win_turns:
            stats.fastest_win_turns = game_length
    else:
        stats.losses += 1
        stats.current_streak = 0

    stats.win_rate = stats.wins / max(stats.games_played, 1)
    # Running average for game length
    n = stats.games_played
    stats.avg_game_length = ((stats.avg_game_length * (n - 1)) + game_length) / n
    stats.total_cards_played += cards_played
    stats.total_draw_actions += draw_actions
    stats.wild_cards_played += wilds_played


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def get_leaderboard(
    session: AsyncSession,
    season: str = "season_1",
    limit: int = 50,
    offset: int = 0,
) -> Sequence[LeaderboardEntry]:
    result = await session.execute(
        select(LeaderboardEntry)
        .where(LeaderboardEntry.season == season)
        .order_by(LeaderboardEntry.elo.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


async def upsert_leaderboard(
    session: AsyncSession,
    user_id: uuid.UUID,
    new_elo: float,
    season: str = "season_1",
    won: bool = False,
) -> None:
    """Create or update a leaderboard entry after a game."""
    result = await session.execute(
        select(LeaderboardEntry).where(
            LeaderboardEntry.user_id == user_id,
            LeaderboardEntry.season == season,
        )
    )
    entry = result.scalar_one_or_none()

    if entry is None:
        entry = LeaderboardEntry(
            user_id=user_id,
            elo=new_elo,
            season=season,
            games_played=1,
            wins=1 if won else 0,
        )
        session.add(entry)
    else:
        entry.elo = new_elo
        entry.games_played += 1
        if won:
            entry.wins += 1


# ─────────────────────────────────────────────────────────────────────────────
# Lobby CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def create_lobby(
    session: AsyncSession,
    name: str,
    host_id: uuid.UUID,
    max_players: int = 4,
    ai_fill: bool = True,
) -> Lobby:
    lobby = Lobby(
        name=name,
        host_id=host_id,
        max_players=max_players,
        ai_fill=ai_fill,
    )
    session.add(lobby)
    await session.flush()
    return lobby


async def get_open_lobbies(session: AsyncSession) -> Sequence[Lobby]:
    result = await session.execute(
        select(Lobby)
        .where(Lobby.status == "waiting")
        .order_by(Lobby.created_at.desc())
    )
    return result.scalars().all()


async def get_lobby_by_id(session: AsyncSession, lobby_id: str) -> Optional[Lobby]:
    result = await session.execute(
        select(Lobby).where(Lobby.id == uuid.UUID(lobby_id))
    )
    return result.scalar_one_or_none()


async def update_lobby_status(
    session: AsyncSession,
    lobby_id: str,
    status: str,
    match_id: Optional[str] = None,
) -> None:
    values: Dict[str, Any] = {"status": status}
    if match_id is not None:
        values["match_id"] = uuid.UUID(match_id)
    await session.execute(
        update(Lobby).where(Lobby.id == uuid.UUID(lobby_id)).values(**values)
    )

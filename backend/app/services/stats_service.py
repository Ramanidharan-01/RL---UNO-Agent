"""
app/services/stats_service.py
─────────────────────────────
Statistics aggregation, ELO calculation, and leaderboard ranking.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from app.core.security import calculate_elo
from app.db.base import get_db_session
from app.db.repository import (
    get_leaderboard,
    get_user_by_id,
    get_user_matches,
    get_user_stats,
    update_user_elo,
    upsert_leaderboard,
)


class StatsService:
    """Service for player statistics and leaderboard operations."""

    async def get_player_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive stats for a player."""
        uid = uuid.UUID(user_id)
        async with get_db_session() as db:
            stats = await get_user_stats(db, uid)
            user = await get_user_by_id(db, uid)

            if stats is None or user is None:
                return None

            return {
                "user_id": user_id,
                "username": user.username,
                "elo_rating": user.elo_rating,
                "games_played": stats.games_played,
                "wins": stats.wins,
                "losses": stats.losses,
                "draws": stats.draws,
                "win_rate": round(stats.win_rate * 100, 1),
                "avg_game_length": round(stats.avg_game_length, 1),
                "total_cards_played": stats.total_cards_played,
                "total_draw_actions": stats.total_draw_actions,
                "wild_cards_played": stats.wild_cards_played,
                "current_streak": stats.current_streak,
                "best_streak": stats.best_streak,
                "fastest_win_turns": stats.fastest_win_turns,
            }

    async def get_match_history(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get paginated match history for a player."""
        uid = uuid.UUID(user_id)
        async with get_db_session() as db:
            matches = await get_user_matches(db, uid, limit, offset)
            return [
                {
                    "match_id": str(m.id),
                    "mode": m.mode,
                    "status": m.status,
                    "winner_seat": m.winner_seat,
                    "total_turns": m.total_turns,
                    "started_at": m.started_at.isoformat() if m.started_at else None,
                    "ended_at": m.ended_at.isoformat() if m.ended_at else None,
                    "duration_seconds": m.duration_seconds,
                }
                for m in matches
            ]

    async def get_leaderboard_page(
        self,
        season: str = "season_1",
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get paginated leaderboard entries."""
        async with get_db_session() as db:
            entries = await get_leaderboard(db, season, limit, offset)
            result = []
            for i, entry in enumerate(entries):
                user = await get_user_by_id(db, entry.user_id)
                result.append({
                    "rank": offset + i + 1,
                    "user_id": str(entry.user_id),
                    "username": user.username if user else "Unknown",
                    "avatar_url": user.avatar_url if user else None,
                    "elo": round(entry.elo, 1),
                    "games_played": entry.games_played,
                    "wins": entry.wins,
                    "win_rate": round(entry.wins / max(entry.games_played, 1) * 100, 1),
                })
            return result

    async def update_elo_after_match(
        self,
        user_id: str,
        opponent_elo: float,
        won: bool,
        season: str = "season_1",
    ) -> float:
        """
        Update a player's ELO rating after a match.

        For AI opponents, uses a fixed ELO (e.g., 1200 for trained agent).
        Returns the new ELO rating.
        """
        uid = uuid.UUID(user_id)
        async with get_db_session() as db:
            user = await get_user_by_id(db, uid)
            if user is None:
                return 0.0

            new_elo = calculate_elo(user.elo_rating, opponent_elo, won)
            await update_user_elo(db, uid, new_elo)
            await upsert_leaderboard(db, uid, new_elo, season, won)
            return new_elo

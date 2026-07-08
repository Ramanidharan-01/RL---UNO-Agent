"""
app/services/game_service.py
────────────────────────────
Orchestrates game wrapper calls and database persistence.

This is the bridge between the stateless FastAPI handlers and the
JAX game engine. Every public method:
1. Delegates game logic to UNOGameWrapper (Redis + JAX).
2. Persists results to PostgreSQL for history/stats/replays.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.db.base import get_db_session
from app.db.repository import (
    complete_match,
    create_match as db_create_match,
    create_turns_batch,
    get_match_turns,
    update_user_stats_after_game,
)
from app.game.wrapper import MatchMode, UNOGameWrapper


class GameService:
    """
    High-level service for Human-vs-AI games.

    One instance is created at app startup and shared across handlers.
    """

    def __init__(self, wrapper: UNOGameWrapper) -> None:
        self.wrapper = wrapper

    async def create_game(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        mode: str = MatchMode.HUMAN_VS_AI,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a new match and persist it to both Redis and PostgreSQL.

        Returns the match_id and initial frontend state.
        """
        match_id, frontend_state = await self.wrapper.create_match(
            mode=mode,
            seed=seed,
            human_seat=0,
        )

        # Persist match metadata to PostgreSQL
        players = [
            {"seat": 0, "player_type": "human", "user_id": user_id, "display_name": username or "You"},
            {"seat": 1, "player_type": "ai", "user_id": None, "display_name": "AI Agent 1"},
            {"seat": 2, "player_type": "ai", "user_id": None, "display_name": "AI Agent 2"},
            {"seat": 3, "player_type": "ai", "user_id": None, "display_name": "AI Agent 3"},
        ]
        async with get_db_session() as db:
            await db_create_match(db, match_id, mode, seed, players)

        return {"match_id": match_id, **frontend_state}

    async def play_action(
        self,
        match_id: str,
        action_idx: int,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a human action, run AI opponents, persist turns.

        Returns the updated frontend state with events, or None if
        the match is not found or it's not the human's turn.
        """
        result = await self.wrapper.apply_human_action(match_id, action_idx)
        if result is None:
            return None

        events = result.get("events", [])

        # Get current step count for turn numbering
        async with get_db_session() as db:
            existing_turns = await get_match_turns(db, match_id)
            start_step = len(existing_turns)
            await create_turns_batch(db, match_id, events, start_step)

            # If game is done, finalize
            if result.get("done"):
                winner_seat = result.get("winner")
                total_turns = start_step + len(events)

                # Get full history for replay
                history = await self.wrapper.get_history(match_id)
                await complete_match(
                    db, match_id, winner_seat, total_turns, replay_data={"events": history}
                )

                # Update user stats if authenticated
                if user_id:
                    import uuid
                    won = winner_seat == 0  # human is always seat 0
                    cards_played = sum(1 for e in history if e.get("player") == 0 and e.get("action_name") != "draw")
                    draw_actions = sum(1 for e in history if e.get("player") == 0 and e.get("action_name") == "draw")
                    await update_user_stats_after_game(
                        db,
                        uuid.UUID(user_id),
                        won=won,
                        game_length=total_turns,
                        cards_played=cards_played,
                        draw_actions=draw_actions,
                    )

        return result

    async def get_state(
        self,
        match_id: str,
        viewing_player: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get current match state from Redis."""
        return await self.wrapper.get_match_state(match_id, viewing_player)

    async def get_history(self, match_id: str) -> List[Dict[str, Any]]:
        """Get full move history from Redis."""
        return await self.wrapper.get_history(match_id)

    async def forfeit(self, match_id: str, user_id: Optional[str] = None) -> bool:
        """Forfeit a match — delete from Redis and mark as abandoned in DB."""
        existed = await self.wrapper.delete_match(match_id)
        if existed:
            async with get_db_session() as db:
                from sqlalchemy import update as sa_update
                from app.db.models import Match
                import uuid
                await db.execute(
                    sa_update(Match)
                    .where(Match.id == uuid.UUID(match_id))
                    .values(status="abandoned")
                )
        return existed

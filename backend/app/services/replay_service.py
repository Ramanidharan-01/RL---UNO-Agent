"""
app/services/replay_service.py
──────────────────────────────
Replay data retrieval and step-by-step reconstruction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.db.base import get_db_session
from app.db.repository import get_match_by_id, get_match_turns


class ReplayService:
    """Service for retrieving and serving match replays."""

    async def get_full_replay(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the full replay data for a completed match.

        Returns match metadata + ordered list of all turns.
        """
        async with get_db_session() as db:
            match = await get_match_by_id(db, match_id)
            if match is None:
                return None

            turns = await get_match_turns(db, match_id)

            return {
                "match_id": match_id,
                "mode": match.mode,
                "seed": match.seed,
                "status": match.status,
                "winner_seat": match.winner_seat,
                "total_turns": match.total_turns,
                "started_at": match.started_at.isoformat() if match.started_at else None,
                "ended_at": match.ended_at.isoformat() if match.ended_at else None,
                "duration_seconds": match.duration_seconds,
                "turns": [
                    {
                        "step": t.step,
                        "player_seat": t.player_seat,
                        "player_type": t.player_type,
                        "action_idx": t.action_idx,
                        "action_name": t.action_name,
                        "ai_value_estimate": t.ai_value_estimate,
                        "ai_top_actions": t.ai_top_actions,
                    }
                    for t in turns
                ],
                # Compact replay data stored at match completion
                "replay_data": match.replay_data,
            }

    async def get_replay_step(
        self,
        match_id: str,
        step: int,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific step from a replay."""
        async with get_db_session() as db:
            turns = await get_match_turns(db, match_id)
            for t in turns:
                if t.step == step:
                    return {
                        "step": t.step,
                        "player_seat": t.player_seat,
                        "player_type": t.player_type,
                        "action_idx": t.action_idx,
                        "action_name": t.action_name,
                        "ai_value_estimate": t.ai_value_estimate,
                        "ai_top_actions": t.ai_top_actions,
                    }
            return None

    async def get_replay_range(
        self,
        match_id: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get a range of steps from a replay."""
        async with get_db_session() as db:
            turns = await get_match_turns(db, match_id)
            filtered = [t for t in turns if t.step >= start]
            if end is not None:
                filtered = [t for t in filtered if t.step <= end]

            return [
                {
                    "step": t.step,
                    "player_seat": t.player_seat,
                    "player_type": t.player_type,
                    "action_idx": t.action_idx,
                    "action_name": t.action_name,
                    "ai_value_estimate": t.ai_value_estimate,
                    "ai_top_actions": t.ai_top_actions,
                }
                for t in filtered
            ]

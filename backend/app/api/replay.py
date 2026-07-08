"""
app/api/replay.py
─────────────────
Replay retrieval endpoints.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.replay_service import ReplayService

router = APIRouter(prefix="/replay", tags=["replay"])

_replay_service = ReplayService()


@router.get(
    "/{match_id}",
    summary="Get full match replay",
)
async def get_replay(match_id: str):
    """Get the full replay data for a completed match."""
    result = await _replay_service.get_full_replay(match_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Match not found")
    return result


@router.get(
    "/{match_id}/step/{step}",
    summary="Get a specific replay step",
)
async def get_replay_step(match_id: str, step: int):
    """Get a specific step from a match replay."""
    result = await _replay_service.get_replay_step(match_id, step)
    if result is None:
        raise HTTPException(status_code=404, detail="Step not found")
    return result


@router.get(
    "/{match_id}/range",
    summary="Get a range of replay steps",
)
async def get_replay_range(
    match_id: str,
    start: int = Query(default=0, ge=0),
    end: Optional[int] = Query(default=None),
):
    """Get a range of steps from a match replay."""
    return await _replay_service.get_replay_range(match_id, start, end)

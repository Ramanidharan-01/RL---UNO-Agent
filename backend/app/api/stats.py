"""
app/api/stats.py
────────────────
Statistics and leaderboard endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_current_user_optional, get_db
from app.db.models import User
from app.services.stats_service import StatsService

router = APIRouter(prefix="/stats", tags=["stats"])

_stats_service = StatsService()


@router.get(
    "/me",
    summary="Get personal statistics",
)
async def get_my_stats(user: User = Depends(get_current_user)):
    """Get the authenticated user's game statistics."""
    result = await _stats_service.get_player_stats(str(user.id))
    if result is None:
        raise HTTPException(status_code=404, detail="Stats not found")
    return result


@router.get(
    "/player/{user_id}",
    summary="Get player statistics",
)
async def get_player_stats(user_id: str):
    """Get game statistics for any player."""
    result = await _stats_service.get_player_stats(user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return result


@router.get(
    "/history",
    summary="Get match history",
)
async def get_match_history(
    user: User = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Get paginated match history for the authenticated user."""
    return await _stats_service.get_match_history(str(user.id), limit, offset)


@router.get(
    "/leaderboard",
    summary="Get leaderboard",
)
async def get_leaderboard(
    season: str = Query(default="season_1"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Get the paginated leaderboard for a season."""
    return await _stats_service.get_leaderboard_page(season, limit, offset)

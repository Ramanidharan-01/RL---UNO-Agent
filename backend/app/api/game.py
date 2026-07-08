"""
app/api/game.py
───────────────
REST endpoints for Human-vs-AI gameplay.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_current_user_optional, get_wrapper
from app.db.models import User
from app.game.wrapper import UNOGameWrapper
from app.services.game_service import GameService

router = APIRouter(prefix="/game", tags=["game"])


# ─── Schemas ──────────────────────────────────────────────────────────────────

class CreateGameRequest(BaseModel):
    mode: str = "human_vs_ai"
    seed: Optional[int] = None


class ActionRequest(BaseModel):
    action_idx: int = Field(ge=0, le=60, description="Action index (0-60)")


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post(
    "/create",
    summary="Create a new Human-vs-AI game",
    status_code=status.HTTP_201_CREATED,
)
async def create_game(
    body: CreateGameRequest,
    request: Request,
    user: User | None = Depends(get_current_user_optional),
):
    """
    Create a new match. Returns match_id and the initial game state.

    Authentication is optional — anonymous users can play but stats
    won't be tracked.
    """
    wrapper = get_wrapper(request)
    service = GameService(wrapper)
    result = await service.create_game(
        user_id=str(user.id) if user else None,
        username=user.username if user else None,
        mode=body.mode,
        seed=body.seed,
    )
    return result


@router.get(
    "/{match_id}",
    summary="Get current game state",
)
async def get_game_state(
    match_id: str,
    request: Request,
    viewing_player: int = 0,
):
    """Get the current state of a match."""
    wrapper = get_wrapper(request)
    service = GameService(wrapper)
    result = await service.get_state(match_id, viewing_player)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found or expired",
        )
    return result


@router.post(
    "/{match_id}/action",
    summary="Submit a player action",
)
async def submit_action(
    match_id: str,
    body: ActionRequest,
    request: Request,
    user: User | None = Depends(get_current_user_optional),
):
    """
    Submit a human action and receive the updated state.

    The response includes an "events" array with all moves that occurred
    (the human's move plus all AI opponent moves until the human's next turn).
    """
    wrapper = get_wrapper(request)
    service = GameService(wrapper)
    result = await service.play_action(
        match_id=match_id,
        action_idx=body.action_idx,
        user_id=str(user.id) if user else None,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Match not found, not your turn, or game is over",
        )
    return result


@router.delete(
    "/{match_id}",
    summary="Forfeit / delete a match",
)
async def forfeit_game(
    match_id: str,
    request: Request,
    user: User | None = Depends(get_current_user_optional),
):
    """Forfeit and delete a match."""
    wrapper = get_wrapper(request)
    service = GameService(wrapper)
    existed = await service.forfeit(
        match_id=match_id,
        user_id=str(user.id) if user else None,
    )
    if not existed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found",
        )
    return {"message": "Match forfeited", "match_id": match_id}


@router.get(
    "/{match_id}/history",
    summary="Get move history",
)
async def get_history(
    match_id: str,
    request: Request,
):
    """Get the full move history for a match."""
    wrapper = get_wrapper(request)
    service = GameService(wrapper)
    history = await service.get_history(match_id)
    return {"match_id": match_id, "events": history}

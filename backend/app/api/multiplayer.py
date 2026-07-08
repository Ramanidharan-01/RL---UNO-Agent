"""
app/api/multiplayer.py
──────────────────────
Multiplayer lobby and real-time gameplay endpoints.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.db.models import User
from app.services.multiplayer_service import MultiplayerService

router = APIRouter(prefix="/multiplayer", tags=["multiplayer"])

# Module-level singleton (created at startup and attached to app.state)
_mp_service: Optional[MultiplayerService] = None


def get_mp_service(request: Request) -> MultiplayerService:
    if not hasattr(request.app.state, "multiplayer_service"):
        request.app.state.multiplayer_service = MultiplayerService()
    return request.app.state.multiplayer_service


# ─── Schemas ──────────────────────────────────────────────────────────────────

class CreateLobbyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    max_players: int = Field(default=4, ge=2, le=4)
    ai_fill: bool = True


class JoinLobbyRequest(BaseModel):
    lobby_id: str


# ─── REST Routes ──────────────────────────────────────────────────────────────

@router.post(
    "/lobby/create",
    summary="Create a multiplayer lobby",
    status_code=status.HTTP_201_CREATED,
)
async def create_lobby(
    body: CreateLobbyRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    service = get_mp_service(request)
    result = await service.create_lobby_room(
        name=body.name,
        host_id=str(user.id),
        max_players=body.max_players,
        ai_fill=body.ai_fill,
    )
    return result


@router.get(
    "/lobby/list",
    summary="List open lobbies",
)
async def list_lobbies(request: Request):
    service = get_mp_service(request)
    return await service.list_lobbies()


@router.post(
    "/lobby/{lobby_id}/join",
    summary="Join an existing lobby",
)
async def join_lobby(
    lobby_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    service = get_mp_service(request)
    result = await service.join_lobby(lobby_id, str(user.id), user.username)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Lobby is full, closed, or not found",
        )
    return result


@router.get(
    "/lobby/{lobby_id}/players",
    summary="Get lobby players",
)
async def lobby_players(lobby_id: str, request: Request):
    service = get_mp_service(request)
    players = await service.get_lobby_players(lobby_id)
    return {"lobby_id": lobby_id, "players": players}


# ─── WebSocket Gameplay ──────────────────────────────────────────────────────

@router.websocket("/{match_id}/play")
async def multiplayer_game(websocket: WebSocket, match_id: str):
    """
    Real-time multiplayer game WebSocket.

    Protocol:
    - Client sends: {"action": "join", "user_id": "...", "seat": 0}
    - Client sends: {"action": "move", "action_idx": 42}
    - Client sends: {"action": "spectate"}
    - Server sends: {"type": "state", "data": {...}}
    - Server sends: {"type": "move", "data": {...}}
    - Server sends: {"type": "player_joined", "seat": 1, "username": "..."}
    - Server sends: {"type": "player_left", "seat": 2}
    """
    await websocket.accept()

    service: MultiplayerService = websocket.app.state.multiplayer_service
    user_id = None
    seat = None

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            action = data.get("action")

            if action == "join":
                user_id = data.get("user_id")
                seat = data.get("seat", 0)
                service.register_connection(match_id, user_id, seat, websocket)
                await websocket.send_json({
                    "type": "joined",
                    "match_id": match_id,
                    "seat": seat,
                })

                # Notify other players
                for uid, conn in service.get_connections(match_id).items():
                    if uid != user_id:
                        try:
                            await conn["ws"].send_json({
                                "type": "player_joined",
                                "user_id": user_id,
                                "seat": seat,
                            })
                        except Exception:
                            pass

            elif action == "spectate":
                user_id = data.get("user_id", "anon")
                service.add_spectator(match_id, user_id)
                await websocket.send_json({
                    "type": "spectating",
                    "match_id": match_id,
                })

            elif action == "move" and seat is not None:
                action_idx = data.get("action_idx", 60)  # default to draw
                # Delegate to game wrapper
                wrapper = websocket.app.state.game_wrapper
                result = await wrapper.apply_human_action(match_id, action_idx)

                if result:
                    # Broadcast to all connected players and spectators
                    msg = {"type": "state_update", "data": result}
                    for uid, conn in service.get_connections(match_id).items():
                        try:
                            await conn["ws"].send_json(msg)
                        except Exception:
                            pass

            elif action == "heartbeat":
                await websocket.send_json({"type": "heartbeat_ack"})

    except WebSocketDisconnect:
        if user_id:
            service.unregister_connection(match_id, user_id)
            service.remove_spectator(match_id, user_id)

            # Notify remaining players
            for uid, conn in service.get_connections(match_id).items():
                try:
                    await conn["ws"].send_json({
                        "type": "player_left",
                        "user_id": user_id,
                        "seat": seat,
                    })
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

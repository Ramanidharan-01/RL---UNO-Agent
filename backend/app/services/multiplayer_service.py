"""
app/services/multiplayer_service.py
───────────────────────────────────
Lobby management, matchmaking, AI substitution on disconnect,
and spectator tracking for multiplayer games.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Set

from app.core.redis_client import get_redis
from app.db.base import get_db_session
from app.db.repository import (
    create_lobby,
    get_lobby_by_id,
    get_open_lobbies,
    update_lobby_status,
)


class MultiplayerService:
    """
    Manages multiplayer lobby state and player connections.

    Lobby membership and connection tracking is stored in Redis for speed.
    Lobby metadata is also persisted to PostgreSQL for history.
    """

    def __init__(self) -> None:
        # In-memory tracking of active WebSocket connections per match
        # Format: {match_id: {user_id: {"seat": int, "ws": websocket}}}
        self._connections: Dict[str, Dict[str, Any]] = {}
        # Spectators: {match_id: set of user_ids}
        self._spectators: Dict[str, Set[str]] = {}

    async def create_lobby_room(
        self,
        name: str,
        host_id: str,
        max_players: int = 4,
        ai_fill: bool = True,
    ) -> Dict[str, Any]:
        """Create a new lobby and return its details."""
        async with get_db_session() as db:
            lobby = await create_lobby(
                db,
                name=name,
                host_id=uuid.UUID(host_id),
                max_players=max_players,
                ai_fill=ai_fill,
            )
            lobby_id = str(lobby.id)

        # Store lobby players in Redis for fast access
        redis = await get_redis()
        lobby_key = f"lobby:{lobby_id}:players"
        await redis.hset(lobby_key, host_id.encode(), json.dumps({
            "seat": 0,
            "ready": False,
            "username": "Host",
        }).encode())
        await redis.expire(lobby_key, 3600)  # 1 hour TTL

        return {
            "lobby_id": lobby_id,
            "name": name,
            "host_id": host_id,
            "max_players": max_players,
            "current_players": 1,
            "ai_fill": ai_fill,
            "status": "waiting",
        }

    async def join_lobby(
        self,
        lobby_id: str,
        user_id: str,
        username: str,
    ) -> Optional[Dict[str, Any]]:
        """Join an existing lobby. Returns lobby state or None if full/not found."""
        redis = await get_redis()
        lobby_key = f"lobby:{lobby_id}:players"

        # Check how many players are in the lobby
        players = await redis.hgetall(lobby_key)
        if not players:
            return None

        # Check max capacity from DB
        async with get_db_session() as db:
            lobby = await get_lobby_by_id(db, lobby_id)
            if lobby is None or lobby.status != "waiting":
                return None
            if len(players) >= lobby.max_players:
                return None

        # Assign next available seat
        taken_seats = set()
        for p_data in players.values():
            p_info = json.loads(p_data)
            taken_seats.add(p_info["seat"])

        next_seat = None
        for s in range(4):
            if s not in taken_seats:
                next_seat = s
                break
        if next_seat is None:
            return None

        # Add player
        await redis.hset(lobby_key, user_id.encode(), json.dumps({
            "seat": next_seat,
            "ready": False,
            "username": username,
        }).encode())

        # Update count in DB
        async with get_db_session() as db:
            lobby = await get_lobby_by_id(db, lobby_id)
            if lobby:
                lobby.current_players = len(players) + 1

        return {
            "lobby_id": lobby_id,
            "seat": next_seat,
            "current_players": len(players) + 1,
        }

    async def list_lobbies(self) -> List[Dict[str, Any]]:
        """List all open lobbies."""
        async with get_db_session() as db:
            lobbies = await get_open_lobbies(db)
            return [
                {
                    "lobby_id": str(lobby.id),
                    "name": lobby.name,
                    "host_id": str(lobby.host_id) if lobby.host_id else None,
                    "max_players": lobby.max_players,
                    "current_players": lobby.current_players,
                    "ai_fill": lobby.ai_fill,
                    "status": lobby.status,
                }
                for lobby in lobbies
            ]

    async def get_lobby_players(self, lobby_id: str) -> List[Dict[str, Any]]:
        """Get all players currently in a lobby."""
        redis = await get_redis()
        lobby_key = f"lobby:{lobby_id}:players"
        players = await redis.hgetall(lobby_key)
        result = []
        for uid, pdata in players.items():
            info = json.loads(pdata)
            info["user_id"] = uid.decode() if isinstance(uid, bytes) else uid
            result.append(info)
        return sorted(result, key=lambda x: x["seat"])

    def register_connection(self, match_id: str, user_id: str, seat: int, ws: Any) -> None:
        """Register a WebSocket connection for a match player."""
        if match_id not in self._connections:
            self._connections[match_id] = {}
        self._connections[match_id][user_id] = {"seat": seat, "ws": ws}

    def unregister_connection(self, match_id: str, user_id: str) -> None:
        """Unregister a WebSocket connection."""
        if match_id in self._connections:
            self._connections[match_id].pop(user_id, None)
            if not self._connections[match_id]:
                del self._connections[match_id]

    def get_connections(self, match_id: str) -> Dict[str, Any]:
        """Get all active connections for a match."""
        return self._connections.get(match_id, {})

    def add_spectator(self, match_id: str, user_id: str) -> None:
        """Add a spectator to a match."""
        if match_id not in self._spectators:
            self._spectators[match_id] = set()
        self._spectators[match_id].add(user_id)

    def remove_spectator(self, match_id: str, user_id: str) -> None:
        """Remove a spectator from a match."""
        if match_id in self._spectators:
            self._spectators[match_id].discard(user_id)

    def get_spectators(self, match_id: str) -> Set[str]:
        """Get spectator user IDs for a match."""
        return self._spectators.get(match_id, set())

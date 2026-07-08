"""
app/services/simulation_service.py
──────────────────────────────────
Manages AI simulation lifecycle: creation, step-by-step advancement,
speed control, and pause/resume state.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from app.core.config import get_settings
from app.core.redis_client import get_redis
from app.game.wrapper import MatchMode, UNOGameWrapper


class SimulationService:
    """
    Service for AI-only simulation matches.

    Simulation state (paused, speed) is stored in Redis alongside the
    game state so it survives worker restarts.
    """

    def __init__(self, wrapper: UNOGameWrapper) -> None:
        self.wrapper = wrapper
        self._settings = get_settings()

    async def create_simulation(
        self,
        mode: str = MatchMode.AGENT_VS_RANDOM,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new simulation match."""
        if mode not in (MatchMode.AGENT_VS_RANDOM, MatchMode.AGENT_VS_GREEDY):
            mode = MatchMode.AGENT_VS_RANDOM

        match_id, frontend_state = await self.wrapper.create_match(
            mode=mode,
            seed=seed,
            human_seat=0,
        )

        # Store simulation control state in Redis
        redis = await get_redis()
        sim_key = f"sim:{match_id}:control"
        await redis.hset(sim_key, mapping={  # type: ignore[arg-type]
            b"paused": b"0",
            b"speed": str(self._settings.sim_default_speed).encode(),
        })
        await redis.expire(sim_key, self._settings.sim_ttl_seconds)

        return {"match_id": match_id, **frontend_state}

    async def step(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Advance the simulation by one turn."""
        return await self.wrapper.simulation_step(match_id)

    async def get_state(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get current simulation state."""
        return await self.wrapper.get_match_state(match_id, viewing_player=0)

    async def set_paused(self, match_id: str, paused: bool) -> bool:
        """Set the pause state. Returns False if match not found."""
        redis = await get_redis()
        sim_key = f"sim:{match_id}:control"
        if not await redis.exists(sim_key):
            return False
        await redis.hset(sim_key, b"paused", b"1" if paused else b"0")
        return True

    async def is_paused(self, match_id: str) -> bool:
        """Check if the simulation is paused."""
        redis = await get_redis()
        val = await redis.hget(f"sim:{match_id}:control", b"paused")
        return val == b"1"

    async def set_speed(self, match_id: str, speed: float) -> bool:
        """Set simulation speed multiplier (0.25x to 10x)."""
        speed = max(0.25, min(10.0, speed))
        redis = await get_redis()
        sim_key = f"sim:{match_id}:control"
        if not await redis.exists(sim_key):
            return False
        await redis.hset(sim_key, b"speed", str(speed).encode())
        return True

    async def get_speed(self, match_id: str) -> float:
        """Get the current simulation speed multiplier."""
        redis = await get_redis()
        val = await redis.hget(f"sim:{match_id}:control", b"speed")
        if val is None:
            return self._settings.sim_default_speed
        return float(val)

    async def get_step_delay(self, match_id: str) -> float:
        """Calculate the delay between steps in seconds based on speed."""
        speed = await self.get_speed(match_id)
        base_delay = self._settings.sim_step_delay_ms / 1000.0
        return base_delay / speed

    async def get_history(self, match_id: str):
        """Get full simulation history."""
        return await self.wrapper.get_history(match_id)

    async def delete(self, match_id: str) -> bool:
        """Delete simulation and its control state."""
        redis = await get_redis()
        await redis.delete(f"sim:{match_id}:control")
        return await self.wrapper.delete_match(match_id)

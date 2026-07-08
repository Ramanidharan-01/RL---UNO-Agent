"""
app/api/simulation.py
─────────────────────
REST + WebSocket endpoints for AI simulation mode.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_wrapper
from app.game.wrapper import MatchMode, UNOGameWrapper
from app.services.simulation_service import SimulationService

router = APIRouter(prefix="/simulation", tags=["simulation"])


# ─── Schemas ──────────────────────────────────────────────────────────────────

class CreateSimRequest(BaseModel):
    mode: str = "agent_vs_random"  # agent_vs_random or agent_vs_greedy
    seed: Optional[int] = None


class SpeedRequest(BaseModel):
    speed: float = Field(ge=0.25, le=10.0, default=1.0)


# ─── REST Routes ──────────────────────────────────────────────────────────────

@router.post(
    "/create",
    summary="Create a new AI simulation",
    status_code=status.HTTP_201_CREATED,
)
async def create_simulation(body: CreateSimRequest, request: Request):
    """Create a new AI-only simulation match."""
    wrapper = get_wrapper(request)
    service = SimulationService(wrapper)
    result = await service.create_simulation(mode=body.mode, seed=body.seed)
    return result


@router.get(
    "/{match_id}",
    summary="Get simulation state",
)
async def get_sim_state(match_id: str, request: Request):
    """Get the current state of a simulation."""
    wrapper = get_wrapper(request)
    service = SimulationService(wrapper)
    result = await service.get_state(match_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return result


@router.post(
    "/{match_id}/step",
    summary="Advance simulation by one step",
)
async def sim_step(match_id: str, request: Request):
    """Manually advance the simulation by one turn."""
    wrapper = get_wrapper(request)
    service = SimulationService(wrapper)
    result = await service.step(match_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Simulation ended or not found")
    return result


@router.post(
    "/{match_id}/pause",
    summary="Pause simulation",
)
async def pause_sim(match_id: str, request: Request):
    wrapper = get_wrapper(request)
    service = SimulationService(wrapper)
    ok = await service.set_paused(match_id, True)
    if not ok:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return {"paused": True}


@router.post(
    "/{match_id}/resume",
    summary="Resume simulation",
)
async def resume_sim(match_id: str, request: Request):
    wrapper = get_wrapper(request)
    service = SimulationService(wrapper)
    ok = await service.set_paused(match_id, False)
    if not ok:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return {"paused": False}


@router.post(
    "/{match_id}/speed",
    summary="Change simulation speed",
)
async def set_speed(match_id: str, body: SpeedRequest, request: Request):
    wrapper = get_wrapper(request)
    service = SimulationService(wrapper)
    ok = await service.set_speed(match_id, body.speed)
    if not ok:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return {"speed": body.speed}


@router.delete(
    "/{match_id}",
    summary="Delete a simulation",
)
async def delete_sim(match_id: str, request: Request):
    wrapper = get_wrapper(request)
    service = SimulationService(wrapper)
    await service.delete(match_id)
    return {"message": "Simulation deleted"}


# ─── WebSocket Streaming ─────────────────────────────────────────────────────

@router.websocket("/{match_id}/stream")
async def stream_simulation(websocket: WebSocket, match_id: str):
    """
    WebSocket endpoint that streams simulation steps to the client.

    The client can send JSON messages to control the simulation:
      {"action": "pause"}
      {"action": "resume"}
      {"action": "step"}      — single step when paused
      {"action": "speed", "value": 2.0}
      {"action": "stop"}

    The server sends game state updates after each step.
    """
    await websocket.accept()

    # Get wrapper from app state
    wrapper: UNOGameWrapper = websocket.app.state.game_wrapper
    service = SimulationService(wrapper)

    try:
        # Send initial state
        state = await service.get_state(match_id)
        if state is None:
            await websocket.send_json({"error": "Simulation not found"})
            await websocket.close()
            return

        await websocket.send_json({"type": "state", "data": state})

        # Main simulation loop
        while True:
            # Check for client messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(msg)
                action = data.get("action")

                if action == "pause":
                    await service.set_paused(match_id, True)
                    await websocket.send_json({"type": "control", "paused": True})
                    continue
                elif action == "resume":
                    await service.set_paused(match_id, False)
                    await websocket.send_json({"type": "control", "paused": False})
                elif action == "step":
                    # Single step regardless of pause state
                    result = await service.step(match_id)
                    if result is None:
                        await websocket.send_json({"type": "done"})
                        break
                    await websocket.send_json({"type": "step", "data": result})
                    continue
                elif action == "speed":
                    new_speed = float(data.get("value", 1.0))
                    await service.set_speed(match_id, new_speed)
                    await websocket.send_json({"type": "control", "speed": new_speed})
                elif action == "stop":
                    await websocket.send_json({"type": "stopped"})
                    break

            except asyncio.TimeoutError:
                pass  # No message from client, continue simulation

            # If paused, wait and retry
            if await service.is_paused(match_id):
                await asyncio.sleep(0.1)
                continue

            # Advance one step
            result = await service.step(match_id)
            if result is None:
                await websocket.send_json({"type": "done"})
                break

            await websocket.send_json({"type": "step", "data": result})

            # Wait based on speed setting
            delay = await service.get_step_delay(match_id)
            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

"""
app/game/wrapper.py
───────────────────
``UNOGameWrapper`` is the single integration point between the JAX/Flax game
engine and the async FastAPI backend.

Architecture overview
─────────────────────

Redis key layout for match ``<mid>``:
  game:<mid>:state    → serialised UNOState  (pickle bytes)
  game:<mid>:carries  → serialised {seat→TrXLCarry}  (pickle bytes)
  game:<mid>:meta     → JSON metadata  (str)
  game:<mid>:history  → Redis List of JSON event strings

Game modes
──────────
  HUMAN_VS_AI       Human sits at seat 0; agent controls seats 1-3.
  AGENT_VS_RANDOM   Agent (hero, seat 0) vs random-policy opponents.
  AGENT_VS_GREEDY   Agent (hero, seat 0) vs greedy-policy opponents.

Thread safety
─────────────
All JAX / numpy computation runs inside ``asyncio.to_thread()`` so the event
loop is never blocked.  Redis I/O uses the native async client.

Concurrency note
────────────────
A full distributed lock (Redlock) would be needed if multiple API workers
update the same match concurrently.  With ``--workers 1`` (the default in
docker-compose.yml) a single-process event loop is sufficient for Phase 1.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from app.core.config import get_settings
from app.core.redis_client import get_redis
from app.game.agent import AgentInference
from app.game.constants import (
    COLOR_NAMES,
    DRAW_ACTION,
    NUM_ACTIONS,
    NUM_PHYSICAL_CARD_TYPES,
    NUM_PLAYERS,
    TrXLCarry,
    UNO61Env,
    UNOState,
    action_scores_for_greedy,
    decode_action,
    decode_physical_card,
)
from app.game.serializer import (
    deserialize_carries,
    deserialize_uno_state,
    serialize_carries,
    serialize_uno_state,
)


# ─────────────────────────────────────────────────────────────────────────────
# Match mode constants
# ─────────────────────────────────────────────────────────────────────────────

class MatchMode:
    HUMAN_VS_AI       = "human_vs_ai"
    AGENT_VS_RANDOM   = "agent_vs_random"
    AGENT_VS_GREEDY   = "agent_vs_greedy"

    ALL = {HUMAN_VS_AI, AGENT_VS_RANDOM, AGENT_VS_GREEDY}


# ─────────────────────────────────────────────────────────────────────────────
# Internal event dataclass (sent over WebSocket in Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MoveEvent:
    """One player's turn, suitable for JSON serialisation."""
    player: int
    player_type: str          # "human" | "ai" | "random" | "greedy"
    action_idx: int
    action_name: str
    done: bool
    winner: int               # -1 if game not over
    value_estimate: Optional[float] = None
    top_actions: Optional[List[Dict[str, Any]]] = None  # top-5 by probability

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "player": self.player,
            "player_type": self.player_type,
            "action_idx": self.action_idx,
            "action_name": self.action_name,
            "done": self.done,
            "winner": self.winner,
        }
        if self.value_estimate is not None:
            d["value_estimate"] = self.value_estimate
        if self.top_actions is not None:
            d["top_actions"] = self.top_actions
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Main wrapper
# ─────────────────────────────────────────────────────────────────────────────

class UNOGameWrapper:
    """
    Async facade over the JAX UNO game engine.

    Parameters
    ----------
    agent:
        A fully-loaded ``AgentInference`` instance.  Shared across all active
        matches (it is read-only).
    """

    # In Human-vs-AI games the human always occupies seat 0.
    HUMAN_SEAT: int = 0

    def __init__(self, agent: AgentInference) -> None:
        self.agent = agent
        self.env   = UNO61Env(
            max_turns       = agent.cfg.max_turns,
            cards_per_player= agent.cfg.cards_per_player,
        )
        self._settings = get_settings()

    # ═════════════════════════════════════════════════════════════════════════
    # Public async API  (called by WebSocket / REST handlers)
    # ═════════════════════════════════════════════════════════════════════════

    async def create_match(
        self,
        mode: str = MatchMode.HUMAN_VS_AI,
        seed: Optional[int] = None,
        human_seat: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Initialise a new match, persist it to Redis, and return the initial
        frontend-ready state.

        Parameters
        ----------
        mode:
            One of ``MatchMode.*``.
        seed:
            Integer seed for reproducible games.  If ``None`` a random seed
            is chosen.
        human_seat:
            Which seat (0-3) the human occupies.  Ignored for simulation modes.

        Returns
        -------
        match_id : str
            UUID string that identifies this match in all subsequent calls.
        frontend_state : dict
            The initial game state in a format ready for JSON serialisation and
            dispatch to the frontend via WebSocket.
        """
        if mode not in MatchMode.ALL:
            raise ValueError(f"Unknown match mode: {mode!r}")

        match_id = str(uuid.uuid4())
        seed = seed if seed is not None else int(np.random.randint(0, 2**31))

        # JAX work is CPU-bound; run it off the event loop.
        state, carries, meta = await asyncio.to_thread(
            self._init_game, mode, seed, human_seat
        )

        ttl = (
            self._settings.sim_ttl_seconds
            if mode != MatchMode.HUMAN_VS_AI
            else self._settings.game_ttl_seconds
        )
        await self._save_to_redis(match_id, state, carries, meta, ttl=ttl)

        viewing_player = human_seat if mode == MatchMode.HUMAN_VS_AI else 0
        frontend_state = await asyncio.to_thread(
            self._state_to_frontend, state, viewing_player, mode, match_id
        )
        return match_id, frontend_state

    async def get_match_state(
        self,
        match_id: str,
        viewing_player: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load the current match state from Redis and return a frontend dict.

        Returns ``None`` if the match is not found (e.g. expired or invalid ID).
        """
        result = await self._load_from_redis(match_id)
        if result is None:
            return None
        state, carries, meta = result
        vp = (
            viewing_player
            if viewing_player is not None
            else meta.get("human_seat", 0)
        )
        return await asyncio.to_thread(
            self._state_to_frontend, state, vp, meta["mode"], match_id
        )

    async def apply_human_action(
        self,
        match_id: str,
        action_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply a human player's chosen action, then automatically play all AI
        opponent turns until it is the human's turn again (or the game ends).

        Parameters
        ----------
        match_id:
            The UUID of an active Human-vs-AI match.
        action_idx:
            Index into the 61-action space.  Illegal actions are silently
            replaced with the DRAW action.

        Returns
        -------
        dict or None
            Updated frontend state (includes an ``"events"`` list describing
            each move that occurred this round), or ``None`` if the match was
            not found or it was not the human's turn.
        """
        result = await self._load_from_redis(match_id)
        if result is None:
            return None

        state, carries, meta = result
        human_seat = int(meta["human_seat"])

        # Guard: only accept the action if it really is the human's turn.
        if int(np.asarray(state.current_player)) != human_seat:
            return None

        # CPU-bound: runs the human move + all opponent turns.
        new_state, new_carries, events = await asyncio.to_thread(
            self._human_step, state, carries, action_idx, human_seat
        )

        ttl = self._settings.game_ttl_seconds
        await self._save_to_redis(match_id, new_state, new_carries, meta, ttl=ttl)
        await self._append_history(match_id, events, ttl=ttl)

        frontend_state = await asyncio.to_thread(
            self._state_to_frontend,
            new_state, human_seat, meta["mode"], match_id,
        )
        frontend_state["events"] = [e.to_dict() for e in events]
        return frontend_state

    async def simulation_step(
        self,
        match_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Advance a simulation match by exactly **one player's turn**.

        The WebSocket handler calls this in a loop, streaming each step to the
        frontend at a configurable cadence.

        Returns
        -------
        dict or None
            Updated frontend state with an ``"event"`` key describing the move
            just taken, or ``None`` when the match is done or not found.
        """
        result = await self._load_from_redis(match_id)
        if result is None:
            return None

        state, carries, meta = result
        if bool(np.asarray(state.done)):
            return None

        new_state, new_carries, event = await asyncio.to_thread(
            self._sim_step, state, carries, meta
        )

        ttl = self._settings.sim_ttl_seconds
        await self._save_to_redis(match_id, new_state, new_carries, meta, ttl=ttl)
        await self._append_history(match_id, [event], ttl=ttl)

        frontend_state = await asyncio.to_thread(
            self._state_to_frontend,
            new_state, 0, meta["mode"], match_id,
        )
        frontend_state["event"] = event.to_dict()
        return frontend_state

    async def delete_match(self, match_id: str) -> bool:
        """Delete all Redis keys for a match. Returns True if keys existed."""
        redis  = await get_redis()
        keys   = [
            f"game:{match_id}:state",
            f"game:{match_id}:carries",
            f"game:{match_id}:meta",
            f"game:{match_id}:history",
        ]
        deleted = await redis.delete(*keys)
        return bool(deleted)

    async def match_exists(self, match_id: str) -> bool:
        """Return True if the match is still alive in Redis."""
        redis = await get_redis()
        return bool(await redis.exists(f"game:{match_id}:meta"))

    async def get_history(self, match_id: str) -> List[Dict[str, Any]]:
        """Return the full move history as a list of event dicts."""
        redis = await get_redis()
        raw   = await redis.lrange(f"game:{match_id}:history", 0, -1)
        return [json.loads(item) for item in raw]

    # ═════════════════════════════════════════════════════════════════════════
    # Synchronous JAX / numpy computation  (all run via asyncio.to_thread)
    # ═════════════════════════════════════════════════════════════════════════

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_game(
        self,
        mode: str,
        seed: int,
        human_seat: int,
    ) -> Tuple[UNOState, Dict[int, TrXLCarry], Dict[str, Any]]:
        """
        Reset the environment and prepare initial carries and metadata.

        Returns
        -------
        state : UNOState
        carries : {seat → TrXLCarry}
        meta : dict  (stored as JSON in Redis)
        """
        key   = jax.random.PRNGKey(seed)
        state = self.env.reset(key, jnp.int32(human_seat))

        if mode == MatchMode.HUMAN_VS_AI:
            # AI controls every seat that is NOT the human.
            ai_seats = [s for s in range(NUM_PLAYERS) if s != human_seat]
        else:
            # Simulation: agent is always at seat 0, opponents handled inline.
            ai_seats = [0]

        carries: Dict[int, TrXLCarry] = {
            seat: self.agent.init_carry() for seat in ai_seats
        }

        meta: Dict[str, Any] = {
            "mode":       mode,
            "seed":       seed,
            "human_seat": human_seat,
            "ai_seats":   ai_seats,
            "created_at": time.time(),
        }
        return state, carries, meta

    # ── Human-vs-AI step ──────────────────────────────────────────────────────

    def _human_step(
        self,
        state: UNOState,
        carries: Dict[int, TrXLCarry],
        action_idx: int,
        human_seat: int,
    ) -> Tuple[UNOState, Dict[int, TrXLCarry], List[MoveEvent]]:
        """
        Apply the human's action, then advance all AI opponents until the game
        is over or it is the human's turn again.

        PRNG management
        ───────────────
        ``UNOState.key`` is the game's internal JAX PRNG key.  Every call to
        ``env.apply_action`` that draws cards advances this key automatically
        via ``_split_keys`` inside the environment.  For *AI action sampling*
        we generate an independent key from numpy, keeping game-state randomness
        and policy-sampling randomness orthogonal.
        """
        events: List[MoveEvent] = []

        # ── Validate action ────────────────────────────────────────────────
        legal_mask = np.asarray(
            self.env.legal_action_mask(state, jnp.int32(human_seat))
        )
        if not legal_mask[action_idx]:
            action_idx = DRAW_ACTION   # graceful fallback

        # ── Apply human move ───────────────────────────────────────────────
        state = self.env.apply_action(
            state, jnp.int32(human_seat), jnp.int32(action_idx)
        )
        events.append(MoveEvent(
            player       = human_seat,
            player_type  = "human",
            action_idx   = action_idx,
            action_name  = decode_action(action_idx),
            done         = bool(np.asarray(state.done)),
            winner       = int(np.asarray(state.winner)),
        ))

        # ── AI opponent loop ───────────────────────────────────────────────
        # Run until the game ends OR control returns to the human's seat.
        while (
            not bool(np.asarray(state.done))
            and int(np.asarray(state.current_player)) != human_seat
        ):
            current_seat = int(np.asarray(state.current_player))
            carry = carries.get(current_seat, self.agent.init_carry())

            obs, _ = self.env.observe_for_player(state, jnp.int32(current_seat))

            ai_action, new_carry, info = self.agent.select_action(
                obs             = obs,
                carry           = carry,
                episode_start   = False,     # mid-game; carry already warmed up
                deterministic   = self._settings.agent_deterministic,
            )

            carries[current_seat] = new_carry
            state = self.env.apply_action(
                state,
                jnp.int32(current_seat),
                jnp.int32(ai_action),
            )

            events.append(MoveEvent(
                player          = current_seat,
                player_type     = "ai",
                action_idx      = ai_action,
                action_name     = decode_action(ai_action),
                done            = bool(np.asarray(state.done)),
                winner          = int(np.asarray(state.winner)),
                value_estimate  = info["value_estimate"],
                top_actions     = info.get("top_actions"),
            ))

        return state, carries, events

    # ── Simulation step ───────────────────────────────────────────────────────

    def _sim_step(
        self,
        state: UNOState,
        carries: Dict[int, TrXLCarry],
        meta: Dict[str, Any],
    ) -> Tuple[UNOState, Dict[int, TrXLCarry], MoveEvent]:
        """
        Advance the simulation by one player turn.

        * Seat 0 (hero): uses ``RecurrentUNOAgent``.
        * Other seats:   random or greedy policy depending on ``meta["mode"]``.
        """
        current_seat = int(np.asarray(state.current_player))
        mode         = meta["mode"]
        obs, _       = self.env.observe_for_player(state, jnp.int32(current_seat))

        if current_seat == 0:
            # ── Agent turn ────────────────────────────────────────────────
            carry = carries.get(0, self.agent.init_carry())
            ai_action, new_carry, info = self.agent.select_action(
                obs           = obs,
                carry         = carry,
                episode_start = False,
                deterministic = self._settings.agent_deterministic,
            )
            carries[0] = new_carry
            player_type = "agent"
            value_est   = info["value_estimate"]

        elif mode == MatchMode.AGENT_VS_RANDOM:
            # ── Random opponent ───────────────────────────────────────────
            legal_mask  = np.asarray(obs.action_mask)
            legal_idxs  = np.where(legal_mask)[0]
            ai_action   = int(np.random.choice(legal_idxs))
            player_type = "random"
            value_est   = None

        else:
            # ── Greedy opponent ───────────────────────────────────────────
            scores    = np.asarray(
                action_scores_for_greedy(state, self.env, jnp.int32(current_seat))
            )
            ai_action   = int(np.argmax(scores))
            player_type = "greedy"
            value_est   = None

        state = self.env.apply_action(
            state,
            jnp.int32(current_seat),
            jnp.int32(ai_action),
        )

        event = MoveEvent(
            player         = current_seat,
            player_type    = player_type,
            action_idx     = ai_action,
            action_name    = decode_action(ai_action),
            done           = bool(np.asarray(state.done)),
            winner         = int(np.asarray(state.winner)),
            value_estimate = value_est,
        )
        return state, carries, event

    # ── Frontend serialisation ─────────────────────────────────────────────────

    def _state_to_frontend(
        self,
        state: UNOState,
        viewing_player: int,
        mode: str,
        match_id: str,
    ) -> Dict[str, Any]:
        """
        Convert a ``UNOState`` into a plain Python dict that can be serialised
        to JSON and sent to the React frontend over a WebSocket.

        The dict is structured around the *viewing player's perspective*:
        - Full hand visibility for ``viewing_player``.
        - Only hand sizes for opponents (hidden information).
        - Legal actions only when it is ``viewing_player``'s turn.
        - Wild-card actions (52-59) are grouped for the frontend colour-picker.

        Parameters
        ----------
        state:
            The current ``UNOState``.
        viewing_player:
            The seat index whose hand is fully revealed (0 = human in HvA mode).
        mode:
            Match mode string (passed through to help the frontend render UI).
        match_id:
            Echoed back so the frontend can route WebSocket messages.
        """
        hands_np        = np.asarray(state.hands)          # (4, 54)
        current_player  = int(np.asarray(state.current_player))
        hero_seat       = int(np.asarray(state.hero_seat))
        discard_type    = int(np.asarray(state.discard_type))
        current_color   = int(np.asarray(state.current_color))
        direction       = int(np.asarray(state.direction))
        done            = bool(np.asarray(state.done))
        winner          = int(np.asarray(state.winner))
        last_action     = int(np.asarray(state.last_action))
        step_count      = int(np.asarray(state.step_count))
        deck_size       = int(np.asarray(state.deck_counts).sum())

        # ── Viewing player's hand ─────────────────────────────────────────
        your_hand: List[Dict[str, Any]] = []
        for card_idx in range(NUM_PHYSICAL_CARD_TYPES):
            count = int(hands_np[viewing_player, card_idx])
            for _ in range(count):
                your_hand.append({
                    "card_idx": card_idx,
                    "name":     decode_physical_card(card_idx),
                })

        # ── Legal actions (only on the viewer's turn) ─────────────────────
        legal_actions: List[Dict[str, Any]] = []
        if current_player == viewing_player and not done:
            legal_mask = np.asarray(
                self.env.legal_action_mask(state, jnp.int32(viewing_player))
            )
            legal_actions = [
                {"action_idx": i, "name": decode_action(i)}
                for i in range(NUM_ACTIONS)
                if legal_mask[i]
            ]

        # ── Grouped wild actions (for the frontend colour-picker widget) ───
        # If the player can play a wild or wild-draw4, we surface the four
        # colour-choice variants as a sub-group so the UI can show a picker.
        wild_groups: List[Dict[str, Any]] = []
        wd4_groups:  List[Dict[str, Any]] = []
        for action in legal_actions:
            ai = action["action_idx"]
            if 52 <= ai <= 55:
                wild_groups.append(action)
            elif 56 <= ai <= 59:
                wd4_groups.append(action)

        # ── Opponent hand sizes (ordered by seat) ─────────────────────────
        opponents: Dict[str, Dict[str, Any]] = {}
        for seat in range(NUM_PLAYERS):
            if seat == viewing_player:
                continue
            opponents[str(seat)] = {
                "seat":       seat,
                "hand_size":  int(hands_np[seat].sum()),
                "is_current": seat == current_player,
                "uno":        int(hands_np[seat].sum()) == 1,
            }

        return {
            # ── Match metadata ────────────────────────────────────────────
            "match_id":       match_id,
            "mode":           mode,
            "step_count":     step_count,
            # ── Turn / game state ─────────────────────────────────────────
            "current_player": current_player,
            "hero_seat":      hero_seat,
            "viewing_player": viewing_player,
            "is_your_turn":   (current_player == viewing_player) and not done,
            "direction":      "clockwise" if direction == 0 else "counter-clockwise",
            "done":           done,
            "winner":         winner,
            "winner_name":    f"Player {winner}" if winner >= 0 else None,
            # ── Discard pile ──────────────────────────────────────────────
            "top_card": {
                "card_idx": discard_type,
                "name":     decode_physical_card(discard_type),
            },
            "current_color": {
                "idx":  current_color,
                "name": COLOR_NAMES[current_color],
            },
            "last_action": {
                "action_idx": last_action,
                "name":       decode_action(last_action),
            },
            # ── Draw pile ─────────────────────────────────────────────────
            "deck_size": deck_size,
            # ── Player hands ──────────────────────────────────────────────
            "your_hand":       your_hand,
            "your_hand_size":  len(your_hand),
            "opponents":       opponents,
            # ── Actions ───────────────────────────────────────────────────
            "legal_actions":   legal_actions,
            "wild_actions":    wild_groups,     # [] unless viewer has a wild
            "wd4_actions":     wd4_groups,      # [] unless viewer has a wd4
        }

    # ═════════════════════════════════════════════════════════════════════════
    # Redis helpers
    # ═════════════════════════════════════════════════════════════════════════

    async def _save_to_redis(
        self,
        match_id: str,
        state: UNOState,
        carries: Dict[int, TrXLCarry],
        meta: Dict[str, Any],
        ttl: int = 7_200,
    ) -> None:
        """Atomically (pipeline) write all match keys with the given TTL."""
        # Serialise in a thread to avoid blocking the event loop.
        state_bytes   = await asyncio.to_thread(serialize_uno_state,  state)
        carries_bytes = await asyncio.to_thread(serialize_carries, carries)
        meta_json     = json.dumps(meta)

        redis = await get_redis()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.setex(f"game:{match_id}:state",   ttl, state_bytes)
            pipe.setex(f"game:{match_id}:carries", ttl, carries_bytes)
            pipe.setex(f"game:{match_id}:meta",    ttl, meta_json)
            await pipe.execute()

    async def _load_from_redis(
        self,
        match_id: str,
    ) -> Optional[Tuple[UNOState, Dict[int, TrXLCarry], Dict[str, Any]]]:
        """
        Load all match keys in a single MGET call.
        Returns ``None`` if *any* key is missing (expired or bad match_id).
        """
        redis = await get_redis()
        state_bytes, carries_bytes, meta_json = await redis.mget(
            f"game:{match_id}:state",
            f"game:{match_id}:carries",
            f"game:{match_id}:meta",
        )

        if any(v is None for v in (state_bytes, carries_bytes, meta_json)):
            return None

        # Deserialise off the event loop.
        state   = await asyncio.to_thread(deserialize_uno_state, state_bytes)
        carries = await asyncio.to_thread(deserialize_carries,   carries_bytes)
        meta    = json.loads(meta_json)
        return state, carries, meta

    async def _append_history(
        self,
        match_id: str,
        events: List[MoveEvent],
        ttl: int = 7_200,
    ) -> None:
        """Append move events to the Redis List for this match."""
        if not events:
            return
        redis = await get_redis()
        key   = f"game:{match_id}:history"
        async with redis.pipeline() as pipe:
            for event in events:
                pipe.rpush(key, json.dumps(event.to_dict()))
            pipe.expire(key, ttl)
            await pipe.execute()

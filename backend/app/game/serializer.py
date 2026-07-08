"""
app/game/serializer.py
──────────────────────
Converts JAX NamedTuples (UNOState, TrXLCarry) to/from raw bytes so they can
be stored in Redis with zero schema migrations.

Design choices
──────────────
* We use Python's `pickle` protocol 5 (the fastest binary format available in
  Python ≥ 3.8) after converting every JAX array to a plain `numpy.ndarray`.
  This avoids platform-specific JAX device buffers in the serialised payload.

* State and carries are stored under *separate* Redis keys so that a human
  player's action (which only mutates state) does not re-serialise the AI
  memory, and vice-versa.

* All round-trips are CPU-only numpy copies – no GPU memory is allocated
  during serialisation.
"""
from __future__ import annotations

import pickle
from typing import Dict

import jax.numpy as jnp
import numpy as np

from app.game.constants import UNOState, TrXLCarry

# Use the fastest pickle protocol available on this Python version.
_PROTOCOL = pickle.HIGHEST_PROTOCOL


# ─────────────────────────────────────────────────────────────────────────────
# UNOState  (14 fields, all JAX arrays)
# ─────────────────────────────────────────────────────────────────────────────

def serialize_uno_state(state: UNOState) -> bytes:
    """
    Serialise a UNOState to a compact byte string for Redis.

    All JAX arrays are first materialised on the host as numpy arrays so that
    the pickle payload is self-contained and device-independent.
    """
    numpy_dict: Dict[str, np.ndarray] = {
        field: np.asarray(getattr(state, field))
        for field in UNOState._fields
    }
    return pickle.dumps(numpy_dict, protocol=_PROTOCOL)


def deserialize_uno_state(data: bytes) -> UNOState:
    """
    Reconstruct a UNOState from bytes previously produced by
    `serialize_uno_state`.  Arrays are placed back on the default JAX device.
    """
    numpy_dict: Dict[str, np.ndarray] = pickle.loads(data)
    return UNOState(**{
        field: jnp.asarray(numpy_dict[field])
        for field in UNOState._fields
    })


# ─────────────────────────────────────────────────────────────────────────────
# TrXLCarry  (one field: memory, shape [num_layers, memory_len, hidden_dim])
# ─────────────────────────────────────────────────────────────────────────────

def serialize_carries(carries: Dict[int, TrXLCarry]) -> bytes:
    """
    Serialise a mapping of  {seat_index → TrXLCarry}  to bytes.

    In Human-vs-AI games the dict contains entries for every AI-controlled
    seat (1, 2, 3).  In simulation mode it contains only seat 0.

    Args:
        carries: ``{seat: TrXLCarry}`` – one carry per AI-controlled player.

    Returns:
        Pickled ``{seat: np.ndarray}`` where the array is the carry memory.
    """
    numpy_dict: Dict[int, np.ndarray] = {
        seat: np.asarray(carry.memory)
        for seat, carry in carries.items()
    }
    return pickle.dumps(numpy_dict, protocol=_PROTOCOL)


def deserialize_carries(data: bytes) -> Dict[int, TrXLCarry]:
    """
    Reconstruct ``{seat → TrXLCarry}`` from bytes produced by
    `serialize_carries`.
    """
    numpy_dict: Dict[int, np.ndarray] = pickle.loads(data)
    return {
        seat: TrXLCarry(memory=jnp.asarray(memory))
        for seat, memory in numpy_dict.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers used in tests / admin tooling
# ─────────────────────────────────────────────────────────────────────────────

def state_byte_size(state: UNOState) -> int:
    """Return the byte size of a serialised UNOState (for monitoring)."""
    return len(serialize_uno_state(state))


def carries_byte_size(carries: Dict[int, TrXLCarry]) -> int:
    """Return the byte size of a serialised carries dict (for monitoring)."""
    return len(serialize_carries(carries))

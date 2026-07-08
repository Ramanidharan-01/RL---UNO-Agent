"""
app/game/agent.py
─────────────────
Wraps ``RecurrentUNOAgent`` for *synchronous, single-step* inference as
required by the web backend.

Key responsibilities
────────────────────
1. Load a trained checkpoint (params + TrainConfig) once at startup.
2. JIT-warm the model on a dummy input so the *first real game turn* is fast.
3. Expose ``select_action()`` that takes an observation + carry and returns
   (action_idx, new_carry, info_dict) without the caller needing to know
   anything about JAX.
4. Expose ``init_carry()`` to create zero-initialised TrXLCarry for new
   game sessions.

Thread safety
─────────────
``AgentInference`` is **read-only** after ``__init__`` (params are frozen).
Multiple async workers can call ``select_action()`` concurrently on the same
instance; JAX handles its own internal locking.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from app.core.config import get_settings
from app.game.constants import (
    NUM_ACTIONS,
    RecurrentUNOAgent,
    TrXLCarry,
    TrainConfig,
    UNOObservation,
    make_dummy_observation,
    masked_categorical_sample,
)


class AgentInference:
    """
    Thin, stateless wrapper around a trained ``RecurrentUNOAgent``.

    After construction the instance is safe to share across coroutines /
    threads – nothing is mutated after ``__init__``.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pkl`` file produced by ``save_checkpoint()`` in
        ``uno_jax.py``.  Must contain keys ``"cfg"`` and ``"main_state"``.
    """

    def __init__(self, checkpoint_path: str) -> None:
        self.cfg: TrainConfig
        self.model: RecurrentUNOAgent
        self.params: Any                  # frozen FrozenDict living on device
        self._load(checkpoint_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        """Deserialise checkpoint → build model → JIT-warm → store params."""
        with open(path, "rb") as fh:
            payload: Dict[str, Any] = pickle.load(fh)

        cfg: TrainConfig = payload["cfg"]
        self.cfg = cfg

        self.model = RecurrentUNOAgent(
            hidden_dim=cfg.hidden_dim,
            hand_embed_dim=cfg.hand_embed_dim,
            num_layers=cfg.num_layers,
            memory_len=cfg.memory_len,
            num_heads=cfg.num_heads,
        )

        # Materialise params on the default accelerator (CPU / GPU / TPU).
        self.params = jax.device_put(payload["main_state"].params)

        # Trigger XLA compilation now so the first in-game call is instant.
        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy forward pass to compile the JIT-traced graph."""
        dummy_obs   = make_dummy_observation()
        dummy_carry = self.init_carry()
        _ = self.model.apply(
            {"params": self.params},
            dummy_obs,
            dummy_carry,
            jnp.bool_(True),          # episode_start = True on first step
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def init_carry(self) -> TrXLCarry:
        """
        Return a zero-initialised ``TrXLCarry`` suitable for the start of a
        new game session.

        Shape: ``(num_layers, memory_len, hidden_dim)``
        """
        return RecurrentUNOAgent.init_carry(
            num_layers=self.cfg.num_layers,
            memory_len=self.cfg.memory_len,
            hidden_dim=self.cfg.hidden_dim,
        )

    def select_action(
        self,
        obs: UNOObservation,
        carry: TrXLCarry,
        episode_start: bool = False,
        key: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, TrXLCarry, Dict[str, Any]]:
        """
        Run one step of recurrent inference.

        Parameters
        ----------
        obs:
            Current observation for the player whose turn it is.
        carry:
            The player's ``TrXLCarry`` accumulated over previous turns.
        episode_start:
            Pass ``True`` only on the absolute *first* turn of a new game
            (zeroes out any stale carry memory).
        key:
            JAX PRNGKey for stochastic sampling.  If ``None`` a key is
            generated from numpy random (sufficient for web use).
        deterministic:
            When ``True``, always pick ``argmax(logits)`` instead of sampling.

        Returns
        -------
        action_idx : int
            Index into the 61-action space (0-60).
        new_carry : TrXLCarry
            Updated memory to persist for the *next* call for this player.
        info : dict
            Diagnostic payload sent to the frontend / logs::

                {
                    "value_estimate": float,       # V(s) from the critic head
                    "top_actions": [               # top-5 by probability
                        {"action_idx": int, "action_name": str, "prob": float},
                        ...
                    ],
                }
        """
        start_flag = jnp.bool_(episode_start)

        # Forward pass (JIT-compiled after first call).
        outs = self.model.apply(
            {"params": self.params},
            obs,
            carry,
            start_flag,
        )

        new_carry: TrXLCarry = outs["trxl_carry"]
        logits: jnp.ndarray  = outs["policy_logits"]

        # ── Action selection ───────────────────────────────────────────────
        if deterministic:
            action = int(np.asarray(jnp.argmax(logits)))
        else:
            if key is None:
                # Use numpy to generate a fresh JAX key each call.  This is
                # safe – individual game sessions are sequential.
                key = jax.random.PRNGKey(int(np.random.randint(0, 2**31)))
            action = int(np.asarray(masked_categorical_sample(key, logits)))

        # ── Diagnostic info ────────────────────────────────────────────────
        probs_np: np.ndarray = np.asarray(jax.nn.softmax(logits))
        top5_indices          = np.argsort(-probs_np)[:5]

        from app.game.constants import decode_action  # local to avoid cycles
        top_actions = [
            {
                "action_idx": int(i),
                "action_name": decode_action(int(i)),
                "prob": float(probs_np[i]),
            }
            for i in top5_indices
        ]

        info: Dict[str, Any] = {
            "value_estimate": float(np.asarray(outs["value"])),
            "top_actions": top_actions,
        }

        return action, new_carry, info


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton – loaded once on first access
# ─────────────────────────────────────────────────────────────────────────────

_agent: Optional[AgentInference] = None


def get_agent() -> AgentInference:
    """
    Return (and lazily construct) the shared ``AgentInference`` singleton.

    Call this from FastAPI dependency injection or application startup.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file specified in ``CHECKPOINT_PATH`` does not exist.
    """
    global _agent
    if _agent is None:
        settings = get_settings()
        _agent = AgentInference(settings.checkpoint_path)
    return _agent

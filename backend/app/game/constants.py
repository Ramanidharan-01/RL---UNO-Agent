"""
app/game/constants.py
─────────────────────
Single import-point for everything from the JAX source file.
All other backend modules import from here, not directly from uno_jax,
so the module path can be reconfigured in one place.
"""
from app.game.uno_jax import (
    # ── Numeric constants ──────────────────────────────────────────────────
    NUM_PLAYERS,
    NUM_COLORS,
    NUM_COLORED_RANKS,
    NUM_COLORED_ACTIONS,
    PHYSICAL_WILD,
    PHYSICAL_WILD_DRAW4,
    NUM_PHYSICAL_CARD_TYPES,
    ACTION_WILD_START,
    ACTION_WILD_DRAW4_START,
    DRAW_ACTION,
    NUM_ACTIONS,
    MAX_HAND_SIZE,
    CARD_FEATURE_DIM,
    SKIP_RANK,
    REVERSE_RANK,
    DRAW2_RANK,
    WILD_RANK,
    WILD_DRAW4_RANK,
    # ── Opponent mode flags ────────────────────────────────────────────────
    OPP_RANDOM,
    OPP_GREEDY,
    OPP_AVERAGE,
    # ── Human-readable name tables ─────────────────────────────────────────
    COLOR_NAMES,
    RANK_NAMES,
    # ── NamedTuples / data structures ─────────────────────────────────────
    UNOState,
    UNOObservation,
    TrXLCarry,
    TrainConfig,
    PPOTrainState,
    # ── Neural network models ──────────────────────────────────────────────
    RecurrentUNOAgent,
    AveragePolicyModel,
    # ── Game environment ───────────────────────────────────────────────────
    UNO61Env,
    # ── Utility functions ──────────────────────────────────────────────────
    decode_physical_card,
    decode_action,
    action_scores_for_greedy,
    masked_categorical_sample,
    make_dummy_observation,
    build_card_feature_table,
    CARD_FEATURE_TABLE,
    INITIAL_DECK_COUNTS,
)

__all__ = [
    "NUM_PLAYERS", "NUM_COLORS", "NUM_COLORED_RANKS", "NUM_COLORED_ACTIONS",
    "PHYSICAL_WILD", "PHYSICAL_WILD_DRAW4", "NUM_PHYSICAL_CARD_TYPES",
    "ACTION_WILD_START", "ACTION_WILD_DRAW4_START", "DRAW_ACTION",
    "NUM_ACTIONS", "MAX_HAND_SIZE", "CARD_FEATURE_DIM",
    "SKIP_RANK", "REVERSE_RANK", "DRAW2_RANK", "WILD_RANK", "WILD_DRAW4_RANK",
    "OPP_RANDOM", "OPP_GREEDY", "OPP_AVERAGE",
    "COLOR_NAMES", "RANK_NAMES",
    "UNOState", "UNOObservation", "TrXLCarry", "TrainConfig", "PPOTrainState",
    "RecurrentUNOAgent", "AveragePolicyModel",
    "UNO61Env",
    "decode_physical_card", "decode_action",
    "action_scores_for_greedy", "masked_categorical_sample",
    "make_dummy_observation", "build_card_feature_table",
    "CARD_FEATURE_TABLE", "INITIAL_DECK_COUNTS",
]

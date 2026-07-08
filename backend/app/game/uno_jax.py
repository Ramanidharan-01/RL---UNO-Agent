from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple
try:
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
except ImportError:
    # matplotlib and IPython are only needed for training/notebook plots.
    # They are not in requirements.txt for the web backend.
    plt = None  # type: ignore[assignment]
    def clear_output(*args, **kwargs):  # type: ignore[misc]
        pass
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


NUM_PLAYERS = 4
NUM_COLORS = 4
NUM_COLORED_RANKS = 13  # 0-9, Skip, Reverse, Draw2
NUM_COLORED_ACTIONS = NUM_COLORS * NUM_COLORED_RANKS  # 52
PHYSICAL_WILD = 52
PHYSICAL_WILD_DRAW4 = 53
NUM_PHYSICAL_CARD_TYPES = 54
ACTION_WILD_START = 52
ACTION_WILD_DRAW4_START = 56
DRAW_ACTION = 60
NUM_ACTIONS = 61
MAX_HAND_SIZE = 32
CARD_FEATURE_DIM = NUM_COLORS + (NUM_COLORED_RANKS + 2)  # colors + ranks + wild + wd4
SKIP_RANK = 10
REVERSE_RANK = 11
DRAW2_RANK = 12
WILD_RANK = 13
WILD_DRAW4_RANK = 14

OPP_RANDOM = 0
OPP_GREEDY = 1
OPP_AVERAGE = 2

COLOR_NAMES = ("red", "yellow", "green", "blue")
RANK_NAMES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "skip", "reverse", "draw2")


def build_initial_deck_counts() -> jnp.ndarray:
    counts = np.zeros((NUM_PHYSICAL_CARD_TYPES,), dtype=np.int32)
    for color in range(NUM_COLORS):
        base = color * NUM_COLORED_RANKS
        counts[base + 0] = 1
        counts[base + 1:base + NUM_COLORED_RANKS] = 2
    counts[PHYSICAL_WILD] = 4
    counts[PHYSICAL_WILD_DRAW4] = 4
    return jnp.array(counts, dtype=jnp.int32)


def build_card_feature_table() -> jnp.ndarray:
    table = np.zeros((NUM_PHYSICAL_CARD_TYPES, CARD_FEATURE_DIM), dtype=np.float32)
    for idx in range(NUM_COLORED_ACTIONS):
        color = idx // NUM_COLORED_RANKS
        rank = idx % NUM_COLORED_RANKS
        table[idx, color] = 1.0
        table[idx, NUM_COLORS + rank] = 1.0
    table[PHYSICAL_WILD, NUM_COLORS + WILD_RANK] = 1.0
    table[PHYSICAL_WILD_DRAW4, NUM_COLORS + WILD_DRAW4_RANK] = 1.0
    return jnp.array(table, dtype=jnp.float32)


INITIAL_DECK_COUNTS = build_initial_deck_counts()
CARD_FEATURE_TABLE = build_card_feature_table()
COLORED_COLORS = jnp.arange(NUM_COLORED_ACTIONS, dtype=jnp.int32) // NUM_COLORED_RANKS
COLORED_RANKS = jnp.arange(NUM_COLORED_ACTIONS, dtype=jnp.int32) % NUM_COLORED_RANKS


def player_advance(player: jnp.ndarray, direction_flag: jnp.ndarray, steps: int) -> jnp.ndarray:
    delta = jnp.where(direction_flag == 0, steps, -steps)
    return jnp.mod(player + delta, NUM_PLAYERS)


def decode_physical_card(card_idx: int) -> str:
    if card_idx < NUM_COLORED_ACTIONS:
        color = COLOR_NAMES[card_idx // NUM_COLORED_RANKS]
        rank = RANK_NAMES[card_idx % NUM_COLORED_RANKS]
        return f"{color}:{rank}"
    if card_idx == PHYSICAL_WILD:
        return "wild"
    return "wild_draw4"


def decode_action(action_idx: int) -> str:
    if action_idx < NUM_COLORED_ACTIONS:
        return decode_physical_card(action_idx)
    if ACTION_WILD_START <= action_idx < ACTION_WILD_DRAW4_START:
        return f"wild->{COLOR_NAMES[action_idx - ACTION_WILD_START]}"
    if ACTION_WILD_DRAW4_START <= action_idx < DRAW_ACTION:
        return f"wild_draw4->{COLOR_NAMES[action_idx - ACTION_WILD_DRAW4_START]}"
    return "draw"


class UNOState(NamedTuple):
    hands: jnp.ndarray
    deck_counts: jnp.ndarray
    played_counts: jnp.ndarray
    seen_counts: jnp.ndarray
    discard_type: jnp.ndarray
    current_color: jnp.ndarray
    current_player: jnp.ndarray
    hero_seat: jnp.ndarray
    direction: jnp.ndarray
    step_count: jnp.ndarray
    done: jnp.ndarray
    winner: jnp.ndarray
    last_action: jnp.ndarray
    key: jnp.ndarray


class UNOObservation(NamedTuple):
    hand: jnp.ndarray
    hand_mask: jnp.ndarray
    top_card: jnp.ndarray
    current_color: jnp.ndarray
    opponent_counts: jnp.ndarray
    uno_flags: jnp.ndarray
    direction: jnp.ndarray
    draw_pile_frac: jnp.ndarray
    turn_frac: jnp.ndarray
    hand_size_frac: jnp.ndarray
    last_action: jnp.ndarray
    played_hist: jnp.ndarray
    action_mask: jnp.ndarray


class TrXLCarry(NamedTuple):
    memory: jnp.ndarray


class TrajectoryBatch(NamedTuple):
    obs: UNOObservation
    bootstrap_obs: UNOObservation
    episode_starts: jnp.ndarray
    bootstrap_starts: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    old_log_probs: jnp.ndarray
    old_values: jnp.ndarray
    belief_targets: jnp.ndarray
    init_memory: jnp.ndarray


class PPOTrainState(NamedTuple):
    params: Any
    opt_state: optax.OptState
    step: jnp.ndarray


class AvgTrainState(NamedTuple):
    params: Any
    opt_state: optax.OptState
    step: jnp.ndarray


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    max_turns: int = 256
    cards_per_player: int = 7
    num_envs: int = 64
    rollout_len: int = 32
    total_updates: int = 4000
    hidden_dim: int = 192
    hand_embed_dim: int = 48
    memory_len: int = 12
    num_layers: int = 2
    num_heads: int = 4
    lr: float = 2.5e-4
    avg_lr: float = 3.0e-4
    warmup_steps: int = 1000
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    belief_coef: float = 0.05
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    ppo_epochs: int = 2
    reservoir_capacity: int = 200_000
    avg_batch_size: int = 512
    avg_updates_per_round: int = 2
    exploiter_every: int = 2
    snapshot_every: int = 250
    history_limit: int = 8
    eval_every: int = 250
    checkpoint_every: int = 500
    eval_episodes: int = 64
    output_dir: str = "cloud_runs/uno_t4"


class UNO61Env:
    def __init__(self, max_turns: int = 256, cards_per_player: int = 7):
        self.max_turns = max_turns
        self.cards_per_player = cards_per_player

    def _split_keys(self, state: UNOState, num: int) -> Tuple[UNOState, jnp.ndarray]:
        keys = jax.random.split(state.key, num + 1)
        new_state = state._replace(key=keys[0])
        return new_state, keys[1:]

    def _maybe_refill_deck(self, state: UNOState) -> UNOState:
        deck_total = jnp.sum(state.deck_counts)
        keep_top = jax.nn.one_hot(state.discard_type, NUM_PHYSICAL_CARD_TYPES, dtype=jnp.int32)
        recycled = jnp.maximum(state.played_counts - keep_top, 0)
        can_recycle = jnp.sum(recycled) > 0
        should_refill = (deck_total == 0) & can_recycle
        deck_counts = jnp.where(should_refill, recycled, state.deck_counts)
        played_counts = jnp.where(should_refill, keep_top, state.played_counts)
        return state._replace(deck_counts=deck_counts, played_counts=played_counts)

    def _draw_one(self, state: UNOState, player: jnp.ndarray) -> UNOState:
        state = self._maybe_refill_deck(state)
        state, keys = self._split_keys(state, 1)
        draw_key = keys[0]
        deck_total = jnp.sum(state.deck_counts)
        logits = jnp.where(
            state.deck_counts > 0,
            jnp.log(state.deck_counts.astype(jnp.float32)),
            jnp.full_like(state.deck_counts, -1e30, dtype=jnp.float32),
        )
        sampled = jax.random.categorical(draw_key, logits)

        def do_draw(s: UNOState) -> UNOState:
            hands = s.hands.at[player, sampled].add(1)
            deck_counts = s.deck_counts.at[sampled].add(-1)
            return s._replace(hands=hands, deck_counts=deck_counts)

        return jax.lax.cond(deck_total > 0, do_draw, lambda s: s, state)

    def _draw_n(self, state: UNOState, player: jnp.ndarray, n: int) -> UNOState:
        def body_fn(_, s: UNOState) -> UNOState:
            return self._draw_one(s, player)

        return jax.lax.fori_loop(0, n, body_fn, state)

    def _expand_hand(self, hand_counts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        total_cards = jnp.sum(hand_counts)
        capped_total = jnp.minimum(total_cards, MAX_HAND_SIZE)
        cumulative = jnp.cumsum(hand_counts)
        slots = jnp.arange(MAX_HAND_SIZE, dtype=jnp.int32)
        mask = slots < capped_total
        first_true = jnp.argmax(slots[:, None] < cumulative[None, :], axis=1)
        tokens = CARD_FEATURE_TABLE[first_true]
        tokens = tokens * mask[:, None].astype(jnp.float32)
        return tokens, mask

    def legal_action_mask(self, state: UNOState, player: jnp.ndarray) -> jnp.ndarray:
        hand = state.hands[player]
        top_is_colored = state.discard_type < NUM_COLORED_ACTIONS
        top_rank = jnp.where(top_is_colored, state.discard_type % NUM_COLORED_RANKS, -1)
        color_match = COLORED_COLORS == state.current_color
        rank_match = COLORED_RANKS == top_rank
        colored_legal = (hand[:NUM_COLORED_ACTIONS] > 0) & (color_match | (top_is_colored & rank_match))
        has_wild = hand[PHYSICAL_WILD] > 0
        color_counts = hand[:NUM_COLORED_ACTIONS].reshape(NUM_COLORS, NUM_COLORED_RANKS).sum(axis=1)
        no_current_color = color_counts[state.current_color] == 0
        has_wd4 = (hand[PHYSICAL_WILD_DRAW4] > 0) & no_current_color
        wild_legal = jnp.full((4,), has_wild, dtype=jnp.bool_)
        wd4_legal = jnp.full((4,), has_wd4, dtype=jnp.bool_)
        draw_legal = jnp.array([True], dtype=jnp.bool_)
        return jnp.concatenate([colored_legal, wild_legal, wd4_legal, draw_legal], axis=0)

    def observe_for_player(self, state: UNOState, player: jnp.ndarray) -> Tuple[UNOObservation, jnp.ndarray]:
        hand = state.hands[player]
        hand_tokens, hand_mask = self._expand_hand(hand)
        top_card = CARD_FEATURE_TABLE[state.discard_type]
        current_color = jax.nn.one_hot(state.current_color, NUM_COLORS, dtype=jnp.float32)
        order = jnp.mod(player + jnp.arange(NUM_PLAYERS, dtype=jnp.int32), NUM_PLAYERS)
        opp_order = order[1:]
        hand_sizes = state.hands.sum(axis=-1).astype(jnp.float32)
        opponent_counts = hand_sizes[opp_order] / float(MAX_HAND_SIZE)
        uno_flags = (hand_sizes[order] == 1).astype(jnp.float32)
        direction = jax.nn.one_hot(state.direction, 2, dtype=jnp.float32)
        draw_pile_frac = jnp.sum(state.deck_counts).astype(jnp.float32) / 108.0
        turn_frac = state.step_count.astype(jnp.float32) / float(self.max_turns)
        hand_size_frac = jnp.minimum(hand_sizes[player], float(MAX_HAND_SIZE)) / float(MAX_HAND_SIZE)
        last_action = jax.nn.one_hot(state.last_action, NUM_ACTIONS, dtype=jnp.float32)
        played_hist = state.seen_counts.astype(jnp.float32) / INITIAL_DECK_COUNTS.astype(jnp.float32)
        action_mask = self.legal_action_mask(state, player)
        belief_targets = (state.hands[opp_order] > 0).astype(jnp.float32)
        obs = UNOObservation(
            hand=hand_tokens,
            hand_mask=hand_mask,
            top_card=top_card,
            current_color=current_color,
            opponent_counts=opponent_counts,
            uno_flags=uno_flags,
            direction=direction,
            draw_pile_frac=jnp.float32(draw_pile_frac),
            turn_frac=jnp.float32(turn_frac),
            hand_size_frac=jnp.float32(hand_size_frac),
            last_action=last_action,
            played_hist=played_hist,
            action_mask=action_mask,
        )
        return obs, belief_targets

    def reset(self, key: jnp.ndarray, hero_seat: jnp.ndarray = jnp.int32(0)) -> UNOState:
        zero_hands = jnp.zeros((NUM_PLAYERS, NUM_PHYSICAL_CARD_TYPES), dtype=jnp.int32)
        state = UNOState(
            hands=zero_hands,
            deck_counts=INITIAL_DECK_COUNTS,
            played_counts=jnp.zeros((NUM_PHYSICAL_CARD_TYPES,), dtype=jnp.int32),
            seen_counts=jnp.zeros((NUM_PHYSICAL_CARD_TYPES,), dtype=jnp.int32),
            discard_type=jnp.int32(0),
            current_color=jnp.int32(0),
            current_player=hero_seat,
            hero_seat=hero_seat,
            direction=jnp.int32(0),
            step_count=jnp.int32(0),
            done=jnp.bool_(False),
            winner=jnp.int32(-1),
            last_action=jnp.int32(DRAW_ACTION),
            key=key,
        )
        for player in range(NUM_PLAYERS):
            for _ in range(self.cards_per_player):
                state = self._draw_one(state, jnp.int32(player))

        state = self._maybe_refill_deck(state)
        state, keys = self._split_keys(state, 1)
        first_logits = jnp.where(
            state.deck_counts[:NUM_COLORED_ACTIONS] > 0,
            jnp.log(state.deck_counts[:NUM_COLORED_ACTIONS].astype(jnp.float32)),
            jnp.full((NUM_COLORED_ACTIONS,), -1e30, dtype=jnp.float32),
        )
        discard = jax.random.categorical(keys[0], first_logits)
        deck_counts = state.deck_counts.at[discard].add(-1)
        played_counts = state.played_counts.at[discard].add(1)
        seen_counts = state.seen_counts.at[discard].add(1)
        return state._replace(
            deck_counts=deck_counts,
            played_counts=played_counts,
            seen_counts=seen_counts,
            discard_type=discard.astype(jnp.int32),
            current_color=(discard // NUM_COLORED_RANKS).astype(jnp.int32),
            current_player=hero_seat,
            hero_seat=hero_seat,
            direction=jnp.int32(0),
            step_count=jnp.int32(0),
            done=jnp.bool_(False),
            winner=jnp.int32(-1),
            last_action=jnp.int32(DRAW_ACTION),
        )

    def _post_play_state(self, state: UNOState, player: jnp.ndarray, next_state: UNOState) -> UNOState:
        hand_empty = jnp.sum(next_state.hands[player]) == 0
        next_player = next_state.current_player
        timeout = next_state.step_count >= self.max_turns
        winner = jnp.where(hand_empty, player, jnp.where(timeout, jnp.int32(-1), next_state.winner))
        done = hand_empty | timeout
        return next_state._replace(done=done, winner=winner, current_player=next_player)

    def apply_action(self, state: UNOState, player: jnp.ndarray, action: jnp.ndarray) -> UNOState:
        mask = self.legal_action_mask(state, player)
        safe_action = jnp.where(mask[action], action, DRAW_ACTION)
        step_state = state._replace(
            step_count=state.step_count + 1,
            last_action=safe_action.astype(jnp.int32),
        )

        def do_draw(s: UNOState) -> UNOState:
            s = self._draw_one(s, player)
            next_player = player_advance(player, s.direction, 1)
            s = s._replace(current_player=next_player)
            return self._post_play_state(s, player, s)

        def do_colored(s: UNOState) -> UNOState:
            color = safe_action // NUM_COLORED_RANKS
            rank = safe_action % NUM_COLORED_RANKS
            hands = s.hands.at[player, safe_action].add(-1)
            played_counts = s.played_counts.at[safe_action].add(1)
            seen_counts = s.seen_counts.at[safe_action].add(1)
            s = s._replace(
                hands=hands,
                played_counts=played_counts,
                seen_counts=seen_counts,
                discard_type=safe_action.astype(jnp.int32),
                current_color=color.astype(jnp.int32),
            )

            def rank_normal(x: UNOState) -> UNOState:
                return x._replace(current_player=player_advance(player, x.direction, 1))

            def rank_skip(x: UNOState) -> UNOState:
                return x._replace(current_player=player_advance(player, x.direction, 2))

            def rank_reverse(x: UNOState) -> UNOState:
                new_direction = jnp.int32(1 - x.direction)
                next_player = player_advance(player, new_direction, 1)
                return x._replace(direction=new_direction, current_player=next_player)

            def rank_draw2(x: UNOState) -> UNOState:
                target = player_advance(player, x.direction, 1)
                x = self._draw_n(x, target, 2)
                return x._replace(current_player=player_advance(player, x.direction, 2))

            s = jax.lax.switch(
                jnp.where(rank <= 9, 0, rank - 9),
                (rank_normal, rank_skip, rank_reverse, rank_draw2),
                s,
            )
            return self._post_play_state(s, player, s)

        def do_wild(s: UNOState) -> UNOState:
            chosen_color = safe_action - ACTION_WILD_START
            hands = s.hands.at[player, PHYSICAL_WILD].add(-1)
            played_counts = s.played_counts.at[PHYSICAL_WILD].add(1)
            seen_counts = s.seen_counts.at[PHYSICAL_WILD].add(1)
            s = s._replace(
                hands=hands,
                played_counts=played_counts,
                seen_counts=seen_counts,
                discard_type=jnp.int32(PHYSICAL_WILD),
                current_color=chosen_color.astype(jnp.int32),
                current_player=player_advance(player, s.direction, 1),
            )
            return self._post_play_state(s, player, s)

        def do_wd4(s: UNOState) -> UNOState:
            chosen_color = safe_action - ACTION_WILD_DRAW4_START
            hands = s.hands.at[player, PHYSICAL_WILD_DRAW4].add(-1)
            played_counts = s.played_counts.at[PHYSICAL_WILD_DRAW4].add(1)
            seen_counts = s.seen_counts.at[PHYSICAL_WILD_DRAW4].add(1)
            s = s._replace(
                hands=hands,
                played_counts=played_counts,
                seen_counts=seen_counts,
                discard_type=jnp.int32(PHYSICAL_WILD_DRAW4),
                current_color=chosen_color.astype(jnp.int32),
            )
            target = player_advance(player, s.direction, 1)
            s = self._draw_n(s, target, 4)
            s = s._replace(current_player=player_advance(player, s.direction, 2))
            return self._post_play_state(s, player, s)

        return jax.lax.cond(
            safe_action == DRAW_ACTION,
            do_draw,
            lambda s: jax.lax.cond(
                safe_action < NUM_COLORED_ACTIONS,
                do_colored,
                lambda x: jax.lax.cond(safe_action < ACTION_WILD_DRAW4_START, do_wild, do_wd4, x),
                s,
            ),
            step_state,
        )


class CardEmbedding(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.Dense(self.embed_dim, name="proj1")(x)
        y = nn.LayerNorm(name="ln1")(y)
        y = nn.gelu(y)
        y = nn.Dense(self.embed_dim, name="proj2")(y)
        return y


class TokenSelfAttentionPool(nn.Module):
    embed_dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        head_dim = self.embed_dim // self.num_heads
        q = nn.DenseGeneral((self.num_heads, head_dim), name="q")(tokens)
        k = nn.DenseGeneral((self.num_heads, head_dim), name="k")(tokens)
        v = nn.DenseGeneral((self.num_heads, head_dim), name="v")(tokens)

        scores = jnp.einsum("thd,shd->hts", q, k) / math.sqrt(float(head_dim))
        valid = mask.astype(jnp.bool_)
        attn_mask = valid[None, :, None] & valid[None, None, :]
        scores = jnp.where(attn_mask, scores, jnp.full_like(scores, -1e9))
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("hts,shd->thd", weights, v).reshape(tokens.shape[0], self.embed_dim)
        out = nn.Dense(self.embed_dim, name="o")(out)
        out = nn.LayerNorm(name="ln_out")(tokens + out)
        mask_f = mask[:, None].astype(jnp.float32)
        denom = jnp.maximum(mask_f.sum(), 1.0)
        pooled = (out * mask_f).sum(axis=0) / denom
        return pooled, weights.mean(axis=0)


class ObservationEncoder(nn.Module):
    hidden_dim: int = 192
    hand_embed_dim: int = 48
    num_heads: int = 4

    @nn.compact
    def __call__(self, obs: UNOObservation) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        hand_emb = CardEmbedding(self.hand_embed_dim, name="hand_card_emb")(obs.hand)
        hand_vec, hand_attention = TokenSelfAttentionPool(
            embed_dim=self.hand_embed_dim,
            num_heads=self.num_heads,
            name="hand_pool",
        )(hand_emb, obs.hand_mask)

        top_vec = CardEmbedding(self.hand_embed_dim // 2, name="top_emb")(obs.top_card)
        scalar_feats = jnp.array(
            [obs.draw_pile_frac, obs.turn_frac, obs.hand_size_frac],
            dtype=jnp.float32,
        )
        extra = jnp.concatenate(
            [
                obs.current_color,
                obs.opponent_counts,
                obs.uno_flags,
                obs.direction,
                obs.last_action,
                obs.played_hist,
                scalar_feats,
            ],
            axis=-1,
        )

        fused = jnp.concatenate([hand_vec, top_vec, extra], axis=-1)
        x = nn.Dense(self.hidden_dim, name="fuse1")(fused)
        x = nn.LayerNorm(name="ln1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, name="fuse2")(x)
        x = nn.LayerNorm(name="ln2")(x)
        x = nn.gelu(x)
        return x, {"hand_attention": hand_attention}


class GRUGatingUnit(nn.Module):
    dim: int
    bias_init: float = 2.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r = nn.Dense(self.dim, use_bias=False, name="wr")(y) + nn.Dense(self.dim, name="ur")(x)
        r = jax.nn.sigmoid(r)
        z = nn.Dense(self.dim, use_bias=False, name="wz")(y) + nn.Dense(
            self.dim,
            bias_init=nn.initializers.constant(-self.bias_init),
            name="uz",
        )(x)
        z = jax.nn.sigmoid(z)
        h_tilde = nn.Dense(self.dim, use_bias=False, name="wh")(y) + nn.Dense(self.dim, name="uh")(r * x)
        h_tilde = jnp.tanh(h_tilde)
        return (1.0 - z) * x + z * h_tilde


class GTrXLLayer(nn.Module):
    hidden_dim: int
    num_heads: int = 4
    mlp_ratio: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, layer_memory: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q = nn.LayerNorm(name="attn_q_ln")(x)[None, :]
        context = jnp.concatenate([nn.LayerNorm(name="attn_mem_ln")(layer_memory), q], axis=0)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            name="mha",
        )(q, context)[0]
        x1 = GRUGatingUnit(self.hidden_dim, name="gru1")(x, attn_out)
        mlp = nn.LayerNorm(name="mlp_ln")(x1)
        mlp = nn.Dense(self.hidden_dim * self.mlp_ratio, name="fc1")(mlp)
        mlp = nn.gelu(mlp)
        mlp = nn.Dense(self.hidden_dim, name="fc2")(mlp)
        x2 = GRUGatingUnit(self.hidden_dim, name="gru2")(x1, mlp)
        new_memory = jnp.concatenate([layer_memory[1:], x2[None, :]], axis=0)
        return x2, new_memory


class RecurrentUNOAgent(nn.Module):
    hidden_dim: int = 192
    hand_embed_dim: int = 48
    num_actions: int = NUM_ACTIONS
    num_layers: int = 2
    memory_len: int = 12
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        obs: UNOObservation,
        carry: TrXLCarry,
        episode_start: Optional[jnp.ndarray] = None,
        capture_aux: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        if episode_start is not None:
            alive = 1.0 - episode_start.astype(jnp.float32)
            carry = TrXLCarry(memory=carry.memory * alive)

        x, aux = ObservationEncoder(
            hidden_dim=self.hidden_dim,
            hand_embed_dim=self.hand_embed_dim,
            num_heads=self.num_heads,
            name="obs_enc",
        )(obs)

        memories = []
        for layer_idx in range(self.num_layers):
            x, mem = GTrXLLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                name=f"gtrxl_{layer_idx}",
            )(x, carry.memory[layer_idx])
            memories.append(mem)
        new_carry = TrXLCarry(memory=jnp.stack(memories, axis=0))

        p = nn.Dense(self.hidden_dim, name="pi_fc")(x)
        p = nn.gelu(p)
        logits = nn.Dense(self.num_actions, name="pi_out")(p)
        masked_logits = jnp.where(obs.action_mask, logits, jnp.full_like(logits, -1e9))

        v = nn.Dense(self.hidden_dim // 2, name="vf_fc1")(x)
        v = nn.gelu(v)
        v = nn.Dense(self.hidden_dim // 4, name="vf_fc2")(v)
        v = nn.gelu(v)
        value = nn.Dense(1, name="vf_out")(v).squeeze(-1)

        b = nn.Dense(self.hidden_dim // 2, name="belief_fc1")(x)
        b = nn.gelu(b)
        belief_logits = nn.Dense((NUM_PLAYERS - 1) * NUM_PHYSICAL_CARD_TYPES, name="belief_out")(b)
        belief_logits = belief_logits.reshape(NUM_PLAYERS - 1, NUM_PHYSICAL_CARD_TYPES)

        out = {
            "policy_logits": masked_logits,
            "value": value,
            "belief_logits": belief_logits,
            "trxl_carry": new_carry,
        }
        if capture_aux:
            out["hand_attention"] = aux["hand_attention"]
        return out

    @staticmethod
    def init_carry(num_layers: int, memory_len: int, hidden_dim: int, batch_size: Optional[int] = None) -> TrXLCarry:
        if batch_size is None:
            shape = (num_layers, memory_len, hidden_dim)
        else:
            shape = (num_layers, batch_size, memory_len, hidden_dim)
        return TrXLCarry(memory=jnp.zeros(shape, dtype=jnp.float32))


class AveragePolicyModel(nn.Module):
    hidden_dim: int = 160
    hand_embed_dim: int = 48
    num_heads: int = 4
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, obs: UNOObservation) -> Dict[str, jnp.ndarray]:
        x, aux = ObservationEncoder(
            hidden_dim=self.hidden_dim,
            hand_embed_dim=self.hand_embed_dim,
            num_heads=self.num_heads,
            name="obs_enc",
        )(obs)
        x = nn.Dense(self.hidden_dim, name="pi_fc")(x)
        x = nn.gelu(x)
        logits = nn.Dense(self.num_actions, name="pi_out")(x)
        masked_logits = jnp.where(obs.action_mask, logits, jnp.full_like(logits, -1e9))
        return {"policy_logits": masked_logits, "hand_attention": aux["hand_attention"]}


def make_dummy_observation() -> UNOObservation:
    return UNOObservation(
        hand=jnp.zeros((MAX_HAND_SIZE, CARD_FEATURE_DIM), dtype=jnp.float32),
        hand_mask=jnp.zeros((MAX_HAND_SIZE,), dtype=jnp.bool_),
        top_card=jnp.zeros((CARD_FEATURE_DIM,), dtype=jnp.float32),
        current_color=jnp.zeros((NUM_COLORS,), dtype=jnp.float32),
        opponent_counts=jnp.zeros((NUM_PLAYERS - 1,), dtype=jnp.float32),
        uno_flags=jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
        direction=jnp.zeros((2,), dtype=jnp.float32),
        draw_pile_frac=jnp.float32(1.0),
        turn_frac=jnp.float32(0.0),
        hand_size_frac=jnp.float32(0.0),
        last_action=jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32),
        played_hist=jnp.zeros((NUM_PHYSICAL_CARD_TYPES,), dtype=jnp.float32),
        action_mask=jnp.ones((NUM_ACTIONS,), dtype=jnp.bool_),
    )


def make_policy_optimizer(lr: float, max_grad_norm: float, warmup_steps: int, total_steps: int) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=lr * 0.1,
    )
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(schedule, eps=1e-5),
    )


def make_avg_optimizer(lr: float) -> optax.GradientTransformation:
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr, eps=1e-5))


def compute_gae(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> jnp.ndarray:
    not_done = 1.0 - dones.astype(jnp.float32)
    next_values = jnp.concatenate([values[1:], bootstrap_value[None]], axis=0)
    deltas = rewards + gamma * next_values * not_done - values

    def body(carry: jnp.ndarray, xs: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        delta, nd = xs
        adv = delta + gamma * gae_lambda * nd * carry
        return adv, adv

    _, advantages_rev = jax.lax.scan(
        body,
        jnp.zeros_like(bootstrap_value),
        (deltas[::-1], not_done[::-1]),
    )
    return advantages_rev[::-1]


def masked_categorical_sample(key: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    return jax.random.categorical(key, logits)


def action_scores_for_greedy(state: UNOState, env: UNO61Env, player: jnp.ndarray) -> jnp.ndarray:
    legal = env.legal_action_mask(state, player)
    hand = state.hands[player]
    hand_size = jnp.sum(hand).astype(jnp.float32)
    next_player = player_advance(player, state.direction, 1)
    next_hand_size = jnp.sum(state.hands[next_player]).astype(jnp.float32)
    color_mass = hand[:NUM_COLORED_ACTIONS].reshape(NUM_COLORS, NUM_COLORED_RANKS).sum(axis=1).astype(jnp.float32)

    scores = jnp.full((NUM_ACTIONS,), -1e9, dtype=jnp.float32)
    scores = scores.at[DRAW_ACTION].set(0.0)

    colored_bonus = 8.0 + 0.5 * color_mass[COLORED_COLORS]
    colored_bonus = colored_bonus + jnp.where(COLORED_RANKS == SKIP_RANK, 14.0, 0.0)
    colored_bonus = colored_bonus + jnp.where(COLORED_RANKS == REVERSE_RANK, 10.0, 0.0)
    colored_bonus = colored_bonus + jnp.where(COLORED_RANKS == DRAW2_RANK, 18.0, 0.0)
    colored_bonus = colored_bonus + jnp.where((next_hand_size <= 2.0) & (COLORED_RANKS >= SKIP_RANK), 18.0, 0.0)
    colored_bonus = colored_bonus + jnp.where(hand_size <= 1.0, 100.0, 0.0)
    scores = scores.at[:NUM_COLORED_ACTIONS].set(colored_bonus)

    wild_bonus = 14.0 + color_mass
    wd4_bonus = 26.0 + color_mass + jnp.where(next_hand_size <= 2.0, 20.0, 0.0)
    scores = scores.at[ACTION_WILD_START:ACTION_WILD_DRAW4_START].set(wild_bonus)
    scores = scores.at[ACTION_WILD_DRAW4_START:DRAW_ACTION].set(wd4_bonus)
    return jnp.where(legal, scores, -1e9)


def make_full_turn_fn(env: UNO61Env, avg_model: AveragePolicyModel):
    def full_turn_single(
        state: UNOState,
        hero_action: jnp.ndarray,
        opponent_params: Any,
        opponent_mode: int,
        shaping_weight: jnp.float32, # <-- New Parameter
    ) -> Tuple[UNOState, jnp.ndarray, jnp.ndarray]:

        # Track hand size before the action
        old_hand_size = jnp.sum(state.hands[state.hero_seat]).astype(jnp.float32)

        state = env.apply_action(state, state.hero_seat, hero_action)

        def select_opponent_action(s: UNOState) -> Tuple[UNOState, jnp.ndarray]:
            # ... (keep existing opponent action logic) ...
            player = s.current_player
            obs, _ = env.observe_for_player(s, player)
            s, keys = env._split_keys(s, 1)
            action_key = keys[0]
            if opponent_mode == OPP_RANDOM:
                logits = jnp.where(obs.action_mask, 0.0, -1e9)
                action = masked_categorical_sample(action_key, logits)
            elif opponent_mode == OPP_GREEDY:
                action = jnp.argmax(action_scores_for_greedy(s, env, player))
            else:
                outs = avg_model.apply({"params": opponent_params}, obs)
                action = masked_categorical_sample(action_key, outs["policy_logits"])
            return s, action.astype(jnp.int32)

        def cond_fn(s: UNOState) -> jnp.ndarray:
            return (~s.done) & (s.current_player != s.hero_seat)

        def body_fn(s: UNOState) -> UNOState:
            s, opp_action = select_opponent_action(s)
            return env.apply_action(s, s.current_player, opp_action)

        final_state = jax.lax.while_loop(cond_fn, body_fn, state)

        # Track hand size after the turn resolves
        new_hand_size = jnp.sum(final_state.hands[final_state.hero_seat]).astype(jnp.float32)

        # --- NEW REWARD LOGIC ---

        # 1. True Sparse Reward
        sparse_reward = jnp.where(
            final_state.done,
            jnp.where(final_state.winner == final_state.hero_seat, 1.0, -1.0),
            0.0,
        ).astype(jnp.float32)

        # 2. Dense Shaped Reward
        step_penalty = -0.01
        hand_delta_reward = (old_hand_size - new_hand_size) * 0.05
        dense_reward = step_penalty + hand_delta_reward

        # 3. Annealed Hybrid Reward
        # When weight is 1.0, dense dominates. When weight is 0.0, only sparse remains.
        reward = sparse_reward + (shaping_weight * dense_reward)

        done = final_state.done
        reset_state = env.reset(final_state.key, final_state.hero_seat)
        out_state = jax.tree_util.tree_map(lambda r, n: jnp.where(done, r, n), reset_state, final_state)

        return out_state, reward, done

    return full_turn_single

def make_rollout_collector(
    env: UNO61Env,
    model: RecurrentUNOAgent,
    avg_model: AveragePolicyModel,
    cfg: TrainConfig,
):
    full_turn_single = make_full_turn_fn(env, avg_model)
    batched_observe = jax.vmap(env.observe_for_player)
    batched_full_turn = jax.vmap(full_turn_single, in_axes=(0, 0, None, None,None))

    def apply_single(obs_i: UNOObservation, mem_i: jnp.ndarray, start_i: jnp.ndarray, params: Any) -> Dict[str, jnp.ndarray]:
        return model.apply({"params": params}, obs_i, TrXLCarry(memory=mem_i), start_i)

    @partial(jax.jit, static_argnames=("opponent_mode",))
    def collect(
        env_states: UNOState,
        carry: TrXLCarry,
        episode_starts: jnp.ndarray,
        params: Any,
        opponent_params: Any,
        key: jnp.ndarray,
        opponent_mode: int,
        shaping_weight: jnp.float32,
    ) -> Tuple[TrajectoryBatch, UNOState, TrXLCarry, jnp.ndarray]:
        init_memory = carry.memory

        def step_fn(
            loop_carry: Tuple[UNOState, TrXLCarry, jnp.ndarray, jnp.ndarray],
            _,
        ) -> Tuple[Tuple[UNOState, TrXLCarry, jnp.ndarray, jnp.ndarray], Tuple[UNOObservation, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            states, trxl_carry, starts, rng = loop_carry
            players = states.hero_seat
            obs, belief_targets = batched_observe(states, players)
            outs = jax.vmap(apply_single, in_axes=(0, 1, 0, None))(obs, trxl_carry.memory, starts, params)
            next_memory = jnp.swapaxes(outs["trxl_carry"].memory, 0, 1)

            rng, sample_rng = jax.random.split(rng)
            action_keys = jax.random.split(sample_rng, cfg.num_envs)
            actions = jax.vmap(masked_categorical_sample)(action_keys, outs["policy_logits"]).astype(jnp.int32)

            log_probs = jax.nn.log_softmax(outs["policy_logits"], axis=-1)
            chosen_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)
            next_states, rewards, dones = batched_full_turn(states, actions, opponent_params, opponent_mode,shaping_weight)
            alive = 1.0 - dones.astype(jnp.float32)
            next_carry = TrXLCarry(memory=next_memory * alive[None, :, None, None])
            next_starts = dones
            output = (
                obs,
                starts,
                actions,
                rewards,
                dones,
                chosen_log_probs,
                outs["value"],
                belief_targets,
            )
            return (next_states, next_carry, next_starts, rng), output

        (final_states, final_carry, final_starts, final_key), traj = jax.lax.scan(
            step_fn,
            (env_states, carry, episode_starts, key),
            None,
            length=cfg.rollout_len,
        )

        bootstrap_obs, _ = batched_observe(final_states, final_states.hero_seat)
        batch = TrajectoryBatch(
            obs=traj[0],
            bootstrap_obs=bootstrap_obs,
            episode_starts=traj[1],
            bootstrap_starts=final_starts,
            actions=traj[2],
            rewards=traj[3],
            dones=traj[4],
            old_log_probs=traj[5],
            old_values=traj[6],
            belief_targets=traj[7],
            init_memory=init_memory,
        )
        return batch, final_states, final_carry, final_starts

    return collect


def ppo_loss(
    params: Any,
    model: RecurrentUNOAgent,
    batch: TrajectoryBatch,
    cfg: TrainConfig,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    init_carry = TrXLCarry(memory=batch.init_memory)

    def apply_single(obs_i: UNOObservation, mem_i: jnp.ndarray, start_i: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return model.apply({"params": params}, obs_i, TrXLCarry(memory=mem_i), start_i)

    def scan_step(carry: TrXLCarry, xs: Tuple[UNOObservation, jnp.ndarray]) -> Tuple[TrXLCarry, Dict[str, jnp.ndarray]]:
        obs_t, start_t = xs
        outs = jax.vmap(apply_single, in_axes=(0, 1, 0))(obs_t, carry.memory, start_t)
        new_carry = TrXLCarry(memory=jnp.swapaxes(outs["trxl_carry"].memory, 0, 1))
        return new_carry, {
            "policy_logits": outs["policy_logits"],
            "value": outs["value"],
            "belief_logits": outs["belief_logits"],
        }

    carry_after, rollout = jax.lax.scan(scan_step, init_carry, (batch.obs, batch.episode_starts))
    bootstrap = jax.vmap(apply_single, in_axes=(0, 1, 0))(batch.bootstrap_obs, carry_after.memory, batch.bootstrap_starts)
    bootstrap_value = jax.lax.stop_gradient(bootstrap["value"])

    log_pi = jax.nn.log_softmax(rollout["policy_logits"], axis=-1)
    new_log_probs = jnp.take_along_axis(log_pi, batch.actions[..., None], axis=-1).squeeze(-1)

    advantages = compute_gae(
        rewards=batch.rewards,
        dones=batch.dones,
        values=rollout["value"],
        bootstrap_value=bootstrap_value,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )
    advantages = jax.lax.stop_gradient(advantages)
    returns = jax.lax.stop_gradient(advantages + rollout["value"])
    norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = jnp.exp(new_log_probs - batch.old_log_probs)
    unclipped = -norm_adv * ratio
    clipped = -norm_adv * jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
    pg_loss = jnp.maximum(unclipped, clipped).mean()

    value_pred = rollout["value"]
    value_clipped = batch.old_values + jnp.clip(value_pred - batch.old_values, -cfg.clip_eps, cfg.clip_eps)
    vf_loss = 0.5 * jnp.maximum((value_pred - returns) ** 2, (value_clipped - returns) ** 2).mean()

    probs = jax.nn.softmax(rollout["policy_logits"], axis=-1)
    entropy = -(probs * log_pi).sum(axis=-1).mean()

    belief_loss = optax.sigmoid_binary_cross_entropy(
        rollout["belief_logits"],
        batch.belief_targets.astype(jnp.float32),
    ).mean()

    total_loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy + cfg.belief_coef * belief_loss
    approx_kl = ((ratio - 1.0) - jnp.log(ratio + 1e-8)).mean()
    expl_var = 1.0 - jnp.var(returns - value_pred) / (jnp.var(returns) + 1e-8)
    metrics = {
        "total_loss": total_loss,
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "entropy": entropy,
        "belief_loss": belief_loss,
        "approx_kl": approx_kl,
        "explained_variance": expl_var,
        "mean_return": returns.mean(),
    }
    return total_loss, metrics


def make_recurrent_train_step(
    model: RecurrentUNOAgent,
    optimizer: optax.GradientTransformation,
    cfg: TrainConfig,
):
    @jax.jit
    def train_step(state: PPOTrainState, batch: TrajectoryBatch) -> Tuple[PPOTrainState, Dict[str, jnp.ndarray]]:
        def loss_fn(p):
            return ppo_loss(p, model, batch, cfg)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return PPOTrainState(params=new_params, opt_state=new_opt_state, step=state.step + 1), metrics

    return train_step


def average_policy_loss(params: Any, model: AveragePolicyModel, obs: UNOObservation, actions: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    outs = jax.vmap(lambda o: model.apply({"params": params}, o))(obs)
    log_pi = jax.nn.log_softmax(outs["policy_logits"], axis=-1)
    nll = -jnp.take_along_axis(log_pi, actions[:, None], axis=-1).squeeze(-1).mean()
    entropy = -(jax.nn.softmax(outs["policy_logits"], axis=-1) * log_pi).sum(axis=-1).mean()
    metrics = {"ce_loss": nll, "entropy": entropy}
    return nll, metrics


def make_average_train_step(model: AveragePolicyModel, optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(state: AvgTrainState, obs: UNOObservation, actions: jnp.ndarray) -> Tuple[AvgTrainState, Dict[str, jnp.ndarray]]:
        def loss_fn(p):
            return average_policy_loss(p, model, obs, actions)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return AvgTrainState(params=new_params, opt_state=new_opt_state, step=state.step + 1), metrics

    return train_step


def flatten_observation(obs: UNOObservation) -> Dict[str, np.ndarray]:
    return {field: np.asarray(getattr(obs, field)) for field in obs._fields}


def unflatten_observation(data: Dict[str, np.ndarray], indices: np.ndarray) -> UNOObservation:
    return UNOObservation(**{field: jnp.asarray(value[indices]) for field, value in data.items()})


class ReservoirBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.seen = 0
        self.obs_storage: Optional[Dict[str, np.ndarray]] = None
        self.action_storage: Optional[np.ndarray] = None

    def _ensure_storage(self, flat_obs: Dict[str, np.ndarray], actions: np.ndarray) -> None:
        if self.obs_storage is not None:
            return
        self.obs_storage = {}
        for key, value in flat_obs.items():
            self.obs_storage[key] = np.zeros((self.capacity,) + value.shape[1:], dtype=value.dtype)
        self.action_storage = np.zeros((self.capacity,), dtype=actions.dtype)

    def add_trajectory(self, batch: TrajectoryBatch) -> None:
        obs_np = flatten_observation(batch.obs)
        actions_np = np.asarray(batch.actions)
        flat_obs = {key: value.reshape((-1,) + value.shape[2:]) for key, value in obs_np.items()}
        flat_actions = actions_np.reshape(-1)
        self._ensure_storage(flat_obs, flat_actions)
        assert self.obs_storage is not None and self.action_storage is not None

        total = flat_actions.shape[0]
        for i in range(total):
            self.seen += 1
            if self.size < self.capacity:
                idx = self.size
                self.size += 1
            else:
                idx = np.random.randint(0, self.seen)
                if idx >= self.capacity:
                    continue
            for key, value in flat_obs.items():
                self.obs_storage[key][idx] = value[i]
            self.action_storage[idx] = flat_actions[i]

    def sample(self, batch_size: int) -> Optional[Tuple[UNOObservation, jnp.ndarray]]:
        if self.size == 0 or self.obs_storage is None or self.action_storage is None:
            return None
        take = min(batch_size, self.size)
        indices = np.random.randint(0, self.size, size=(take,))
        obs = unflatten_observation(self.obs_storage, indices)
        actions = jnp.asarray(self.action_storage[indices])
        return obs, actions


@dataclass
class OpponentSpec:
    label: str
    mode: int
    params: Any = None


class LeagueManager:
    def __init__(self, history_limit: int = 8):
        self.history_limit = history_limit
        self.history: List[OpponentSpec] = []
        self.ratings: Dict[str, float] = {
            "main": 1200.0,
            "exploiter": 1200.0,
            "random": 1200.0,
            "greedy": 1200.0,
            "main_avg": 1200.0,
            "exploiter_avg": 1200.0,
        }

    def snapshot_main_average(self, params: Any, update: int) -> None:
        copied = jax.tree_util.tree_map(lambda x: np.array(x), jax.device_get(params))
        self.history.append(OpponentSpec(label=f"hist_avg_{update}", mode=OPP_AVERAGE, params=copied))
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]
        self.ratings.setdefault(f"hist_avg_{update}", 1200.0)

    def sample_for_main(self, main_avg_params: Any, exploiter_avg_params: Any) -> OpponentSpec:
        pool = [
            (0.15, OpponentSpec("random", OPP_RANDOM, None)),
            (0.15, OpponentSpec("greedy", OPP_GREEDY, None)),
            (0.30, OpponentSpec("main_avg", OPP_AVERAGE, main_avg_params)),
            (0.20, OpponentSpec("exploiter_avg", OPP_AVERAGE, exploiter_avg_params)),
        ]
        if self.history:
            hist = self.history[np.random.randint(0, len(self.history))]
            pool.append((0.20, hist))
        probs = np.array([w for w, _ in pool], dtype=np.float64)
        probs /= probs.sum()
        idx = np.random.choice(len(pool), p=probs)
        return pool[idx][1]

    def opponent_for_exploiter(self, main_avg_params: Any) -> OpponentSpec:
        return OpponentSpec("main_avg", OPP_AVERAGE, main_avg_params)


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 24.0) -> Tuple[float, float]:
    ea = elo_expected(rating_a, rating_b)
    eb = elo_expected(rating_b, rating_a)
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1.0 - score_a) - eb)
    return new_a, new_b


def update_elo_table(ratings: Dict[str, float], main_label: str, results: Dict[str, float]) -> Dict[str, float]:
    updated = dict(ratings)
    for opp_label, win_rate in results.items():
        if opp_label not in updated:
            updated[opp_label] = 1200.0
        updated[main_label], updated[opp_label] = elo_update(updated.get(main_label, 1200.0), updated[opp_label], win_rate)
    return updated


def init_recurrent_state(model: RecurrentUNOAgent, cfg: TrainConfig, key: jnp.ndarray) -> PPOTrainState:
    dummy_obs = make_dummy_observation()
    dummy_carry = RecurrentUNOAgent.init_carry(cfg.num_layers, cfg.memory_len, cfg.hidden_dim)
    params = model.init(key, dummy_obs, dummy_carry, jnp.bool_(False))["params"]
    optimizer = make_policy_optimizer(cfg.lr, cfg.max_grad_norm, cfg.warmup_steps, cfg.total_updates * cfg.ppo_epochs)
    return PPOTrainState(params=params, opt_state=optimizer.init(params), step=jnp.int32(0))


def init_average_state(model: AveragePolicyModel, cfg: TrainConfig, key: jnp.ndarray) -> AvgTrainState:
    dummy_obs = make_dummy_observation()
    params = model.init(key, dummy_obs)["params"]
    optimizer = make_avg_optimizer(cfg.avg_lr)
    return AvgTrainState(params=params, opt_state=optimizer.init(params), step=jnp.int32(0))


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(jax.device_get(payload), fh)


def load_checkpoint(path: str) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def run_single_episode(
    env: UNO61Env,
    model: RecurrentUNOAgent,
    avg_model: AveragePolicyModel,
    params: Any,
    opponent: OpponentSpec,
    cfg: TrainConfig,
    seed: int,
) -> float:
    full_turn = make_full_turn_fn(env, avg_model)
    key = jax.random.PRNGKey(seed)
    state = env.reset(key, jnp.int32(0))
    carry = RecurrentUNOAgent.init_carry(cfg.num_layers, cfg.memory_len, cfg.hidden_dim)
    episode_start = jnp.bool_(True)

    for _ in range(cfg.max_turns):
        obs, _ = env.observe_for_player(state, state.hero_seat)
        outs = model.apply({"params": params}, obs, carry, episode_start)
        action = int(np.asarray(jnp.argmax(outs["policy_logits"])))
        state, reward, done = full_turn(state, jnp.int32(action), opponent.params, opponent.mode,jnp.float32(0.0))
        carry = TrXLCarry(memory=outs["trxl_carry"].memory * (1.0 - done.astype(jnp.float32)))
        episode_start = done
        if bool(np.asarray(done)):
            return float(np.asarray(reward))
    return -1.0


def evaluate_suite(
    env: UNO61Env,
    model: RecurrentUNOAgent,
    avg_model: AveragePolicyModel,
    params: Any,
    main_avg_params: Any,
    exploiter_avg_params: Any,
    league: LeagueManager,
    cfg: TrainConfig,
    episodes: int,
) -> Dict[str, float]:
    opponents = [
        OpponentSpec("random", OPP_RANDOM, None),
        OpponentSpec("greedy", OPP_GREEDY, None),
        OpponentSpec("main_avg", OPP_AVERAGE, main_avg_params),
        OpponentSpec("exploiter_avg", OPP_AVERAGE, exploiter_avg_params),
    ]
    if league.history:
        opponents.append(league.history[-1])

    results: Dict[str, float] = {}
    for opponent in opponents:
        wins = 0.0
        for episode in range(episodes):
            reward = run_single_episode(env, model, avg_model, params, opponent, cfg, seed=cfg.seed + episode)
            wins += 1.0 if reward > 0 else 0.0
        results[opponent.label] = wins / float(episodes)
    return results


def compute_saliency_report(
    env: UNO61Env,
    model: RecurrentUNOAgent,
    params: Any,
    cfg: TrainConfig,
    seed: int,
) -> Dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    state = env.reset(key, jnp.int32(0))
    obs, _ = env.observe_for_player(state, state.hero_seat)
    carry = RecurrentUNOAgent.init_carry(cfg.num_layers, cfg.memory_len, cfg.hidden_dim)

    def chosen_logit(hand_tokens: jnp.ndarray) -> jnp.ndarray:
        edited = obs._replace(hand=hand_tokens)
        outs = model.apply({"params": params}, edited, carry, jnp.bool_(True))
        action = jnp.argmax(outs["policy_logits"])
        return outs["policy_logits"][action]

    grads = jax.grad(chosen_logit)(obs.hand)
    saliency = jnp.linalg.norm(grads, axis=-1) * obs.hand_mask.astype(jnp.float32)
    outs = model.apply({"params": params}, obs, carry, jnp.bool_(True), capture_aux=True)
    chosen_action = int(np.asarray(jnp.argmax(outs["policy_logits"])))

    hand_card_indices = []
    mask_np = np.asarray(obs.hand_mask)
    hand_np = np.asarray(obs.hand)
    for i in range(int(mask_np.sum())):
        matches = np.where((np.asarray(CARD_FEATURE_TABLE) == hand_np[i]).all(axis=1))[0]
        hand_card_indices.append(int(matches[0]) if len(matches) else -1)

    ranked = np.argsort(-np.asarray(saliency))[: min(5, int(mask_np.sum()))]
    top_cards = [
        {
            "slot": int(idx),
            "card": decode_physical_card(hand_card_indices[idx]) if idx < len(hand_card_indices) and hand_card_indices[idx] >= 0 else "unknown",
            "saliency": float(np.asarray(saliency[idx])),
        }
        for idx in ranked
    ]
    return {
        "chosen_action": decode_action(chosen_action),
        "top_salient_cards": top_cards,
        "hand_attention_shape": list(np.asarray(outs["hand_attention"]).shape),
    }


def train_command(args: argparse.Namespace) -> None:
    total_updates = args.total_updates
    if getattr(args, "total_steps", None) is not None:
        total_updates = max(1, math.ceil(args.total_steps / max(1, args.num_envs * args.rollout_len)))

    cfg = TrainConfig(
        seed=args.seed,
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        total_updates=total_updates,
        hidden_dim=args.hidden_dim,
        memory_len=args.memory_len,
        num_layers=args.num_layers,
        reservoir_capacity=args.reservoir_capacity,
        avg_batch_size=args.avg_batch_size,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
    )
    env = UNO61Env(max_turns=cfg.max_turns, cards_per_player=cfg.cards_per_player)
    model = RecurrentUNOAgent(
        hidden_dim=cfg.hidden_dim,
        hand_embed_dim=cfg.hand_embed_dim,
        num_layers=cfg.num_layers,
        memory_len=cfg.memory_len,
        num_heads=cfg.num_heads,
    )
    avg_model = AveragePolicyModel(
        hidden_dim=cfg.hidden_dim - 32,
        hand_embed_dim=cfg.hand_embed_dim,
        num_heads=cfg.num_heads,
    )

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_main, k_expl, k_avg_main, k_avg_expl, k_env = jax.random.split(rng, 6)

    main_state = init_recurrent_state(model, cfg, k_main)
    exploiter_state = init_recurrent_state(model, cfg, k_expl)
    main_avg_state = init_average_state(avg_model, cfg, k_avg_main)
    exploiter_avg_state = init_average_state(avg_model, cfg, k_avg_expl)

    recurrent_optimizer = make_policy_optimizer(cfg.lr, cfg.max_grad_norm, cfg.warmup_steps, cfg.total_updates * cfg.ppo_epochs)
    avg_optimizer = make_avg_optimizer(cfg.avg_lr)
    recurrent_step = make_recurrent_train_step(model, recurrent_optimizer, cfg)
    average_step = make_average_train_step(avg_model, avg_optimizer)
    collect_rollout = make_rollout_collector(env, model, avg_model, cfg)

    env_keys = jax.random.split(k_env, cfg.num_envs)
    batched_reset = jax.vmap(env.reset, in_axes=(0, None))
    env_states = batched_reset(env_keys, jnp.int32(0))
    exploiter_keys = jax.random.split(jax.random.fold_in(k_env, 1), cfg.num_envs)
    exploiter_env_states = batched_reset(exploiter_keys, jnp.int32(0))
    main_carry = RecurrentUNOAgent.init_carry(cfg.num_layers, cfg.memory_len, cfg.hidden_dim, batch_size=cfg.num_envs)
    exploiter_carry = RecurrentUNOAgent.init_carry(cfg.num_layers, cfg.memory_len, cfg.hidden_dim, batch_size=cfg.num_envs)
    main_starts = jnp.ones((cfg.num_envs,), dtype=jnp.bool_)
    exploiter_starts = jnp.ones((cfg.num_envs,), dtype=jnp.bool_)

    main_reservoir = ReservoirBuffer(cfg.reservoir_capacity)
    exploiter_reservoir = ReservoirBuffer(cfg.reservoir_capacity // 2)
    league = LeagueManager(history_limit=cfg.history_limit)

    os.makedirs(cfg.output_dir, exist_ok=True)
    # Dictionary to store metrics for our live plot
    plot_history = {"update": [], "return": [], "entropy": [], "belief_loss": []}
    for update in range(1, cfg.total_updates + 1):
        decay_fraction = max(0.0, 1.0 - (update / (cfg.total_updates * 0.5)))
        current_shaping_weight = jnp.float32(decay_fraction)
        step_start = time.perf_counter()
        opponent = league.sample_for_main(main_avg_state.params, exploiter_avg_state.params)
        rng, rollout_key = jax.random.split(rng)
        batch, env_states, main_carry, main_starts = collect_rollout(
            env_states,
            main_carry,
            main_starts,
            main_state.params,
            main_avg_state.params if opponent.params is None else opponent.params,
            rollout_key,
            opponent.mode,
            current_shaping_weight,
        )

        last_metrics = {}
        for _ in range(cfg.ppo_epochs):
            main_state, last_metrics = recurrent_step(main_state, batch)
        jax.block_until_ready(last_metrics["total_loss"])

        main_reservoir.add_trajectory(batch)
        avg_metrics = {"ce_loss": jnp.array(0.0), "entropy": jnp.array(0.0)}
        for _ in range(cfg.avg_updates_per_round):
            sample = main_reservoir.sample(cfg.avg_batch_size)
            if sample is None:
                break
            obs_batch, act_batch = sample
            main_avg_state, avg_metrics = average_step(main_avg_state, obs_batch, act_batch)

        expl_metrics = {"total_loss": jnp.array(0.0), "mean_return": jnp.array(0.0)}
        if update % cfg.exploiter_every == 0:
            expl_opp = league.opponent_for_exploiter(main_avg_state.params)
            rng, expl_rollout_key = jax.random.split(rng)
            expl_batch, exploiter_env_states, exploiter_carry, exploiter_starts = collect_rollout(
                env_states=exploiter_env_states,
                carry=exploiter_carry,
                episode_starts=exploiter_starts,
                params=exploiter_state.params,
                opponent_params=expl_opp.params,
                key=expl_rollout_key,
                opponent_mode=expl_opp.mode,
                shaping_weight=current_shaping_weight,
            )
            for _ in range(cfg.ppo_epochs):
                exploiter_state, expl_metrics = recurrent_step(exploiter_state, expl_batch)
            jax.block_until_ready(expl_metrics["total_loss"])
            exploiter_reservoir.add_trajectory(expl_batch)
            sample = exploiter_reservoir.sample(cfg.avg_batch_size)
            if sample is not None:
                obs_batch, act_batch = sample
                exploiter_avg_state, _ = average_step(exploiter_avg_state, obs_batch, act_batch)

        if update % cfg.snapshot_every == 0:
            league.snapshot_main_average(main_avg_state.params, update)

        eval_results: Dict[str, float] = {}
        if update % cfg.eval_every == 0:
            eval_results = evaluate_suite(
                env=env,
                model=model,
                avg_model=avg_model,
                params=main_state.params,
                main_avg_params=main_avg_state.params,
                exploiter_avg_params=exploiter_avg_state.params,
                league=league,
                cfg=cfg,
                episodes=cfg.eval_episodes,
            )
            league.ratings = update_elo_table(league.ratings, "main", eval_results)

        if update % cfg.checkpoint_every == 0 or update == cfg.total_updates:
            payload = {
                "cfg": cfg,
                "main_state": main_state,
                "main_avg_state": main_avg_state,
                "exploiter_state": exploiter_state,
                "exploiter_avg_state": exploiter_avg_state,
                "ratings": league.ratings,
                "history": league.history,
            }
            save_checkpoint(os.path.join(cfg.output_dir, f"ckpt_{update:06d}.pkl"), payload)
            save_checkpoint(os.path.join(cfg.output_dir, "latest.pkl"), payload)

        fps = (cfg.num_envs * cfg.rollout_len) / max(time.perf_counter() - step_start, 1e-6)
        line = {
            "update": update,
            "opponent": opponent.label,
            "loss": float(np.asarray(last_metrics["total_loss"])),
            "return": float(np.asarray(last_metrics["mean_return"])),
            "entropy": float(np.asarray(last_metrics["entropy"])),
            "belief_loss": float(np.asarray(last_metrics["belief_loss"])),
            "avg_ce": float(np.asarray(avg_metrics["ce_loss"])),
            "expl_return": float(np.asarray(expl_metrics["mean_return"])),
            "fps": float(fps),
        }
        if eval_results:
            line["eval"] = eval_results
            line["elo"] = league.ratings.get("main", 1200.0)
        # Store the metrics instead of printing them
        plot_history["update"].append(update)
        plot_history["return"].append(float(np.asarray(last_metrics["mean_return"])))
        plot_history["entropy"].append(float(np.asarray(last_metrics["entropy"])))
        plot_history["belief_loss"].append(float(np.asarray(last_metrics["belief_loss"])))

        # Update the visual graph every 25 updates to save frontend memory
        if update % 25 == 0:
            clear_output(wait=True)
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            # Plot 1: Agent Return
            axs[0].plot(plot_history["update"], plot_history["return"], color="#1f77b4", linewidth=2)
            axs[0].set_title("Agent Mean Return")
            axs[0].set_xlabel("Update Step")
            axs[0].grid(True, linestyle="--", alpha=0.7)

            # Plot 2: Policy Entropy
            axs[1].plot(plot_history["update"], plot_history["entropy"], color="#ff7f0e", linewidth=2)
            axs[1].set_title("Policy Entropy (Exploration)")
            axs[1].set_xlabel("Update Step")
            axs[1].grid(True, linestyle="--", alpha=0.7)

            # Plot 3: Belief State Loss
            axs[2].plot(plot_history["update"], plot_history["belief_loss"], color="#2ca02c", linewidth=2)
            axs[2].set_title("Belief State Loss")
            axs[2].set_xlabel("Update Step")
            axs[2].grid(True, linestyle="--", alpha=0.7)

            plt.tight_layout()
            plt.show()
            plt.close(fig)


def eval_command(args: argparse.Namespace) -> None:
    payload = load_checkpoint(args.checkpoint)
    cfg = payload["cfg"]
    env = UNO61Env(max_turns=cfg.max_turns, cards_per_player=cfg.cards_per_player)
    model = RecurrentUNOAgent(
        hidden_dim=cfg.hidden_dim,
        hand_embed_dim=cfg.hand_embed_dim,
        num_layers=cfg.num_layers,
        memory_len=cfg.memory_len,
        num_heads=cfg.num_heads,
    )
    avg_model = AveragePolicyModel(
        hidden_dim=cfg.hidden_dim - 32,
        hand_embed_dim=cfg.hand_embed_dim,
        num_heads=cfg.num_heads,
    )
    league = LeagueManager(history_limit=cfg.history_limit)
    league.history = payload.get("history", [])
    results = evaluate_suite(
        env=env,
        model=model,
        avg_model=avg_model,
        params=payload["main_state"].params,
        main_avg_params=payload["main_avg_state"].params,
        exploiter_avg_params=payload["exploiter_avg_state"].params,
        league=league,
        cfg=cfg,
        episodes=args.episodes,
    )
    print(json.dumps(results, indent=2))


def saliency_command(args: argparse.Namespace) -> None:
    payload = load_checkpoint(args.checkpoint)
    cfg = payload["cfg"]
    env = UNO61Env(max_turns=cfg.max_turns, cards_per_player=cfg.cards_per_player)
    model = RecurrentUNOAgent(
        hidden_dim=cfg.hidden_dim,
        hand_embed_dim=cfg.hand_embed_dim,
        num_layers=cfg.num_layers,
        memory_len=cfg.memory_len,
        num_heads=cfg.num_heads,
    )
    report = compute_saliency_report(env, model, payload["main_state"].params, cfg, seed=args.seed)
    print(json.dumps(report, indent=2))


def run_colab_train(
    output_dir: str = "/content/uno_t4",
    num_envs: int = 128,
    rollout_len: int = 32,
    total_updates: int = 25,
    total_steps: Optional[int] = 100_000,
    hidden_dim: int = 256,
    memory_len: int = 16,
    num_layers: int = 2,
    reservoir_capacity: int = 50_000,
    avg_batch_size: int = 256,
    eval_every: int = 25,
    checkpoint_every: int = 25,
    seed: int = 0,
) -> int:
    argv = [
        "train",
        "--output-dir",
        output_dir,
        "--num-envs",
        str(num_envs),
        "--rollout-len",
        str(rollout_len),
        "--total-updates",
        str(total_updates),
        "--hidden-dim",
        str(hidden_dim),
        "--memory-len",
        str(memory_len),
        "--num-layers",
        str(num_layers),
        "--reservoir-capacity",
        str(reservoir_capacity),
        "--avg-batch-size",
        str(avg_batch_size),
        "--eval-every",
        str(eval_every),
        "--checkpoint-every",
        str(checkpoint_every),
        "--seed",
        str(seed),
    ]
    if total_steps is not None:
        argv.extend(["--total-steps", str(total_steps)])
    return main(argv)


def run_colab_eval(checkpoint: str, episodes: int = 64) -> int:
    return main(["eval", "--checkpoint", checkpoint, "--episodes", str(episodes)])


def run_colab_saliency(checkpoint: str, seed: int = 0) -> int:
    return main(["saliency", "--checkpoint", checkpoint, "--seed", str(seed)])


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud-ready UNO RL trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="train the league")
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--num-envs", type=int, default=32)
    train_p.add_argument("--rollout-len", type=int, default=16)
    train_p.add_argument("--total-updates", type=int, default=2000)
    train_p.add_argument("--total-steps", type=int, default=None)
    train_p.add_argument("--hidden-dim", type=int, default=192)
    train_p.add_argument("--memory-len", type=int, default=12)
    train_p.add_argument("--num-layers", type=int, default=2)
    train_p.add_argument("--reservoir-capacity", type=int, default=50_000)
    train_p.add_argument("--avg-batch-size", type=int, default=256)
    train_p.add_argument("--eval-every", type=int, default=250)
    train_p.add_argument("--checkpoint-every", type=int, default=500)
    train_p.add_argument("--output-dir", type=str, default="cloud_runs/uno_t4")

    eval_p = subparsers.add_parser("eval", help="evaluate a checkpoint")
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument("--episodes", type=int, default=64)

    sal_p = subparsers.add_parser("saliency", help="produce a saliency report")
    sal_p.add_argument("--checkpoint", type=str, required=True)
    sal_p.add_argument("--seed", type=int, default=0)

    return parser


def _running_inside_ipykernel() -> bool:
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def _sanitize_argv(argv: Optional[Sequence[str]] = None) -> List[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    cleaned: List[str] = []
    skip_next = False
    for token in raw:
        if skip_next:
            skip_next = False
            continue
        if token == "-f":
            skip_next = True
            continue
        if token.startswith("--f="):
            continue
        if token.endswith(".json") and "kernel-" in token:
            continue
        cleaned.append(token)
    return cleaned


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    cleaned_argv = _sanitize_argv(argv)

    if _running_inside_ipykernel() and not cleaned_argv:
        parser.print_help()
        print(
            "\nColab usage examples:\n"
            "  main(['train', '--num-envs', '32', '--rollout-len', '16', '--total-updates', '2000', '--output-dir', '/content/uno_t4'])\n"
            "  main(['eval', '--checkpoint', '/content/uno_t4/latest.pkl', '--episodes', '64'])\n"
            "  main(['saliency', '--checkpoint', '/content/uno_t4/latest.pkl'])"
        )
        return 0

    args = parser.parse_args(cleaned_argv)
    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    else:
        saliency_command(args)
    return 0

# ── Notebook / CLI entrypoint ─────────────────────────────────────────────────
# Guarded so that importing this module for web serving does NOT start training.

if __name__ == "__main__":
    # Notebook entrypoint for Google Colab
    # This avoids argparse/CLI issues completely.

    RUN_MODE = "train"  # change to "eval" or "saliency" later if needed

    NOTEBOOK_ARGS = argparse.Namespace(
        command=RUN_MODE,
        seed=0,
        num_envs=128,
        rollout_len=32,
        total_updates=2000,
        total_steps=None,          # keep None when you want exactly 2000 updates
        hidden_dim=256,
        memory_len=16,
        num_layers=2,
        reservoir_capacity=50_000, # trimmed to save Colab RAM
        avg_batch_size=256,        # trimmed to save Colab RAM
        eval_every=100,
        checkpoint_every=100,
        output_dir="/content/uno_t4",
        checkpoint="/content/uno_t4/latest.pkl",
        episodes=64,
    )

    if RUN_MODE == "train":
        train_command(NOTEBOOK_ARGS)
    elif RUN_MODE == "eval":
        eval_command(NOTEBOOK_ARGS)
    elif RUN_MODE == "saliency":
        saliency_command(NOTEBOOK_ARGS)
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

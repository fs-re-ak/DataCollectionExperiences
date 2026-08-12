"""
Trial-order randomiser for the Emotions Model Experience.

Generates one full permutation of the 9 emotions per block subject to:
  1. No emotion repeats at block boundaries
     (last trial of block N ≠ first trial of block N+1).
  2. Neither CONTEMPT variant is the first or last trial in a block (best-effort).
  3. The RNG seed for each block is recorded for full reproducibility.

Usage
-----
    from session.randomizer import build_session_order

    blocks = build_session_order(n_blocks=3)
    for b in blocks:
        print(b.block_number, b.trial_order, b.rng_seed, b.mirror)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


@dataclass
class BlockOrder:
    block_number: int           # 1-indexed
    mirror:       bool          # True if webcam mirror is on for this block
    trial_order:  list[str]     # e.g. ["HAPPINESS", "ANGER", ..., "CONTEMPT_LEFT"]
    rng_seed:     int


def build_session_order(
    n_blocks: int = config.N_BLOCKS,
    mirror_blocks: set[int] = config.MIRROR_BLOCKS,
    seed_override: int | None = None,
) -> list[BlockOrder]:
    """
    Generate trial orders for all blocks.

    Parameters
    ----------
    n_blocks      : total number of blocks (default from config)
    mirror_blocks : set of 1-indexed block numbers with mirror ON
    seed_override : if given, seeds the master RNG (for testing / replay)
    """
    master_rng = random.Random(seed_override)

    blocks: list[BlockOrder] = []
    prev_last: str | None = None

    for b_idx in range(n_blocks):
        block_num  = b_idx + 1
        block_seed = master_rng.randint(0, 2**31 - 1)

        order = _generate_block_order(
            block_seed=block_seed,
            prev_last=prev_last,
        )

        blocks.append(BlockOrder(
            block_number=block_num,
            mirror=(block_num in mirror_blocks),
            trial_order=order,
            rng_seed=block_seed,
        ))
        prev_last = order[-1]

    return blocks


# ── internal helpers ──────────────────────────────────────────────────────────

_MAX_ATTEMPTS = 1000


def _check_contempt_position(order: list[str]) -> bool:
    """Return True if no CONTEMPT variant is first or last (best-effort)."""
    contempt_positions = [i for i, e in enumerate(order) if "CONTEMPT" in e]
    if not contempt_positions:
        return True
    return contempt_positions[0] != 0 and contempt_positions[-1] != len(order) - 1


def _generate_block_order(
    block_seed: int,
    prev_last: str | None,
) -> list[str]:
    """
    Produce a valid permutation for one block.

    Tries up to _MAX_ATTEMPTS shuffles; if constraints cannot be satisfied
    it falls back gracefully.
    """
    rng  = random.Random(block_seed)
    pool = list(config.EMOTIONS_BASE)
    best: list[str] | None = None

    for _ in range(_MAX_ATTEMPTS):
        rng.shuffle(pool)
        order = list(pool)

        # Hard constraint: no repeat at block boundary
        if prev_last is not None and order[0] == prev_last:
            continue

        if best is None:
            best = list(order)

        # Soft constraint: contempt variants not first or last
        if _check_contempt_position(order):
            return order

    if best is not None:
        return best

    rng.shuffle(pool)
    return list(pool)

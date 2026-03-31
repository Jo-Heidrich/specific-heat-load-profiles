# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
import numpy as np
import pandas as pd

from soft.config_io import ConfigError


def _contiguous_segments(sorted_idx: Sequence[int]) -> List[List[int]]:
    """Split sorted indices into contiguous runs (by +1)."""
    if not sorted_idx:
        return []
    segs = []
    cur = [sorted_idx[0]]
    for a, b in zip(sorted_idx[:-1], sorted_idx[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            segs.append(cur)
            cur = [b]
    segs.append(cur)
    return segs


@dataclass
class ValidationSelection:
    train_core_idx: List[int]
    val_blocks: List[List[int]]  # e.g. 2 blocks


def select_validation_blocks_from_train(
    train_idx: List[int],
    *,
    val_frac: float = 0.0,
    n_blocks: int = 2,
    dominance_ratio: float = 1.6,
    min_block_size: int = 24,
) -> ValidationSelection:
    """
    Pick validation blocks from the largest available contiguous training segments.

    Rules:
    - Compute contiguous segments within train_idx.
    - Sort segments by length desc.
    - If the largest segment is 'dominant' (len1 >= dominance_ratio * len2),
      take all validation blocks from the largest segment.
    - Else take from the top segments (one block from each), up to n_blocks.
    - Each block is taken from the *end* of the chosen segment (time-aware).
    """
    if n_blocks != 2:
        raise ValueError("This implementation currently assumes n_blocks=2.")

    train_sorted = sorted(map(int, train_idx))
    segs = _contiguous_segments(train_sorted)
    if not segs:
        return ValidationSelection(train_core_idx=train_sorted, val_blocks=[])

    segs = sorted(segs, key=len, reverse=True)

    n_train = len(train_sorted)
    n_val_total = max(
        n_blocks * min_block_size, int(round(n_train * val_frac))
    )
    # split total validation into two equal-ish blocks
    n_val_1 = n_val_total // 2
    n_val_2 = n_val_total - n_val_1

    # Decide whether one segment dominates
    len1 = len(segs[0])
    len2 = len(segs[1]) if len(segs) > 1 else 0
    use_one_segment = (len2 == 0) or (len1 >= dominance_ratio * max(1, len2))

    val_blocks: List[List[int]] = []

    if use_one_segment:
        seg = segs[0]
        if len(seg) < (n_val_1 + n_val_2 + min_block_size):
            # if too short, shrink validation to fit reasonably
            n_val_1 = min(n_val_1, max(min_block_size, len(seg) // 4))
            n_val_2 = min(n_val_2, max(min_block_size, len(seg) // 4))

        # Take two blocks from the tail of the biggest segment
        block2 = seg[-n_val_2:]
        block1 = seg[-(n_val_1 + n_val_2) : -n_val_2]
        val_blocks = [block1, block2]
    else:
        # Take one block from each of the two biggest segments
        seg_a, seg_b = segs[0], segs[1]

        block_a = seg_a[-n_val_1:]
        block_b = seg_b[-n_val_2:]
        val_blocks = [block_a, block_b]

    # Build train_core = train_idx minus validation indices
    val_all = set(i for blk in val_blocks for i in blk)
    train_core = [i for i in train_sorted if i not in val_all]

    return ValidationSelection(
        train_core_idx=train_core, val_blocks=val_blocks
    )


def make_eval_sets_for_split(
    df: pd.DataFrame,
    *,
    features: List[str],
    target_col: str,
    train_idx: List[int],
    test_idx: List[int],
    mode: str,
    val_frac: float,
) -> Tuple[pd.DataFrame, pd.Series, Optional[list]]:
    """
    Returns:
      X_fit, y_fit, eval_sets

    - mode=off: eval_sets=None (fit on full train_idx)
    - mode=test: eval_sets=[(X_test, y_test)] or include train too (optional)
    - mode=train: eval_sets=[(X_val1,y_val1),(X_val2,y_val2)] and fit on reduced train_core
    """
    mode = str(mode).lower().strip()

    df_train = df.iloc[train_idx]
    X_train = df_train[features]
    y_train = df_train[target_col]

    df_test = df.iloc[test_idx]
    X_test = df_test[features]
    y_test = df_test[target_col]

    if mode == "off":
        return X_train, y_train, None

    if mode == "test":
        # reproduce: validate on X_test (leaky)
        return X_train, y_train, [(X_test, y_test)]

    if mode == "train":
        sel = select_validation_blocks_from_train(train_idx, val_frac=val_frac)
        df_core = df.iloc[sel.train_core_idx]
        X_core = df_core[features]
        y_core = df_core[target_col]

        eval_sets = []
        for blk in sel.val_blocks:
            dfv = df.iloc[blk]
            eval_sets.append((dfv[features], dfv[target_col]))

        return X_core, y_core, eval_sets

    raise ValueError("training.validation.mode must be one of: off|train|test")


def select_validation_blocks_distributed(
    train_idx: List[int],
    *,
    val_frac: float = 0.10,
    n_blocks: int = 8,
    block_min_size: int = 24,
    rng_seed: int = 42,
) -> tuple[list[int], list[list[int]]]:
    """
    Choose multiple small validation blocks distributed over available training segments.

    Returns:
      train_core_idx, val_blocks
    """
    train_sorted = sorted(map(int, train_idx))
    segs = _contiguous_segments(train_sorted)
    segs = [
        s for s in segs if len(s) >= block_min_size * 3
    ]  # ignore tiny segments

    if not segs:
        # fallback: no validation
        return train_sorted, []

    n_train = len(train_sorted)
    n_val_total = max(
        n_blocks * block_min_size, int(round(n_train * val_frac))
    )
    block_size = max(block_min_size, n_val_total // n_blocks)

    rng = np.random.default_rng(rng_seed)

    # Allocate blocks across segments proportional to segment length
    seg_lengths = np.array([len(s) for s in segs], dtype=float)
    probs = seg_lengths / seg_lengths.sum()

    blocks = []
    # sample which segment each block comes from
    seg_choices = rng.choice(len(segs), size=n_blocks, replace=True, p=probs)

    for si in seg_choices:
        seg = segs[si]
        # pick a block end somewhere not too close to segment start
        # keep it within segment
        end_max = len(seg) - 1
        end_min = max(
            block_size, int(0.3 * len(seg))
        )  # avoid super-early part
        if end_min >= end_max:
            end_min = block_size

        end = int(rng.integers(end_min, end_max + 1))
        start = max(0, end - block_size + 1)
        blk = seg[start : end + 1]
        blocks.append(blk)

    # de-duplicate indices (blocks might overlap) -> keep unique and then re-chunk
    val_all = sorted(set(i for blk in blocks for i in blk))
    if len(val_all) < block_min_size:
        return train_sorted, []

    # rebuild blocks as contiguous runs (for plotting clarity)
    val_blocks = _contiguous_segments(val_all)

    val_set = set(val_all)
    train_core = [i for i in train_sorted if i not in val_set]

    return train_core, val_blocks


def normalize_validation_mode(x) -> str:
    """
    Normalize YAML / CLI inputs to one of:
      off | train | train_distributed | test
    Accepts booleans and common aliases.
    """
    if x is None:
        return "off"
    if isinstance(x, bool):
        return "train" if x else "off"

    s = str(x).strip().lower()
    if s in {"off", "false", "0", "none", ""}:
        return "off"
    if s in {"test"}:
        return "test"
    if s in {"train"}:
        return "train"
    if s in {"train_distributed", "distributed"}:
        return "train_distributed"

    raise ConfigError(
        "training.validation.mode must be one of: off|train|train_distributed|test"
    )


def combine_val_blocks_to_one_eval_set(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    val_blocks: List[List[int]],
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """
    Flatten val blocks to unique indices and return a single (X_val, y_val).
    """
    if not val_blocks:
        return None
    val_idx = sorted({int(i) for blk in val_blocks for i in blk})
    if not val_idx:
        return None
    df_val = df.iloc[val_idx].reset_index(drop=True)
    return df_val[features], df_val[target_col]

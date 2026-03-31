# -*- coding: utf-8 -*-
"""
@author: heidrich

Seasonal splitters.

This module provides two seasonal cross-validation splitters:

1) LegacySeasonalPairSplitter:
   Reproduces the behavior of the original `create_splitting(df)` from splitting.py:
   - Two test blocks per split (each block is test_frac_each of total length).
   - The two blocks must be from two different seasons.
   - Each season-pair is used at most once (max 6 unique pairs).
   - Blocks are sampled by choosing a start index within the season indices,
     but the contiguous block may spill over beyond that season (legacy behavior).

2) WithinSeasonSeasonalPairSplitter:
   Similar interface, but enforces that each sampled block lies fully within a
   *contiguous season segment* (i.e., does not cross season boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

Season = str
IndexRange = Tuple[int, int]
SplitIdx = Tuple[List[int], List[int]]


def assign_season(ts: pd.Timestamp) -> Season:
    """Map timestamp to meteorological season (Winter/Spring/Summer/Autumn)."""
    m = int(ts.month)
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _ranges_overlap(a: IndexRange, b: IndexRange) -> bool:
    """Return True if ranges [a0,a1) and [b0,b1) overlap."""
    return a[0] < b[1] and a[1] > b[0]


def _cyclic_month_gap_ok(
    d1: pd.Timestamp, d2: pd.Timestamp, min_month_gap: int
) -> bool:
    """
    Check cyclic distance between months (0..11 wrap-around).
    Example: Jan vs Nov => diff=2 (wrap).
    """
    diff = abs(int(d1.month) - int(d2.month))
    diff = min(diff, 12 - diff)
    return diff >= int(min_month_gap)


@dataclass
class LegacySeasonalPairSplitter:
    """
    Legacy seasonal block-wise CV with two test windows.

    Reproduces old splitting.py behavior:
    - Sample 2 seasons (different)
    - Sample block1 start within season1 indices (block may spill to next seasons)
    - Sample block2 start within season2 indices (block may spill)
    - Reject if month distance between block start months < min_month_gap
    - Use each season-pair at most once (max 6 splits)
    """

    date_col: str = "date"
    n_splits: int = 10
    test_frac_each: float = 0.10
    min_month_gap: int = 4
    seed: int = 42
    max_attempts: int = 300

    def split(self, df: pd.DataFrame) -> List[SplitIdx]:
        """Return list of (train_idx, test_idx) for each split."""
        df = df.sort_values(self.date_col).reset_index(drop=True).copy()
        n = len(df)
        if n == 0:
            return []

        df["_season"] = df[self.date_col].apply(assign_season)

        block_size = int(n * self.test_frac_each)
        block_size = max(1, block_size)

        season_blocks: Dict[Season, np.ndarray] = {
            s: df.index[df["_season"] == s].to_numpy()
            for s in df["_season"].unique()
        }

        rng = np.random.RandomState(self.seed)
        used_pairs = set()
        splits_ranges: List[Tuple[IndexRange, IndexRange]] = []

        def find_valid_block(
            start_indices: np.ndarray, taken: List[IndexRange]
        ) -> Optional[IndexRange]:
            # Same logic as in your splitting.py: permute candidates, pick first non-overlapping
            for start in rng.permutation(start_indices):
                start = int(start)
                end = start + block_size
                if end >= n:
                    continue
                if any(_ranges_overlap((start, end), r) for r in taken):
                    continue
                return (start, end)
            return None

        attempts = 0
        seasons = list(season_blocks.keys())
        while (
            len(splits_ranges) < self.n_splits and attempts < self.max_attempts
        ):
            if len(seasons) < 2:
                break

            s1, s2 = rng.choice(seasons, 2, replace=False)
            pair = tuple(sorted([s1, s2]))
            if pair in used_pairs:
                attempts += 1
                continue

            taken: List[IndexRange] = []
            b1 = find_valid_block(season_blocks[s1], taken)
            if b1 is None:
                attempts += 1
                continue
            taken.append(b1)

            b2 = find_valid_block(season_blocks[s2], taken)
            if b2 is None:
                attempts += 1
                continue

            d1 = pd.Timestamp(df.loc[b1[0], self.date_col])
            d2 = pd.Timestamp(df.loc[b2[0], self.date_col])
            if not _cyclic_month_gap_ok(d1, d2, self.min_month_gap):
                attempts += 1
                continue

            used_pairs.add(pair)
            splits_ranges.append((b1, b2))

        # Convert ranges -> indices
        out: List[SplitIdx] = []
        all_idx = np.arange(n)
        for b1, b2 in splits_ranges:
            test_idx = list(range(b1[0], b1[1])) + list(range(b2[0], b2[1]))
            mask = np.zeros(n, dtype=bool)
            mask[test_idx] = True
            train_idx = all_idx[~mask].tolist()
            out.append((train_idx, test_idx))

        df.drop(columns=["_season"], inplace=True, errors="ignore")
        return out


@dataclass
class WithinSeasonSeasonalPairSplitter:
    """
    Seasonal splitter that enforces blocks to be fully inside contiguous season segments.

    Differences vs legacy:
    - Candidate blocks are sampled from *season segments* (continuous index ranges
      where season is constant).
    - A block is only valid if [start, start+block_size) stays within the same segment.
    """

    date_col: str = "date"
    n_splits: int = 6
    test_frac_each: float = 0.10
    min_month_gap: int = 4
    seed: int = 42
    max_attempts: int = 500
    allow_pair_reuse: bool = (
        False  # False => max 6 unique pairs, True => repeats allowed
    )

    def split(self, df: pd.DataFrame) -> List[SplitIdx]:
        df = df.sort_values(self.date_col).reset_index(drop=True).copy()
        n = len(df)
        if n == 0:
            return []

        df["_season"] = df[self.date_col].apply(assign_season)
        block_size = max(1, int(n * self.test_frac_each))

        # Build contiguous season segments: list of (season, start, end) with [start,end)
        segs: List[Tuple[Season, int, int]] = []
        cur_s = df.loc[0, "_season"]
        start = 0
        for i in range(1, n):
            s = df.loc[i, "_season"]
            if s != cur_s:
                segs.append((cur_s, start, i))
                cur_s = s
                start = i
        segs.append((cur_s, start, n))

        # Per season: list of segments
        season_to_segs: Dict[Season, List[Tuple[int, int]]] = {}
        for s, a, b in segs:
            season_to_segs.setdefault(s, []).append((a, b))

        seasons = list(season_to_segs.keys())
        if len(seasons) < 2:
            raise ValueError(
                "Need at least two seasons present to create seasonal splits."
            )

        # All unique pairs
        pairs: List[Tuple[Season, Season]] = []
        for i in range(len(seasons)):
            for j in range(i + 1, len(seasons)):
                pairs.append((seasons[i], seasons[j]))

        rng = np.random.default_rng(self.seed)
        used_pairs = set()
        out: List[SplitIdx] = []
        all_idx = np.arange(n)

        def sample_block_from_season(
            season: Season, taken: List[IndexRange]
        ) -> Optional[IndexRange]:
            # sample a segment, then sample a start within it such that block fits
            seg_list = season_to_segs.get(season, [])
            if not seg_list:
                return None

            # try multiple times
            for _ in range(200):
                a, b = seg_list[int(rng.integers(0, len(seg_list)))]
                if b - a < block_size:
                    continue
                start = int(rng.integers(a, b - block_size + 1))
                block = (start, start + block_size)
                if any(_ranges_overlap(block, t) for t in taken):
                    continue
                return block
            return None

        attempts = 0
        while len(out) < self.n_splits and attempts < self.max_attempts:
            s1, s2 = rng.choice(seasons, 2, replace=False)
            pair = tuple(sorted([s1, s2]))
            if (not self.allow_pair_reuse) and (pair in used_pairs):
                attempts += 1
                continue

            taken: List[IndexRange] = []
            b1 = sample_block_from_season(s1, taken)
            if b1 is None:
                attempts += 1
                continue
            taken.append(b1)

            b2 = sample_block_from_season(s2, taken)
            if b2 is None:
                attempts += 1
                continue

            d1 = pd.Timestamp(df.loc[b1[0], self.date_col])
            d2 = pd.Timestamp(df.loc[b2[0], self.date_col])
            if not _cyclic_month_gap_ok(d1, d2, self.min_month_gap):
                attempts += 1
                continue

            test_idx = list(range(b1[0], b1[1])) + list(range(b2[0], b2[1]))
            mask = np.zeros(n, dtype=bool)
            mask[test_idx] = True
            train_idx = all_idx[~mask].tolist()
            out.append((train_idx, test_idx))

            used_pairs.add(pair)

        df.drop(columns=["_season"], inplace=True, errors="ignore")
        return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt

    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parents[3]  # anpassen falls nötig
    sys.path.insert(0, str(PROJECT_ROOT))

    # apply style BEFORE creating figures
    STYLE_FILE = PROJECT_ROOT / "style" / "fraunhofer.mplstyle"
    plt.style.use(str(STYLE_FILE))

    from style import fhg_style as fhg

    fhg.set_font_scale(1.0)

    fraunhofer = [
        "#005b7f",
        "#39c1cd",
        "#179c7d",
        "#b2d235",
        "#008598",
        "#7c154d",
    ]

    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = THIS_DIR.parent  # "Ordner drüber"
    sys.path.insert(0, str(PROJECT_ROOT))
    from style import fhg_style as fhg

    fhg.set_font_scale(1.0)

    date_range_base = pd.date_range(
        start="2018-01-01", end="2019-12-31 23:00:00", freq="h"
    )
    df_base = pd.DataFrame({"date": date_range_base}).reset_index(drop=True)
    n = len(df_base)

    # --- 1) Legacy (reproduces splitting.py look/behavior) ---
    splitter = LegacySeasonalPairSplitter(
        date_col="date",
        n_splits=6,
        test_frac_each=0.10,
        min_month_gap=4,
        seed=42,
    )
    splits = splitter.split(df_base)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for i, (_train_idx, test_idx) in enumerate(splits):
        row = np.zeros(n, dtype=float)
        row[test_idx] = 1.0
        ax.plot(
            df_base["date"],
            row + i * 1.2,
            label=f"Split {i+1}",
            color=fraunhofer[i % len(fraunhofer)],
            linewidth=1.8,
        )
    ax.set_yticks([])
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, alpha=0.3)
    fhg.legend(ax, loc="upper left")
    # leg = ax.legend(loc="upper left")  # oder "lower right"
    # # fraunhofer look (wie in mplstyle)
    # frame = leg.get_frame()
    # frame.set_facecolor("#f2f3f4")
    # frame.set_edgecolor("#c7cdd1")
    # frame.set_alpha(1.0)

    # # Größe konsistent (statt fhg.legend default 7)
    # for t in leg.get_texts():
    #     t.set_fontsize(11)
    plt.show()

    # --- 2) Within-season strict blocks (no season boundary crossing) ---
    splitter2 = WithinSeasonSeasonalPairSplitter(
        date_col="date",
        n_splits=6,
        test_frac_each=0.10,
        min_month_gap=4,
        seed=42,
        allow_pair_reuse=False,
    )
    splits2 = splitter2.split(df_base)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for i, (_train_idx, test_idx) in enumerate(splits2):
        row = np.zeros(n, dtype=float)
        row[test_idx] = 1.0
        ax.plot(
            df_base["date"],
            row + i * 1.2,
            label=f"Strict Split {i+1}",
            color=fraunhofer[i % len(fraunhofer)],
            linewidth=1.8,
        )
    ax.set_yticks([])
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True)
    ax.legend(
        loc="lower right",
        fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.85,
    )
    plt.show()

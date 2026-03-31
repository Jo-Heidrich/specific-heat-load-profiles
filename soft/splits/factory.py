# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional

import pandas as pd

from soft.config_io import get, ConfigError
from .splits_seasonal import (
    LegacySeasonalPairSplitter,
    WithinSeasonSeasonalPairSplitter,
)
from .splits import DualEdgeSplitter


SplitIdx = Tuple[List[int], List[int]]


def build_time_raster(
    cfg: dict, date_col: str, df: pd.DataFrame
) -> pd.DataFrame:
    """Build a complete time raster DataFrame for defining split indices."""
    freq = str(cfg.get("freq", "h"))
    start = cfg.get("start", None)
    end = cfg.get("end", None)

    if start is None:
        raise ValueError(
            "split_index_space.start is required when enabled=true"
        )

    start_ts = pd.Timestamp(start)
    if end is None:
        end_ts = pd.Timestamp(df[date_col].max())
    else:
        end_ts = pd.Timestamp(end)

    idx = pd.date_range(start=start_ts, end=end_ts, freq=freq)
    return pd.DataFrame({date_col: idx})


def map_splits_from_raster_to_df(
    df: pd.DataFrame,
    df_raster: pd.DataFrame,
    splits_on_raster: List[SplitIdx],
    date_col: str,
    strict: bool = False,
) -> List[SplitIdx]:
    """
    Convert splits expressed in raster row indices to splits expressed in df row indices,
    matching by timestamp values in `date_col`.

    If strict=False: missing timestamps are dropped from train/test sets.
    If strict=True: raises if any raster test timestamp is missing in df.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    df_raster = df_raster.sort_values(date_col).reset_index(drop=True)

    # map timestamp -> df index
    df_ts_to_idx: Dict[pd.Timestamp, int] = {
        pd.Timestamp(t): i for i, t in enumerate(df[date_col].tolist())
    }

    mapped: List[SplitIdx] = []

    for train_idx_r, test_idx_r in splits_on_raster:
        # timestamps in raster space
        train_ts = df_raster.loc[train_idx_r, date_col].tolist()
        test_ts = df_raster.loc[test_idx_r, date_col].tolist()

        # map to df indices
        train_idx = []
        for t in train_ts:
            ti = df_ts_to_idx.get(pd.Timestamp(t))
            if ti is not None:
                train_idx.append(ti)

        test_idx = []
        missing_test = []
        for t in test_ts:
            ti = df_ts_to_idx.get(pd.Timestamp(t))
            if ti is not None:
                test_idx.append(ti)
            else:
                missing_test.append(pd.Timestamp(t))

        if strict and missing_test:
            raise ValueError(
                f"Raster-based split contains {len(missing_test)} test timestamps "
                f"missing in dataset. Example: {missing_test[:5]}"
            )

        # de-duplicate + sort
        train_idx = sorted(set(train_idx))
        test_idx = sorted(set(test_idx))

        mapped.append((train_idx, test_idx))

    return mapped


def make_splits_from_cfg(
    cfg: dict,
    df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> Tuple[str, Optional[pd.DataFrame], List[Tuple[List[int], List[int]]]]:
    """
    Returns:
      split_name, df_raster_or_None, splits (list of (train_idx, test_idx))
    """
    split = str(get(cfg, "training.split", "dualedge")).lower().strip()

    if split == "none":
        return "none", None, []

    if split == "dualedge":
        test_fraction = float(get(cfg, "training.test_fraction", 0.2))
        splitter = DualEdgeSplitter(date_col=date_col)
        df_train, df_test, train_idx, test_idx = splitter.split(
            df, test_fraction
        )
        return "dualedge", None, [(train_idx, test_idx)]

    if split == "seasonal":
        seasonal = get(cfg, "training.seasonal", {}) or {}
        variant = str(seasonal.get("variant", "legacy")).lower().strip()
        n_splits = int(seasonal.get("n_splits", 6))
        test_frac_each = float(seasonal.get("test_frac_each", 0.10))
        min_month_gap = int(seasonal.get("min_month_gap", 4))
        seed = int(seasonal.get("seed", 42))

        if variant == "legacy":
            splitter = LegacySeasonalPairSplitter(
                date_col=date_col,
                n_splits=n_splits,
                test_frac_each=test_frac_each,
                min_month_gap=min_month_gap,
                seed=seed,
            )
        elif variant in ("within", "strict", "within_season"):
            splitter = WithinSeasonSeasonalPairSplitter(
                date_col=date_col,
                n_splits=n_splits,
                test_frac_each=test_frac_each,
                min_month_gap=min_month_gap,
                seed=seed,
                allow_pair_reuse=False,
            )
        else:
            raise ConfigError(
                "training.seasonal.variant must be one of: legacy, within"
            )

        split_space = get(cfg, "split_index_space", {}) or {}
        use_raster = bool(split_space.get("enabled", False))

        if use_raster:
            df_raster = build_time_raster(split_space, date_col, df)
            splits_raster = splitter.split(df_raster)
            splits = map_splits_from_raster_to_df(
                df=df,
                df_raster=df_raster,
                splits_on_raster=splits_raster,
                date_col=date_col,
                strict=bool(split_space.get("strict", False)),
            )
            return "seasonal", df_raster, splits

        return "seasonal", None, splitter.split(df)

    raise ConfigError(
        "training.split must be one of: dualedge, seasonal, none"
    )

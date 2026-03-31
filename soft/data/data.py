# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from .feature_builders import HourlyFeatureBuilder, DailyMeanFeatureBuilder


class FeatureDataLoader:
    def __init__(self, date_col="date"):
        self.date_col = date_col

    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)
        return df

    def drop_year(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        if year is None:
            return df
        return df[~(df[self.date_col].dt.year == year)].reset_index(drop=True)


def read_raw_csv(path: str) -> pd.DataFrame:
    """Read your raw format: date, consumption_kWh, temperature_hourly."""
    df = pd.read_csv(path)
    return df


def prepare_from_raw(
    raw_csv_path: str,
    mode: Literal["train", "predict"],
    feature_mode: Literal["hourly", "daily_mean"],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Raw -> features. Optionally save to output_path."""
    df_raw = read_raw_csv(raw_csv_path)

    require_target = mode == "train"

    if feature_mode == "hourly":
        builder = HourlyFeatureBuilder()
        df_feat = builder.build_from_raw(df_raw, require_target=require_target)
    elif feature_mode == "daily_mean":
        builder = DailyMeanFeatureBuilder()
        df_feat = builder.build_from_raw(df_raw, require_target=require_target)
    else:
        raise ValueError("feature_mode must be 'hourly' or 'daily_mean'")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(out, index=False)

    return df_feat

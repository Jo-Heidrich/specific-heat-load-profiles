# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 22:02:05 2026

@author: heidrich
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
import joblib

from soft.data.feature_builders import HourlyFeatureBuilder
from models.profil_regression import regression_results


@dataclass(frozen=True)
class CompareConfig:
    date_col: str = "date"
    target_col: Optional[str] = (
        "consumption_kWh"  # set None for prediction-only
    )
    temp_col: str = "temperature_hourly"


class LightGBMAdapter:
    """
    LightGBM model wrapper that can predict from:
    (A) a df_test that already contains the engineered feature columns, OR
    (B) raw hourly input (date + temperature_hourly), in which case we build features.
    """

    def __init__(self, model_path: str, fb: HourlyFeatureBuilder):
        self.model = joblib.load(model_path)
        self.fb = (
            fb  # defines feature column names + optional build_from_raw()
        )

    def _has_all_features(self, df: pd.DataFrame) -> bool:
        cols = set(df.columns)
        return all(c in cols for c in self.fb.feature_columns())

    def predict_aligned(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Return predictions aligned to df_test rows:
        - rows without required features -> NaN
        - predictions only where feature rows are complete
        """
        d = df_test.copy()
        d[self.fb.date_col] = pd.to_datetime(
            d[self.fb.date_col], errors="coerce"
        )
        d = d.sort_values(self.fb.date_col)

        # 1) Get a feature frame
        if self._has_all_features(d):
            feat_df = d
        else:
            # Build features from raw (requires temperature_hourly column etc.)
            feat_df = self.fb.build_from_raw(d, require_target=False)

        feat_cols = self.fb.feature_columns()

        # 2) Predict only on valid rows (no NaNs in features)
        valid = feat_df[feat_cols].notna().all(axis=1)
        yhat = np.full(len(d), np.nan, dtype=float)

        # We need to map predictions back to the original df_test row positions.
        # Best: use the datetime key (merge_asof) or keep original index.
        # Here we assume feat_df rows correspond to a subset of d rows in the same order
        # (true if feat_df==d or if build_from_raw kept timestamps).
        pred = self.model.predict(feat_df.loc[valid, feat_cols])
        row_ids = feat_df.loc[valid, "_row_id"].to_numpy()
        yhat[row_ids] = np.asarray(pred, dtype=float)
        yhat_idx = feat_df.index[valid]  # indices in 'd' if feat_df==d

        # yhat[yhat_idx] = np.asarray(pred, dtype=float)
        return yhat


@dataclass(frozen=True)
class NonlinConfig:
    date_col: str = "date"
    temp_hourly_col: str = "temperature_hourly"
    params: list[float] = None  # pass your 20 params


class NonlinAdapter:
    """Non-linear regression adapter producing an hourly series aligned to df_test."""

    def __init__(self, cfg: NonlinConfig):
        if not cfg.params:
            raise ValueError("NonlinConfig.params must be provided")
        self.cfg = cfg

    def predict_aligned(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Return predictions aligned to df_test rows:
        - rows without required features -> NaN
        - predictions only where feature rows are complete
        """
        d = df_test.copy()

        # Always keep row identity for alignment back to ORIGINAL df_test order
        d["_row_id"] = d.index.to_numpy()

        d[self.fb.date_col] = pd.to_datetime(
            d[self.fb.date_col], errors="coerce"
        )
        d = d.sort_values(self.fb.date_col)

        # 1) Get a feature frame
        if self._has_all_features(d):
            feat_df = d
        else:
            feat_df = self.fb.build_from_raw(d, require_target=False)
            # build_from_raw already carries _row_id (as you implemented)

        feat_cols = self.fb.feature_columns()

        # 2) Predict only on valid rows (no NaNs in features)
        valid = feat_df[feat_cols].notna().all(axis=1)
        yhat = np.full(len(df_test), np.nan, dtype=float)

        pred = self.model.predict(feat_df.loc[valid, feat_cols])
        row_ids = feat_df.loc[valid, "_row_id"].to_numpy()

        # Place predictions back into original df_test order
        yhat[row_ids] = np.asarray(pred, dtype=float)
        return yhat

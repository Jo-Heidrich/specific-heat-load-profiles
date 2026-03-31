# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 21:03:54 2025

@author: heidrich
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings


class BaselineModel:
    def fit(self, df_train: pd.DataFrame, target: str):
        raise NotImplementedError

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class LinearRegressionBaseline(BaselineModel):
    def __init__(self, features):
        self.features = list(features)
        self.model = LinearRegression()

    def fit(self, df_train, target):
        self.model.fit(df_train[self.features], df_train[target])
        return self

    def predict(self, df_test):
        return self.model.predict(df_test[self.features])


class Rolling24hBaseline(BaselineModel):
    """
    Rolling mean auf dem TRUE target (klassische naive baseline).
    Wichtig: braucht echte y-Werte im Testzeitraum,
    daher wird diese baseline in Evaluation berechnet, nicht "fit".
    """

    def __init__(self, window=24):
        self.window = window

    def fit(self, df_train, target):
        return self

    def predict_from_series(self, y_true: pd.Series) -> np.ndarray:
        return y_true.rolling(self.window, min_periods=1).mean().to_numpy()


def make_baseline_predictions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    date_col: str,
    target_col: str,
    baseline_type: str,
    baseline_features: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Create baseline predictions for df_test.

    baseline_type:
      - "rolling24h": rolling mean over last 24 hours on the test target (uses y_test history)
      - "linreg": linear regression trained on df_train[baseline_features]
      - "both": both baselines
      - "none": returns empty dict
    """
    out: Dict[str, np.ndarray] = {}
    y_test = df_test[target_col].astype(float)

    if baseline_type in ("rolling24h", "both"):
        # rolling baseline computed on y_test itself (as in your original code)
        out["baseline_rolling24h"] = (
            y_test.rolling(window=24, min_periods=1).mean().to_numpy()
        )

    if baseline_type in ("linreg", "both"):
        if not baseline_features:
            raise ValueError(
                "baseline_features must be provided for linreg baseline."
            )

        Xtr = df_train[baseline_features].astype(float)
        ytr = df_train[target_col].astype(float)
        Xte = df_test[baseline_features].astype(float)

        # --- TRAIN: drop rows with NaN/Inf in X or y ---
        Xtr_np = Xtr.to_numpy()
        ytr_np = ytr.to_numpy()

        m_tr = np.isfinite(Xtr_np).all(axis=1) & np.isfinite(ytr_np)
        drop_tr = 1.0 - (m_tr.mean() if len(m_tr) else 0.0)

        if drop_tr > 0.05:
            warnings.warn(
                f"[baseline_linreg] Dropping {drop_tr*100:.1f}% of TRAIN rows due to NaN/Inf "
                f"(kept {int(m_tr.sum())}/{len(m_tr)}). Consider adjusting baseline_features.",
                RuntimeWarning,
            )

        Xtr2 = Xtr.loc[m_tr]
        ytr2 = ytr.loc[m_tr]

        # Guard: too little training data
        if len(Xtr2) < 10:
            warnings.warn(
                f"[baseline_linreg] Too few valid TRAIN rows after dropping NaNs "
                f"({len(Xtr2)}). Skipping linreg baseline for this split.",
                RuntimeWarning,
            )
        else:
            lr = LinearRegression()
            lr.fit(Xtr2, ytr2)

            # --- TEST: predict only where X is finite; others -> NaN ---
            Xte_np = Xte.to_numpy()
            m_te = np.isfinite(Xte_np).all(axis=1)
            drop_te = 1.0 - (m_te.mean() if len(m_te) else 0.0)

            if drop_te > 0.05:
                warnings.warn(
                    f"[baseline_linreg] Cannot predict for {drop_te*100:.1f}% of TEST rows due to NaN/Inf "
                    f"(predicted {int(m_te.sum())}/{len(m_te)}).",
                    RuntimeWarning,
                )

            yhat = np.full(len(df_test), np.nan, dtype=float)
            if m_te.any():
                yhat[m_te] = lr.predict(Xte.loc[m_te])

            out["baseline_linreg"] = yhat

    return out

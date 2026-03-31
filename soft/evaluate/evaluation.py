# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from dataclasses import dataclass
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from .metrics import compute_metrics
from .baselines import LinearRegressionBaseline, Rolling24hBaseline
from .model import TrainerFactory


def mape_safe(y_true, y_pred, eps=1e-6) -> float:
    """Robust MAPE (avoids division by zero via eps)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps=1e-6) -> float:
    """Symmetric MAPE."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


@dataclass
class Metrics:
    RMSE: float
    MAE: float
    MAPE: float
    SMAPE: float
    R2: float


class Evaluator:
    """Compute standard regression metrics for model evaluation."""

    def compute_metrics(self, y_true, y_pred) -> Metrics:
        """Compute RMSE/MAE/MAPE/SMAPE/R2."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return Metrics(
            RMSE=rmse,
            MAE=mae,
            MAPE=mape_safe(y_true, y_pred),
            SMAPE=smape(y_true, y_pred),
            R2=r2,
        )

    def metrics_table(
        self,
        main: Metrics,
        baseline: Optional[Metrics] = None,
        other: Optional[Metrics] = None,
    ) -> pd.DataFrame:
        """Return a comparison table for main model vs optional baselines."""
        rows = {"Model": main.__dict__}
        if baseline is not None:
            rows["Baseline"] = baseline.__dict__
        if other is not None:
            rows["OtherModel"] = other.__dict__
        return pd.DataFrame(rows).T


def _baseline_linreg(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    baseline_features: List[str],
) -> np.ndarray:
    """Linear regression baseline using baseline_features."""
    X_tr = df_train[baseline_features]
    y_tr = df_train[target_col]
    X_te = df_test[baseline_features]
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _baseline_rolling24h(df_test: pd.DataFrame, target_col: str) -> np.ndarray:
    """
    Rolling 24h mean baseline on the test slice.

    Note: This is not a causal forecasting baseline if your test contains two distant blocks.
    It is still useful as a naive comparator, but interpret with care.
    """
    y = df_test[target_col].astype(float).reset_index(drop=True)
    return y.rolling(window=24, min_periods=1).mean().to_numpy()


def cross_validate_over_splits(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    trainer_type: str,
    trainer_params: Dict[str, Any],
    splits: List[Tuple[List[int], List[int]]],
    baseline_features: Optional[List[str]] = None,
    baseline_type: str = "linreg",
) -> pd.DataFrame:
    """
    Train/evaluate across externally provided splits and return per-split metrics + mean row.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    date_col : str
        Datetime column name.
    target_col : str
        Target column.
    feature_cols : List[str]
        Feature columns used for model training.
    trainer_type : str
        Trainer identifier (e.g. "lgbm", "xgb", ...).
    trainer_params : Dict[str, Any]
        Hyperparameters for the trainer.
    splits : List[(train_idx, test_idx)]
        Precomputed index splits (as returned by SeasonalBlockCVSplitter.split()).
    baseline_features : Optional[List[str]]
        Feature columns for linear regression baseline (if used).
    baseline_type : str
        "linreg", "rolling24h", "both", or "none".

    Returns
    -------
    pd.DataFrame
        Per-split metrics and an additional "mean" row at the end.
    """
    df = df.sort_values(date_col).reset_index(drop=True)

    evaluator = Evaluator()
    rows: List[Dict[str, Any]] = []

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        X_test = df_test[feature_cols]
        y_test = df_test[target_col].to_numpy()

        trainer = TrainerFactory(trainer_type, trainer_params).make()
        model = trainer.fit(
            X_train, y_train, X_valid=X_test, y_valid=df_test[target_col]
        )

        y_pred = model.predict(X_test)
        m_model = evaluator.compute_metrics(y_test, y_pred)

        # --- baseline handling ---
        baseline_metrics = None
        if baseline_type != "none":
            y_base = None

            if baseline_type in ("linreg", "both"):
                if not baseline_features:
                    raise ValueError(
                        "baseline_features must be provided for linreg baseline."
                    )
                base = LinearRegressionBaseline(baseline_features).fit(
                    df_train, target_col
                )
                y_base = base.predict(df_test)

            if baseline_type in ("rolling24h", "both"):
                # naive baseline computed from true test values
                y_roll = Rolling24hBaseline(window=24).predict_from_series(
                    df_test[target_col]
                )
                y_base = (
                    y_roll if y_base is None else y_base
                )  # keep linreg as primary if both

            if y_base is not None:
                baseline_metrics = evaluator.compute_metrics(y_test, y_base)

        row = {
            "split": i,
            **{f"model_{k}": v for k, v in m_model.__dict__.items()},
        }
        if baseline_metrics is not None:
            row.update(
                {f"baseline_{k}": v for k, v in baseline_metrics.__dict__.items()}
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    # Add mean row
    mean_row = {"split": "mean"}
    for c in out.columns:
        if c == "split":
            continue
        mean_row[c] = float(pd.to_numeric(out[c], errors="coerce").mean())
    out = pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)

    return out

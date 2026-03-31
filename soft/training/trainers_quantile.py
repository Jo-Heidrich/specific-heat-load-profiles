# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class QuantileIntervalModel:
    """Wraps 3 models: lower quantile, center (median/mean), upper quantile."""

    model_lo: Any
    model_mid: Any
    model_hi: Any
    q_lo: float
    q_hi: float
    center: str = "median"  # "median" or "mean"

    def predict(self, X, kind: str = "mid", **kwargs):
        if kind == "lo":
            return self.model_lo.predict(X, **kwargs)
        if kind == "hi":
            return self.model_hi.predict(X, **kwargs)
        return self.model_mid.predict(X, **kwargs)

    def predict_interval(self, X, **kwargs):
        lo = self.model_lo.predict(X, **kwargs)
        mid = self.model_mid.predict(X, **kwargs)
        hi = self.model_hi.predict(X, **kwargs)
        return lo, mid, hi


class LGBMQuantileIntervalTrainer:
    """
    Train LightGBM models for lower/upper quantiles + center (median or mean).
    - lower/upper use objective='quantile'
    - center uses objective='quantile' with alpha=0.5 (median) OR 'regression' (mean)
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        q_lo: float = 0.1,
        q_hi: float = 0.9,
        center: str = "median",  # "median" or "mean"
        early_stopping_rounds: int = 50,
        log_period: int = 50,
    ):
        self.params = params or {}
        self.q_lo = float(q_lo)
        self.q_hi = float(q_hi)
        self.center = str(center).lower().strip()
        self.early_stopping_rounds = early_stopping_rounds
        self.log_period = log_period

    def _fit_one(
        self,
        X_train,
        y_train,
        objective: str,
        alpha: Optional[float],
        X_valid=None,
        y_valid=None,
        eval_sets=None,
        verbose: bool = True,
    ):
        from lightgbm import LGBMRegressor
        from lightgbm import early_stopping, log_evaluation, record_evaluation

        p = dict(self.params)

        # silence LightGBM if verbose=False
        if verbose:
            p.setdefault("verbosity", -1)
        else:
            p["verbosity"] = -1
            p["verbose"] = -1

        # objective setup
        p["objective"] = objective
        if objective == "quantile":
            if alpha is None:
                raise ValueError("alpha must be set for quantile objective")
            p["alpha"] = float(alpha)

        model = LGBMRegressor(**p)

        eval_set = [(X_train, y_train)]
        if eval_sets is not None:
            for Xv, yv in eval_sets:
                if Xv is None or yv is None:
                    continue
                eval_set.append((Xv, yv))
        elif X_valid is not None and y_valid is not None:
            eval_set.append((X_valid, y_valid))

        eval_result = {}
        callbacks = None
        if len(eval_set) >= 2:
            callbacks = [
                early_stopping(
                    stopping_rounds=int(self.early_stopping_rounds)
                ),
                record_evaluation(eval_result),
            ]
            if verbose:
                callbacks.insert(
                    1, log_evaluation(period=int(self.log_period))
                )

        if callbacks is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)

        return model

    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        eval_sets=None,
        verbose: bool = True,
    ) -> QuantileIntervalModel:
        # lower / upper quantile models
        m_lo = self._fit_one(
            X_train,
            y_train,
            objective="quantile",
            alpha=self.q_lo,
            X_valid=X_valid,
            y_valid=y_valid,
            eval_sets=eval_sets,
            verbose=verbose,
        )
        m_hi = self._fit_one(
            X_train,
            y_train,
            objective="quantile",
            alpha=self.q_hi,
            X_valid=X_valid,
            y_valid=y_valid,
            eval_sets=eval_sets,
            verbose=verbose,
        )

        # center model
        if self.center == "mean":
            m_mid = self._fit_one(
                X_train,
                y_train,
                objective="regression",
                alpha=None,
                X_valid=X_valid,
                y_valid=y_valid,
                eval_sets=eval_sets,
                verbose=verbose,
            )
        else:
            # median as quantile 0.5
            m_mid = self._fit_one(
                X_train,
                y_train,
                objective="quantile",
                alpha=0.5,
                X_valid=X_valid,
                y_valid=y_valid,
                eval_sets=eval_sets,
                verbose=verbose,
            )

        return QuantileIntervalModel(
            model_lo=m_lo,
            model_mid=m_mid,
            model_hi=m_hi,
            q_lo=self.q_lo,
            q_hi=self.q_hi,
            center=self.center,
        )

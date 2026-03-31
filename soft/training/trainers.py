# -*- coding: utf-8 -*-
"""
@author: heidrich

Training abstractions.

- BaseModel: what a fitted model must provide (predict)
- BaseTrainer: trainer interface (fit + optional eval_result_)
- LGBMTrainer: LightGBM wrapper
- TrainerFactory: constructs trainers by name
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np

from soft.config_io import get


@runtime_checkable
class BaseModel(Protocol):
    """
    Minimal interface a trained model must support.
    """

    def predict(self, X) -> np.ndarray:  # pragma: no cover
        """Return predictions for X."""


class BaseTrainer(Protocol):
    """
    Minimal trainer interface.

    Trainers encapsulate:
    - model construction
    - fitting logic (optionally with validation + eval logging)
    """

    eval_result_: Optional[Dict[str, Any]]

    def fit(
        self, X_train, y_train, X_valid=None, y_valid=None, eval_sets=None
    ) -> BaseModel:
        """Fit and return a trained model."""


@dataclass
class LGBMTrainer:
    """
    LightGBM trainer wrapper.

    Notes:
    - If validation data is provided, EarlyStopping + eval logging is enabled.
    - eval_result_ is populated when callbacks are used.
    """

    params: Dict[str, Any]
    early_stopping_rounds: int = 50
    log_period: int = 50

    eval_result_: Optional[Dict[str, Any]] = None

    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        eval_sets=None,  # list of (X_val, y_val)
        verbose: bool = True,
    ) -> BaseModel:
        from lightgbm import LGBMRegressor
        from lightgbm import early_stopping, log_evaluation, record_evaluation

        # Copy params so caller dict is never mutated
        p = dict(self.params or {})

        # Silence LightGBM internals when verbose=False
        # (sklearn wrapper honors both keys depending on version)
        if verbose:
            p.setdefault(
                "verbosity", 0
            )  # still keep model fairly quiet by default
        else:
            p["verbosity"] = -1
            p["verbose"] = -1

        model = LGBMRegressor(**p)

        # Build eval_set
        eval_set = [(X_train, y_train)]  # keep like your old script

        if eval_sets is not None:
            for Xv, yv in eval_sets:
                if Xv is None or yv is None:
                    continue
                eval_set.append((Xv, yv))
        elif X_valid is not None and y_valid is not None:
            eval_set.append((X_valid, y_valid))

        self.eval_result_ = {}

        callbacks = None
        if len(eval_set) >= 2:
            callbacks = [
                early_stopping(
                    stopping_rounds=int(self.early_stopping_rounds),
                    verbose=bool(
                        verbose
                    ),  # <-- KEY: suppress ES prints when verbose=False
                ),
                record_evaluation(self.eval_result_),
            ]

            # Only add evaluation logging when verbose=True
            if verbose:
                callbacks.insert(
                    1, log_evaluation(period=int(self.log_period))
                )

        # Fit
        if callbacks is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)

        return model


class TrainerFactory:
    """
    Factory for trainers.

    Supports:
    - lgbm / lightgbm

    Extend here to add:
    - xgb: XGBoost trainer
    - cat: CatBoost trainer
    - linreg/ridge/etc.
    """

    def __init__(
        self,
        cfg,
        trainer_type: str,
        trainer_params: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.trainer_type = str(trainer_type).lower().strip()
        self.trainer_params = trainer_params or {}

    def make(self) -> BaseTrainer:
        t = self.trainer_type

        if t in ("lgbm", "lightgbm"):
            return LGBMTrainer(params=self.trainer_params)

        if t in ("lgbm_quantile", "lgbm_interval", "quantile_lgbm"):
            # Import here to keep optional dependency clean
            from soft.training.trainers_quantile import (
                LGBMQuantileIntervalTrainer,
            )

            # Split params: "interval" (our meta-params) vs. LightGBM params
            p = dict(self.trainer_params or {})

            interval = dict(p.pop("interval", {}) or {})
            q_lo = float(interval.get("q_lo", 0.10))
            q_hi = float(interval.get("q_hi", 0.90))
            center = str(interval.get("center", "median"))

            # For quantile training we control objective/alpha internally.
            # Guard against accidental carry-over.
            p.pop("objective", None)
            p.pop("alpha", None)

            return LGBMQuantileIntervalTrainer(
                params=p,
                q_lo=q_lo,
                q_hi=q_hi,
                center=center,
                early_stopping_rounds=int(
                    get(self.cfg, "trainer.early_stopping_rounds", 50)
                ),
                log_period=int(get(self.cfg, "trainer.log_period", 50)),
            )
        raise ValueError(f"Unknown trainer_type: {self.trainer_type}")

    @staticmethod
    def create(
        trainer_type: str, trainer_params: Optional[Dict[str, Any]] = None
    ) -> BaseTrainer:
        """
        Convenience constructor (static), to match the calling style:
          TrainerFactory.create(...)
        """
        return TrainerFactory(trainer_type, trainer_params).make()

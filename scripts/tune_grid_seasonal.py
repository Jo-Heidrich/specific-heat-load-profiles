# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import argparse
import datetime as _dt

from soft.config_io import get, require
from soft.splits.factory import make_splits_from_cfg
from soft.evaluate.evaluate_model_general import evaluate_model_general
from soft.evaluate.cv import summarize_metrics
from soft.training.trainers import TrainerFactory

from soft.training.validation import (
    select_validation_blocks_from_train,
    select_validation_blocks_distributed,
    normalize_validation_mode,
    combine_val_blocks_to_one_eval_set,
)

"""
Run:
  python -m scripts.tune_grid_seasonal --config configs/tuning/tuning_Aalborg_SFH_NMAE.yaml
"""


def winkler_score(y, lo, hi, alpha: float):
    """
    Winkler score for (1-alpha) PI: [lo, hi]
    y, lo, hi: 1D arrays
    """
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    width = hi - lo
    below = y < lo
    above = y > hi
    penalty = (2.0 / alpha) * (lo - y) * below + (2.0 / alpha) * (
        y - hi
    ) * above
    return width + penalty


def _predict_point(model, X):
    """
    Representative point forecast:
    - quantile wrapper -> median (q50)
    - deterministic    -> normal predict
    """
    # Prefer explicit API if present
    if hasattr(model, "predict") and hasattr(model, "predict_interval"):
        # your wrapper: use median model for point metrics
        inner = getattr(model, "model_mid", model)
    else:
        inner = model

    best_it = getattr(inner, "best_iteration_", None)
    if best_it is not None and best_it > 0:
        return inner.predict(X, num_iteration=best_it)
    return inner.predict(X)


def _predict_interval_with_best_iter(model, X):
    """
    Returns (lo, mid, hi).
    Works for your QuantileIntervalModel-like wrapper.
    Uses best_iteration_ of each inner model if available.
    """
    if not hasattr(model, "predict_interval"):
        raise ValueError("Model has no predict_interval()")

    # assumes wrapper exposes model_lo/model_mid/model_hi
    lo_m = getattr(model, "model_lo", None)
    mid_m = getattr(model, "model_mid", None)
    hi_m = getattr(model, "model_hi", None)

    if lo_m is None or mid_m is None or hi_m is None:
        # fallback: wrapper handles it internally
        return model.predict_interval(X)

    def pred(inner):
        best_it = getattr(inner, "best_iteration_", None)
        if best_it is not None and best_it > 0:
            return inner.predict(X, num_iteration=best_it)
        return inner.predict(X)

    return pred(lo_m), pred(mid_m), pred(hi_m)


def _param_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    return [
        dict(zip(keys, vals))
        for vals in itertools.product(*[grid[k] for k in keys])
    ]


def _score_from_summary(
    summary_df: pd.DataFrame, optimize_metric: str
) -> float:
    if summary_df.empty:
        raise ValueError("Empty summary_df")
    if optimize_metric not in summary_df.columns:
        raise KeyError(
            f"optimize_metric '{optimize_metric}' not in summary_df columns: "
            f"{list(summary_df.columns)}"
        )
    return float(summary_df.iloc[0][optimize_metric])


def _is_valid_combo(p: dict) -> bool:
    """
    Enforce structural constraints between tree depth and number of leaves.

    Rationale:
    - Shallow trees cannot support a large number of leaves without
      creating degenerate or highly unstable splits.
    - Restricting this space reduces meaningless configurations and
      improves the efficiency of both grid search and Bayesian optimization.
    """
    d = int(p.get("max_depth", -1))
    l = int(p.get("num_leaves", 31))

    if d > 0:
        if d <= 4:
            return l <= 31
        if d >= 6:
            return l <= 64
    return True


def make_optuna_objective(
    *,
    cfg: dict,
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    splits: list,
    validation_mode: str,
    optimize_metric: str,
):
    """
    Factory that builds an Optuna objective function.

    The returned function:
    - samples hyperparameters from predefined ranges,
    - performs seasonal cross-validation,
    - evaluates each split using the same metrics as grid search,
    - and returns the mean CV score to Optuna.

    This guarantees that Bayesian optimization is fully
    comparable to the deterministic grid-search results.
    """

    def objective(trial):

        # --------------------------------------------------------
        # Sample hyperparameters
        # --------------------------------------------------------
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.1, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.6, 1.0
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.6, 1.0
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        }

        # Enforce structural constraints (depth vs. leaves)
        if not _is_valid_combo(params):
            # Return a very bad score to discourage this region
            return float("inf")

        # Merge with base trainer params
        trainer_params = dict(get(cfg, "trainer.params", {}) or {})
        trainer_params.update(params)

        trainer_type = str(get(cfg, "trainer.type", "lgbm"))
        trainer = TrainerFactory(cfg, trainer_type, trainer_params).make()

        rows = []  # store per-split metric dicts (like gridsearch)

        for si, (train_idx, test_idx) in enumerate(splits, 1):
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)

            X_train = df_train[features]
            y_train = df_train[target_col]
            X_test = df_test[features]
            y_test = df_test[target_col]

            eval_sets, X_fit, y_fit = build_validation_sets(
                cfg=cfg,
                df=df,
                features=features,
                target_col=target_col,
                train_idx=train_idx,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                validation_mode=validation_mode,
            )

            model = trainer.fit(
                X_fit, y_fit, eval_sets=eval_sets, verbose=False
            )

            # Score on validation if available, else test
            X_score, y_score = X_test, y_test
            if validation_mode != "off" and eval_sets:
                X_score, y_score = eval_sets[-1]

            opt = optimize_metric.upper().strip()
            is_interval_metric = opt.startswith("WINKLER")

            if is_interval_metric:
                if not hasattr(model, "predict_interval"):
                    raise ValueError(
                        "Interval optimization requires a quantile model "
                        "with predict_interval()."
                    )
                # IMPORTANT: use best_iteration if wrapper exposes inner models
                lo, mid, hi = _predict_interval_with_best_iter(model, X_score)

                # if you later want alpha configurable: derive from trainer_params["interval"]
                alpha = 0.10
                ws = winkler_score(y_score, lo, hi, alpha=alpha)
                met = {"WINKLER90": float(np.mean(ws))}
            else:
                y_pred = _predict_point(model, X_score)
                met = evaluate_model_general(
                    y_true=y_score,
                    y_pred=y_pred,
                    y_baseline=None,
                    plot=False,
                    verbose=False,
                )

            met["model"] = "model"
            met["split"] = si
            rows.append(met)

        per_split_df = pd.DataFrame(rows)
        summary_df = summarize_metrics(per_split_df, group_key="model")
        return _score_from_summary(summary_df, optimize_metric)

    return objective


def build_validation_sets(
    *,
    cfg: dict,
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    train_idx,
    X_train,
    y_train,
    X_test,
    y_test,
    validation_mode: str,
):
    """
    Build (eval_sets, X_fit, y_fit) according to the framework's
    validation logic.

    Returns:
        eval_sets : list[(X_val, y_val)] | None
        X_fit     : DataFrame
        y_fit     : Series
    """

    eval_sets = None
    X_fit, y_fit = X_train, y_train

    if validation_mode == "off":
        return None, X_fit, y_fit

    if validation_mode == "test":
        eval_sets = [(X_test, y_test)]
        return eval_sets, X_fit, y_fit

    if validation_mode == "train":
        val_frac = float(get(cfg, "training.validation.val_frac", 0.10))
        dom = float(get(cfg, "training.validation.dominance_ratio", 1.6))

        sel = select_validation_blocks_from_train(
            train_idx,
            val_frac=val_frac,
            n_blocks=2,
            dominance_ratio=dom,
        )

        # train on reduced core
        df_core = df.iloc[sel.train_core_idx].reset_index(drop=True)
        X_fit = df_core[features]
        y_fit = df_core[target_col]

        agg = combine_val_blocks_to_one_eval_set(
            df, features, target_col, sel.val_blocks
        )
        eval_sets = [agg] if agg is not None else None
        return eval_sets, X_fit, y_fit

    if validation_mode == "train_distributed":
        val_frac = float(get(cfg, "training.validation.val_frac", 0.10))
        n_blocks = int(get(cfg, "training.validation.n_blocks", 8))
        min_size = int(get(cfg, "training.validation.block_min_size", 24))
        seed = int(get(cfg, "training.seasonal.seed", 42))

        train_core_idx, val_blocks = select_validation_blocks_distributed(
            train_idx,
            val_frac=val_frac,
            n_blocks=n_blocks,
            block_min_size=min_size,
            rng_seed=seed,
        )

        df_core = df.iloc[train_core_idx].reset_index(drop=True)
        X_fit = df_core[features]
        y_fit = df_core[target_col]

        agg = combine_val_blocks_to_one_eval_set(
            df, features, target_col, val_blocks
        )
        eval_sets = [agg] if agg is not None else None
        return eval_sets, X_fit, y_fit

    raise ValueError(f"Unknown validation_mode: {validation_mode}")


def save_tuning_results(
    *,
    out_dir: Path,
    cfg: dict,
    method: str,
    result: dict,
    optimize_metric: str,
    validation_mode: str,
):
    """
    Persist all tuning artifacts in a structured, reproducible way.

    For each tuning run this function stores:
    - meta-information (method, metric, validation mode),
    - best hyperparameters,
    - aggregated CV performance,
    - and, if applicable, the full Optuna trial history.

    All outputs are written under:
        outputs/tuning/<experiment>/<run-id>/<method>/
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "method": method,
        "name": require(cfg, "name"),
        "optimize_metric": optimize_metric,
        "validation_mode": validation_mode,
    }

    with (out_dir / "meta.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    if result.get("best_params") is not None:
        with (out_dir / "best_params.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(result["best_params"], f, sort_keys=False)

    if result.get("best_summary") is not None:
        result["best_summary"].to_csv(
            out_dir / "best_summary.csv", index=False
        )

    if result.get("all_results"):
        pd.DataFrame(result["all_results"]).to_csv(
            out_dir / "results.csv", index=False
        )

    if result.get("study") is not None:
        result["study"].trials_dataframe().to_csv(
            out_dir / "optuna_trials.csv", index=False
        )


def run_gridsearch(
    *,
    cfg: dict,
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    splits: list,
    validation_mode: str,
    optimize_metric: str,
) -> dict:
    """
    Run a classical grid search over hyperparameters using the same
    seasonal cross-validation logic as in training and evaluation.

    The function:
    - iterates over all parameter combinations,
    - trains a model for each split,
    - evaluates on either validation or test sets (depending on validation_mode),
    - aggregates metrics across splits,
    - and returns the best configuration according to optimize_metric.

    This implementation is intentionally deterministic and fully reproducible,
    making it well-suited for scientific reporting and ablation studies.
    """

    # ------------------------------------------------------------
    # Build parameter grid
    # ------------------------------------------------------------
    grid = dict(get(cfg, "tuning.grid", {}) or {})
    combos = _param_product(grid)

    # Remove structurally invalid parameter combinations
    combos = [p for p in combos if _is_valid_combo(p)]
    print(f"Filtered grid size: {len(combos)} valid combinations")

    best_score = float("inf")
    best_params = None
    best_summary = None

    all_results = []

    # ------------------------------------------------------------
    # Iterate over all hyperparameter combinations
    # ------------------------------------------------------------
    for ci, params in enumerate(combos, 1):

        # Merge base trainer params with current grid parameters
        trainer_params = dict(get(cfg, "trainer.params", {}) or {})
        trainer_params.update(params)

        print(
            f"[Grid {ci:03d}/{len(combos)}] "
            + ", ".join(f"{k}={v}" for k, v in params.items())
        )

        # Build trainer instance
        trainer_type = str(get(cfg, "trainer.type", "lgbm"))
        trainer = TrainerFactory(cfg, trainer_type, trainer_params).make()

        rows = []  # per-split metric rows

        # --------------------------------------------------------
        # Cross-validation loop
        # --------------------------------------------------------
        for si, (train_idx, test_idx) in enumerate(splits, 1):

            # --- build train / test frames ---
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)

            X_train = df_train[features]
            y_train = df_train[target_col]
            X_test = df_test[features]
            y_test = df_test[target_col]

            # ----------------------------------------------------
            # Build validation sets according to framework logic
            # ----------------------------------------------------
            eval_sets, X_fit, y_fit = build_validation_sets(
                cfg=cfg,
                df=df,
                features=features,
                target_col=target_col,
                train_idx=train_idx,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                validation_mode=validation_mode,
            )

            # --- train ---
            model = trainer.fit(
                X_fit, y_fit, eval_sets=eval_sets, verbose=False
            )

            # ----------------------------------------------------
            # Decide which data to score on
            # ----------------------------------------------------
            # Score ALWAYS on the seasonal test window (outer evaluation)
            X_score, y_score = X_test, y_test

            # ----------------------------------------------------
            # Metric computation
            # ----------------------------------------------------
            # If optimizing an interval metric (e.g. Winkler score),
            # use predictive intervals; otherwise use point forecasts.
            opt = optimize_metric.upper().strip()
            is_interval_metric = opt.startswith("WINKLER")

            if is_interval_metric:
                if not hasattr(model, "predict_interval"):
                    raise ValueError(
                        "Interval optimization requires a quantile model "
                        "with predict_interval()."
                    )

                lo, mid, hi = model.predict_interval(X_score)
                alpha = 0.10  # for 90% interval
                ws = winkler_score(y_score, lo, hi, alpha=alpha)
                met = {"WINKLER90": float(np.mean(ws))}

            else:
                y_pred = _predict_point(model, X_score)
                met = evaluate_model_general(
                    y_true=y_score,
                    y_pred=y_pred,
                    y_baseline=None,
                    plot=False,
                    verbose=False,
                )

            met["model"] = "model"
            met["split"] = si
            rows.append(met)

        # --------------------------------------------------------
        # Aggregate metrics across splits
        # --------------------------------------------------------
        per_split_df = pd.DataFrame(rows)
        summary_df = summarize_metrics(per_split_df, group_key="model")

        score = float(summary_df.iloc[0][optimize_metric])

        all_results.append(
            {
                "combo_id": ci,
                "score": score,
                "score_metric": optimize_metric,
                "validation_mode": validation_mode,
                **trainer_params,
            }
        )

        # Track best configuration
        if score < best_score:
            best_score = score
            best_params = trainer_params
            best_summary = summary_df.copy()
            print(f"  ↳ New best {optimize_metric}: {score:.4f}")

    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_summary": best_summary,
        "all_results": all_results,
    }


def run_bayesianopt(
    *,
    cfg: dict,
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    splits: list,
    validation_mode: str,
    optimize_metric: str,
) -> dict:
    """
    Run Bayesian hyperparameter optimization using Optuna.

    Compared to grid search, Bayesian optimization:
    - explores the hyperparameter space adaptively,
    - concentrates trials in promising regions,
    - and is more efficient in higher-dimensional parameter spaces.

    The objective function uses the *same* cross-validation logic
    as grid search to ensure methodological consistency.
    """

    import optuna

    n_trials = int(get(cfg, "tuning.n_trials", 50))
    seed = int(get(cfg, "tuning.seed", 42))

    # Build objective function (closure over cfg, data, splits, etc.)
    objective = make_optuna_objective(
        cfg=cfg,
        df=df,
        features=features,
        target_col=target_col,
        splits=splits,
        validation_mode=validation_mode,
        optimize_metric=optimize_metric,
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"Starting Bayesian optimization with {n_trials} trials …")
    study.optimize(objective, n_trials=n_trials)

    print("Best Optuna score:", study.best_value)
    print("Best parameters:", study.best_params)

    # Re-run best configuration once to get a clean CV summary
    best_params = dict(get(cfg, "trainer.params", {}) or {})
    best_params.update(study.best_params)

    # Reuse grid-search evaluation logic for final summary
    res = run_gridsearch(
        cfg={
            **cfg,
            "trainer": {**cfg.get("trainer", {}), "params": best_params},
        },
        df=df,
        features=features,
        target_col=target_col,
        splits=splits,
        validation_mode=validation_mode,
        optimize_metric=optimize_metric,
    )

    return {
        "best_score": study.best_value,
        "best_params": best_params,
        "best_summary": res["best_summary"],
        "all_results": res.get("all_results"),
        "study": study,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tuning_default.yaml",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)  # <<< DAS FEHLT
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_name = require(cfg, "name")
    tuning_root = Path("outputs") / "tuning" / base_name
    tuning_root.mkdir(parents=True, exist_ok=True)

    # optional: run-id, damit "both" + mehrere runs nicht überschreiben
    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_root = tuning_root / run_id
    tuning_root.mkdir(parents=True, exist_ok=True)

    # --- choose tuning validation mode here ---
    # off: clean/no ES; train/train_distributed: leakage-free ES; test: optimistic ES
    validation_mode = normalize_validation_mode(
        get(cfg, "tuning.validation_mode", "off")
    )

    cfg.setdefault("trainer", {}).setdefault("params", {})

    grid_raw = get(cfg, "tuning.grid", None)
    grid = dict(grid_raw or {})

    if validation_mode != "off":
        # early stopping decides n_estimators
        cfg.setdefault("trainer", {}).setdefault("params", {})[
            "n_estimators"
        ] = 10000
        grid.pop("n_estimators", None)
        cfg.setdefault("tuning", {})["grid"] = grid

    # metric to minimize (from summarize_metrics output)
    optimize_metric = str(get(cfg, "tuning.optimize_metric", "MAPE_mean"))

    # --- grid: if no early stopping, include n_estimators ---
    if not grid:
        grid = {
            "learning_rate": [0.02, 0.03, 0.05],
            "num_leaves": [31, 64],
            "max_depth": [6, 10],
            "min_data_in_leaf": [30, 50],
            "feature_fraction": [0.85],
            "bagging_fraction": [0.9],
            "bagging_freq": [1],
        }
        if validation_mode == "off":
            grid["n_estimators"] = [1000, 2000, 3000, 4500]

    combos = _param_product(grid)
    print(
        f"Grid size: {len(combos)} combinations | validation_mode={validation_mode}"
    )

    combos = [p for p in combos if _is_valid_combo(p)]
    print(f"Filtered grid size: {len(combos)} valid combinations")

    # load data via runner helpers (adapt if you have a dedicated loader)
    from soft.experiment import load_dataset_from_cfg, apply_target_transform

    df, features, target_col, _meta = load_dataset_from_cfg(cfg)
    df, _factor = apply_target_transform(df, target_col, cfg)

    split_name, _df_raster, splits = make_splits_from_cfg(
        cfg, df, date_col=str(get(cfg, "data.date_col", "date"))
    )
    if split_name != "seasonal":
        raise RuntimeError(f"Expected seasonal splits, got {split_name}")

    search_mode = (
        str(get(cfg, "tuning.search_mode", "gridsearch")).lower().strip()
    )
    if search_mode not in ("gridsearch", "bayesianopt", "both"):
        raise ValueError(
            "tuning.search_mode must be one of: gridsearch|bayesianopt|both"
        )

    if split_name != "seasonal":
        raise RuntimeError(f"Expected seasonal splits, got {split_name}")

    results = {}

    if search_mode in ("gridsearch", "both"):
        out_grid = tuning_root / "gridsearch"
        out_grid.mkdir(parents=True, exist_ok=True)
        res_grid = run_gridsearch(
            cfg=cfg,
            df=df,
            features=features,
            target_col=target_col,
            splits=splits,
            validation_mode=validation_mode,
            optimize_metric=optimize_metric,
        )
        save_tuning_results(
            out_dir=out_grid,
            cfg=cfg,
            method="gridsearch",
            result=res_grid,
            optimize_metric=optimize_metric,
            validation_mode=validation_mode,
        )

    if search_mode in ("bayesianopt", "both"):
        out_bayes = tuning_root / "bayesianopt"
        out_bayes.mkdir(parents=True, exist_ok=True)
        res_bayes = run_bayesianopt(
            cfg=cfg,
            df=df,
            features=features,
            target_col=target_col,
            splits=splits,
            validation_mode=validation_mode,
            optimize_metric=optimize_metric,
        )
        save_tuning_results(
            out_dir=out_bayes,
            cfg=cfg,
            method="bayesianopt",
            result=res_bayes,
            optimize_metric=optimize_metric,
            validation_mode=validation_mode,
        )


if __name__ == "__main__":
    main()
    print("Tuning finished.")

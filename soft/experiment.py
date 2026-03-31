# -*- coding: utf-8 -*-
"""
@author: heidrich

python -m scripts.train --config configs/train_Aalborg_HEF_seasonal_legacy_hourly.yaml
python -m scripts.evaluate --config configs/evaluate_Aalborg_HEF_seasonal_legacy_hourly.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import (
    skew,
    kurtosis,
    shapiro,
    normaltest,
    jarque_bera,
    anderson,
)

from soft.config_io import get, require, ConfigError
from soft.data import FeatureDataLoader, prepare_from_raw
from soft.data import HourlyFeatureBuilder, DailyMeanFeatureBuilder
from soft.training.trainers import TrainerFactory
from soft.model import ModelBundle, save_bundle, load_bundle
from soft.evaluate.evaluate_model_general import evaluate_model_general
from soft.splits.factory import make_splits_from_cfg
from soft.plotting.plotting_results import ResultPlotter
from soft.plotting.plotting_models import ProfileComparison
from soft.evaluate.baselines import make_baseline_predictions
from soft.evaluate.cv import (
    summarize_metrics,
    save_per_split_with_summary,
    print_cv_summary_block,
)
from soft.training.validation import (
    select_validation_blocks_from_train,
    select_validation_blocks_distributed,
    normalize_validation_mode,
    combine_val_blocks_to_one_eval_set,
)
from soft.evaluate.residual_analysis import ResidualAnalyzer
from statistics import median


@dataclass
class TrainResult:
    split_model_paths: List[str]
    fulltrain_model_path: str | None
    per_split_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    splits: List[Tuple[List[int], List[int]]]
    df: pd.DataFrame


@dataclass
class EvaluateResult:
    per_split_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame


# ----------------------------
# Feature-mode helpers
# ----------------------------
def select_builder(cfg: dict):
    fm = str(require(cfg, "features.mode")).lower().strip()

    hol_country = str(get(cfg, "data.holiday_country", "Denmark"))
    hol_subdiv = get(cfg, "data.holiday_subdiv", None)
    hol_subdiv = None if hol_subdiv in (None, "", "null") else str(hol_subdiv)

    if fm == "hourly":
        return HourlyFeatureBuilder(
            holiday_country=hol_country,
            holiday_subdiv=hol_subdiv,
        )
    if fm in ("daily"):
        return DailyMeanFeatureBuilder(
            holiday_country=hol_country,
            holiday_subdiv=hol_subdiv,
        )
    raise ConfigError("features.mode must be one of: hourly, daily")


# ----------------------------
# Data loading / transforms
# ----------------------------


def _save_residuals_csv(df_res: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(path, index=False)


def load_dataset_from_cfg(
    cfg: dict,
) -> Tuple[pd.DataFrame, List[str], str, Dict[str, Any]]:
    """
    Supports:
      data.kind: processed|raw
      data.path
      data.date_col
      data.target_col
      data.drop_year (processed only)
      features.mode (hourly|daily)
    """
    meta: Dict[str, Any] = {}

    kind = str(get(cfg, "data.kind", "processed")).lower().strip()
    path = str(require(cfg, "data.path"))
    date_col = str(get(cfg, "data.date_col", "date"))
    target_col = str(get(cfg, "data.target_col", "consumption_kWh"))
    drop_year = get(cfg, "data.drop_year", None)

    builder = select_builder(cfg)
    inferred_features = builder.feature_columns()

    # You CAN override feature list explicitly (optional)
    features = get(cfg, "features.columns", None)
    if features is None:
        features = inferred_features
    else:
        if not isinstance(features, list) or not features:
            raise ConfigError(
                "features.columns must be a non-empty list if provided"
            )

    if kind == "raw":
        df = prepare_from_raw(
            raw_csv_path=path,
            mode="train",
            feature_mode=str(require(cfg, "features.mode")),
            output_path=str(get(cfg, "data.save_processed", "")) or None,
        )
        meta["data_source"] = "raw"
        meta["raw_path"] = path
        if get(cfg, "data.save_processed", ""):
            meta["saved_processed_path"] = str(get(cfg, "data.save_processed"))
    else:
        loader = FeatureDataLoader(date_col=date_col)
        df = loader.load_csv(path)
        if drop_year is not None:
            df = loader.drop_year(df, int(drop_year))
        meta["data_source"] = "processed"
        meta["processed_path"] = path
        meta["drop_year"] = drop_year

    df = df.sort_values(date_col).reset_index(drop=True)

    # sanity checks
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing required feature columns: {missing}"
        )
    if target_col not in df.columns:
        raise ValueError(f"Dataset missing target column '{target_col}'")

    return df, list(features), target_col, meta


def apply_target_transform(
    df: pd.DataFrame, target_col: str, cfg: dict
) -> None:
    """
    Supports:
      data.target_scale: float (multiply target by scale)
    """
    target_scale = float(get(cfg, "data.target_scale", 1.0))
    if target_scale != 1.0:
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found for scaling"
            )
        df[target_col] = df[target_col].astype(float) * target_scale

    return df, target_scale


def window_in_test_splits(time_index: int, n_days: int, splits) -> list[int]:
    win = set(range(time_index, time_index + n_days * 24))
    hits = []
    for si, (_tr, te) in enumerate(splits, 1):
        te_set = set(map(int, te.tolist() if hasattr(te, "tolist") else te))
        if win.issubset(te_set):
            hits.append(si)
    return hits


# ---------------------------------
def winkler_score(y, lo, hi, alpha: float):
    """
    Winkler score for (1-alpha) PI: [lo, hi]
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


def resolve_bundles_from_cfg(cfg: dict):
    """
    Same resolution logic as ExperimentRunner.run_evaluate():
    supports entries:
      - {path: "...", label: "..."}
      - {name: "<train_exp_name>", label: "..."}  -> auto-pick fulltrain or split01
      - "outputs/.../models/...pkl" (plain string)
    """
    bundles_cfg = require(cfg, "bundles")

    resolved = []
    for b in bundles_cfg:
        if isinstance(b, dict) and "path" in b:
            p = str(b["path"])
            label = str(b.get("label", "")).strip() or Path(p).stem

        elif isinstance(b, dict) and "name" in b:
            n = str(b["name"])
            base = Path("outputs") / n / "models"
            p_full = base / f"{n}__fulltrain.pkl"
            p_split1 = base / f"{n}__split01.pkl"

            if p_full.exists():
                p = str(p_full)
            elif p_split1.exists():
                p = str(p_split1)
            else:
                raise FileNotFoundError(
                    f"No model found for '{n}'. Expected {p_full} or {p_split1}."
                )
            label = str(b.get("label", n))

        else:
            p = str(b)
            label = Path(p).stem

        resolved.append((label, p))

    bundles = [(label, p, load_bundle(p)) for (label, p) in resolved]
    return bundles, resolved


def significance_report(
    df_metrics: pd.DataFrame,
    metric: str,
    model_a: str,
    model_b: str,
    n_boot: int = 10000,
):
    """
    Split-level paired comparison: model_a vs model_b on a metric (lower is better for MAE/RMSE/NMAE/NRMSE/MAPE/SMAPE).
    Prints:
      - win/tie/loss over splits
      - mean diff and bootstrap CI for mean diff
      - paired t-test and Wilcoxon signed-rank (if available)
    """
    import numpy as np

    piv = df_metrics.pivot_table(
        index="split", columns="model", values=metric, aggfunc="mean"
    )
    piv = piv.dropna(subset=[model_a, model_b])
    if piv.empty:
        print(
            f"[significance] No overlapping splits for {model_a} vs {model_b} on {metric}"
        )
        return

    a = piv[model_a].to_numpy()
    b = piv[model_b].to_numpy()
    diff = a - b  # negative => A better (smaller error)

    wins = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    losses = int(np.sum(diff > 0))
    mean_diff = float(np.mean(diff))

    # bootstrap CI over splits (resample splits with replacement)
    rng = np.random.default_rng(0)
    boot = np.array(
        [
            np.mean(rng.choice(diff, size=len(diff), replace=True))
            for _ in range(n_boot)
        ]
    )
    ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(
        np.quantile(boot, 0.975)
    )

    print(
        f"\n=== Significance over splits: {model_a} vs {model_b} | metric={metric} ==="
    )
    print(f"n_splits={len(diff)} | wins/ties/losses = {wins}/{ties}/{losses}")
    print(
        f"mean({model_a}-{model_b}) = {mean_diff:.6g}  | 95% bootstrap CI [{ci_lo:.6g}, {ci_hi:.6g}]"
    )

    # optional tests
    try:
        from scipy.stats import ttest_rel, wilcoxon

        t_p = float(ttest_rel(a, b).pvalue)
        print(f"paired t-test p={t_p:.4g}")
        try:
            w_p = float(wilcoxon(diff).pvalue)
            print(f"wilcoxon p={w_p:.4g}")
        except Exception:
            pass
    except Exception:
        pass


def resolve_training_residuals_path(
    self, bundle_path: str, bundle, unique: bool
) -> Path:
    """
    bundle_path points to an anchor model bundle inside:
      outputs/<train_exp>/models/<...>.pkl

    We resolve residuals relative to that train experiment folder.
    """
    p = Path(bundle_path)
    train_exp_dir = p.parent.parent  # .../outputs/<train_exp>/

    name = str(bundle.meta.get("experiment", train_exp_dir.name))
    # name sollte bei dir exp_base sein, nicht "__fulltrain"

    fname = (
        f"{name}__residuals__all_unique.csv"
        if unique
        else f"{name}__residuals__all.csv"
    )
    return train_exp_dir / "residuals" / fname


def resolve_fixed_window(
    cfg: dict, df_all: pd.DataFrame, date_col: str
) -> tuple[int, int] | None:
    """
    Returns (time_index, n_days) in GLOBAL df_all index space.
    Accepts either:
      - fixed_window: {start: ..., end: ...}
      - fixed_window: {index: ..., days: ...}
    """
    fw = get(cfg, "plot.fixed_window", None)
    if fw in (None, "", "null"):
        return None

    if not isinstance(fw, dict):
        raise ConfigError("plot.fixed_window must be a dict or null")

    # --- case 1: date-based ---
    if ("start" in fw) and ("end" in fw):
        t0 = pd.to_datetime(fw["start"])
        t1 = pd.to_datetime(fw["end"])
        if t1 <= t0:
            raise ConfigError("plot.fixed_window.end must be after start")

        dts = pd.to_datetime(df_all[date_col])
        # find first index >= t0
        idx0_series = np.where(dts >= t0)[0]
        if len(idx0_series) == 0:
            raise ValueError("fixed_window.start is after dataset end")
        time_index = int(idx0_series[0])

        # compute days from duration
        duration_hours = (t1 - t0).total_seconds() / 3600.0
        n_days = int(np.ceil(duration_hours / 24.0))
        n_days = max(n_days, 1)

        return time_index, n_days

    # --- case 2: index-based ---
    if ("index" in fw) and ("days" in fw):
        time_index = int(fw["index"])
        n_days = int(fw["days"])
        if n_days <= 0:
            raise ConfigError("plot.fixed_window.days must be > 0")
        return time_index, n_days

    raise ConfigError(
        "plot.fixed_window must provide either (start,end) or (index,days)"
    )


# ----------------------------
# Training / Evaluation core
# ----------------------------


class ExperimentRunner:
    def __init__(self, cfg: dict, *, config_path: str = ""):
        self.cfg = cfg
        self.config_path = config_path
        self.name = require(cfg, "name")
        self.exp_base = self.name

        self.date_col = str(get(cfg, "data.date_col", "date"))
        self.target_col = str(get(cfg, "data.target_col", "consumption_kWh"))

        is_eval_cfg = "bundles" in cfg  # evaluate configs define bundles:

        # base output name (train vs evaluate)
        out_name = f"{self.name}_eval" if is_eval_cfg else self.name

        # auto-suffix for quantile training runs (no new yaml key needed)
        if not is_eval_cfg:
            tt = str(get(cfg, "trainer.type", "")).lower().strip()
            if tt in ("lgbm_quantile", "lgbm_interval", "quantile_lgbm"):
                out_name = f"{out_name}__quantile"

        self.out_dir = Path("outputs") / out_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # subdirs
        self.plots_dir = self.out_dir / "plots"
        self.models_dir = self.out_dir / "models"
        self.metrics_dir = self.out_dir / "metrics"

        for d in (self.plots_dir, self.models_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.residuals_dir = self.out_dir / "residuals"
        self.residuals_dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    def _plot_path(self, stem: str, suffix: str = ".png") -> Path:
        return self.plots_dir / f"{self.name}__{stem}{suffix}"

    def _save_splits_json(
        self, split_name: str, splits, date_col: str
    ) -> Path:
        import json

        payload = {
            "experiment": self.name,
            "split_name": split_name,
            "date_col": date_col,
            "n_splits": len(splits) if splits is not None else 0,
            "splits": [
                {
                    "split": i,
                    "train_idx": list(map(int, train_idx)),
                    "test_idx": list(map(int, test_idx)),
                }
                for i, (train_idx, test_idx) in enumerate(splits, 1)
            ],
        }

        out_path = self.out_dir / "splits.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved splits: {out_path}")
        return out_path

    # ---------- train ----------
    def run_train(self) -> TrainResult:
        cfg = self.cfg
        date_col = self.date_col

        df, features, target_col, meta_add = load_dataset_from_cfg(cfg)

        # apply scaling
        df, factor = apply_target_transform(df, target_col, cfg)
        meta_add["target_scale"] = factor

        trainer_type = require(cfg, "trainer.type")
        trainer_params = get(cfg, "trainer.params", {})
        trainer = TrainerFactory(cfg, trainer_type, trainer_params).make()

        # interval metrics enabled if model supports predict_interval AND interval is configured
        interval_cfg = get(cfg, "trainer.params.interval", None)
        use_interval_metrics = interval_cfg is not None

        best_iters: List[int] = []

        split_name, _df_raster, splits = make_splits_from_cfg(
            cfg, df, date_col=date_col
        )
        n_splits = len(splits) if splits is not None else 0
        full_train = bool(get(cfg, "training.full_train", False))

        print(f"Split strategy: {split_name}")
        print(f"Number of splits: {n_splits}")

        if splits is not None:
            self._save_splits_json(split_name, splits, date_col)

        all_rows: List[Dict[str, Any]] = []
        models: List[Any] = []
        val_blocks_by_split = []

        split_model_paths: List[str] = []
        fulltrain_model_path: Optional[str] = None

        residual_rows: List[pd.DataFrame] = []

        # train
        if split_name == "none":
            X_all, y_all = df[features], df[target_col]
            model = trainer.fit(
                X_all, y_all, X_valid=None, y_valid=None, verbose=True
            )
            models = [model]
        else:
            val_cfg = get(cfg, "training.validation", {})
            val_mode = normalize_validation_mode(get(val_cfg, "mode", "off"))
            val_frac = float(get(val_cfg, "val_frac", 0.15))
            dominance_ratio = float(get(val_cfg, "dominance_ratio", 1.6))
            n_blocks = int(get(val_cfg, "n_blocks", 2))

            for si, (train_idx, test_idx) in enumerate(splits, 1):
                # --- keep GLOBAL indices as numpy arrays ---
                train_idx_arr = np.asarray(train_idx, dtype=int)
                test_idx_arr = np.asarray(test_idx, dtype=int)

                # you may still reset for clean X/y matrices,
                # BUT keep global indices separately
                df_train = df.iloc[train_idx_arr].reset_index(drop=True)
                df_test = df.iloc[test_idx_arr].reset_index(drop=True)

                X_train = df_train[features]
                y_train = df_train[target_col]
                X_test = df_test[features]
                y_test = df_test[target_col]

                # --- choose validation mode ---
                eval_sets = None

                if val_mode == "off":
                    # no validation -> no early stopping
                    X_fit, y_fit = X_train, y_train
                    val_blocks_by_split.append([])

                elif val_mode == "test":
                    # reproduce / leaky mode: validate on test
                    X_fit, y_fit = X_train, y_train
                    eval_sets = [(X_test, y_test)]
                    val_blocks_by_split.append([test_idx])

                elif val_mode == "train":
                    # pick 2 blocks but COMBINE them into ONE validation set for stable early stopping
                    sel = select_validation_blocks_from_train(
                        train_idx, val_frac=val_frac, n_blocks=2
                    )
                    val_blocks = sel.val_blocks
                    train_core_idx = sel.train_core_idx

                    df_core = df.iloc[train_core_idx].reset_index(drop=True)
                    X_fit = df_core[features]
                    y_fit = df_core[target_col]

                    # combine blocks -> single validation set
                    agg = combine_val_blocks_to_one_eval_set(
                        df, features, target_col, sel.val_blocks
                    )
                    eval_sets = [agg] if agg is not None else None

                    val_blocks_by_split.append(sel.val_blocks)

                elif val_mode == "train_distributed":
                    # demo: many small blocks
                    train_core_idx, val_blocks = (
                        select_validation_blocks_distributed(
                            train_idx,
                            val_frac=val_frac,
                            n_blocks=8,
                            block_min_size=24,
                            rng_seed=int(
                                get(cfg, "training.seasonal.seed", 42)
                            ),
                        )
                    )

                    df_core = df.iloc[train_core_idx].reset_index(drop=True)
                    X_fit = df_core[features]
                    y_fit = df_core[target_col]

                    # IMPORTANT: combine blocks to ONE eval set (otherwise early stopping becomes unstable)
                    agg = combine_val_blocks_to_one_eval_set(
                        df, features, target_col, val_blocks
                    )
                    eval_sets = [agg] if agg is not None else None

                    val_blocks_by_split.append(val_blocks)

                else:
                    raise ConfigError(
                        "training.validation.mode must be one of: off|train|test|train_distributed"
                    )

                # --- fit ---
                m = trainer.fit(
                    X_fit, y_fit, eval_sets=eval_sets, verbose=True
                )
                best_it = getattr(m, "best_iteration_", None)
                if best_it:
                    best_iters.append(int(best_it))

                models.append(m)

                split_tag = f"{self.name}__split{si:02d}"
                meta_split = {
                    "feature_mode": require(cfg, "features.mode"),
                    "trainer_type": trainer_type,
                    "trainer_params": trainer_params,
                    "split": split_name,
                    "split_id": si,
                    "config_path": self.config_path,
                    **meta_add,
                }
                meta_split["target_scale"] = meta_add.get("target_scale", 1.0)
                meta_split["experiment"] = self.exp_base
                if hasattr(m, "predict_interval"):
                    meta_split["interval"] = {
                        "q_lo": float(m.q_lo),
                        "q_hi": float(m.q_hi),
                        "center": str(m.center),
                    }

                bundle_split = ModelBundle(
                    model=m,
                    features=list(features),
                    target=target_col,
                    meta=meta_split,
                )

                out_split = self.models_dir / f"{split_tag}.pkl"
                save_bundle(bundle_split, str(out_split))
                # also export native LightGBM model to .txt (optional, keep .pkl for pipeline)
                try:
                    booster = getattr(m, "booster_", None)
                    if booster is not None:
                        out_txt = out_split.with_suffix(".txt")
                        booster.save_model(str(out_txt))
                        print(f"Saved LightGBM txt: {out_txt}")
                except Exception as e:
                    print(
                        f"[warn] could not export LightGBM txt for split {si}: {e}"
                    )

                split_model_paths.append(str(out_split))

                best_it = getattr(m, "best_iteration_", None)
                if eval_sets is not None:
                    print(f"  Split {si}: best_iteration={best_it}")

                # --- predict with best_iteration if available ---

                yhat = _predict_point(m, X_test)

                # --- residuals per split (GLOBAL index!) ---
                train_start = int(train_idx_arr.min())
                train_end = int(train_idx_arr.max())

                # date sanity (optional)
                # df_test[date_col] is already in df_test, so we use it for plotting/time features
                # global_index comes from test_idx_arr

                # --- residual base: take ALL columns from original df (global index preserved) ---
                df_res = df.iloc[
                    test_idx_arr
                ].copy()  # <-- IMPORTANT: all columns, global index
                df_res["global_index"] = df_res.index.astype(int)

                # add model outputs
                df_res["y_pred"] = yhat
                df_res["y_true"] = df_res[
                    target_col
                ].to_numpy()  # same as y_test but aligned to df
                df_res["residual"] = (
                    df_res["y_true"].to_numpy() - df_res["y_pred"].to_numpy()
                )

                # meta
                df_res["model"] = self.name
                df_res["split"] = si
                df_res["train_start_idx"] = int(train_idx_arr.min())
                df_res["train_end_idx"] = int(train_idx_arr.max())

                # optional: move global_index to front (nicer csv)
                front = [
                    "global_index",
                    date_col,
                    "y_true",
                    "y_pred",
                    "residual",
                ]
                cols = front + [c for c in df_res.columns if c not in front]
                df_res = df_res[cols]

                out_res_split = (
                    self.residuals_dir
                    / f"{self.name}__residuals__split{si:02d}.csv"
                )
                _save_residuals_csv(
                    df_res.reset_index(drop=True), out_res_split
                )  # reset only for file cleanliness
                residual_rows.append(df_res.reset_index(drop=True))

                # sanity: global_index must match test_idx_arr (order-wise)
                missing = [c for c in df.columns if c not in df_res.columns]
                if missing:
                    print(
                        f"[residuals][warn] still missing {len(missing)} df columns, e.g.: {missing[:8]}"
                    )

                # must match test_idx_arr exactly
                assert np.array_equal(
                    df_res["global_index"].to_numpy(dtype=int), test_idx_arr
                )

                # --- SANITY CHECKS (per split) ---
                # 1) ensure we truly stored global indices
                assert df_res["global_index"].min() >= 0
                # 2) check local-vs-global symptom: local indices would start at 0 each split
                if df_res["global_index"].min() == 0 and si > 1:
                    print(
                        f"[sanity] WARNING: split {si} global_index starts at 0. "
                        "This would indicate local indices; expected global positions."
                    )

                # 3) quick coverage print (optional but helpful)
                dmin = pd.to_datetime(df_res["date"]).min()
                dmax = pd.to_datetime(df_res["date"]).max()
                print(
                    f"[residuals] split {si:02d}: "
                    f"global_index=[{df_res['global_index'].min()}..{df_res['global_index'].max()}], "
                    f"dates=[{dmin}..{dmax}], n={len(df_res)}"
                )

                metrics = evaluate_model_general(
                    y_true=y_test.values,
                    y_pred=yhat,
                    y_baseline=None,
                    plot=False,
                )
                metrics["split"] = si
                metrics["model"] = "model"

                # --- interval metrics (optional) ---
                if use_interval_metrics and hasattr(m, "predict_interval"):
                    q_lo = float(get(interval_cfg, "q_lo", 0.05))
                    q_hi = float(get(interval_cfg, "q_hi", 0.95))
                    alpha = 1.0 - (q_hi - q_lo)  # 0.10 for 0.05..0.95

                    lo, mid, hi = _predict_interval_with_best_iter(m, X_test)
                    w = winkler_score(y_test.values, lo, hi, alpha=alpha)

                    metrics[f"WINKLER{int(round((q_hi-q_lo)*100)):02d}"] = (
                        float(np.mean(w))
                    )
                    # for 0.05..0.95 -> key "WINKLER90"

                all_rows.append(metrics)

        full_train = bool(get(cfg, "training.full_train", False))
        if full_train:
            # If early stopping was used during CV splits, align the final model size
            # to the median best_iteration_ across splits.
            if best_iters:
                med_it = int(median(best_iters))
                med_it = max(1, med_it)

                # IMPORTANT: full_train has no validation -> no early stopping,
                # so we must set n_estimators explicitly.
                trainer.params = dict(trainer.params or {})
                trainer.params["n_estimators"] = med_it

                print(
                    f"Full-train: using n_estimators = median(best_iteration) = {med_it} "
                    f"(from {len(best_iters)} splits)"
                )

            else:
                print(
                    "Full-train: no best_iteration values found -> keeping trainer.params.n_estimators"
                )

            X_all, y_all = df[features], df[target_col]
            m_full = trainer.fit(
                X_all, y_all, X_valid=None, y_valid=None, verbose=True
            )

            meta_full = {
                "feature_mode": require(cfg, "features.mode"),
                "trainer_type": trainer_type,
                "trainer_params": trainer_params,
                "split": split_name,
                "split_id": None,
                "trained_on": "full",
                "config_path": self.config_path,
                **meta_add,
            }
            meta_full["target_scale"] = meta_add.get("target_scale", 1.0)
            meta_full["experiment"] = self.exp_base
            if hasattr(m_full, "predict_interval"):
                meta_full["interval"] = {
                    "q_lo": float(m.q_lo),
                    "q_hi": float(m.q_hi),
                    "center": str(m.center),
                }

            bundle_full = ModelBundle(
                model=m_full,
                features=list(features),
                target=target_col,
                meta=meta_full,
            )

            out_full = self.models_dir / f"{self.name}__fulltrain.pkl"
            save_bundle(bundle_full, str(out_full))
            fulltrain_model_path = str(out_full)
            print(f"Saved full-train ModelBundle: {out_full}")
            # also export native LightGBM model to .txt (optional, keep .pkl for pipeline)
            try:
                booster = getattr(m_full, "booster_", None)
                if booster is not None:
                    out_txt = out_full.with_suffix(".txt")
                    booster.save_model(str(out_txt))
                    print(f"Saved LightGBM txt: {out_txt}")
            except Exception as e:
                print(
                    f"[warn] could not export LightGBM txt for fulltrain: {e}"
                )

        print("Saved per-split ModelBundles under models/{name}__splitXX.pkl")

        # --- global residuals over all split test points (deduplicated) ---
        if residual_rows:
            df_res_all = pd.concat(residual_rows, ignore_index=True)

            # Sanity: global index range must match dataset scale (not just ~3500)
            gi_min = int(df_res_all["global_index"].min())
            gi_max = int(df_res_all["global_index"].max())
            nuniq = int(df_res_all["global_index"].nunique())
            nall = int(len(df_res_all))
            print(
                f"[sanity] residual ALL: global_index range = [{gi_min}..{gi_max}]"
            )
            print(
                f"[sanity] residual ALL: unique global_index = {nuniq} / rows = {nall}"
            )

            # If this is way too small (e.g. max ~3500), something is still local
            if gi_max < 0.5 * len(df):  # heuristic
                print(
                    "[sanity] WARNING: global_index max is suspiciously small vs len(df). "
                    "Likely still storing per-split local indices somewhere."
                )

            # Check overlap amount
            n_dups = nall - nuniq
            print(
                f"[sanity] residual ALL: duplicates due to overlapping test sets = {n_dups}"
            )

            def _distance_to_training(row) -> int:
                # distance from this test point to the training set of THAT split
                # using train_end as proxy (works well for contiguous training cores)
                t = int(row["global_index"])
                te = int(row["train_end_idx"])
                ts = int(row.get("train_start_idx", te))

                # if training is before test (typical), distance to end is meaningful
                if t >= te:
                    return t - te
                # if test is before training (rare, but safe)
                if t <= ts:
                    return ts - t
                return 0

            df_res_all["dist_to_train"] = df_res_all.apply(
                _distance_to_training, axis=1
            )

            # choose, for each (global_index, model), the entry with maximum distance
            df_res_unique = (
                df_res_all.sort_values(
                    ["global_index", "model", "dist_to_train"],
                    ascending=[True, True, False],
                )
                .drop_duplicates(
                    subset=["global_index", "model"], keep="first"
                )
                .drop(columns=["dist_to_train"])
                .sort_values("global_index")
            )

            out_res_all = (
                self.residuals_dir / f"{self.name}__residuals__all.csv"
            )
            _save_residuals_csv(df_res_all, out_res_all)

            out_res_unique = (
                self.residuals_dir / f"{self.name}__residuals__all_unique.csv"
            )
            _save_residuals_csv(df_res_unique, out_res_unique)

        per_split_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        summary_df = (
            summarize_metrics(per_split_df, group_key="model")
            if not per_split_df.empty
            else pd.DataFrame()
        )

        if not summary_df.empty:
            print_cv_summary_block(summary_df, group_key="model", decimals=3)

        cv_out = self.metrics_dir / f"{self.name}__cv_metrics.csv"
        save_per_split_with_summary(
            per_split_df,
            out_path=str(cv_out),
            group_key="model",
        )
        print(f"Saved CV metrics (per-split + mean/std): {cv_out}")

        plot_splits = bool(get(cfg, "output.plot_splits", True))
        if split_name == "seasonal" and plot_splits:
            from soft.plotting.plotting_splits import (
                plot_seasonal_splits,
                plot_splits_with_validation,
            )

            val_cfg = get(cfg, "training.validation", {})
            val_mode = normalize_validation_mode(get(val_cfg, "mode", "off"))
            variant = str(get(cfg, "training.seasonal.variant", "legacy"))

            # always save the plain seasonal split plot
            plot_seasonal_splits(
                df=df,
                splits=splits,
                date_col=date_col,
                title="",
                save_path=self.plots_dir / f"{self.name}__seasonal_splits.png",
            )

            # save the "with validation" overlay, depending on mode
            if val_mode == "off":
                vblocks = None
                subtitle = "val=off"
            elif val_mode == "train":
                vblocks = val_blocks_by_split
                subtitle = f"val=train (val_frac={float(get(val_cfg,'val_frac',0.15))})"
            elif val_mode in ("train", "train_distributed"):
                vblocks = val_blocks_by_split
                subtitle = f"val={val_mode} (val_frac={val_frac})"
            else:  # test
                vblocks = val_blocks_by_split
                subtitle = "val=test (leaky)"

            plot_splits_with_validation(
                df=df,
                splits=splits,
                val_blocks_by_split=vblocks,
                date_col=date_col,
                title="",
                save_path=str(
                    self.plots_dir
                    / f"{self.name}__seasonal_splits_with_validation.png"
                ),
            )

        return TrainResult(
            split_model_paths=split_model_paths,
            fulltrain_model_path=fulltrain_model_path,
            per_split_metrics=per_split_df,
            summary_metrics=summary_df,
            splits=splits,
            df=df,
        )

    # ---------- evaluate ----------
    def run_evaluate(self) -> EvaluateResult:
        fraunhofer_colors = {
            "green": "#00a19a",  # ggf. an deine exakten Farben anpassen
            "graphit": "#3b3b3b",
        }

        cfg = self.cfg
        date_col = self.date_col

        df, _features, target_col, meta_add = load_dataset_from_cfg(cfg)
        df, factor = apply_target_transform(df, target_col, cfg)

        df_all = df.copy()

        # bundles to evaluate
        bundles, resolved = resolve_bundles_from_cfg(cfg)

        # load exact splits from training artifact
        import json

        if "evaluation.splits_path" in cfg:
            splits_path = Path(cfg["evaluation"]["splits_path"])
        else:
            # default: take splits.json next to the first bundle's experiment dir
            first_bundle_path = Path(resolved[0][1])
            exp_dir = first_bundle_path.parent.parent  # .../outputs/<exp>
            splits_path = exp_dir / "splits.json"

        if not splits_path.exists():
            raise FileNotFoundError(
                f"splits.json not found at {splits_path}. "
                "Train must save splits.json, or set evaluation.splits_path."
            )

        with splits_path.open("r", encoding="utf-8") as f:
            splits_payload = json.load(f)

        split_name = splits_payload["split_name"]
        splits = [
            (
                np.array(s["train_idx"], dtype=int),
                np.array(s["test_idx"], dtype=int),
            )
            for s in splits_payload["splits"]
        ]
        print(
            f"Loaded splits: {splits_path} | split_name={split_name} | n_splits={len(splits)}"
        )

        n_splits = len(splits) if splits is not None else 0

        print("\n=== Bundle sanity & split resolution check ===\n")

        cfg_scale = float(get(cfg, "data.target_scale", 1.0))
        n_splits = len(splits)

        for label, p, bundle in bundles:
            pth = Path(p)

            print(f"Bundle '{label}':")
            print(f"  anchor path     : {pth}")
            print(f"  path exists     : {pth.exists()}")

            # --- basic bundle sanity ---
            b_scale = float(bundle.meta.get("target_scale", 1.0))
            print(f"  target          : {bundle.target}")
            print(f"  n_features      : {len(bundle.features)}")
            print(f"  target_scale    : bundle={b_scale} | cfg={cfg_scale}")

            tr = bundle.meta.get("trainer_type", None)
            print(f"  trainer_type    : {tr}")

            if abs(b_scale - cfg_scale) > 1e-12:
                raise ValueError(
                    f"[{label}] target_scale mismatch: bundle={b_scale} vs cfg={cfg_scale}. "
                    "Use matching evaluate config or retrain."
                )

            # --- split model resolution ---
            base_models_dir = pth.parent
            prefix = str(bundle.meta.get("experiment", label))

            print(f"  experiment      : {prefix}")
            print(f"  models dir      : {base_models_dir}")
            print(f"  splits expected : {n_splits}")

            missing = []
            for si in range(1, n_splits + 1):
                sp = base_models_dir / f"{prefix}__split{si:02d}.pkl"
                if sp.exists():
                    print(f"    split{si:02d} -> OK")
                else:
                    print(f"    split{si:02d} -> MISSING ({sp.name})")
                    missing.append(sp)

            if missing:
                raise FileNotFoundError(
                    f"\nBundle '{label}' is missing {len(missing)} split models.\n"
                    f"Expected all of:\n"
                    + "\n".join(f"  - {p}" for p in missing)
                )

            print("")

        print("=== Bundle sanity & split resolution OK ===\n")

        # --- validation mode controls plotting overlay only (evaluate doesn't train) ---
        val_cfg = get(cfg, "training.validation", {})
        val_mode = str(get(val_cfg, "mode", "off")).lower().strip()
        val_frac = float(get(val_cfg, "val_frac", 0.15))
        dominance_ratio = float(get(val_cfg, "dominance_ratio", 1.6))
        n_blocks = int(get(val_cfg, "n_blocks", 2))

        if n_blocks != 2:
            raise ConfigError("Currently only n_blocks=2 is supported.")

        val_blocks_by_split = []
        for train_idx, test_idx in splits:
            if val_mode == "off":
                val_blocks_by_split.append([])
            elif val_mode == "test":
                # mark test also as validation (leaky mode visualization)
                val_blocks_by_split.append([test_idx])
            elif val_mode == "train":
                sel = select_validation_blocks_from_train(
                    train_idx,
                    val_frac=val_frac,
                    dominance_ratio=dominance_ratio,
                )
                val_blocks_by_split.append(sel.val_blocks)
            else:
                raise ConfigError(
                    "training.validation.mode must be one of: off|train|test"
                )

        plot_splits = bool(get(cfg, "output.plot_splits", True))
        if split_name == "seasonal" and plot_splits:
            from soft.plotting.plotting_splits import (
                plot_splits_with_validation,
            )

            variant = str(get(cfg, "training.seasonal.variant", "legacy"))

            # overlay plot
            vblocks = None if val_mode == "off" else val_blocks_by_split
            subtitle = (
                "val=off"
                if val_mode == "off"
                else (
                    "val=test (leaky)"
                    if val_mode == "test"
                    else f"val=train (val_frac={val_frac})"
                )
            )

            plot_splits_with_validation(
                df=df,
                splits=splits,
                val_blocks_by_split=vblocks,
                date_col=date_col,
                title="",
                save_path=str(
                    self._plot_path(
                        "evaluate__seasonal_splits_with_validation"
                    )
                ),
            )
            print(
                f"Saved plot: {self._plot_path('evaluate__seasonal_splits_with_validation')}"
            )

        # Collect per split per model
        rows: List[Dict[str, Any]] = []
        preds_for_plot: Dict[str, np.ndarray] = {}
        plotter = ResultPlotter()

        # For plotting “comparison to real consumption”: only sensible for a single test window set.
        # In seasonal CV there are many disjoint windows; we’ll still allow “multi windows plot” if you want.
        plot_enabled = bool(get(cfg, "plot.enabled", True))
        window_days = int(get(cfg, "plot.window_days", 10))
        n_windows = int(get(cfg, "plot.n_windows", 2))

        pool = {}  # model -> dict of sums

        def _pool_init():
            return {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_y_abs": 0.0}

        for si, (train_idx, test_idx) in enumerate(splits, 1):
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].copy()
            df_test["global_index"] = df_test.index.astype(int)
            df_test = df_test.reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)
            df_test = df_test.sort_values(date_col).reset_index(drop=True)
            y_true = df_test[target_col].to_numpy()

            # store preds for plotting only for first split (optional), to avoid huge plots
            do_plot_split = si == 2

            base_cfg = get(cfg, "evaluation.baselines", {})
            baseline_type = str(get(base_cfg, "type", "none")).lower().strip()
            baseline_features = get(base_cfg, "linreg_features", None)

            baseline_preds = {}
            if baseline_type != "none":
                baseline_preds = make_baseline_predictions(
                    df_train=df_train,
                    df_test=df_test,
                    date_col=date_col,
                    target_col=target_col,
                    baseline_type=baseline_type,
                    baseline_features=baseline_features,
                )

            # --- evaluate baselines like models ---
            for bname, y_bpred in baseline_preds.items():
                mb = evaluate_model_general(
                    y_true=y_true,
                    y_pred=y_bpred,
                    y_baseline=None,
                    plot=False,
                    verbose=False,
                )
                mb["model"] = bname
                mb["split"] = si
                rows.append(mb)

                st = pool.setdefault(bname, _pool_init())
                y_bpred = np.asarray(y_bpred, float)
                mfin = np.isfinite(y_true) & np.isfinite(y_bpred)
                if mfin.any():
                    err = y_true[mfin] - y_bpred[mfin]
                    st["n"] += int(mfin.sum())
                    st["sum_abs"] += float(np.sum(np.abs(err)))
                    st["sum_sq"] += float(np.sum(err**2))
                    st["sum_y_abs"] += float(np.sum(np.abs(y_true[mfin])))

                if plot_enabled and do_plot_split:
                    preds_for_plot[bname] = y_bpred

            for label, p, bundle in bundles:
                base_models_dir = Path(
                    p
                ).parent  # .../outputs/<train_exp>/models

                prefix = str(bundle.meta.get("experiment", label))
                split_model_path = (
                    base_models_dir / f"{prefix}__split{si:02d}.pkl"
                )
                if not split_model_path.exists():
                    raise FileNotFoundError(
                        f"Missing split model: {split_model_path}. "
                        "CV-evaluation requires one model per split."
                    )

                bundle_i = load_bundle(str(split_model_path))
                model = bundle_i.model
                feat_i = bundle_i.features

                missing = set(feat_i) - set(df_test.columns)
                if missing:
                    raise ValueError(
                        f"[{prefix}] Missing features in df_test: {sorted(missing)[:10]}"
                    )

                X_test = df_test[feat_i]

                y_pred = _predict_point(model, X_test)

                m = evaluate_model_general(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_baseline=None,
                    plot=False,
                    verbose=False,
                )
                m["model"] = label
                m["split"] = si

                st = pool.setdefault(label, _pool_init())
                err = y_true - y_pred
                st["n"] += len(y_true)
                st["sum_abs"] += float(np.sum(np.abs(err)))
                st["sum_sq"] += float(np.sum(err**2))
                st["sum_y_abs"] += float(np.sum(np.abs(y_true)))

                # interval metrics if the loaded model supports it and cfg requests it
                interval_cfg = bundle_i.meta.get("interval", None)
                if interval_cfg is not None and hasattr(
                    model, "predict_interval"
                ):
                    q_lo = float(interval_cfg.get("q_lo", 0.05))
                    q_hi = float(interval_cfg.get("q_hi", 0.95))
                    alpha = 1.0 - (q_hi - q_lo)

                    lo, mid, hi = _predict_interval_with_best_iter(
                        model, X_test
                    )
                    w = winkler_score(y_true, lo, hi, alpha=alpha)
                    m[f"WINKLER{int(round((q_hi-q_lo)*100)):02d}"] = float(
                        np.mean(w)
                    )

                rows.append(m)

                if plot_enabled and do_plot_split:
                    preds_for_plot[label] = y_pred

            if plot_enabled and do_plot_split:
                segment = str(get(cfg, "plot.segment", "last"))

                # Multi-model window plot saved under plots/
                out_path = self._plot_path(
                    f"evaluate_compare__{split_name}__split{si}"
                )
                plotter.plot_windows(
                    df_test=df_test,
                    y_true=y_true,
                    preds=preds_for_plot,
                    date_col=date_col,
                    window_days=window_days,
                    n_windows=n_windows,
                    title=f"Model comparison ({split_name}) - split {si}",
                    save_path=str(out_path),
                    show=False,
                )

                stem_path = self._plot_path(
                    f"evaluate_compare_temp__{split_name}__split{si}",
                    suffix="",  # <- wichtig
                )

                plotter.plot_windows_temp(
                    df_test=df_test,
                    y_true=y_true,
                    preds=preds_for_plot,
                    date_col=date_col,
                    temperature_col="temperature_day_mean",
                    window_days=window_days,
                    n_windows=n_windows,
                    title=f"Model comparison ({split_name}) - split {si}, segment=last",
                    save_stem=stem_path.name,  # nur Name
                    out_dir=str(stem_path.parent),  # plots dir
                    show=False,
                    segment=segment,
                    gap_hours=2,
                )

                out_prof = self._plot_path(
                    f"profile_compare__{split_name}__split{si}"
                )
                plotter.plot_profile_compare_multi(
                    df_test=df_test,
                    date_col=date_col,
                    target_col=target_col,  # evaluate => vorhanden
                    preds=preds_for_plot,  # label -> y_pred
                    temp_col="temperature_hourly",  # <- anpassen an deine Spalte
                    start=None,
                    hours=24
                    * window_days,  # z.B. gleiche Länge wie window_days
                    ylim_temp=None,
                    title=f"Profile compare ({split_name}) - split {si}",
                    save_path=str(out_prof),
                    show=False,
                )

                # Demo window settings
                fw_res = resolve_fixed_window(
                    cfg, df_all=df_all, date_col=date_col
                )

                if fw_res is not None:
                    time_index, paper_days = fw_res

                    # check coverage in test splits
                    hits = window_in_test_splits(
                        time_index, paper_days, splits
                    )
                    if hits:
                        print(
                            f"📌 Fixed window [idx {time_index}..{time_index+paper_days*24-1}] "
                            f"is FULLY in test set of split(s): {hits}"
                        )
                    else:
                        print(
                            f"⚠️ Fixed window [idx {time_index}..{time_index+paper_days*24-1}] "
                            f"is NOT fully contained in any test split"
                        )

                    # choose which split model to use for prediction curve in the paper plot
                    paper_split_id = len(
                        splits
                    )  # last split by default (robust)
                    train_idx_p, test_idx_p = splits[-1]

                    paper_label, paper_p, paper_bundle = bundles[
                        0
                    ]  # first YAML model
                    paper_base_models_dir = Path(paper_p).parent
                    paper_prefix = str(
                        paper_bundle.meta.get("experiment", paper_label)
                    )

                    split_model_path = (
                        paper_base_models_dir
                        / f"{paper_prefix}__split{paper_split_id:02d}.pkl"
                    )
                    b = load_bundle(str(split_model_path))
                    m = b.model
                    feat = b.features

                    X_all = df_all[feat]
                    y_pred_all = _predict_point(m, X_all)

                    preds_all = {paper_label: y_pred_all}

                    out_path = self._plot_path(
                        f"paper_window_idx{time_index}_days{paper_days}"
                    ).as_posix()
                    plotter.plot_fixed_global_window_temp(
                        df_all=df_all,
                        y_true_all=df_all[target_col].to_numpy(),
                        preds_all=preds_all,
                        date_col=date_col,
                        time_index=time_index,
                        n_days=paper_days,
                        temperature_col="temperature_day_mean",
                        title=f"Paper example window (idx={time_index}, {paper_days} days)",
                        save_path=out_path,
                        show=False,
                    )
                    print(f"✅ Saved paper window plot: {out_path}")

        per_split_df = pd.DataFrame(rows)
        summary_df = summarize_metrics(per_split_df, group_key="model")

        if not summary_df.empty:
            print_cv_summary_block(
                summary_df,
                group_key="model",
                decimals=3,
                title="Evaluation metrics (mean ± std over splits)",
            )

        # single CSV: per-split + appended mean/std
        eval_out = self.metrics_dir / f"{self.name}__eval_metrics.csv"
        save_per_split_with_summary(
            per_split_df,
            out_path=str(eval_out),
            group_key="model",
        )
        print(f"Saved evaluation metrics (per-split + mean/std): {eval_out}")

        main_metric = str(get(cfg, "evaluation.significance_metric", "NMAE"))
        col = (
            f"{main_metric}_mean"
            if f"{main_metric}_mean" in summary_df.columns
            else main_metric
        )

        best = summary_df.sort_values(col, ascending=True).iloc[0]
        print(
            f"🏆 Best model by {main_metric}: {best['model']} ({best[f'{main_metric}_mean']:.4g} ± {best[f'{main_metric}_std']:.4g})"
        )

        # pick main metric for selection (you can change)
        main_metric = str(get(cfg, "evaluation.significance_metric", "NMAE"))

        # compare the two LGBM models
        labels = [b[0] for b in bundles]  # eval labels
        if len(labels) >= 2:
            significance_report(
                per_split_df, main_metric, labels[0], labels[1]
            )

        # optionally compare to baselines (if present)
        for b in ["baseline_rolling24h", "baseline_linreg"]:
            if b in per_split_df["model"].unique():
                significance_report(per_split_df, main_metric, labels[0], b)
                significance_report(per_split_df, main_metric, labels[1], b)

        pooled_rows = []
        for model_name, st in pool.items():
            n = st["n"]
            mae = st["sum_abs"] / n
            rmse = (st["sum_sq"] / n) ** 0.5
            mean_abs_y = st["sum_y_abs"] / n
            nmae = mae / mean_abs_y
            nrmse = rmse / mean_abs_y
            pooled_rows.append(
                {
                    "model": model_name,
                    "MAE_pooled": mae,
                    "RMSE_pooled": rmse,
                    "NMAE_pooled": nmae,
                    "NRMSE_pooled": nrmse,
                    "n_test_points": n,
                }
            )

        df_pooled = pd.DataFrame(pooled_rows)
        mm = main_metric.replace("_mean", "")
        df_pooled = df_pooled.sort_values(f"{mm}_pooled")
        out_pooled = self.metrics_dir / f"{self.name}__eval_metrics_pooled.csv"
        df_pooled.to_csv(out_pooled, index=False)
        print(f"Saved pooled metrics: {out_pooled}")

        mm = main_metric.replace(
            "_mean", ""
        )  # falls jemand "NMAE" oder "NMAE_mean" angibt
        col = f"{mm}_pooled"
        if col in df_pooled.columns:
            best = df_pooled.sort_values(col).iloc[0]
            print(
                f"🏆 Best model (pooled) by {mm}: {best['model']} ({best[col]:.4g})"
            )

        def pooled_diff_statement(df_pooled, metric, a, b):
            col = f"{metric}_pooled"
            pa = float(df_pooled.loc[df_pooled.model == a, col].iloc[0])
            pb = float(df_pooled.loc[df_pooled.model == b, col].iloc[0])
            print(
                f"Overall pooled {metric}: {a}={pa:.4g}, {b}={pb:.4g}, diff={pa-pb:.4g}"
            )

        pooled_diff_statement(df_pooled, "NMAE", labels[0], labels[1])

        """
        # --- residual analysis (global, per bundle) ---
        analyzer = ResidualAnalyzer(
            date_col="date",
            residual_col="residual",
            y_pred_col="y_pred",
            quiet=False,
        )

        residual_summary_rows = []

        for label, p, bundle in bundles:
            # p = anchor bundle path, e.g. outputs/<train_exp>/models/<...>.pkl
            base_models_dir = Path(p).parent  # .../outputs/<train_exp>/models
            train_out_dir = base_models_dir.parent  # .../outputs/<train_exp>

            # where training wrote residuals
            train_res_dir = train_out_dir / "residuals"

            # use TRAIN experiment name from bundle meta (most robust)
            train_exp = str(bundle.meta.get("experiment", label))

            res_all_path = train_res_dir / f"{train_exp}__residuals__all.csv"
            res_unique_path = (
                train_res_dir / f"{train_exp}__residuals__all_unique.csv"
            )

            # write analysis artifacts to EVAL output (so evaluation is self-contained)
            base_out = self.plots_dir / "residual_analysis" / label

            res_cfg = dict(get(cfg, "residual_analysis", {}) or {})

            # 1) ALL residuals (diagnostics)
            if res_all_path.exists():
                result_all = analyzer.analyze_from_csv(
                    label=f"{label} (all)",
                    residual_csv_path=res_all_path,
                    out_dir=base_out / "all",
                    tag=f"{self.name}__{label}__all",
                    residual_cfg=res_cfg,
                )
                residual_summary_rows.append(
                    {
                        "bundle": label,
                        "set": "all",
                        "n": result_all.n,
                        "mean": result_all.mean,
                        "std": result_all.std,
                        "skew": result_all.skew,
                        "kurtosis_fisher": result_all.kurtosis_fisher,
                        "shapiro_p": result_all.shapiro_p,
                        "dagostino_p": result_all.dagostino_p,
                        "jarque_bera_p": result_all.jarque_bera_p,
                        "anderson_stat": result_all.anderson_stat,
                        "normality_conclusion": result_all.normality_conclusion,
                        "bias_conclusion": result_all.bias_conclusion,
                        "heteroskedasticity_hint": result_all.heteroskedasticity_hint,
                        "source_csv": str(res_all_path),
                    }
                )
            else:
                print(
                    f"[residuals] missing (ALL) for '{label}': {res_all_path}"
                )

            # 2) UNIQUE residuals (modeling base)
            if res_unique_path.exists():
                result_uni = analyzer.analyze_from_csv(
                    label=f"{label} (unique)",
                    residual_csv_path=res_unique_path,
                    out_dir=base_out / "unique",
                    tag=f"{self.name}__{label}__unique",
                    residual_cfg=res_cfg,
                )
                residual_summary_rows.append(
                    {
                        "bundle": label,
                        "set": "unique",
                        "n": result_uni.n,
                        "mean": result_uni.mean,
                        "std": result_uni.std,
                        "skew": result_uni.skew,
                        "kurtosis_fisher": result_uni.kurtosis_fisher,
                        "shapiro_p": result_uni.shapiro_p,
                        "dagostino_p": result_uni.dagostino_p,
                        "jarque_bera_p": result_uni.jarque_bera_p,
                        "anderson_stat": result_uni.anderson_stat,
                        "normality_conclusion": result_uni.normality_conclusion,
                        "bias_conclusion": result_uni.bias_conclusion,
                        "heteroskedasticity_hint": result_uni.heteroskedasticity_hint,
                        "source_csv": str(res_unique_path),
                    }
                )
            else:
                print(
                    f"[residuals] missing (UNIQUE) for '{label}': {res_unique_path}"
                )

        # write one compact summary table under eval/metrics
        if residual_summary_rows:
            df_sum = pd.DataFrame(residual_summary_rows)

            eval_metrics_dir = self.out_dir / "eval" / "metrics"
            eval_metrics_dir.mkdir(parents=True, exist_ok=True)

            out_csv = (
                eval_metrics_dir
                / f"{self.name}__residual_analysis_summary.csv"
            )
            df_sum.to_csv(out_csv, index=False)
            print(f"[residuals] Saved residual analysis summary: {out_csv}")
        """
        return EvaluateResult(
            per_split_metrics=per_split_df, summary_metrics=summary_df
        )

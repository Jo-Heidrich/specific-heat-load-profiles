# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from pathlib import Path


def summarize_metrics(
    per_split_df: pd.DataFrame, group_key: str = "model"
) -> pd.DataFrame:
    """
    Input: per-split metrics with columns: group_key, split, MAE, RMSE, ...
    Output: one row per group_key with metric_mean and metric_std columns.

    Notes:
    - Only numeric metric columns are aggregated.
    - 'split' can be int/str; it is always excluded from aggregation.
    """
    if per_split_df is None or per_split_df.empty:
        return pd.DataFrame()

    df = per_split_df.copy()

    # metric columns = everything except identifiers
    metric_cols = [c for c in df.columns if c not in (group_key, "split")]

    # numeric only
    metric_cols_num = [
        c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols_num:
        return df[[group_key]].drop_duplicates().reset_index(drop=True)

    g = df.groupby(group_key, dropna=False)
    mean_df = g[metric_cols_num].mean(numeric_only=True).add_suffix("_mean")
    std_df = g[metric_cols_num].std(numeric_only=True).add_suffix("_std")

    out = pd.concat([mean_df, std_df], axis=1).reset_index()

    # optional: sort by a metric if present
    for key in ("WINKLER90_mean", "WINKLER80_mean", "MAPE_mean", "MAE_mean"):
        if key in out.columns:
            out = out.sort_values(key, ascending=True)
            break
    return out


def save_per_split_with_summary(
    per_split_df: pd.DataFrame,
    *,
    out_path: str,
    group_key: str = "model",
) -> None:
    """
    Save per-split rows and append one block of summary rows:
      - one 'mean' row per model
      - one 'std' row per model
    This makes a single CSV that is still readable in Excel.
    """
    if per_split_df is None or per_split_df.empty:
        raise ValueError("per_split_df empty; nothing to save")

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df = per_split_df.copy()

    # build summary
    summary = summarize_metrics(df, group_key=group_key)
    if summary.empty:
        df.to_csv(p, index=False)
        return

    # Convert summary cols back into rows for Excel friendliness
    metric_means = [c for c in summary.columns if c.endswith("_mean")]
    metric_stds = [c for c in summary.columns if c.endswith("_std")]

    rows = []
    for _, r in summary.iterrows():
        model = r[group_key]
        mean_row = {group_key: model, "split": "mean"}
        std_row = {group_key: model, "split": "std"}
        for c in metric_means:
            mean_row[c.replace("_mean", "")] = r[c]
        for c in metric_stds:
            std_row[c.replace("_std", "")] = r[c]
        rows.append(mean_row)
        rows.append(std_row)

    df_out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df_out.to_csv(p, index=False)


def print_cv_summary_block(
    summary_df: pd.DataFrame,
    *,
    group_key: str = "model",
    decimals: int = 3,
    title: str = "Cross validation results (mean ± std over splits)",
) -> None:
    """
    Pretty console output for summarize_metrics() output.

    summary_df columns like:
      model, MAE_mean, MAE_std, RMSE_mean, RMSE_std, ...

    Prints one block per model:
      MAE : 0.123 ± 0.045
      ...
    """
    if summary_df is None or summary_df.empty:
        print("⚠️ No CV summary metrics to print.")
        return

    # preferred order (adjust as you like)
    metric_order = [
        "MAE",
        "RMSE",
        "R2",
        "MAPE",
        "SMAPE",
        "MdAE",
        "Bias",
        "NMAE",
        "NRMSE",
        "WINKLER90",
        "WINKLER80",
    ]

    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        try:
            return f"{float(x):.{decimals}f}"
        except Exception:
            return str(x)

    print(f"\n{title}:")

    for _, r in summary_df.iterrows():
        name = r.get(group_key, "model")
        print(f"\nModel: {name}")
        for m in metric_order:
            mk = f"{m}_mean"
            sk = f"{m}_std"
            if mk in summary_df.columns:
                mean_v = r.get(mk, None)
                std_v = r.get(sk, None) if sk in summary_df.columns else None
                if std_v is None:
                    print(f"  {m:<6}: {fmt(mean_v)}")
                else:
                    print(f"  {m:<6}: {fmt(mean_v)} ± {fmt(std_v)}")

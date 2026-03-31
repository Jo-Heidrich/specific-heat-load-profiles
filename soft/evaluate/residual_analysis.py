# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Any

try:
    from scipy.stats import (
        shapiro,
        normaltest,
        jarque_bera,
        skew,
        kurtosis,
        anderson,
    )
except Exception:  # pragma: no cover
    shapiro = normaltest = jarque_bera = skew = kurtosis = anderson = None

from soft.plotting.plotting_residuals import ResidualPlotter


@dataclass
class ResidualAnalysisResult:
    label: str
    n: int
    mean: float
    std: float
    skew: float | None
    kurtosis_fisher: float | None

    # tests
    shapiro_p: float | None
    dagostino_p: float | None
    jarque_bera_p: float | None
    anderson_stat: float | None
    anderson_critical_values: list[float] | None
    anderson_significance_levels: list[float] | None

    # interpretation
    normality_conclusion: str
    bias_conclusion: str
    heteroskedasticity_hint: str

    # saved artifacts
    artifacts: dict


def _sample_if_large(
    s: pd.Series, max_n: int = 5000, seed: int = 42
) -> pd.Series:
    if len(s) > max_n:
        return s.sample(max_n, random_state=seed)
    return s


class ResidualAnalyzer:
    """
    Encapsulated residual analysis: loads residual CSV, computes statistics & tests,
    prints conclusions, and saves artifacts (csv/json/plots) via ResidualPlotter.
    """

    def __init__(
        self,
        date_col: str = "date",
        residual_col: str = "residual_scaled",
        y_pred_col: str = "y_pred_scaled",
        temperature_col: str = "temperature_day_mean",
        quiet: bool = False,
    ):
        self.date_col = date_col
        self.residual_col = residual_col
        self.y_pred_col = y_pred_col
        self.temperature_col = temperature_col
        self.quiet = quiet
        self.plotter = ResidualPlotter()

    def analyze_from_csv(
        self,
        label: str,
        residual_csv_path: str | Path,
        out_dir: str | Path,
        tag: str,
        max_n_tests: int | None = None,
        alpha: float | None = None,
        residual_cfg: dict[str, Any] | None = None,
    ) -> ResidualAnalysisResult:
        # --- YAML overrides (only if not explicitly provided) ---
        if residual_cfg is not None:
            if alpha is None:
                alpha = float(residual_cfg.get("alpha", 0.05))
            if max_n_tests is None:
                max_n_tests = int(residual_cfg.get("max_n_tests", 5000))

        # --- hard defaults (if neither arg nor YAML provided) ---
        if alpha is None:
            alpha = 0.05
        if max_n_tests is None:
            max_n_tests = 5000

        p = Path(residual_csv_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not p.exists():
            raise FileNotFoundError(f"Residual file not found: {p}")

        df = pd.read_csv(p)
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(
                df[self.date_col], errors="coerce"
            )

        res = pd.Series(df[self.residual_col]).dropna()
        res_test = _sample_if_large(res, max_n=max_n_tests)

        mean = float(res.mean())
        std = float(res.std())

        sk = float(skew(res, bias=False)) if skew is not None else None
        ku = (
            float(kurtosis(res, fisher=True, bias=False))
            if kurtosis is not None
            else None
        )

        sh_p = dg_p = jb_p = ad_stat = None
        ad_cv = ad_sig = None

        if shapiro is not None and len(res_test) >= 3:
            sh_stat, sh_p = shapiro(res_test)
            dg_stat, dg_p = normaltest(res_test)
            jb_stat, jb_p = jarque_bera(res_test)

            ad = anderson(res_test, dist="norm")
            ad_stat = float(ad.statistic)
            ad_cv = list(map(float, ad.critical_values))
            ad_sig = list(map(float, ad.significance_level))

        # --- Interpretations ---
        normality_conclusion = self._interpret_normality(
            sh_p, dg_p, jb_p, alpha=alpha
        )
        bias_conclusion = self._interpret_bias(mean, std)
        hetero_hint = self._interpret_heteroskedasticity_hint(df)

        # --- Plots saved by separate plotter ---
        artifacts = {}
        artifacts["hist"] = self.plotter.save_histogram(
            df=df,
            residual_col=self.residual_col,
            out_path=out_dir / f"{tag}__hist.png",
        )
        artifacts["qq"] = self.plotter.save_qqplot(
            df=df,
            residual_col=self.residual_col,
            out_path=out_dir / f"{tag}__qq.png",
        )
        artifacts["resid_vs_pred"] = self.plotter.save_residual_vs_pred(
            df=df,
            residual_col=self.residual_col,
            y_pred_col=self.y_pred_col,
            out_path=out_dir / f"{tag}__resid_vs_pred.png",
        )
        artifacts["rolling_std"] = self.plotter.save_rolling_std(
            df=df,
            date_col=self.date_col,
            residual_col=self.residual_col,
            out_path=out_dir / f"{tag}__rolling_std.png",
            window=200,
        )

        # --- group table (important for heteroskedasticity) ---
        grp = self._group_std_table(df)
        out_grp = out_dir / f"{tag}__group_std.csv"
        grp.to_csv(out_grp, index=False)
        artifacts["group_std_csv"] = str(out_grp)

        # --- Save JSON summary ---
        result = ResidualAnalysisResult(
            label=label,
            n=int(res.shape[0]),
            mean=mean,
            std=std,
            skew=sk,
            kurtosis_fisher=ku,
            shapiro_p=float(sh_p) if sh_p is not None else None,
            dagostino_p=float(dg_p) if dg_p is not None else None,
            jarque_bera_p=float(jb_p) if jb_p is not None else None,
            anderson_stat=ad_stat,
            anderson_critical_values=ad_cv,
            anderson_significance_levels=ad_sig,
            normality_conclusion=normality_conclusion,
            bias_conclusion=bias_conclusion,
            heteroskedasticity_hint=hetero_hint,
            artifacts=artifacts,
        )

        out_json = out_dir / f"{tag}__summary.json"
        out_json.write_text(
            json.dumps(result.__dict__, indent=2), encoding="utf-8"
        )
        artifacts["summary_json"] = str(out_json)

        # --- Console prints (human-readable) ---
        if not self.quiet:
            self._print_report(result, alpha=alpha)

        return result

    # ------------------------- interpretations -------------------------

    @staticmethod
    def _interpret_normality(sh_p, dg_p, jb_p, alpha: float) -> str:
        if sh_p is None and dg_p is None and jb_p is None:
            return "Normality tests unavailable (SciPy missing)."

        # Reject normality if ANY test rejects at alpha (conservative)
        rejects = []
        if sh_p is not None and sh_p < alpha:
            rejects.append("Shapiro")
        if dg_p is not None and dg_p < alpha:
            rejects.append("D’Agostino K²")
        if jb_p is not None and jb_p < alpha:
            rejects.append("Jarque–Bera")

        if rejects:
            return (
                f"Residuals deviate from Normality at α={alpha:g} "
                f"(rejected by: {', '.join(rejects)})."
            )
        return f"No evidence against Normality at α={alpha:g} (all available tests non-significant)."

    @staticmethod
    def _interpret_bias(mean: float, std: float) -> str:
        # simple rule-of-thumb: |mean| > 0.1*std indicates noticeable bias
        if std <= 0:
            return "Residual std is zero; bias interpretation not meaningful."
        if abs(mean) > 0.1 * std:
            return f"Non-zero mean residual suggests bias (mean={mean:.4g} ≈ {mean/std:.3g}·std)."
        return "Mean residual close to zero (no strong bias indication)."

    def _interpret_heteroskedasticity_hint(self, df: pd.DataFrame) -> str:
        # Rule-of-thumb: residual scale changes strongly across prediction quantiles
        if (
            self.y_pred_col not in df.columns
            or self.residual_col not in df.columns
        ):
            return "No heteroskedasticity check (missing columns)."

        d = df[[self.y_pred_col, self.residual_col]].dropna()
        if len(d) < 50:
            return "Insufficient data for heteroskedasticity hint."

        q = np.quantile(d[self.y_pred_col], [0.1, 0.9])
        low = d[d[self.y_pred_col] <= q[0]][self.residual_col].std()
        high = d[d[self.y_pred_col] >= q[1]][self.residual_col].std()

        if low <= 0 or high <= 0:
            return "Heteroskedasticity hint unavailable (degenerate std)."

        ratio = float(high / low)
        if ratio >= 1.5:
            return f"Strong heteroskedasticity indication: std(res) increases with ŷ (std_hi/std_lo ≈ {ratio:.2f})."
        if ratio <= 0.67:
            return f"Scale change detected: std(res) decreases with ŷ (std_hi/std_lo ≈ {ratio:.2f})."
        return f"Only mild scale change across prediction levels (std_hi/std_lo ≈ {ratio:.2f})."

    def _group_std_table(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if self.date_col in d.columns:
            dt = pd.to_datetime(d[self.date_col], errors="coerce")
            d["month"] = dt.dt.month
            d["weekday"] = dt.dt.weekday
            d["hour"] = dt.dt.hour

        rows = []

        def add_group(name: str, key: str):
            g = d.groupby(key, observed=False)[self.residual_col]
            tmp = g.agg(["count", "mean", "std", "median"]).reset_index()
            tmp.insert(0, "group", name)
            tmp = tmp.rename(columns={key: "key"})
            rows.append(tmp)

        if "month" in d.columns:
            add_group("month", "month")
        if "weekday" in d.columns:
            add_group("weekday", "weekday")
        if "hour" in d.columns:
            add_group("hour", "hour")

        if self.temperature_col in d.columns:
            t = pd.to_numeric(d[self.temperature_col], errors="coerce")
            bins = [-50, -10, -5, 0, 5, 10, 15, 20, 50]
            labels = [f"{bins[i]}..{bins[i+1]}" for i in range(len(bins) - 1)]
            d["temp_bin"] = pd.cut(
                t, bins=bins, labels=labels, include_lowest=True
            )
            add_group("temperature_bin", "temp_bin")

        if rows:
            out = pd.concat(rows, ignore_index=True)
            out["key"] = out["key"].astype(str)
            return out[["group", "key", "count", "mean", "std", "median"]]

        return pd.DataFrame(
            columns=["group", "key", "count", "mean", "std", "median"]
        )

    def _print_report(self, r: ResidualAnalysisResult, alpha: float) -> None:
        print("\n" + "=" * 72)
        print(f"Residual analysis: {r.label}")
        print("=" * 72)
        print(f"n = {r.n}")
        print(f"mean = {r.mean:.6g}, std = {r.std:.6g}")

        if r.skew is not None:
            print(
                f"skew = {r.skew:.4f}, kurtosis(Fisher) = {r.kurtosis_fisher:.4f}"
            )

        print("\nStatistical tests (H0: Gaussian residuals):")
        if r.shapiro_p is None:
            print("  (tests unavailable)")
        else:
            print(f"  Shapiro–Wilk     p = {r.shapiro_p:.3g}")
            print(f"  D’Agostino K²    p = {r.dagostino_p:.3g}")
            print(f"  Jarque–Bera     p = {r.jarque_bera_p:.3g}")

        # --- Unified interpretation & conclusion ---
        print("\nInterpretation and conclusions:")

        print(
            f"  • Normality: {r.normality_conclusion} " f"(α = {alpha:.2g})."
        )
        print(
            f"  • Bias: {r.bias_conclusion} "
            f"(mean residual = {r.mean:.4g})."
        )
        print(f"  • Heteroskedasticity: {r.heteroskedasticity_hint}.")

        print("=" * 72 + "\n")

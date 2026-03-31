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

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass
class SuitabilityResult:
    label: str
    n: int

    # Stationarity
    adf_stat: float | None
    adf_p: float | None
    adf_crit: dict | None

    # Autocorrelation
    ljungbox_lags: list[int]
    ljungbox_pvalues: list[float]

    # Positivity diagnostics (CIR-on-level warning)
    min_residual: float
    share_negative: float
    suggested_shift: float

    # conclusions
    ou_suitability: str
    cir_suitability: str

    artifacts: dict


class ResidualSuitabilityAnalyzer:
    """
    OU/CIR suitability pretests for residuals:
      - ADF (stationarity)
      - ACF/PACF plots
      - Ljung-Box (serial correlation)
      - positivity diagnostics for CIR (if modeling level directly)
    """

    def __init__(
        self, date_col="date", residual_col="residual", quiet: bool = False
    ):
        self.date_col = date_col
        self.residual_col = residual_col
        self.quiet = quiet

    def analyze_from_csv(
        self,
        label: str,
        residual_csv_path: str | Path,
        out_dir: str | Path,
        tag: str,
        alpha: float = 0.05,
        acf_lags: int = 72,
        pacf_lags: int | None = None,
        ljungbox_lags: list[int] | None = None,
    ) -> SuitabilityResult:
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

        res = pd.Series(df[self.residual_col]).dropna().astype(float)
        n = int(res.shape[0])

        # --- ADF ---
        adf_stat = adf_p = None
        adf_crit = None
        try:
            adf = adfuller(res.values, autolag="AIC")
            adf_stat = float(adf[0])
            adf_p = float(adf[1])
            adf_crit = {k: float(v) for k, v in adf[4].items()}
        except Exception:
            pass

        # --- Ljung-Box ---
        if ljungbox_lags is None:
            ljungbox_lags = [12, 24, 48, 72]
        lb = acorr_ljungbox(res.values, lags=ljungbox_lags, return_df=True)
        lb_p = [float(x) for x in lb["lb_pvalue"].values]

        # --- ACF/PACF plots (like residuen_lightgbm.py) ---
        if pacf_lags is None:
            pacf_lags = acf_lags

        artifacts = {}
        artifacts["acf_pacf"] = self._save_acf_pacf_plot(
            res.values,
            out_path=out_dir / f"{tag}__acf_pacf.png",
            acf_lags=acf_lags,
            pacf_lags=pacf_lags,
        )

        # --- CIR positivity diagnostics (if residuals used as CIR level) ---
        min_r = float(np.min(res.values))
        share_neg = float(np.mean(res.values < 0.0))
        suggested_shift = float(max(0.0, -min_r + 1e-3))

        ou_suitability = self._conclude_ou(adf_p, lb_p, alpha=alpha)
        cir_suitability = self._conclude_cir(share_neg, suggested_shift)

        result = SuitabilityResult(
            label=label,
            n=n,
            adf_stat=adf_stat,
            adf_p=adf_p,
            adf_crit=adf_crit,
            ljungbox_lags=list(map(int, ljungbox_lags)),
            ljungbox_pvalues=lb_p,
            min_residual=min_r,
            share_negative=share_neg,
            suggested_shift=suggested_shift,
            ou_suitability=ou_suitability,
            cir_suitability=cir_suitability,
            artifacts=artifacts,
        )

        out_json = out_dir / f"{tag}__suitability_summary.json"
        out_json.write_text(
            json.dumps(result.__dict__, indent=2), encoding="utf-8"
        )
        artifacts["suitability_json"] = str(out_json)

        if not self.quiet:
            self._print_report(result, alpha=alpha)

        return result

    @staticmethod
    def _save_acf_pacf_plot(
        x: np.ndarray, out_path: Path, acf_lags: int, pacf_lags: int
    ) -> str:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        sm.graphics.tsa.plot_acf(x, lags=acf_lags, ax=ax[0])
        sm.graphics.tsa.plot_pacf(x, lags=pacf_lags, ax=ax[1], method="ywm")
        ax[0].set_title("ACF der Residuen")
        ax[1].set_title("PACF der Residuen")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    @staticmethod
    def _conclude_ou(
        adf_p: float | None, lb_pvalues: list[float], alpha: float
    ) -> str:
        if adf_p is None:
            return "OU: Stationarity untested (ADF unavailable)."

        stationary = adf_p < alpha
        has_autocorr = any(p < alpha for p in lb_pvalues)

        if stationary and has_autocorr:
            return f"OU: Stationarity supported (ADF p<{alpha}); autocorrelation present (Ljung–Box rejects at some lags)."
        if stationary and not has_autocorr:
            return "OU: Stationary but little serial correlation (weak OU signal)."
        return f"OU: Stationarity not supported (ADF p≥{alpha}); OU in level likely problematic."

    @staticmethod
    def _conclude_cir(share_negative: float, suggested_shift: float) -> str:
        if share_negative == 0.0:
            return "CIR: Residuals are non-negative; CIR on level would be feasible."
        return (
            f"CIR: Residuals contain negatives (share≈{share_negative:.2%}); "
            f"direct CIR on residual level violates X_t≥0. "
            f"Consider modeling a positive state (e.g. residual^2 / rolling var) "
            f"or shift residuals by c≈{suggested_shift:.6g} (changes interpretation)."
        )

    def _print_report(self, r: SuitabilityResult, alpha: float) -> None:
        print("\n--- Residual suitability pretests ---")
        print(f"label = {r.label}")
        print(f"n     = {r.n}")

        print("\nADF (H0: unit root / non-stationary):")
        if r.adf_p is None:
            print("  ADF unavailable.")
        else:
            print(f"  stat={r.adf_stat:.3f}, p={r.adf_p:.3g} (alpha={alpha})")
            if r.adf_crit:
                for k, v in r.adf_crit.items():
                    print(f"  crit({k})={v:.3f}")

        print("\nLjung–Box (H0: no autocorrelation):")
        for lag, p in zip(r.ljungbox_lags, r.ljungbox_pvalues):
            print(f"  lag {lag:>3}: p={p:.3g}")

        print("\nCIR positivity diagnostics (if modeling residual level):")
        print(f"  min(res)={r.min_residual:.6g}")
        print(f"  share(res<0)={r.share_negative:.2%}")
        print(f"  suggested shift c≈{r.suggested_shift:.6g}")

        print("\nConclusions:")
        print(f"  • {r.ou_suitability}")
        print(f"  • {r.cir_suitability}")

# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys

THIS_DIR = Path(__file__).resolve().parent
for p in THIS_DIR.parents:
    if (p / "style" / "__init__.py").exists() or (p / "style").is_dir():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError(
        "Konnte das 'style' Package nicht finden (kein 'style/' in Parents)."
    )


import style
from style import fhg_style as fhg

STYLE_FILE = Path(style.__file__).resolve().parent / "fraunhofer.mplstyle"

import matplotlib.pyplot as plt

plt.style.use(str(STYLE_FILE))
fhg.set_font_scale(1.0)

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None


class ResidualPlotter:
    def __init__(self):
        self.colors = {
            "green": "#179c7d",
            "orange": "#f58220",
            "graphit": "#1c3f52",
        }

    def save_histogram(
        self,
        df: pd.DataFrame,
        residual_col: str,
        out_path: str | Path,
        bins: int = 50,
    ) -> str:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path

        res = pd.Series(df[residual_col]).dropna().to_numpy()
        mu = float(np.mean(res)) if res.size else 0.0
        sig = float(np.std(res, ddof=1)) if res.size > 1 else 0.0

        fig = plt.figure(figsize=(8, 5))

        # histogram (counts)
        counts, bin_edges, _ = plt.hist(
            res,
            bins=bins,
            edgecolor="black",
            alpha=0.7,
            color=self.colors["green"],
            label="Residuals",
        )

        # Normal overlay scaled to counts
        if sig > 0 and len(bin_edges) >= 2:
            bin_w = float(bin_edges[1] - bin_edges[0])
            x = np.linspace(bin_edges[0], bin_edges[-1], 400)

            # Normal PDF
            pdf = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(
                -0.5 * ((x - mu) / sig) ** 2
            )

            # scale pdf to histogram counts
            y = pdf * (res.size * bin_w)

            plt.plot(
                x,
                y,
                linewidth=2.0,
                label=f"Normal(mu={mu:.4g}, sigma={sig:.4g})",
            )

        plt.title("Histogram of residuals", color=self.colors["graphit"])
        plt.xlabel(
            "Residual (true - prediction)", color=self.colors["graphit"]
        )
        plt.ylabel("Frequency", color=self.colors["graphit"])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best")
        plt.tight_layout()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    def save_qqplot(
        self, df: pd.DataFrame, residual_col: str, out_path: str | Path
    ) -> str | None:
        if sm is None:
            return None
        res = pd.Series(df[residual_col]).dropna()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        sm.qqplot(res, line="s", ax=ax)
        ax.set_title("Q–Q plot vs. Normal")
        plt.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    def save_residual_vs_pred(
        self,
        df: pd.DataFrame,
        residual_col: str,
        y_pred_col: str,
        out_path: str | Path,
        alpha: float = 0.35,
    ) -> str:
        d = df[[residual_col, y_pred_col]].dropna()
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(
            d[y_pred_col].to_numpy(),
            d[residual_col].to_numpy(),
            s=6,
            alpha=alpha,
            color=self.colors["orange"],
        )
        plt.axhline(0.0, color=self.colors["graphit"], lw=1)
        plt.title("Residuals vs prediction", color=self.colors["graphit"])
        plt.xlabel("Prediction ŷ", color=self.colors["graphit"])
        plt.ylabel("Residual", color=self.colors["graphit"])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    def save_rolling_std(
        self,
        df: pd.DataFrame,
        date_col: str,
        residual_col: str,
        out_path: Path,
        window: int = 200,
        max_gap_hours: int = 2,
    ):
        d = df.copy()
        if date_col not in d.columns:
            return None

        d = d.dropna(subset=[date_col, residual_col]).sort_values(date_col)
        d[date_col] = pd.to_datetime(d[date_col])

        # --- segment breaks if time gap too large ---
        dt = d[date_col].diff().dt.total_seconds().div(3600.0)
        d["__seg"] = (dt > max_gap_hours).cumsum()

        # --- rolling std per segment ---
        d["rolling_std"] = (
            d.groupby("__seg")[residual_col]
            .rolling(window=window, min_periods=max(30, window // 5))
            .std()
            .reset_index(level=0, drop=True)
        )

        fig = plt.figure(figsize=(9, 4))
        plt.plot(d[date_col], d["rolling_std"])
        plt.title("Rolling standard deviation of residuals")
        plt.xlabel("Date")
        plt.ylabel("STD")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return str(out_path)

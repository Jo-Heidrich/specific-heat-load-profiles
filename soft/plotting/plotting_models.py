# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 23:33:33 2026

@author: heidrich
"""
from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from soft.evaluate.adapters import (
    NonlinAdapter,
    LightGBMAdapter,
    CompareConfig,
)
from soft.plotting.plotting_results import ResultPlotter

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


class ProfileComparison:
    """Build and plot profile comparisons in the plotting_results style."""

    def __init__(
        self,
        cfg: CompareConfig,
        gbm: LightGBMAdapter,
        nonlin: NonlinAdapter,
        plotter: Optional[ResultPlotter] = None,
    ):

        self.cfg = cfg
        self.gbm = gbm
        self.nonlin = nonlin
        self.plotter = plotter or ResultPlotter()

    def build_preds(self, df_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "Gradient boosting": self.gbm.predict_aligned(df_test),
            "Non-linear regression": self.nonlin.predict_aligned(df_test),
        }

    def plot_profile(
        self,
        df_test: pd.DataFrame,
        start: str | None = None,
        hours: int = 240,
        ylim_temp: tuple[float, float] | None = None,
        save_path: str | None = None,
        show: bool = True,
        extra_preds: dict[str, np.ndarray] | None = None,
    ) -> None:

        # Base predictions from internal adapters (GBM, non-linear regression)+
        preds = self.build_preds(df_test)

        # Optional: add externally computed model predictions
        # These must already be aligned to df_test rows
        if extra_preds:
            preds.update(extra_preds)

        # target_col can be None (prediction-only) after the small patch in plot_profile_compare_multi
        self.plotter.plot_profile_compare_multi(
            df_test=df_test,
            date_col=self.cfg.date_col,
            target_col=self.cfg.target_col,
            preds=preds,
            temp_col=(
                self.cfg.temp_col
                if self.cfg.temp_col in df_test.columns
                else None
            ),
            start=start,
            hours=hours,
            ylim_temp=ylim_temp,
            title=(
                "Consumption: Real vs Model Predictions"
                if self.cfg.target_col
                else "Consumption: Model Predictions"
            ),
            save_path=save_path,
            show=show,
        )

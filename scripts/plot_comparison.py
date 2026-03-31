# -*- coding: utf-8 -*-
"""
@author: heidrich

Example script: profile-style plot with multiple model predictions.

This script demonstrates how to use ProfileComparison as a high-level
orchestrator for plotting:
- real consumption (optional)
- GBM prediction
- non-linear regression
- additional externally computed model predictions (e.g. linear regression, SLP)

All predictions must be aligned to df_test rows.
"""

"""
Prediction-only profile plot (no real consumption).

This script generates a profile-style plot for future predictions:
- Gradient Boosting
- Non-linear regression
- additional external model predictions

No ground truth / target column is required.
"""

import numpy as np
import pandas as pd

from feature_builders import HourlyFeatureBuilder
from plotting_results import ResultPlotter
from profile_compare import (
    ProfileComparison,
    CompareConfig,
    LightGBMAdapter,
)
from nonlin_adapter import NonlinAdapter, NonlinConfig


# ------------------------------------------------------------
# 1) Prepare future dataframe (NO real consumption)
# ------------------------------------------------------------
# df_future must define the time grid and exogenous inputs
# Example columns:
# - date (hourly timestamps)
# - temperature_hourly
# - optional: precomputed GBM feature columns

df_future = pd.read_parquet("data/forecast_input.parquet")


# ------------------------------------------------------------
# 2) Plotting backend
# ------------------------------------------------------------
plotter = ResultPlotter()


# ------------------------------------------------------------
# 3) Model adapters
# ------------------------------------------------------------
fb = HourlyFeatureBuilder()

gbm = LightGBMAdapter(
    model_path="models/lgbm_fulltrain.pkl",
    fb=fb,
)

nonlin = NonlinAdapter(
    NonlinConfig(
        date_col="date",
        temp_hourly_col="temperature_hourly",
        params=[
            # calibrated non-linear regression parameters
            ...
        ],
    )
)


# ------------------------------------------------------------
# 4) Profile comparison orchestrator
# ------------------------------------------------------------
pc = ProfileComparison(
    cfg=CompareConfig(
        date_col="date",
        target_col=None,  # <-- prediction-only mode
        temp_col="temperature_hourly",
    ),
    gbm=gbm,
    nonlin=nonlin,
    plotter=plotter,
)


# ------------------------------------------------------------
# 5) Optional: additional model predictions
# ------------------------------------------------------------
# These must be aligned to df_future rows
yhat_linear = np.asarray(yhat_linear_forecast, dtype=float)
yhat_slp = np.asarray(yhat_slp_forecast, dtype=float)

extra_preds = {
    "Linear regression": yhat_linear,
    "SLP": yhat_slp,
}


# ------------------------------------------------------------
# 6) Create prediction-only profile plot
# ------------------------------------------------------------
pc.plot_profile(
    df_test=df_future,  # still a DataFrame, but WITHOUT target column
    start=None,
    hours=24 * 14,
    ylim_temp=None,
    save_path="out/profile_forecast.png",
    show=False,
    extra_preds=extra_preds,
)

# -*- coding: utf-8 -*-
"""
@author: heidrich

General-purpose regression model evaluation utilities.

Includes standard error metrics and optional plotting for
model vs. ground truth (and optional baseline).
"""

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

import numpy as np
import matplotlib.pyplot as plt


def safe_mape(y_true, y_pred):
    """
    Robust Mean Absolute Percentage Error (MAPE).

    Ignores time steps where y_true == 0 to avoid division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )


def mase(y_true, y_pred, y_naive):
    """
    Mean Absolute Scaled Error (MASE).

    Scales MAE of the model by the MAE of a naive baseline forecast.
    """
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))
    return mae_model / mae_naive if mae_naive != 0 else np.nan


def evaluate_model_general(
    y_true, y_pred, y_baseline=None, plot=False, verbose=True
):
    """
    Evaluate regression predictions using a standard set of metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Model predictions.
    y_baseline : array-like, optional
        Baseline predictions (for MASE and optional plotting).
    plot : bool, default False
        If True, plot prediction vs. ground truth (and baseline).

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    if verbose:
        print("Prediction min / max:", y_pred.min(), y_pred.max())
        print("Ground truth min / max:", y_true.min(), y_true.max())

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if y_baseline is not None:
        y_baseline = np.asarray(y_baseline, dtype=float)
        mask = mask & np.isfinite(y_baseline)

    if not mask.all():
        dropped = (~mask).sum()
        if verbose:
            print(
                f"[evaluate_model_general] Dropping {dropped}/{len(mask)} rows due to NaN/Inf in inputs."
            )
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_baseline is not None:
            y_baseline = y_baseline[mask]

    if len(y_true) == 0:
        raise ValueError(
            "No valid (finite) samples left after NaN/Inf filtering."
        )

    # Standard metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    # Normalized metrics
    nmae = mae / np.mean(np.abs(y_true))
    nrmse = rmse / np.mean(np.abs(y_true))

    # Scaled metric (optional)
    mase_val = (
        mase(y_true, y_pred, y_baseline) if y_baseline is not None else None
    )
    if verbose:
        print("\nModel evaluation results:")
        print(f"  MAE         : {mae:.3f}")
        print(f"  RMSE        : {rmse:.3f}")
        print(f"  R²          : {r2:.3f}")
        print(f"  MAPE        : {mape:.2f}%")
        print(f"  sMAPE       : {smape_val:.2f}%")
        print(f"  Median AE   : {mdae:.3f}")
        print(f"  Bias        : {bias:.3f}")
        print(f"  NMAE        : {nmae:.3f}")
        print(f"  NRMSE       : {nrmse:.3f}")
        if mase_val is not None:
            print(f"  MASE        : {mase_val:.3f} (vs. naive baseline)")

    # Optional visualization
    if plot:
        plt.figure(figsize=(14, 5))
        plt.plot(y_true, label="Ground truth", color="blue")
        plt.plot(y_pred, label="Model prediction", color="orange")

        if y_baseline is not None:
            plt.plot(
                y_baseline,
                label="Baseline",
                color="green",
                linestyle="--",
            )

        plt.xlabel("Time / index")
        plt.ylabel("Target value")
        plt.title("Model vs. baseline vs. ground truth")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2,
        "SMAPE": smape_val,
        "MdAE": mdae,
        "Bias": bias,
        "NMAE": nmae,
        "NRMSE": nrmse,
        "MASE": mase_val,
    }


# Example usage
if __name__ == "__main__":
    # Example test data
    y_true = np.array([100, 120, 130, 110, 115])
    y_pred = np.array([98, 123, 128, 105, 117])
    y_baseline = np.array([102, 119, 125, 108, 116])

    # Run evaluation
    metrics = evaluate_model_general(
        y_true, y_pred, y_baseline=y_baseline, plot=True
    )
    print("\nMetrics dictionary:", metrics)

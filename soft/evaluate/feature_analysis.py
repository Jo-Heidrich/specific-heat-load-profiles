# -*- coding: utf-8 -*-
"""
@author: heidrich

Feature analysis utilities (sanity checks, correlation filtering, SHAP, PCA).

This module is optional: you can run it during training for diagnostics,
or keep it as a separate analysis tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureIssue:
    """Container for a detected feature quality issue."""

    feature: str
    issues: List[str]


class FeatureAnalyzer:
    """
    Provides diagnostics on feature matrices:
    - constant / nearly-constant features
    - majority-value dominance
    - NaN ratio checks
    - correlation filtering
    """

    def __init__(
        self,
        threshold_unique: int = 1,
        threshold_majority: float = 0.99,
        nan_threshold: float = 0.5,
    ):
        self.threshold_unique = threshold_unique
        self.threshold_majority = threshold_majority
        self.nan_threshold = nan_threshold

    def analyze(self, X: pd.DataFrame) -> List[FeatureIssue]:
        """
        Scan features and return a list of issues.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        List[FeatureIssue]
            List of detected issues per feature.
        """
        issues: List[FeatureIssue] = []
        for col in X.columns:
            s = X[col]
            nunique = s.nunique(dropna=True)
            majority = s.value_counts(normalize=True, dropna=False).max()
            nan_ratio = float(s.isna().mean())

            probs: List[str] = []
            if nunique <= self.threshold_unique:
                probs.append("Constant or nearly-constant values")
            if majority >= self.threshold_majority:
                probs.append(f"Dominant majority value: {majority:.2%}")
            if nan_ratio > self.nan_threshold:
                probs.append(f"High NaN ratio: {nan_ratio:.2%}")

            if probs:
                issues.append(FeatureIssue(feature=col, issues=probs))
        return issues

    def correlation_filter(
        self,
        X: pd.DataFrame,
        corr_threshold: float = 0.95,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove one of each pair of highly correlated features (absolute Pearson corr).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (numeric).
        corr_threshold : float
            Absolute correlation threshold.

        Returns
        -------
        (X_reduced, dropped_features)
        """
        corr = X.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
        return X.drop(columns=drop_cols), drop_cols


def plot_corr_heatmap(X: pd.DataFrame, title: str = "Correlation heatmap"):
    """
    Plot a correlation heatmap (matplotlib only).

    Note: kept lightweight (no seaborn dependency).
    """
    import matplotlib.pyplot as plt

    corr = X.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def run_shap_summary(model, X: pd.DataFrame, max_display: int = 15):
    """
    Compute and plot SHAP summary plots.

    Parameters
    ----------
    model : Any
        Trained model compatible with shap.Explainer.
    X : pd.DataFrame
        Feature matrix used for explanation.
    max_display : int
        Max features shown.
    """
    import shap

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display)
    shap.summary_plot(shap_values, X, max_display=max_display)
    return shap_values


def apply_pca(X: pd.DataFrame, n_components: float = 0.95):
    """
    Apply PCA after standard scaling.

    Returns
    -------
    (X_pca, pca, scaler)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, scaler

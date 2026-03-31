# -*- coding: utf-8 -*-
"""
@author: heidrich

Model bundle persistence.

A ModelBundle contains:
- model: trained model object (must be picklable)
- features: list of feature column names (order matters)
- target: target column name
- meta: free-form metadata dict (yaml config path, split info, target_scale, etc.)

Includes backward compatible loading:
- legacy pickle that stored only the model
- dict with keys {"model","features","target"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import joblib


@dataclass
class ModelBundle:
    """Serializable bundle containing the trained model and metadata."""

    model: Any
    features: List[str]
    target: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Persist this bundle to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ModelBundle":
        """Load a bundle from disk (see load_bundle for backward compatibility)."""
        obj = joblib.load(path)
        if isinstance(obj, ModelBundle):
            return obj
        # Fallback to global loader logic
        return load_bundle(path)


def save_bundle(bundle: ModelBundle, path: str) -> None:
    """Persist a ModelBundle to disk."""
    bundle.save(path)


def load_bundle(path: str) -> ModelBundle:
    """
    Backward compatible loader.

    Supported formats:
    1) ModelBundle object
    2) dict with {"model","features","target"} (meta optional)
    3) raw model object only -> wrapped with empty features/target
    """
    obj = joblib.load(path)

    if isinstance(obj, ModelBundle):
        return obj

    if isinstance(obj, dict):
        keys = set(obj.keys())
        if {"model", "features", "target"} <= keys:
            return ModelBundle(
                model=obj["model"],
                features=list(obj["features"]),
                target=str(obj["target"]),
                meta=dict(obj.get("meta", {})),
            )

    # Legacy: only model persisted
    return ModelBundle(
        model=obj,
        features=[],
        target="",
        meta={"warning": "legacy model file without features/target"},
    )

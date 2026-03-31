# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    pass


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("YAML config must be a mapping (top-level dict).")
    return cfg


def require(cfg: Dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"Missing required config key: '{dotted}'")
        cur = cur[part]
    return cur


def get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# evaluate.py
# -*- coding: utf-8 -*-
"""
@author: heidrich

Evaluate one or more saved ModelBundles on a dataset.

Supports:
- DualEdge test split (classic: test at beginning + end)
- Seasonal block-wise CV with two seasonal test windows per split
  -> metrics are averaged over splits for each model
  -> optional baselines (rolling24h / linreg / both)

Bundles contain NO baseline info by design.

YAML-only evaluator.

Run:
  python -m scripts.evaluate --config configs/evaluate/evaluate_compare.yaml
"""

from __future__ import annotations

import argparse
from soft.config_io import load_yaml
from soft.experiment import ExperimentRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    runner = ExperimentRunner(cfg, config_path=args.config)
    runner.run_evaluate()


if __name__ == "__main__":
    main()

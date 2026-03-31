# -*- coding: utf-8 -*-
"""
@author: heidrich

Train a model bundle on a given dataset using the TrainingPipeline, YAML-only training entrypoint.

Run:
  python -m scripts.train --config configs/training/train_Aalborg_HEF_seasonal_legacy_hourly.yaml
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
    runner.run_train()


if __name__ == "__main__":
    main()

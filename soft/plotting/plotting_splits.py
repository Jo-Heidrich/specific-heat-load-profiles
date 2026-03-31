# -*- coding: utf-8 -*-
"""
@author: heidrich
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
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


def plot_seasonal_splits(
    df: pd.DataFrame,
    splits,
    date_col: str,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot seasonal CV splits similar to soft/splits_seasonal.py __main__:
    - each split is a row
    - test indices are marked as 1.0, others 0.0
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if n == 0 or not splits:
        print("⚠️ No splits or empty dataset -> skip split plot.")
        return

    # Fraunhofer color palette
    colors = ["#005b7f", "#39c1cd", "#179c7d", "#b2d235", "#008598", "#7c154d"]

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    for i, (_train_idx, test_idx) in enumerate(splits):
        row = np.zeros(n, dtype=float)
        row[np.array(test_idx, dtype=int)] = 1.0
        ax.plot(
            df[date_col],
            row + i * 1.2,
            label=f"Split {i+1}",
            color=colors[i % len(colors)],
            linewidth=1.8,
        )

    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    ax.legend(loc="upper left", frameon=True)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # always write PNG (or the given suffix) + PDF
        base = p.with_suffix("")  # remove suffix
        fig.savefig(base.with_suffix(".png"), dpi=150)
        fig.savefig(base.with_suffix(".pdf"))

    if show:
        plt.show()

    plt.close(fig)


def plot_splits_with_validation(
    df: pd.DataFrame,
    splits: List[Tuple[List[int], List[int]]],
    val_blocks_by_split: Optional[List[List[List[int]]]],
    date_col: str,
    title: str = "",  # keep param, but default empty -> no title
    save_path: Optional[str] = None,
    show: bool = False,
    val_level: float = 0.6,
) -> None:
    """
    Seasonal-style plot:
      - one row per split
      - test indices plotted at 1.0
      - validation indices plotted at val_level (default 0.6)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if n == 0 or not splits:
        print("⚠️ No splits or empty dataset -> skip split plot.")
        return

    # Fraunhofer color palette (same as plot_seasonal_splits)
    colors = ["#005b7f", "#39c1cd", "#179c7d", "#b2d235", "#008598", "#7c154d"]

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    for i, (_train_idx, test_idx) in enumerate(splits):
        row = np.zeros(n, dtype=float)

        # validation (optional)
        if val_blocks_by_split is not None and i < len(val_blocks_by_split):
            for blk in val_blocks_by_split[i]:
                if blk:
                    row[np.array(blk, dtype=int)] = val_level

        # test on top
        if test_idx is not None and len(test_idx) > 0:
            row[np.array(test_idx, dtype=int)] = 1.0

        ax.plot(
            df[date_col],
            row + i * 1.2,
            label=f"Split {i+1}",
            color=colors[i % len(colors)],
            linewidth=1.8,
        )

    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    # no title by default (to match your request)
    if title:
        ax.set_title(title)

    ax.legend(
        loc="upper left",
        frameon=True,
    )

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # always write PNG (or the given suffix) + PDF
        base = p.with_suffix("")  # remove suffix
        fig.savefig(base.with_suffix(".png"), dpi=150)
        fig.savefig(base.with_suffix(".pdf"))

    if show:
        plt.show()

    plt.close(fig)

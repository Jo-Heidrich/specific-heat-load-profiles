# -*- coding: utf-8 -*-
"""
@author: heidrich

Plotting utilities for evaluation and prediction windows.
"""

from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from soft.data.feature_builders import HourlyFeatureBuilder

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

COLOR_TEMP = "#bb0056"
COLOR_TRUE = "#005b7f"
COLOR_BASELINE = "#179c7d"  # Fraunhofer green (matches your split palette)

# Fraunhofer-ish accent colors for user-specified models (order matters)
ACCENT_MODEL_COLORS = [
    "#f58220",  # orange
    "#39c1cd",  # aqua
    "#b2d235",  # lime
    "#fdb913",  # yellow
    "#d3c7ae",  # sand
    "#a6bbc8",  # silvergrey
]


def build_color_map(model_names_in_order: list[str]) -> dict[str, str]:
    def is_baseline(name: str) -> bool:
        s = str(name).lower()
        return (
            s.startswith("baseline_")
            or s
            in {
                "linreg",
                "linearregression",
                "linear_regression",
                "rolling24h",
                "rolling_24h",
            }
            or "baseline" in s
        )

    cmap: dict[str, str] = {}
    acc_i = 0
    for name in model_names_in_order:
        if is_baseline(name):
            cmap[name] = COLOR_BASELINE
        else:
            cmap[name] = ACCENT_MODEL_COLORS[acc_i % len(ACCENT_MODEL_COLORS)]
            acc_i += 1
    return cmap


class ResultPlotter:
    """
    Produces time-window plots for true series and one or multiple prediction series.
    """

    def _pick_window_starts(
        self, df: pd.DataFrame, n_windows: int
    ) -> list[int]:
        """Pick start indices roughly uniformly across the dataframe length."""
        n = len(df)
        if n == 0:
            return []
        step = max(1, n // n_windows)
        return [min(i * step, n - 1) for i in range(n_windows)]

    def plot_windows(
        self,
        df_test: pd.DataFrame,
        y_true,
        preds: Dict[str, np.ndarray],
        date_col: str = "date",
        window_days: int = 7,
        n_windows: int = 3,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> list[str]:
        """
        Plot multiple prediction series against the true series in several time windows.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test dataframe containing a datetime column.
        y_true : array-like
            True target values (aligned to df_test rows).
        preds : dict[str, np.ndarray]
            Mapping model_name -> predictions aligned to df_test.
        date_col : str
            Name of the datetime column.
        window_days : int
            Width of each plotted window in days.
        n_windows : int
            Number of windows to plot.
        title : Optional[str]
            Custom plot title prefix.

        If save_path is given, it is treated as:
          - a directory (if endswith '/' or has no suffix) -> one file per window
          - a file path ending with .png/.pdf -> base name, window index appended
        Returns list of written file paths (may be empty).
        """
        df = df_test.copy().reset_index(drop=True)
        df["y_true"] = np.asarray(y_true)

        for k, v in preds.items():
            df[f"pred_{k}"] = np.asarray(v)

        starts = self._pick_window_starts(df, n_windows)
        if not starts:
            print("❌ Empty test set; nothing to plot.")
            return []

        written: list[str] = []

        out_dir: Optional[Path] = None
        out_base: Optional[Path] = None

        if save_path:
            p = Path(save_path)
            if p.suffix.lower() in (".png", ".pdf"):
                out_base = p
                out_dir = p.parent
            else:
                out_dir = p
            out_dir.mkdir(parents=True, exist_ok=True)

        for wi, s in enumerate(starts, 1):
            start_t = df.loc[s, date_col]
            end_t = start_t + pd.Timedelta(days=window_days)
            mask = (df[date_col] >= start_t) & (df[date_col] < end_t)
            w = df.loc[mask]
            if w.empty:
                continue

            plt.figure(figsize=(14, 4))
            plt.plot(
                w[date_col],
                w["y_true"],
                color=COLOR_TRUE,
                label="Real consumption",
            )
            cmap = build_color_map(list(preds.keys()))

            plt.plot(
                w[date_col],
                w["y_true"],
                label="Real Consumption",
                color=COLOR_TRUE,
            )
            for k in preds.keys():
                plt.plot(
                    w[date_col], w[f"pred_{k}"], label=k, color=cmap.get(k)
                )

            plt.xlabel("Time")
            plt.ylabel("Consumption [kWh]")
            base = title or "Test window"
            plt.title(f"{base}: {start_t} to {end_t}")
            plt.grid(True)
            plt.legend(frameon=True)
            plt.tight_layout()

            if out_dir is not None:
                if out_base is not None:
                    # base file: foo.png -> foo_win01.png
                    fn = out_base.with_name(
                        f"{out_base.stem}_win{wi:02d}{out_base.suffix}"
                    )
                else:
                    fn = out_dir / f"windows_win{wi:02d}.png"
                plt.savefig(fn, dpi=150)
                written.append(str(fn))

            if show:
                plt.show()
            else:
                plt.close()

        return written

    def plot_prediction_windows(
        self,
        df_test: pd.DataFrame,
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray,
        date_col: str = "date",
        window_days: int = 10,
        n_windows: int = 2,
        title: str = "Predictions",
    ):
        """
        Plot prediction windows (optionally with true target if available).

        Parameters
        ----------
        df_test : pd.DataFrame
            Dataframe containing the datetime column.
        y_true : Optional[np.ndarray]
            True target values aligned to df_test (or None).
        y_pred : np.ndarray
            Prediction values aligned to df_test.
        date_col : str
            Datetime column name.
        window_days : int
            Width of each plotted window in days.
        n_windows : int
            Number of windows to plot.
        title : str
            Title prefix.
        """
        preds = {"Prediction": np.asarray(y_pred)}
        if y_true is None:
            # Create a dummy y_true just to reuse plot_windows logic (won't be plotted)
            dummy_true = np.full(len(df_test), np.nan)
            self.plot_windows(
                df_test=df_test,
                y_true=dummy_true,
                preds=preds,
                date_col=date_col,
                window_days=window_days,
                n_windows=n_windows,
                title=title,
            )
            return

        self.plot_windows(
            df_test=df_test,
            y_true=np.asarray(y_true),
            preds=preds,
            date_col=date_col,
            window_days=window_days,
            n_windows=n_windows,
            title=title,
        )

    def plot_profile_compare_multi(
        self,
        df_test: pd.DataFrame,
        date_col: str,
        target_col: str | None,
        preds: Dict[str, np.ndarray],
        temp_col: str | None = None,
        start: str | None = None,
        hours: int = 240,
        ylim_temp: tuple[float, float] | None = None,
        title: str = "Consumption: Real vs Model Predictions",
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot real consumption + multiple model predictions on the same window.
        Optionally overlays daily mean temperature points (right axis).

        - preds: dict(name -> yhat array aligned with df_test rows)
        - save_path: if set, saves PNG (creates parent dirs)
        - show: if False, closes figure after saving
        """
        import matplotlib.pyplot as plt

        d = df_test.sort_values(date_col).reset_index(drop=True)

        # pick window start
        if start:
            start_ts = pd.Timestamp(start)
            if (d[date_col] >= start_ts).any():
                idx0 = int((d[date_col] >= start_ts).idxmax())
            else:
                idx0 = 0
        else:
            idx0 = 0

        idx1 = min(len(d), idx0 + int(hours))
        w = d.iloc[idx0:idx1].reset_index(drop=True)
        if len(w) == 0:
            print("⚠️ profile_plot window empty; skipping.")
            return

        fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)
        if target_col and (target_col in w.columns):
            ax1.plot(
                w[date_col],
                w[target_col],
                label="Real consumption",
                color=COLOR_TRUE,
            )

        cmap = build_color_map(list(preds.keys()))

        for name, yhat in preds.items():
            yhat_w = np.asarray(yhat)[idx0:idx1]
            ax1.plot(w[date_col], yhat_w, label=name, color=cmap.get(name))

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Consumption")
        ax1.grid(True, alpha=0.3)

        # optional temperature overlay
        if temp_col and (temp_col in w.columns):
            ax2 = ax1.twinx()

            # midday points for readability (assumes hourly data -> 24 rows/day)
            n_days = max(1, len(w) // 24)
            xs = [min(len(w) - 1, 12 + 24 * i) for i in range(n_days)]
            ax2.plot(
                w.loc[xs, date_col],
                w.loc[xs, temp_col].astype(float).to_numpy(),
                marker="*",
                linestyle="--",
                color=COLOR_TEMP,
                label="Daily mean temperature",
            )
            ax2.set_ylabel("Temperature [°C]", color=COLOR_TEMP)
            ax2.tick_params(axis="y", colors=COLOR_TEMP)
            if ylim_temp is not None:
                ax2.set_ylim(float(ylim_temp[0]), float(ylim_temp[1]))

            # merge legends
            l1, lab1 = ax1.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", frameon=True)
        else:
            ax1.legend(loc="upper left", frameon=True)

        ax1.set_title(title)

        # save + show
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=200)
            print(f"✅ saved plot: {p}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_windows_temp(
        self,
        df_test: pd.DataFrame,
        y_true: np.ndarray,
        preds: dict,  # label -> y_pred
        date_col: str,
        temperature_col: str = "temperature_day_mean",
        window_days: int = 10,
        n_windows: int = 2,
        title: str = "",
        save_stem: str = "",
        out_dir: str | None = None,
        show: bool = False,
        # NEW:
        segment: str = "all",  # "first" | "last" | "all"
        gap_hours: int = 2,  # gap threshold to split blocks
        fixed_window: tuple[int, int] | None = None,
        color_map: dict[str, str] | None = None,
    ) -> list[str]:
        """
        Fraunhofer-style windows with optional temperature overlay (midday points),
        robust to disjoint test blocks (splits into contiguous segments first).

        - segment="last": plot windows from the last contiguous block (recommended for seasonal 2-block tests)
        - segment="first": from first block
        - segment="all": spread windows across all blocks (will not draw ramps)
        """
        if out_dir is None:
            out_dir = "."
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if len(df_test) == 0:
            return []

        df = df_test.copy().reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        y_true = np.asarray(y_true)
        preds_np = {k: np.asarray(v) for k, v in preds.items()}

        # color mapping (stable, based on the order of preds as passed in)
        if color_map is None:
            color_map = build_color_map(list(preds_np.keys()))

        # --------- 1) split into contiguous segments (avoid ramp across gaps) ---------
        x = df[date_col]
        gap = x.diff() > pd.Timedelta(hours=gap_hours)
        seg_id = gap.cumsum().fillna(0).astype(int)

        segments = []
        for sid in sorted(seg_id.unique()):
            idx = np.where(seg_id.to_numpy() == sid)[0]
            if len(idx) == 0:
                continue
            segments.append((sid, idx[0], idx[-1] + 1))  # [start, end)

        if not segments:
            return []

        if segment == "first":
            segments_to_plot = [segments[0]]
        elif segment == "last":
            segments_to_plot = [segments[-1]]
        elif segment == "all":
            segments_to_plot = segments
        else:
            raise ValueError("segment must be one of: 'first', 'last', 'all'")

        print("Segments:")
        for sid, s0, s1 in segments:
            print(
                f"  segment {sid}: "
                f"{df.loc[s0, date_col]} → {df.loc[s1-1, date_col]} "
                f"({s1-s0} rows)"
            )

        # --------- 2) choose window starts within chosen segments ---------
        win_len = window_days * 24  # hourly data
        starts: list[int] = []

        if segment in ("first", "last"):
            _, s0, s1 = segments_to_plot[0]
            seg_len = s1 - s0
            if seg_len <= 1:
                return []

            # If segment shorter than window, plot what we have once/twice
            if seg_len <= win_len:
                starts = [s0]
                if n_windows > 1:
                    starts.append(max(s0, s1 - win_len))
            else:
                step = max(1, (seg_len - win_len) // max(1, n_windows))
                starts = [s0 + i * step for i in range(n_windows)]
                starts = [min(st, s1 - win_len) for st in starts]
        else:
            # segment == "all": distribute windows across segments
            # (simple strategy: cycle segments)
            seg_cycle = segments_to_plot
            for i in range(n_windows):
                _, s0, s1 = seg_cycle[i % len(seg_cycle)]
                seg_len = s1 - s0
                if seg_len <= 1:
                    continue
                if seg_len <= win_len:
                    starts.append(s0)
                else:
                    # place window roughly in the middle of the segment
                    mid = s0 + (seg_len - win_len) // 2
                    starts.append(mid)

        if fixed_window is not None:
            time_index, n_days = fixed_window
            if "global_index" not in df_test.columns:
                raise ValueError(
                    "df_test must contain 'global_index' for fixed_window plots"
                )

            gi = df_test["global_index"].to_numpy(dtype=int)
            start_g = time_index
            end_g = time_index + n_days * 24

            # indices within df_test where global_index is inside window
            loc = np.where((gi >= start_g) & (gi < end_g))[0]
            if len(loc) == 0:
                raise ValueError(
                    "fixed_window does not intersect this df_test"
                )

            # take contiguous span in local index
            start = int(loc.min())
            end = int(loc.max()) + 1

            # plot exactly this one window and return (skip auto-window logic)
            # (use your existing plotting code with xw/y slices)

        starts = [int(s) for s in starts if 0 <= s < len(df)]
        if not starts:
            return []

        written: list[str] = []

        # --------- 3) plot each window ---------
        for wi, start in enumerate(starts, 1):
            end = min(len(df), start + win_len)
            if end <= start + 1:
                continue

            dwin = df.iloc[start:end].copy()
            xw = dwin[date_col]
            yw_true = y_true[start:end]

            fig, ax1 = plt.subplots(figsize=(11, 4))

            # Real consumption
            ax1.plot(
                xw,
                yw_true,
                label="Real consumption",
                color=COLOR_TRUE,
            )

            # Predictions (colored via Fraunhofer accents / baseline green)
            for lab, ypred in preds_np.items():
                ypw = ypred[start:end]
                ax1.plot(xw, ypw, label=lab, color=color_map.get(lab, None))

            ax1.set_xlabel("Date")
            ax1.set_ylabel(
                "Consumption (scaled)" if "kWh" not in title else "Consumption"
            )
            ax1.grid(True, alpha=0.25)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
            fig.autofmt_xdate(rotation=0)

            # Temperature overlay: points at 12:00 local time (robust)
            if temperature_col in dwin.columns:
                ax2 = ax1.twinx()

                # pick rows at exactly 12:00 within the window
                mid_mask = pd.to_datetime(dwin[date_col]).dt.hour.eq(12)
                w_mid = dwin.loc[mid_mask, [date_col, temperature_col]].copy()

                # fallback: if there is no exact 12:00 (shouldn't happen for hourly), pick closest-to-12 per day
                if w_mid.empty:
                    tmp = dwin.copy()
                    tmp["__date"] = pd.to_datetime(tmp[date_col]).dt.date
                    tmp["__dist"] = (
                        pd.to_datetime(tmp[date_col]).dt.hour - 12
                    ).abs()
                    w_mid = (
                        tmp.sort_values(["__date", "__dist"])
                        .groupby("__date", as_index=False)
                        .head(1)[[date_col, temperature_col]]
                    )

                ax2.plot(
                    pd.to_datetime(w_mid[date_col]),
                    w_mid[temperature_col].astype(float).to_numpy(),
                    label="Daily mean temperature",
                    color=COLOR_TEMP,
                    linestyle="--",
                    marker="*",
                    markersize=6,
                )
                ax2.set_ylabel("Temperature [°C]", color=COLOR_TEMP)
                ax2.tick_params(axis="y", colors=COLOR_TEMP)

            # Legend merge
            h1, l1 = ax1.get_legend_handles_labels()
            if temperature_col in dwin.columns:
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)
            else:
                ax1.legend(loc="upper left", frameon=True)

            if title:
                ax1.set_title(title)

            fname = (
                f"{save_stem}_win{wi:02d}.png"
                if save_stem
                else f"fraunhofer_win{wi:02d}.png"
            )
            out_path = str(Path(out_dir) / fname)
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            if show:
                plt.show()
            plt.close(fig)

            written.append(out_path)

        return written

    def plot_fixed_global_window_temp(
        self,
        df_all: pd.DataFrame,
        y_true_all: np.ndarray,
        preds_all: dict[str, np.ndarray],
        date_col: str,
        time_index: int,
        n_days: int = 9,
        temperature_col: str = "temperature_day_mean",
        title: str = "",
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        df = df_all.copy().reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col])

        start = int(time_index)
        end = min(len(df), start + n_days * 24)
        if end <= start + 1:
            return None

        w = df.iloc[start:end].copy()
        xw = w[date_col]
        yw_true = np.asarray(y_true_all)[start:end]

        fig, ax1 = plt.subplots(figsize=(11, 4))
        ax1.plot(xw, yw_true, label="Real consumption", color=COLOR_TRUE)

        cmap = build_color_map(list(preds_all.keys()))
        for mi, (lab, ypred) in enumerate(preds_all.items()):
            ypw = np.asarray(ypred)[start:end]
            if mi == 0:
                ax1.plot(xw, ypw, label=lab, color=cmap.get(lab))
            else:
                ax1.plot(xw, ypw, label=lab)

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Consumption (scaled)")
        ax1.grid(True, alpha=0.25)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
        fig.autofmt_xdate(rotation=0)

        if temperature_col in w.columns:
            ax2 = ax1.twinx()

            # pick rows at exactly 12:00 within the window
            mid_mask = pd.to_datetime(w[date_col]).dt.hour.eq(12)
            w_mid = w.loc[mid_mask, [date_col, temperature_col]].copy()

            # fallback: if there is no exact 12:00 (shouldn't happen for hourly), pick closest-to-12 per day
            if w_mid.empty:
                tmp = w.copy()
                tmp["__date"] = pd.to_datetime(tmp[date_col]).dt.date
                tmp["__dist"] = (
                    pd.to_datetime(tmp[date_col]).dt.hour - 12
                ).abs()
                w_mid = (
                    tmp.sort_values(["__date", "__dist"])
                    .groupby("__date", as_index=False)
                    .head(1)[[date_col, temperature_col]]
                )

            ax2.plot(
                pd.to_datetime(w_mid[date_col]),
                w_mid[temperature_col].astype(float).to_numpy(),
                label="Daily mean temperature",
                color=COLOR_TEMP,
                linestyle="--",
                marker="*",
                markersize=6,
            )
            ax2.set_ylabel("Temperature [°C]", color="#bb0056")
            ax2.tick_params(axis="y", colors="#bb0056")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)
        else:
            ax1.legend(loc="upper left", frameon=True)

        if title:
            ax1.set_title(title)

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            if show:
                plt.show()
            plt.close(fig)
            return str(p)

        if show:
            plt.show()
        plt.close(fig)
        return None

# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


def is_heating_period_dk(ts: pd.Timestamp) -> int:
    """Return 1 if ts is inside DK heating period (Oct 1 – May 15), else 0."""
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)

    year = int(ts.year)
    if ts.month <= 5:
        start = pd.Timestamp(f"{year-1}-10-01")
        end = pd.Timestamp(f"{year}-05-15")
    else:
        start = pd.Timestamp(f"{year}-10-01")
        end = pd.Timestamp(f"{year+1}-05-15")
    return int(start <= ts <= end)


def _holiday_flag(
    ts: pd.Series, country: str = "Denmark", subdiv: Optional[str] = None
) -> np.ndarray:
    years = sorted(set(ts.dt.year.dropna().astype(int).tolist()))
    try:
        import holidays as _holidays  # type: ignore

        c = country.lower()
        if c in ("denmark", "dk"):
            hol = _holidays.Denmark(years=years)
        else:
            # holidays supports e.g. holidays.Germany(subdiv="BY")
            cls = getattr(_holidays, country)
            hol = (
                cls(years=years, subdiv=subdiv) if subdiv else cls(years=years)
            )

        return ts.dt.date.isin(hol).astype(int).to_numpy()
    except Exception:
        return np.zeros(len(ts), dtype=int)


@dataclass
class BaseFeatureBuilder:
    """Base class for feature builders."""

    date_col: str = "date"
    target_col: str = "consumption_kWh"

    def feature_columns(self) -> List[str]:
        raise NotImplementedError

    def build_from_raw(
        self, df_raw: pd.DataFrame, require_target: bool
    ) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class HourlyFeatureBuilder(BaseFeatureBuilder):
    """Feature builder for models that use hourly temperature as input."""

    temp_hourly_col: str = "temperature_hourly"
    holiday_country: str = "Denmark"
    holiday_subdiv: Optional[str] = None

    def feature_columns(self) -> List[str]:
        return [
            "temperature_hourly",
            "temperature_day_mean_smoothed",
            "temp_rolling_24h",
            "temp_yesterday",
            "temp_day_before_yesterday",
            "temp_trend",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "is_heating_period",
            "is_holiday",
            "is_weekend",
        ]

    def build_from_raw(
        self, df_raw: pd.DataFrame, require_target: bool
    ) -> pd.DataFrame:
        df = df_raw.copy()
        # Preserve the original row identity so predictions can be aligned back to df_raw/df_test.
        df["_row_id"] = df.index
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df = (
            df.dropna(subset=[self.date_col])
            .sort_values(self.date_col)
            .reset_index(drop=True)
        )

        if self.temp_hourly_col not in df.columns:
            raise ValueError(f"Missing '{self.temp_hourly_col}'")

        df[self.temp_hourly_col] = pd.to_numeric(
            df[self.temp_hourly_col], errors="coerce"
        )

        # daily mean from hourly
        day = df[self.date_col].dt.floor("D")
        daily_means = df.groupby(day)[self.temp_hourly_col].mean()
        df["temperature_day_mean"] = day.map(daily_means)

        # time features
        ts = df[self.date_col]
        hour = ts.dt.hour
        weekday = ts.dt.weekday
        month = ts.dt.month

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        # lags on daily mean (24h/48h)
        df["temp_yesterday"] = df["temperature_day_mean"].shift(24)
        df["temp_day_before_yesterday"] = df["temperature_day_mean"].shift(48)
        df["temp_trend"] = (
            df["temp_yesterday"] - df["temp_day_before_yesterday"]
        )

        # smoothed daily mean curve (simple hourly cubic interpolation anchored at 12:00)
        d = df[[self.date_col, "temperature_day_mean"]].copy()
        d["date_only"] = d[self.date_col].dt.floor("D")
        daily = d.groupby("date_only")["temperature_day_mean"].first()
        anchors = daily.index + pd.Timedelta(hours=12)
        pad = pd.DataFrame({"datetime": anchors, "temp": daily.values})
        prepend = pad["datetime"].iloc[0] - pd.Timedelta(hours=12)
        append = pad["datetime"].iloc[-1] + pd.Timedelta(hours=12)
        pad = pd.concat(
            [
                pd.DataFrame(
                    {"datetime": [prepend], "temp": [pad["temp"].iloc[0]]}
                ),
                pad,
                pd.DataFrame(
                    {"datetime": [append], "temp": [pad["temp"].iloc[-1]]}
                ),
            ],
            ignore_index=True,
        )
        interp = pad.set_index("datetime").resample("h").asfreq()
        try:
            interp = interp.interpolate(method="cubic")
        except Exception:
            interp = interp.interpolate(method="linear")
        interp = interp.reset_index().rename(
            columns={
                "datetime": self.date_col,
                "temp": "temperature_day_mean_smoothed",
            }
        )
        df = pd.merge_asof(
            df.sort_values(self.date_col),
            interp.sort_values(self.date_col),
            on=self.date_col,
            direction="nearest",
        )

        # flags
        df["is_heating_period"] = ts.apply(is_heating_period_dk)
        df["is_weekend"] = (ts.dt.weekday >= 5).astype(int)
        df["is_holiday"] = _holiday_flag(ts, self.holiday_country)

        df["temp_rolling_24h"] = (
            df["temperature_day_mean"].rolling(24, min_periods=1).mean()
        )

        # target handling
        if require_target:
            if self.target_col not in df.columns:
                raise ValueError(
                    f"Missing target '{self.target_col}' in raw data"
                )
            df[self.target_col] = pd.to_numeric(
                df[self.target_col], errors="coerce"
            )

        # drop NaNs in feature columns (+ target if require_target)
        need = self.feature_columns()
        if require_target:
            need = need + [self.target_col]
        df = df.dropna(subset=need)
        return df


@dataclass
class DailyMeanFeatureBuilder(BaseFeatureBuilder):
    """Feature builder for models trained WITHOUT hourly temperature.

    Idea:
    - We accept daily mean temperatures (one value per day) + start_date
    - We expand them to 24 rows per day
    - Features per hour use:
        - temperature_day_mean (constant within the day)
        - lagged daily means: yesterday, day_before_yesterday, trend
        - hour/weekday/month cyclic encodings
        - weekend/holiday/heating-period flags
    """

    holiday_country: str = "Denmark"
    holiday_subdiv: Optional[str] = None

    def feature_columns(self) -> List[str]:
        return [
            "temperature_day_mean_smoothed",
            "temp_yesterday",
            "temp_day_before_yesterday",
            "temp_trend",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "is_heating_period",
            "is_holiday",
            "is_weekend",
        ]

    def build_from_daily_means(
        self,
        start_date: str,
        daily_means_future: List[float],
        daily_means_history: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Create an hourly feature frame (24*horizon) from daily mean temps.

        For lag features, you should provide at least 2 history days.
        If you provide fewer, the first predicted day(s) may be dropped due to NaNs.
        """
        if daily_means_history is None:
            daily_means_history = []
        all_days = list(daily_means_history) + list(daily_means_future)
        if len(all_days) == 0:
            raise ValueError("No daily means provided")

        start_future = pd.to_datetime(start_date)
        start_all = start_future - pd.Timedelta(days=len(daily_means_history))

        rows = []
        for d, mean in enumerate(all_days):
            day_date = (start_all + pd.Timedelta(days=d)).floor("D")
            for h in range(24):
                ts = day_date + pd.Timedelta(hours=h)
                rows.append({"date": ts, "temperature_day_mean": float(mean)})

        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)

        # lags (24h and 48h, since we expanded to hourly)
        df["temp_yesterday"] = df["temperature_day_mean"].shift(24)
        df["temp_day_before_yesterday"] = df["temperature_day_mean"].shift(48)
        df["temp_trend"] = (
            df["temp_yesterday"] - df["temp_day_before_yesterday"]
        )

        ts = df["date"]
        hour = ts.dt.hour
        weekday = ts.dt.weekday
        month = ts.dt.month

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        df["is_heating_period"] = ts.apply(is_heating_period_dk)
        df["is_weekend"] = (ts.dt.weekday >= 5).astype(int)
        df["is_holiday"] = _holiday_flag(
            ts, self.holiday_country, self.holiday_subdiv
        )

        # keep only future part
        df = df[df["date"] >= start_future].reset_index(drop=True)

        # drop rows with NaNs in required features
        df = df.dropna(subset=self.feature_columns()).reset_index(drop=True)
        return df

    def build_from_raw(
        self, df_raw: pd.DataFrame, require_target: bool
    ) -> pd.DataFrame:
        """Build training dataset from raw HOURLY CSV by collapsing to daily means, then expanding to hourly.

        This is used to train a daily-mean model on the same raw file format you already have.
        """
        df = df_raw.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = (
            df.dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        # compute daily mean temperature from hourly temperatures
        if "temperature_hourly" not in df.columns:
            raise ValueError("Missing 'temperature_hourly' in raw data")
        df["temperature_hourly"] = pd.to_numeric(
            df["temperature_hourly"], errors="coerce"
        )

        day = df["date"].dt.floor("D")
        daily_temp = df.groupby(day)["temperature_hourly"].mean()

        # daily mean consumption (optional choice); here: keep HOURLY target if present
        # Your original was hourly target. We want an hourly target for a 24h shape model.
        # So we simply keep the original hourly 'consumption_kWh' as target, but temperature is daily mean constant.
        df2 = df[["date"]].copy()
        df2["temperature_day_mean"] = day.map(daily_temp).values

        if require_target:
            if self.target_col not in df.columns:
                raise ValueError(
                    f"Missing target '{self.target_col}' in raw data"
                )
            df2[self.target_col] = pd.to_numeric(
                df[self.target_col], errors="coerce"
            )

        # Expand features on hourly dataframe (lags etc. are hourly shifts of day-constant values)
        df2["temp_yesterday"] = df2["temperature_day_mean"].shift(24)
        df2["temp_day_before_yesterday"] = df2["temperature_day_mean"].shift(
            48
        )
        df2["temp_trend"] = (
            df2["temp_yesterday"] - df2["temp_day_before_yesterday"]
        )

        ts = df2["date"]
        hour = ts.dt.hour
        weekday = ts.dt.weekday
        month = ts.dt.month

        df2["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df2["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df2["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        df2["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        df2["month_sin"] = np.sin(2 * np.pi * month / 12)
        df2["month_cos"] = np.cos(2 * np.pi * month / 12)

        df2["is_heating_period"] = ts.apply(is_heating_period_dk)
        df2["is_weekend"] = (ts.dt.weekday >= 5).astype(int)
        df2["is_holiday"] = _holiday_flag(ts, self.holiday_country)

        need = self.feature_columns()
        if require_target:
            need = need + [self.target_col]
        df2 = df2.dropna(subset=need).reset_index(drop=True)
        return df2

"""Causal temporal feature engineering (present and past of each engine only)."""

from __future__ import annotations

import pandas as pd
import numpy as np




class CausalFeatureEngineer:
    """Stateful temporal feature engineer that learns healthy baseline variance for startup imputation.

    [CAUSALITY] Every feature at cycle t (lags, rolling mean/std, slopes) is built from cycles
    <= t only; rolling statistics use shift(1) so they never include the current row's future.
    At deployment you cannot see tomorrow's sensor values, so neither may the model in testing.
    Why these features: single-row models cannot see TREND, and degradation is a trend. They let
    tabular models (XGBoost, ...) get the same temporal context as LSTM/CNN/Transformer.
    The class is stateful (fit/transform) so start-up imputation values come from training only.
    """

    def __init__(self, sensor_cols: list[str], windows: tuple[int, ...] = (3, 5, 10, 20)):
        self.sensor_cols = list(sensor_cols)
        self.windows = tuple(windows)
        self.healthy_rollstd_medians_: dict[tuple[str, int], float] = {}

    def fit(self, train_frame: pd.DataFrame, healthy_cycle_limit: int = 50) -> "CausalFeatureEngineer":
        """Compute median rolling std in healthy training phase for initial imputation."""
        healthy = train_frame.loc[train_frame["cycle"] <= healthy_cycle_limit].sort_values(["unit_id", "cycle"])
        for sensor in self.sensor_cols:
            grouped = healthy.groupby("unit_id", sort=False)[sensor]
            for window in self.windows:
                roll_std = grouped.transform(
                    lambda values: values.shift(1).rolling(window, min_periods=2).std()
                ).dropna()
                median_std = float(roll_std.median()) if not roll_std.empty else 0.01
                if not np.isfinite(median_std) or median_std < 1e-6:
                    median_std = 0.01
                self.healthy_rollstd_medians_[(sensor, window)] = median_std
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add lags, deltas, startup indicator, rolling metrics, and causal window slopes with learned startup imputation."""
        result = frame.sort_values(["unit_id", "cycle"]).copy().reset_index(drop=True)
        result["is_startup"] = (result["cycle"] <= 2).astype(float)
        new_cols: dict[str, pd.Series | np.ndarray] = {}
        for sensor in self.sensor_cols:
            grouped = result.groupby("unit_id", sort=False)[sensor]
            lag1 = grouped.shift(1)
            lag2 = grouped.shift(2)
            new_cols[f"{sensor}_lag1"] = lag1.fillna(result[sensor])
            # Boundary fix at t=2: fill lag2 with lag1 first, then current sensor
            new_cols[f"{sensor}_lag2"] = lag2.fillna(lag1).fillna(result[sensor])
            new_cols[f"{sensor}_delta"] = (result[sensor] - lag1).fillna(0.0)
            for window in self.windows:
                fallback_std = self.healthy_rollstd_medians_.get((sensor, window), 0.0)
                new_cols[f"{sensor}_rollmean_{window}"] = grouped.transform(
                    lambda values: values.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(result[sensor])
                new_cols[f"{sensor}_rollstd_{window}"] = grouped.transform(
                    lambda values: values.shift(1).rolling(window, min_periods=2).std()
                ).fillna(fallback_std)
                # Causal slope over window w: (x(t) - x(t-w)) / w
                lag_w = grouped.shift(window).fillna(grouped.transform("first"))
                new_cols[f"{sensor}_slope_{window}"] = ((result[sensor] - lag_w) / float(window)).fillna(0.0)
        if new_cols:
            result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        return result

    def fit_transform(self, train_frame: pd.DataFrame, healthy_cycle_limit: int = 50) -> pd.DataFrame:
        """Fit on train_frame and transform it."""
        return self.fit(train_frame, healthy_cycle_limit=healthy_cycle_limit).transform(train_frame)

def engineer_causal_temporal_features(
    frame: pd.DataFrame,
    sensor_cols: list[str],
    windows: tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    """Add lags, deltas, trailing statistics, and causal slopes; startup values use current/past data only."""
    result = frame.sort_values(["unit_id", "cycle"]).copy().reset_index(drop=True)
    new_cols: dict[str, pd.Series | np.ndarray] = {}
    for sensor in sensor_cols:
        grouped = result.groupby("unit_id", sort=False)[sensor]
        lag1 = grouped.shift(1)
        lag2 = grouped.shift(2)
        new_cols[f"{sensor}_lag1"] = lag1.fillna(result[sensor])
        # Boundary fix at t=2: fill lag2 with lag1 (x at t=1), NOT x at t=2
        new_cols[f"{sensor}_lag2"] = lag2.fillna(lag1).fillna(result[sensor])
        new_cols[f"{sensor}_delta"] = (result[sensor] - lag1).fillna(0.0)
        for window in windows:
            new_cols[f"{sensor}_rollmean_{window}"] = grouped.transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(result[sensor])
            new_cols[f"{sensor}_rollstd_{window}"] = grouped.transform(
                lambda values: values.shift(1).rolling(window, min_periods=2).std()
            ).fillna(0.0)
            lag_w = grouped.shift(window).fillna(grouped.transform("first"))
            new_cols[f"{sensor}_slope_{window}"] = ((result[sensor] - lag_w) / float(window)).fillna(0.0)
    if new_cols:
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
    return result

"""Univariate statistical process control: EWMA/3-sigma and envelope detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT



class StatisticalAnomalyDetector:
    """Calibrate residual location/scale and summarize per-engine EWMA alarms.

    Statistical process control in four lines:
      1. z = (residual - mean_healthy) / std_healthy       (how unusual is this reading?)
      2. EWMA of |z| smooths noise: s_t = a*|z_t| + (1-a)*s_{t-1}   (a = 0.10 -> long memory)
      3. Alarm when the EWMA exceeds mean + k*sigma of its own in-control distribution (k=3).
      4. Persistence (Western-Electric style): require `min_consecutive_breaches` in a row, so a
         single noisy spike is not an alarm. Trade-off: fewer false alarms, slightly later alarm.
    [CALIBRATION] mean/std come from healthy residuals only. Alarms are only counted AFTER
    the healthy window (cycle > H): that is the "online monitoring" phase.
    Lead time = end-of-life cycle - first alarm cycle (bigger is better, if alarms are real).
    """

    def __init__(
        self,
        threshold_sigma: float = 3.0,
        ewma_alpha: float = 0.10,
        healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
        min_consecutive_breaches: int = 3,
    ):
        if threshold_sigma <= 0 or not 0 < ewma_alpha <= 1:
            raise ValueError("threshold_sigma must be positive and ewma_alpha in (0, 1]")
        if min_consecutive_breaches < 1:
            raise ValueError("min_consecutive_breaches must be >= 1")
        self.threshold_sigma = threshold_sigma
        self.ewma_alpha = ewma_alpha
        self.healthy_cycle_limit = healthy_cycle_limit
        self.min_consecutive_breaches = int(min_consecutive_breaches)
        self.mu_e = 0.0
        self.sigma_e = 1.0
        self.abs_z_mean = np.sqrt(2.0 / np.pi)
        self.abs_z_variance = 1.0 - 2.0 / np.pi

    def calibrate(self, healthy_residuals) -> "StatisticalAnomalyDetector":
        """Estimate mean and standard deviation from healthy calibration residuals."""
        residuals = np.asarray(healthy_residuals, dtype=float)
        if residuals.size == 0 or not np.isfinite(residuals).all():
            raise ValueError("healthy_residuals must be a non-empty finite array")
        self.mu_e = float(residuals.mean())
        self.sigma_e = float(residuals.std())
        if self.sigma_e < 1e-8:
            self.sigma_e = 1.0
        self.calibrate_ewma_distribution(self.compute_z_scores(residuals))
        return self

    def calibrate_ewma_distribution(self, z_scores) -> "StatisticalAnomalyDetector":
        """Estimate the in-control distribution of ``|Z|`` from calibration scores."""
        absolute_scores = np.abs(np.asarray(z_scores, dtype=float))
        if absolute_scores.size == 0 or not np.isfinite(absolute_scores).all():
            raise ValueError("z_scores must be a non-empty finite array")
        self.abs_z_mean = float(absolute_scores.mean())
        self.abs_z_variance = float(absolute_scores.var())
        if self.abs_z_variance < 1e-12:
            self.abs_z_variance = 1e-12
        return self

    def compute_z_scores(self, residuals) -> np.ndarray:
        """Standardize residuals against the fitted healthy baseline."""
        return (np.asarray(residuals, dtype=float) - self.mu_e) / self.sigma_e

    def ewma_upper_limits(self, n: int) -> np.ndarray:
        """Finite-horizon control limits from the calibrated distribution of ``|Z|``."""
        t = np.arange(1, n + 1, dtype=float)
        alpha = self.ewma_alpha
        variance = self.abs_z_variance * alpha / (2.0 - alpha) * (1.0 - (1.0 - alpha) ** (2.0 * t))
        return self.abs_z_mean + self.threshold_sigma * np.sqrt(variance)

    def ewma_abs_z(self, z_scores) -> np.ndarray:
        """Compute EWMA(|Z|) from its calibrated in-control mean."""
        values = np.abs(np.asarray(z_scores, dtype=float))
        result = np.empty_like(values)
        state = self.abs_z_mean
        for i, value in enumerate(values):
            state = self.ewma_alpha * value + (1.0 - self.ewma_alpha) * state
            result[i] = state
        return result

    @staticmethod
    def test_residual_normality(residuals) -> dict[str, float]:
        """Perform Shapiro-Wilk and D'Agostino-Pearson tests on healthy residuals."""
        res = np.asarray(residuals, dtype=float)
        res_sample = res[: min(5000, len(res))]
        shapiro_stat, shapiro_p = stats.shapiro(res_sample)
        norm_stat, norm_p = stats.normaltest(res_sample)
        return {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "dagostino_stat": float(norm_stat),
            "dagostino_p": float(norm_p),
        }

    def calibrate_empirical_ucl(self, z_fleet_scores, alpha_level: float = 0.01) -> float:
        """Calibrate non-parametric empirical UCL from healthy fleet absolute z-scores."""
        abs_z = np.abs(np.asarray(z_fleet_scores, dtype=float))
        if abs_z.size == 0:
            return float(self.abs_z_mean + self.threshold_sigma * np.sqrt(self.abs_z_variance))
        return float(np.quantile(abs_z, 1.0 - alpha_level))

    def score_engine(self, engine_frame: pd.DataFrame, residual_col: str = "residual") -> tuple[pd.DataFrame, dict]:
        """Add Z/EWMA columns and return first operational alarm (cycle > H) and healthy FAR."""
        ordered = engine_frame.sort_values("cycle").copy().reset_index(drop=True)
        z = self.compute_z_scores(ordered[residual_col].to_numpy())
        ewma = self.ewma_abs_z(z)
        limits = self.ewma_upper_limits(len(ewma))

        healthy_mask = ordered["cycle"].to_numpy() <= self.healthy_cycle_limit
        monitoring_mask = ordered["cycle"].to_numpy() > self.healthy_cycle_limit

        # Check breaches: raw Z breach (Shewhart) and EWMA breach
        raw_breaches = np.abs(z) >= self.threshold_sigma
        ewma_breaches = ewma >= limits
        breaches = raw_breaches & ewma_breaches

        far = float(np.mean(ewma_breaches[healthy_mask])) if healthy_mask.any() else 0.0
        premature = bool(np.any(ewma_breaches[healthy_mask])) if healthy_mask.any() else False

        # Operational alarms are strictly identified during online monitoring (cycle > healthy_cycle_limit)
        # using Western Electric persistence rule (min_consecutive_breaches consecutive cycles)
        if self.min_consecutive_breaches > 1:
            online_breaches = breaches & monitoring_mask
            consec = (
                pd.Series(online_breaches)
                .rolling(window=self.min_consecutive_breaches, min_periods=self.min_consecutive_breaches)
                .sum()
                .to_numpy()
            )
            sustained_ends = np.flatnonzero(consec >= self.min_consecutive_breaches)
            online_alarms = sustained_ends - (self.min_consecutive_breaches - 1) if sustained_ends.size else np.array([], dtype=int)
            persistent_mask = np.zeros(len(ewma), dtype=bool)
            for end_idx in sustained_ends:
                start_idx = max(0, end_idx - self.min_consecutive_breaches + 1)
                persistent_mask[start_idx : end_idx + 1] = True
        else:
            online_alarms = np.flatnonzero(breaches & monitoring_mask)
            persistent_mask = breaches & monitoring_mask

        first_cycle = int(ordered.iloc[online_alarms[0]]["cycle"]) if online_alarms.size else None
        eol = int(ordered["cycle"].max())
        lead_time = eol - first_cycle if first_cycle is not None else 0

        ordered["z_score"] = z
        ordered["ewma_abs_z"] = ewma
        ordered["ewma_ucl"] = limits
        ordered["persistent_alarm"] = persistent_mask

        summary = {
            "unit_id": int(ordered["unit_id"].iloc[0]),
            "eol_cycle": eol,
            "first_alarm_cycle": first_cycle,
            "lead_time_cycles": lead_time,
            "healthy_far": far,
            "healthy_far_pct": far * 100.0,
            "premature_alarm": premature,
            "premature_alarm_pct": 100.0 if premature else 0.0,
            "detected": first_cycle is not None,
        }
        return ordered, summary

    def evaluate_fleet(self, fleet_frame: pd.DataFrame, residual_col: str = "residual") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Return scored residual rows, per-engine results, and fleet summary."""
        scored = []
        summaries = []
        for _, engine in fleet_frame.groupby("unit_id", sort=False):
            engine_scores, summary = self.score_engine(engine, residual_col)
            scored.append(engine_scores)
            summaries.append(summary)
        per_engine = pd.DataFrame(summaries)
        detected_engines = per_engine.loc[per_engine["detected"]]
        mean_lead_time_detected = float(detected_engines["lead_time_cycles"].mean()) if not detected_engines.empty else 0.0

        return pd.concat(scored, ignore_index=True), per_engine, {
            "total_engines": len(per_engine),
            "detection_rate_pct": float(per_engine["detected"].mean() * 100.0),
            "mean_lead_time_cycles": float(per_engine["lead_time_cycles"].mean()),
            "mean_lead_time_detected_cycles": mean_lead_time_detected,
            "mean_healthy_false_alarm_pct": float(per_engine["healthy_far"].mean() * 100.0),
            "premature_alarm_pct": float(per_engine["premature_alarm"].mean() * 100.0),
        }

class EnvelopeAnomalyDetector:
    """Envelope / 'embrulho' anomaly detector for time series and digital signals.

    Implements two complementary envelope strategies:
    1. Time Series Prediction Envelope (Power BI-style prediction intervals):
       Upper/lower prediction boundaries based on healthy calibration residual variance:
       Upper(t) = y_pred(t) + k * sigma
       Lower(t) = y_pred(t) - k * sigma
       Points outside this shaded 'embrulho' represent structural anomalies.

    2. Geometric Signal Envelope (Rolling peak/valley hull):
       Wraps signal peaks and valleys using causal rolling extrema:
       Upper_Hull(t) = max_{tau in [t-w, t]} y(tau)
       Lower_Hull(t) = min_{tau in [t-w, t]} y(tau)
       Monitors amplitude expansion (modulation) beyond healthy mechanical limits.
    """

    def __init__(
        self,
        k_sigma: float = 3.0,
        healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
        window_size: int = 5,
        min_consecutive_breaches: int = 3,
    ):
        if k_sigma <= 0:
            raise ValueError("k_sigma must be positive")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.k_sigma = float(k_sigma)
        self.healthy_cycle_limit = int(healthy_cycle_limit)
        self.window_size = int(window_size)
        self.min_consecutive_breaches = int(min_consecutive_breaches)

        self.mu_e = 0.0
        self.sigma_e = 1.0
        self.mu_signal = 0.0
        self.sigma_signal = 1.0
        self.mu_amp = 0.0
        self.sigma_amp = 1.0
        self.mu_res_amp = 0.0
        self.sigma_res_amp = 1.0
        self.mu_res_abs = 0.0
        self.sigma_res_abs = 1.0

    def calibrate(self, healthy_actual, healthy_pred) -> "EnvelopeAnomalyDetector":
        """Estimate residual and geometric envelope baselines from healthy data."""
        actual = np.asarray(healthy_actual, dtype=float)
        pred = np.asarray(healthy_pred, dtype=float)
        if actual.size == 0 or pred.size == 0 or actual.shape != pred.shape:
            raise ValueError("healthy_actual and healthy_pred must be non-empty matching arrays")

        residuals = actual - pred
        self.mu_e = float(np.mean(residuals))
        self.sigma_e = float(np.std(residuals))
        if self.sigma_e < 1e-8:
            self.sigma_e = 1.0

        self.mu_signal = float(np.mean(actual))
        self.sigma_signal = float(np.std(actual))
        if self.sigma_signal < 1e-8:
            self.sigma_signal = 1.0

        series = pd.Series(actual)
        roll_max = series.rolling(window=self.window_size, min_periods=1).max().to_numpy()
        roll_min = series.rolling(window=self.window_size, min_periods=1).min().to_numpy()
        amps = roll_max - roll_min
        self.mu_amp = float(np.mean(amps))
        self.sigma_amp = float(np.std(amps))
        if self.sigma_amp < 1e-8:
            self.sigma_amp = 1.0

        # Calibrate hull amplitude/magnitude on the regime-decoupled RESIDUAL (not the raw signal):
        # on the raw multi-regime signal the hull would mostly measure regime switching.
        res_series = pd.Series(residuals)
        res_roll_max = res_series.rolling(window=self.window_size, min_periods=1).max().to_numpy()
        res_roll_min = res_series.rolling(window=self.window_size, min_periods=1).min().to_numpy()
        res_amps = res_roll_max - res_roll_min
        self.mu_res_amp = float(np.mean(res_amps))
        self.sigma_res_amp = float(np.std(res_amps))
        if self.sigma_res_amp < 1e-8:
            self.sigma_res_amp = 1.0
        abs_res = np.abs(residuals - self.mu_e)
        self.mu_res_abs = float(np.mean(abs_res))
        self.sigma_res_abs = float(np.std(abs_res))
        if self.sigma_res_abs < 1e-8:
            self.sigma_res_abs = 1.0

        return self

    def score_time_series_envelope(
        self,
        df: pd.DataFrame,
        actual_col: str,
        pred_col: str,
        cycle_col: str = "cycle",
        unit_col: str = "unit_id",
    ) -> tuple[pd.DataFrame, dict]:
        """Compute the prediction envelope corridor, flag breaches, and compute lead time to breakage."""
        ordered = df.sort_values(cycle_col).copy().reset_index(drop=True)
        actual = ordered[actual_col].to_numpy(dtype=float)
        pred = ordered[pred_col].to_numpy(dtype=float)

        upper = pred + (self.mu_e + self.k_sigma * self.sigma_e)
        lower = pred + (self.mu_e - self.k_sigma * self.sigma_e)

        out_upper = actual > upper
        out_lower = actual < lower
        out_of_envelope = out_upper | out_lower

        breach_dist = np.maximum(0.0, actual - upper) + np.maximum(0.0, lower - actual)

        ordered["envelope_upper"] = upper
        ordered["envelope_lower"] = lower
        ordered["out_of_envelope"] = out_of_envelope
        ordered["breach_distance"] = breach_dist

        # Operational alarms are identified during online monitoring (cycle > healthy_cycle_limit)
        is_monitoring = ordered[cycle_col].to_numpy() > self.healthy_cycle_limit
        if self.min_consecutive_breaches > 1:
            consecutive = pd.Series(out_of_envelope).rolling(window=self.min_consecutive_breaches, min_periods=self.min_consecutive_breaches).sum()
            alarm_mask = (consecutive.to_numpy() >= self.min_consecutive_breaches) & is_monitoring
            alarm_indices = np.flatnonzero(alarm_mask)
            # Offset to the first cycle of the consecutive sequence
            alarm_indices = alarm_indices - (self.min_consecutive_breaches - 1) if alarm_indices.size else np.array([])
        else:
            alarm_mask = out_of_envelope & is_monitoring
            alarm_indices = np.flatnonzero(alarm_mask)

        eol_cycle = int(ordered[cycle_col].max())
        first_alarm = int(ordered.iloc[alarm_indices[0]][cycle_col]) if len(alarm_indices) > 0 else None
        lead_time = eol_cycle - first_alarm if first_alarm is not None else 0

        healthy_mask = ordered[cycle_col].to_numpy() <= self.healthy_cycle_limit
        healthy_breaches = int(np.sum(out_of_envelope[healthy_mask])) if healthy_mask.any() else 0
        healthy_far = float(np.mean(out_of_envelope[healthy_mask]) * 100.0) if healthy_mask.any() else 0.0

        summary = {
            "unit_id": int(ordered[unit_col].iloc[0]) if unit_col in ordered else 0,
            "eol_cycle": eol_cycle,
            "first_alarm_cycle": first_alarm,
            "lead_time_cycles": lead_time,
            "detected": first_alarm is not None,
            "total_cycles": len(ordered),
            "out_of_envelope_count": int(np.sum(out_of_envelope)),
            "in_envelope_pct": float(np.mean(~out_of_envelope) * 100.0),
            "healthy_far_pct": healthy_far,
            "healthy_breaches": healthy_breaches,
            "envelope_margin": float(self.k_sigma * self.sigma_e),
        }
        return ordered, summary

    def score_geometric_envelope(
        self,
        df: pd.DataFrame,
        signal_col: str,
        cycle_col: str = "cycle",
        unit_col: str = "unit_id",
        use_decoupled_signal: bool = True,
    ) -> tuple[pd.DataFrame, dict]:
        """Compute rolling peak-valley geometric envelope, modulation, and physical breaches.

        When ``use_decoupled_signal=True`` and a regime-standardized column (``{signal_col}_norm``)
        or virtual sensor residual/prediction is available in ``df``, the rolling hull is computed
        on the regime-decoupled series rather than the raw multi-regime signal, which would
        mostly measure regime switching instead of degradation.
        """
        ordered = df.sort_values(cycle_col).copy().reset_index(drop=True)
        raw_signal = ordered[signal_col].to_numpy(dtype=float)

        norm_col = f"{signal_col}_norm"
        if use_decoupled_signal and "residual" in ordered.columns:
            # Option B: Decoupled virtual sensor residual e(t) = y(t) - y_hat(t)
            signal = ordered["residual"].to_numpy(dtype=float)
            amp_threshold = self.mu_res_amp + self.k_sigma * self.sigma_res_amp
            upper_ceiling = abs(self.mu_e) + self.k_sigma * self.sigma_e
            hull_max_abs = (
                pd.Series(np.abs(signal - self.mu_e))
                .rolling(window=self.window_size, min_periods=1)
                .max()
                .to_numpy()
            )
            roll_max = pd.Series(signal).rolling(window=self.window_size, min_periods=1).max().to_numpy()
            roll_min = pd.Series(signal).rolling(window=self.window_size, min_periods=1).min().to_numpy()
            amp = roll_max - roll_min
            geom_breach = (amp > amp_threshold) | (hull_max_abs > upper_ceiling)
            decoupled_mode = "residual"
        elif use_decoupled_signal and ("prediction" in ordered.columns or "pred" in ordered.columns):
            pred_col_name = "prediction" if "prediction" in ordered.columns else "pred"
            signal = raw_signal - ordered[pred_col_name].to_numpy(dtype=float)
            amp_threshold = self.mu_res_amp + self.k_sigma * self.sigma_res_amp
            upper_ceiling = abs(self.mu_e) + self.k_sigma * self.sigma_e
            hull_max_abs = (
                pd.Series(np.abs(signal - self.mu_e))
                .rolling(window=self.window_size, min_periods=1)
                .max()
                .to_numpy()
            )
            roll_max = pd.Series(signal).rolling(window=self.window_size, min_periods=1).max().to_numpy()
            roll_min = pd.Series(signal).rolling(window=self.window_size, min_periods=1).min().to_numpy()
            amp = roll_max - roll_min
            geom_breach = (amp > amp_threshold) | (hull_max_abs > upper_ceiling)
            decoupled_mode = "residual"
        elif use_decoupled_signal and norm_col in ordered.columns:
            # Option A: Regime-standardized Z_k(t)
            signal = ordered[norm_col].to_numpy(dtype=float)
            healthy_mask = ordered[cycle_col].to_numpy() <= self.healthy_cycle_limit
            h_sig = signal[healthy_mask] if healthy_mask.any() else signal
            h_rmax = pd.Series(h_sig).rolling(window=self.window_size, min_periods=1).max().to_numpy()
            h_rmin = pd.Series(h_sig).rolling(window=self.window_size, min_periods=1).min().to_numpy()
            h_amp = h_rmax - h_rmin
            amp_threshold = float(np.mean(h_amp) + self.k_sigma * max(float(np.std(h_amp)), 0.1))
            upper_ceiling = float(self.k_sigma)
            roll_max = pd.Series(signal).rolling(window=self.window_size, min_periods=1).max().to_numpy()
            roll_min = pd.Series(signal).rolling(window=self.window_size, min_periods=1).min().to_numpy()
            amp = roll_max - roll_min
            hull_max_abs = np.maximum(np.abs(roll_max), np.abs(roll_min))
            geom_breach = (amp > amp_threshold) | (hull_max_abs > upper_ceiling)
            decoupled_mode = "regime_z"
        else:
            signal = raw_signal
            roll_max = pd.Series(signal).rolling(window=self.window_size, min_periods=1).max().to_numpy()
            roll_min = pd.Series(signal).rolling(window=self.window_size, min_periods=1).min().to_numpy()
            amp = roll_max - roll_min
            amp_threshold = self.mu_amp + self.k_sigma * self.sigma_amp
            upper_ceiling = self.mu_signal + self.k_sigma * self.sigma_signal
            lower_ceiling = self.mu_signal - self.k_sigma * self.sigma_signal
            geom_breach = (amp > amp_threshold) | (roll_max > upper_ceiling) | (roll_min < lower_ceiling)
            decoupled_mode = "raw"

        ordered["geom_signal"] = signal
        ordered["geom_upper_hull"] = roll_max
        ordered["geom_lower_hull"] = roll_min
        ordered["geom_amplitude"] = amp
        ordered["geom_breach"] = geom_breach

        is_monitoring = ordered[cycle_col].to_numpy() > self.healthy_cycle_limit
        if self.min_consecutive_breaches > 1:
            online_breaches = geom_breach & is_monitoring
            consec = (
                pd.Series(online_breaches)
                .rolling(window=self.min_consecutive_breaches, min_periods=self.min_consecutive_breaches)
                .sum()
                .to_numpy()
            )
            sustained_ends = np.flatnonzero(consec >= self.min_consecutive_breaches)
            alarm_indices = (
                sustained_ends - (self.min_consecutive_breaches - 1)
                if sustained_ends.size
                else np.array([], dtype=int)
            )
        else:
            alarm_indices = np.flatnonzero(geom_breach & is_monitoring)

        eol_cycle = int(ordered[cycle_col].max())
        first_alarm = int(ordered.iloc[alarm_indices[0]][cycle_col]) if len(alarm_indices) > 0 else None
        lead_time = eol_cycle - first_alarm if first_alarm is not None else 0

        summary = {
            "unit_id": int(ordered[unit_col].iloc[0]) if unit_col in ordered else 0,
            "eol_cycle": eol_cycle,
            "first_alarm_cycle": first_alarm,
            "lead_time_cycles": lead_time,
            "detected": first_alarm is not None,
            "amp_threshold": float(amp_threshold),
            "upper_ceiling": float(upper_ceiling),
            "decoupled_mode": decoupled_mode,
        }
        return ordered, summary

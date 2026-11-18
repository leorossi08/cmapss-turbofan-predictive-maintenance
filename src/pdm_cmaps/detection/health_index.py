"""Fusion of sensor residuals into a single health index."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pdm_cmaps.detection.spc import StatisticalAnomalyDetector
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE, NEGATIVE_DRIFT_SENSORS, POSITIVE_DRIFT_SENSORS



class HealthIndexFuser:
    """Multivariate Health Index (HI) synthesis and anomaly monitoring.

    Concept: PCA (principal component analysis) finds the direction of largest variance in
    standardized sensors; a monotone drift of many sensors together shows up as PC1. We map
    PC1 linearly so that the healthy median = 1.0 and the late-life median = 0.0, then smooth
    with an EWMA. Pre-condition: sensors must be regime-standardized (`*_norm`), otherwise PC1
    would capture flight regime, the largest source of variance in FD002/FD004.

    Synthesizes multiple degraded physical sensor channels into an interpretable,
    monotonic Health Index [1.0 (Healthy) -> 0.0 (Failure)] using Principal Component
    Analysis (PCA) calibrated strictly on the healthy window (cycle <= H).
    """

    def __init__(
        self,
        sensors: list[str] | tuple[str, ...] | None = None,
        healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
        ewma_alpha: float = 0.10,
        min_consecutive_breaches: int = 3,
        threshold_sigma: float = 3.0,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.sensors = list(sensors) if sensors is not None else [
            s for s in [*POSITIVE_DRIFT_SENSORS, *NEGATIVE_DRIFT_SENSORS]
        ]
        self.healthy_cycle_limit = int(healthy_cycle_limit)
        self.ewma_alpha = float(ewma_alpha)
        self.min_consecutive_breaches = int(min_consecutive_breaches)
        self.threshold_sigma = float(threshold_sigma)
        self.random_state = int(random_state)

        self.pca_: PCA | None = None
        self.scaler_: StandardScaler | None = None
        self.pc1_sign_: float = 1.0
        self.hi_min_: float = 0.0
        self.hi_max_: float = 1.0
        self.calibrated_sensors_: list[str] = []
        self.detector_: StatisticalAnomalyDetector | None = None

    def fit(self, train_frame: pd.DataFrame) -> "HealthIndexFuser":
        """Fit PCA on regime-standardized or raw healthy sensor observations."""
        healthy = train_frame.loc[train_frame["cycle"] <= self.healthy_cycle_limit].copy()
        if healthy.empty:
            healthy = train_frame.copy()

        # Choose standardized columns ({s}_norm) if present, else fallback to raw sensor
        active_cols = []
        for s in self.sensors:
            norm_col = f"{s}_norm"
            if norm_col in healthy.columns and healthy[norm_col].nunique() > 1 and healthy[norm_col].std() > 1e-6:
                active_cols.append(norm_col)
            elif s in healthy.columns and healthy[s].nunique() > 1 and healthy[s].std() > 1e-6:
                active_cols.append(s)

        if not active_cols:
            raise ValueError("No informative sensor channels available to fit HealthIndexFuser")

        self.calibrated_sensors_ = active_cols
        self.scaler_ = StandardScaler()
        X_h = self.scaler_.fit_transform(healthy[active_cols].to_numpy(dtype=float))

        self.pca_ = PCA(n_components=1, random_state=self.random_state)
        self.pca_.fit(X_h)

        # Orient PC1 empirically (not from hard-coded drift tables): HI must decrease with age,
        # so the projection must be negatively correlated with the cycle index on training engines.
        X_all = self.scaler_.transform(train_frame[active_cols].to_numpy(dtype=float))
        unsigned_proj = self.pca_.transform(X_all)[:, 0]
        cycles = train_frame["cycle"].to_numpy(dtype=float)
        if np.std(unsigned_proj) > 1e-12 and np.std(cycles) > 1e-12:
            age_corr = float(np.corrcoef(unsigned_proj, cycles)[0, 1])
        else:
            age_corr = 0.0
        self.pc1_sign_ = -1.0 if age_corr > 0 else 1.0

        # Calibrate baseline bounds on healthy and degraded profiles
        raw_proj = unsigned_proj * self.pc1_sign_
        
        # Upper anchor (healthy phase median) -> HI = 1.0
        h_proj = (self.pca_.transform(X_h)[:, 0]) * self.pc1_sign_
        self.hi_max_ = float(np.median(h_proj))

        # Lower anchor (EOL cycles: max cycle - 5 per engine in train)
        eol_idx = train_frame.groupby("unit_id")["cycle"].transform("max")
        late_mask = train_frame["cycle"] >= np.maximum(self.healthy_cycle_limit, eol_idx - 10)
        late_proj = raw_proj[late_mask.to_numpy()] if late_mask.any() else raw_proj
        self.hi_min_ = float(np.median(late_proj))

        if abs(self.hi_max_ - self.hi_min_) < 1e-6:
            self.hi_min_ = self.hi_max_ - 1.0

        # Calibrate SPC detector on healthy HI deviation from 1.0
        healthy_hi = np.clip((h_proj - self.hi_min_) / (self.hi_max_ - self.hi_min_), 0.0, 1.0)
        healthy_deviation = 1.0 - healthy_hi
        self.detector_ = StatisticalAnomalyDetector(
            threshold_sigma=self.threshold_sigma,
            ewma_alpha=self.ewma_alpha,
            healthy_cycle_limit=self.healthy_cycle_limit,
            min_consecutive_breaches=self.min_consecutive_breaches,
        )
        self.detector_.calibrate(healthy_deviation)

        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute continuous Health Index and EWMA smoothed HI [1.0 -> 0.0]."""
        if self.pca_ is None or self.scaler_ is None or not self.calibrated_sensors_:
            raise RuntimeError("Fit HealthIndexFuser before transform")

        result = frame.sort_values(["unit_id", "cycle"]).copy().reset_index(drop=True)
        X = self.scaler_.transform(result[self.calibrated_sensors_].to_numpy(dtype=float))
        proj = (self.pca_.transform(X)[:, 0]) * self.pc1_sign_

        # Normalize to [0.0, 1.0]
        denom = self.hi_max_ - self.hi_min_
        raw_hi = np.clip((proj - self.hi_min_) / denom, 0.0, 1.0)
        result["health_index_raw"] = raw_hi

        # Causal EWMA per engine
        ewma_hi = np.empty_like(raw_hi)
        for _, engine in result.groupby("unit_id", sort=False):
            series = pd.Series(engine["health_index_raw"].to_numpy())
            # EWMA smoothing with initial value 1.0 (assumed healthy at t=0)
            smoothed = series.ewm(alpha=self.ewma_alpha, adjust=False).mean().to_numpy()
            ewma_hi[engine.index] = np.clip(smoothed, 0.0, 1.0)

        result["health_index"] = ewma_hi
        return result

    def evaluate(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Fit on train, score on test, and generate fleet health index anomaly summaries."""
        self.fit(train_frame)
        scored = self.transform(test_frame)

        records = []
        for unit_id, engine in scored.groupby("unit_id", sort=False):
            ordered = engine.sort_values("cycle").reset_index(drop=True)
            hi_vals = ordered["health_index"].to_numpy()
            deviation = 1.0 - hi_vals
            z_scores = self.detector_.compute_z_scores(deviation)
            ewma_z = self.detector_.ewma_abs_z(z_scores)
            limits = self.detector_.ewma_upper_limits(len(ewma_z))

            ordered["hi_z_score"] = z_scores
            ordered["hi_ewma_z"] = ewma_z
            ordered["hi_ucl"] = limits

            healthy_mask = ordered["cycle"].to_numpy() <= self.healthy_cycle_limit
            monitoring_mask = ordered["cycle"].to_numpy() > self.healthy_cycle_limit
            breaches = ewma_z >= limits

            far = float(np.mean(breaches[healthy_mask])) if healthy_mask.any() else 0.0
            premature = bool(np.any(breaches[healthy_mask])) if healthy_mask.any() else False

            if self.min_consecutive_breaches > 1:
                online_breaches = breaches & monitoring_mask
                consec = (
                    pd.Series(online_breaches)
                    .rolling(window=self.min_consecutive_breaches, min_periods=self.min_consecutive_breaches)
                    .sum()
                    .to_numpy()
                )
                sustained_ends = np.flatnonzero(consec >= self.min_consecutive_breaches)
                alarm_positions = (
                    sustained_ends - (self.min_consecutive_breaches - 1)
                    if sustained_ends.size
                    else np.array([], dtype=int)
                )
            else:
                alarm_positions = np.flatnonzero(breaches & monitoring_mask)

            eol = int(ordered["cycle"].max())
            alarm_cycle = int(ordered.iloc[alarm_positions[0]]["cycle"]) if alarm_positions.size else None
            lead_time = eol - alarm_cycle if alarm_cycle is not None else 0

            records.append({
                "unit_id": int(unit_id),
                "eol_cycle": eol,
                "first_alarm_cycle": alarm_cycle,
                "lead_time_cycles": lead_time,
                "healthy_far": far,
                "premature_alarm": premature,
                "detected": alarm_cycle is not None,
                "final_health_index": float(hi_vals[-1]),
                "healthy_health_index": float(np.mean(hi_vals[healthy_mask])) if healthy_mask.any() else 1.0,
            })

        per_engine = pd.DataFrame(records)
        detected_engines = per_engine.loc[per_engine["detected"]]
        mean_lead_time_detected = (
            float(detected_engines["lead_time_cycles"].mean()) if not detected_engines.empty else 0.0
        )

        summary = {
            "total_engines": len(per_engine),
            "detection_rate_pct": float(per_engine["detected"].mean() * 100.0),
            "mean_lead_time_cycles": float(per_engine["lead_time_cycles"].mean()),
            "mean_lead_time_detected_cycles": mean_lead_time_detected,
            "mean_healthy_false_alarm_pct": float(per_engine["healthy_far"].mean() * 100.0),
            "premature_alarm_pct": float(per_engine["premature_alarm"].mean() * 100.0),
            "mean_final_health_index": float(per_engine["final_health_index"].mean()),
        }
        return scored, per_engine, summary

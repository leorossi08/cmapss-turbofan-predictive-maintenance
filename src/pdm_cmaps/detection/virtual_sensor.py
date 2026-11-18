"""Regime-conditioned virtual sensor and its residual."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pdm_cmaps.detection.spc import StatisticalAnomalyDetector
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT
from pdm_cmaps.modeling.registry import get_model



CANONICAL_EXOGENOUS_FEATURES = [
    "setting_1", "setting_2", "setting_3", "T2", "P2", "P15", "epr", "Nf_dmd", "PCNfR_dmd"
]

class VirtualSensor:
    """Fit one regression model per operating regime using healthy training rows.

    Concept: a "virtual" (soft) sensor predicts a physical reading from OTHER inputs. Here the
    inputs are exogenous only (settings + ambient channels that degradation does not move), so
    the prediction answers "what would a HEALTHY engine read under these conditions?".
    The residual (actual - predicted) is therefore ~0 + noise when healthy and drifts when the
    hardware wears. [ASSUMPTION] Because those inputs are near-constant inside a regime, the
    model is close to a per-regime mean; the value of the method is regime removal, not
    sophisticated regression.
    [DEFAULT] ``engine_offset=True`` (padrão desde a adoção do deslocamento do motor): o z de cada motor é centrado na média dos seus
    primeiros W ciclos (``offset_window``, padrão H) e a escala é renormalizada, porque o σ agrupado da frota inclui o deslocamento
    entre motores. Com ``engine_offset=False`` volta o comportamento anterior (μ e σ agrupados da frota).
    Design choices: one model per regime (+ global fallback when a regime has <30 rows or <2
    engines); out-of-fold residuals for calibration ([LEAKAGE]); ExtraTrees as a cheap,
    non-linear, low-tuning default.
    """

    def __init__(
        self,
        target_sensor: str = "T50",
        feature_cols: list[str] | None = None,
        mode: str = "exogenous",
        threshold_sigma: float = 3.0,
        ewma_alpha: float = 0.10,
        healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
        min_consecutive_breaches: int = 3,
        estimator=None,
        random_state: int = 42,
        engine_offset: bool = True,
        offset_window: int | None = None,
        calibrate_out_of_window: bool = False,
    ):
        self.target_sensor = target_sensor
        self.feature_cols = feature_cols
        self.mode = mode
        self.threshold_sigma = threshold_sigma
        self.ewma_alpha = ewma_alpha
        self.healthy_cycle_limit = healthy_cycle_limit
        self.min_consecutive_breaches = int(min_consecutive_breaches)
        self.estimator = estimator if estimator is not None else get_model(
            "extra_trees", n_estimators=60, max_depth=12, random_state=random_state
        )
        self.engine_offset = bool(engine_offset)
        self.offset_window = offset_window
        # [DEFAULT OFF] com offset_window < healthy_cycle_limit, a escala e o limite da EWMA são calibrados só nos ciclos FORA da janela que
        # estima o deslocamento (W, H], e o falso alarme da janela saudável também é medido ali: a calibração vê o mesmo erro de estimativa
        # do deslocamento que o monitoramento depois de H verá (com a janela inteira, o z calibrado é centrado nela e o falso alarme sai otimista).
        self.calibrate_out_of_window = bool(calibrate_out_of_window)
        self.engine_z_scale_ = 1.0
        self.models: dict[int, object] = {}
        self.detectors: dict[int, StatisticalAnomalyDetector] = {}
        self.ewma_detector: StatisticalAnomalyDetector | None = None
        self.calibration_predictions_: pd.DataFrame | None = None
        self.fallback_model: object | None = None
        self.fallback_detector: StatisticalAnomalyDetector | None = None

    def _offset_w(self) -> int:
        """Janela (em ciclos) para estimar o deslocamento de cada motor; padrão: a janela saudável inteira."""
        return int(self.offset_window if self.offset_window is not None else self.healthy_cycle_limit)

    _MIN_OFFSET_ROWS = 3

    def _out_of_window(self) -> bool:
        return bool(self.engine_offset and self.calibrate_out_of_window and self._offset_w() < self.healthy_cycle_limit)

    def _engine_regime_offsets(self, frame: pd.DataFrame) -> pd.Series:
        """Deslocamento de cada (motor, regime) em unidades de z: média do z nos primeiros W ciclos do motor naquele regime.

        Por regime porque o σ de cada regime é diferente (a mesma diferença física vira z diferente) e misturar regimes numa média única
        distorcia a previsão nos sensores de pouca variância. Regimes com menos de 3 linhas na janela ficam sem correção (0)."""
        win = frame.loc[frame["cycle"] <= self._offset_w()].groupby(["unit_id", "regime_id"])["z_score"].agg(["mean", "size"])
        return win["mean"].where(win["size"] >= self._MIN_OFFSET_ROWS, 0.0)

    @staticmethod
    def _lookup_offsets(frame: pd.DataFrame, offsets: pd.Series) -> pd.Series:
        idx = pd.MultiIndex.from_arrays([frame["unit_id"].to_numpy(), frame["regime_id"].to_numpy()])
        return pd.Series(offsets.reindex(idx).fillna(0.0).to_numpy(), index=frame.index)

    def fit(self, train_frame: pd.DataFrame) -> "VirtualSensor":
        """Fit regime-specific estimators on cycles within the healthy window with hierarchical fallback."""
        if "regime_id" not in train_frame:
            raise ValueError("train_frame must have regime_id assigned by a train-fitted regime model")
        if self.feature_cols is not None:
            features = list(self.feature_cols)
        elif self.mode == "exogenous":
            features = [c for c in CANONICAL_EXOGENOUS_FEATURES if c in train_frame.columns and c != self.target_sensor]
            if not features:
                features = [
                    column for column in train_frame.columns
                    if column not in {self.target_sensor, "unit_id", "cycle", "regime_id", "RUL", "subpart"}
                ]
        else:
            features = [
                column for column in train_frame.columns
                if column not in {self.target_sensor, "unit_id", "cycle", "regime_id", "RUL", "subpart"}
            ]
        self.feature_cols = list(features)
        healthy = train_frame.loc[train_frame["cycle"] <= self.healthy_cycle_limit]
        if healthy.empty:
            healthy = train_frame

        # Train global hierarchical fallback model across all healthy data
        self.fallback_model = clone(self.estimator)
        self.fallback_model.fit(healthy[self.feature_cols], healthy[self.target_sensor].to_numpy(dtype=float))
        fb_resid = healthy[self.target_sensor].to_numpy(dtype=float) - self.fallback_model.predict(healthy[self.feature_cols])
        self.fallback_detector = StatisticalAnomalyDetector(
            threshold_sigma=self.threshold_sigma,
            ewma_alpha=self.ewma_alpha,
            healthy_cycle_limit=self.healthy_cycle_limit,
            min_consecutive_breaches=self.min_consecutive_breaches,
        )
        self.fallback_detector.calibrate(fb_resid)

        self.models = {}
        self.detectors = {}
        calibration_parts = []

        all_regimes = sorted(train_frame["regime_id"].unique())
        for regime_id in all_regimes:
            rows = healthy.loc[healthy["regime_id"] == regime_id]
            groups = rows["unit_id"].to_numpy() if not rows.empty else np.array([])
            n_engines = np.unique(groups).size if groups.size else 0

            if len(rows) < 30 or n_engines < 2:
                warnings.warn(
                    f"Regime {regime_id} has {len(rows)} samples and {n_engines} engines (< 30 samples or < 2 engines); "
                    "using global hierarchical fallback estimator.",
                    UserWarning,
                )
                self.models[int(regime_id)] = self.fallback_model
                self.detectors[int(regime_id)] = self.fallback_detector
                if not rows.empty:
                    oof_rows = rows.copy()
                    pred = self.fallback_model.predict(rows[self.feature_cols])
                    oof_rows["prediction"] = pred
                    oof_rows["residual"] = rows[self.target_sensor] - pred
                    calibration_parts.append(oof_rows)
                continue

            n_splits = min(5, n_engines)
            oof_prediction = np.full(len(rows), np.nan)
            reg_features = rows[self.feature_cols]
            target = rows[self.target_sensor].to_numpy(dtype=float)

            for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(reg_features, target, groups):
                oof_model = clone(self.estimator)
                oof_model.fit(reg_features.iloc[train_idx], target[train_idx])
                oof_prediction[valid_idx] = oof_model.predict(reg_features.iloc[valid_idx])

            if not np.isfinite(oof_prediction).all():
                nan_mask = ~np.isfinite(oof_prediction)
                oof_prediction[nan_mask] = self.fallback_model.predict(reg_features.iloc[nan_mask])

            model = clone(self.estimator)
            model.fit(reg_features, target)
            self.models[int(regime_id)] = model
            oof_rows = rows.copy()
            oof_rows["prediction"] = oof_prediction
            oof_rows["residual"] = target - oof_prediction
            calibration_parts.append(oof_rows)
            detector = StatisticalAnomalyDetector(
                threshold_sigma=self.threshold_sigma,
                ewma_alpha=self.ewma_alpha,
                healthy_cycle_limit=self.healthy_cycle_limit,
                min_consecutive_breaches=self.min_consecutive_breaches,
            )
            detector.calibrate(oof_rows["residual"].to_numpy())
            self.detectors[int(regime_id)] = detector

        if not self.models:
            raise ValueError("No regime could be fitted or calibrated")

        if calibration_parts:
            self.calibration_predictions_ = pd.concat(calibration_parts, ignore_index=True).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
            calibration_z = []
            self.calibration_predictions_["z_score"] = np.nan
            for regime_id, rows in self.calibration_predictions_.groupby("regime_id", sort=False):
                det = self.detectors.get(int(regime_id), self.fallback_detector)
                z_scores = det.compute_z_scores(rows["residual"].to_numpy())
                self.calibration_predictions_.loc[rows.index, "z_score"] = z_scores
                calibration_z.append(z_scores)
            if self.engine_offset:
                # [LEAKAGE] o deslocamento de cada motor de treino usa só os primeiros W ciclos (W <= H); depois dele a escala
                # volta a ter desvio 1 para que o limiar de 3 sigma e o portão bruto mantenham o significado.
                cp = self.calibration_predictions_
                off = self._engine_regime_offsets(cp)
                z_adj = cp["z_score"] - self._lookup_offsets(cp, off)
                if self._out_of_window():
                    z_adj = z_adj[cp["cycle"] > self._offset_w()]
                self.engine_z_scale_ = max(float(z_adj.std()), 1e-8)
                calibration_z = [(z_adj / self.engine_z_scale_).to_numpy()]
            self.ewma_detector = StatisticalAnomalyDetector(
                threshold_sigma=self.threshold_sigma,
                ewma_alpha=self.ewma_alpha,
                healthy_cycle_limit=self.healthy_cycle_limit,
                min_consecutive_breaches=self.min_consecutive_breaches,
            )
            self.ewma_detector.calibrate_ewma_distribution(np.concatenate(calibration_z))
        else:
            self.ewma_detector = self.fallback_detector

        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return rows supported by a fitted regime model with prediction/residual columns."""
        if not self.models and self.fallback_model is None:
            raise RuntimeError("Fit the virtual sensor before predicting")
        if "regime_id" not in frame:
            raise ValueError("frame must have regime_id assigned")
        parts = []
        for regime_id, rows in frame.groupby("regime_id"):
            model = self.models.get(int(regime_id), self.fallback_model)
            part = rows.copy()
            part["prediction"] = model.predict(part[self.feature_cols])
            part["residual"] = part[self.target_sensor] - part["prediction"]
            parts.append(part)
        if not parts:
            return frame.iloc[0:0].assign(prediction=pd.Series(dtype=float), residual=pd.Series(dtype=float))
        result = pd.concat(parts).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        return result

    def evaluate(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Fit on training engines and score lifecycle anomalies on held-out engines."""
        self.fit(train_frame)
        lifecycle_predictions = self.predict(test_frame)
        calibrated_parts = []
        for regime_id, rows in lifecycle_predictions.groupby("regime_id", sort=False):
            detector = self.detectors.get(int(regime_id), self.fallback_detector)
            part = rows.copy()
            part["z_score"] = detector.compute_z_scores(part["residual"].to_numpy())
            calibrated_parts.append(part)
        if not calibrated_parts:
            raise ValueError("No held-out healthy rows were available to calibrate residuals")
        scored = pd.concat(calibrated_parts, ignore_index=True).sort_values(["unit_id", "cycle"])
        if self.engine_offset:
            # deslocamento do motor = média do z nos primeiros W ciclos dele (causal: o alarme só vale depois da janela saudável)
            off_row = self._lookup_offsets(scored, self._engine_regime_offsets(scored))
            sigma_row = scored["regime_id"].map(lambda r: self.detectors.get(int(r), self.fallback_detector).sigma_e).astype(float)
            scored["prediction"] = scored["prediction"] + off_row * sigma_row
            scored["residual"] = scored["residual"] - off_row * sigma_row
            scored["z_score"] = (scored["z_score"] - off_row) / self.engine_z_scale_
        if self.ewma_detector is None:
            raise RuntimeError("Fit the virtual sensor before scoring EWMA")
        scored["ewma_abs_z"] = np.nan
        records = []
        for unit_id, engine in scored.groupby("unit_id", sort=False):
            ordered = engine.sort_values("cycle")
            ewma = self.ewma_detector.ewma_abs_z(ordered["z_score"].to_numpy())
            scored.loc[ordered.index, "ewma_abs_z"] = ewma
            limits = self.ewma_detector.ewma_upper_limits(len(ewma))
            scored.loc[ordered.index, "ewma_ucl"] = limits

            healthy_mask = ordered["cycle"].to_numpy() <= self.healthy_cycle_limit
            if self._out_of_window():
                healthy_mask = healthy_mask & (ordered["cycle"].to_numpy() > self._offset_w())  # só fora da janela que estima o deslocamento
            monitoring_mask = ordered["cycle"].to_numpy() > self.healthy_cycle_limit

            far = float(np.mean(ewma[healthy_mask] >= limits[healthy_mask])) if healthy_mask.any() else 0.0
            premature = bool(np.any(ewma[healthy_mask] >= limits[healthy_mask])) if healthy_mask.any() else False

            # Online alarms restricted to cycle > healthy_cycle_limit with persistence
            raw_breaches_bool = np.abs(ordered["z_score"].to_numpy()) >= self.threshold_sigma
            breaches = (ewma >= limits) & raw_breaches_bool
            if self.min_consecutive_breaches > 1:
                online_breaches = breaches & monitoring_mask
                consec = (
                    pd.Series(online_breaches)
                    .rolling(window=self.min_consecutive_breaches, min_periods=self.min_consecutive_breaches)
                    .sum()
                    .to_numpy()
                )
                sustained_ends = np.flatnonzero(consec >= self.min_consecutive_breaches)
                online_alarm_positions = (
                    sustained_ends - (self.min_consecutive_breaches - 1)
                    if sustained_ends.size
                    else np.array([], dtype=int)
                )
            else:
                online_alarm_positions = np.flatnonzero(breaches & monitoring_mask)

            eol = int(ordered["cycle"].max())
            alarm_cycle = int(ordered.iloc[online_alarm_positions[0]]["cycle"]) if online_alarm_positions.size else None
            lead_time = eol - alarm_cycle if alarm_cycle is not None else 0

            records.append({
                "unit_id": int(unit_id),
                "eol_cycle": eol,
                "first_alarm_cycle": alarm_cycle,
                "lead_time_cycles": lead_time,
                "healthy_far": far,
                "premature_alarm": premature,
                "detected": alarm_cycle is not None,
            })
        per_engine = pd.DataFrame(records)
        detected_engines = per_engine.loc[per_engine["detected"]]
        mean_lead_time_detected = float(detected_engines["lead_time_cycles"].mean()) if not detected_engines.empty else 0.0

        summary = {
            "total_engines": len(per_engine),
            "detection_rate_pct": float(per_engine["detected"].mean() * 100.0),
            "mean_lead_time_cycles": float(per_engine["lead_time_cycles"].mean()),
            "mean_lead_time_detected_cycles": mean_lead_time_detected,
            "mean_healthy_false_alarm_pct": float(per_engine["healthy_far"].mean() * 100.0),
            "premature_alarm_pct": float(per_engine["premature_alarm"].mean() * 100.0),
        }
        return scored.reset_index(drop=True), per_engine, summary

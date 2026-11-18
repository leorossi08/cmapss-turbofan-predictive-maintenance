"""Multivariate residual monitor (MEWMA) over 13 sensors."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone

from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE, SENSOR_COLUMNS, SETTING_COLUMNS



MULTIVARIATE_SENSORS = ("T24", "T30", "T50", "P30", "Ps30", "phi", "Nf", "Nc", "NRc", "BPR", "htBleed", "W31", "W32")

class MultivariateResidualMonitor:
    """MEWMA chart on the standardized residuals of several virtual sensors (Lowry et al., 1992).

    Why: one sensor (T50) may miss a failure mode that shows up elsewhere (e.g. the fan fault in
    FD003/FD004). The MEWMA combines p residuals, accounting for their correlation:
        w_t   = lam * (z_t - mu) + (1 - lam) * w_{t-1},   w_0 = 0
        T2_t  = w_t' Sigma_t^-1 w_t,   Sigma_t = lam / (2 - lam) * (1 - (1 - lam)^(2t)) * Sigma
    [CALIBRATION][LEAKAGE] mu, Sigma and the control limit come from OUT-OF-FOLD healthy residuals of
    training engines; the limit is the empirical (1 - alpha) quantile of T2 on those healthy sequences,
    so no normality is assumed. An alarm needs ``k`` consecutive cycles above the limit after cycle H.
    """

    def __init__(self, sensors=MULTIVARIATE_SENSORS, lam: float = 0.10, alpha: float = 0.0027, k: int = 2,
                 healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT, random_state: int = DEFAULT_RANDOM_STATE,
                 estimator=None, mode: str = "exogenous", engine_offset: bool = True, offset_window: int | None = None,
                 calibrate_out_of_window: bool = False):
        self.estimator = estimator  # None = the VirtualSensor default (Extra Trees); cloned per sensor
        # "exogenous" (default, thesis detector): expected value from settings + ambient channels only.
        # "cross_sensor": expected value also from the OTHER sensors (a consistency check; a fault that moves
        # all sensors coherently is partly absorbed, an isolated sensor fault is not).
        if mode not in {"exogenous", "cross_sensor"}:
            raise ValueError("mode must be 'exogenous' or 'cross_sensor'")
        self.mode = mode
        self.sensors = tuple(sensors)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.k = int(k)
        self.healthy_cycle_limit = int(healthy_cycle_limit)
        self.random_state = random_state
        # [DEFAULT ON desde 2026-11-18] engine_offset=True remove de cada resíduo padronizado o nível próprio do motor: a média do z do motor,
        # por regime, nos primeiros W ciclos (W = offset_window, padrão H; mínimo de 3 linhas por regime). Causal: só a janela saudável do
        # próprio motor. É o mesmo tratamento do T50 (ver virtual_sensor.engine_offset); engine_offset=False reproduz o protocolo anterior.
        self.engine_offset = bool(engine_offset)
        self.offset_window = offset_window
        # [DEFAULT OFF] com offset_window < H: μ, Σ e o limite do T² saem só dos ciclos FORA da janela do deslocamento (W, H] (a recursão do T²
        # roda desde o ciclo 1, como no monitoramento), e o falso alarme da janela saudável é medido nesses mesmos ciclos.
        self.calibrate_out_of_window = bool(calibrate_out_of_window)

    def _out_of_window(self) -> bool:
        return bool(self.engine_offset and self.calibrate_out_of_window and self._offset_w() < self.healthy_cycle_limit)

    def residual_z(self, vs: "VirtualSensor", rows: pd.DataFrame) -> np.ndarray:
        """z do resíduo de um sensor com o deslocamento do motor removido (se ligado), sem renormalizar: é o que entra no T². ``rows`` = saída de ``vs.predict``."""
        return self._remove_offset(self._z(vs, rows).to_numpy(), rows)

    def chart_z(self, vs: "VirtualSensor", rows: pd.DataFrame) -> np.ndarray:
        """z do gráfico EWMA de um sensor isolado: o de ``residual_z`` na escala em que ``vs.ewma_detector`` foi calibrado."""
        z = self.residual_z(vs, rows)
        return z / vs.engine_z_scale_ if self.engine_offset else z

    def offset_uncovered_pct(self, frame: pd.DataFrame) -> float:
        """% das linhas monitoradas (ciclo > H) cujo (motor, regime) tem menos de 3 linhas na janela do deslocamento e fica sem correção."""
        w = self._offset_w()
        win = frame.loc[frame["cycle"] <= w].groupby(["unit_id", "regime_id"]).size()
        mon = frame.loc[frame["cycle"] > self.healthy_cycle_limit]
        idx = pd.MultiIndex.from_arrays([mon["unit_id"].to_numpy(), mon["regime_id"].to_numpy()])
        return float(100 * (win.reindex(idx).fillna(0).to_numpy() < 3).mean()) if len(mon) else float("nan")

    def _offset_w(self) -> int:
        return int(self.offset_window if self.offset_window is not None else self.healthy_cycle_limit)

    def _remove_offset(self, z: np.ndarray, keys: pd.DataFrame) -> np.ndarray:
        """Subtrai de ``z`` a média do próprio motor (por regime) nos primeiros W ciclos; ``keys`` traz unit_id, cycle e regime_id alinhados a ``z``."""
        if not self.engine_offset:
            return z
        df = pd.DataFrame({"unit_id": keys["unit_id"].to_numpy(), "regime_id": keys["regime_id"].to_numpy(), "cycle": keys["cycle"].to_numpy(), "z": np.asarray(z, dtype=float)})
        win = df.loc[df["cycle"] <= self._offset_w()].groupby(["unit_id", "regime_id"])["z"].agg(["mean", "size"])
        off = win["mean"].where(win["size"] >= 3, 0.0)
        idx = pd.MultiIndex.from_arrays([df["unit_id"].to_numpy(), df["regime_id"].to_numpy()])
        return df["z"].to_numpy() - off.reindex(idx).fillna(0.0).to_numpy()

    def _z(self, vs: "VirtualSensor", rows: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=rows.index)
        for regime_id, part in rows.groupby("regime_id"):
            det = vs.detectors.get(int(regime_id), vs.fallback_detector)
            out.loc[part.index] = det.compute_z_scores(part["residual"].to_numpy())
        return out

    def _t2(self, z: np.ndarray, units: np.ndarray) -> np.ndarray:
        lam, mu = self.lam, self.mu_
        out = np.empty(len(z))
        start = 0
        for _, n in pd.Series(units).groupby(units, sort=False).size().items():
            w = np.zeros(z.shape[1])
            for t in range(n):
                w = lam * (z[start + t] - mu) + (1 - lam) * w
                factor = lam / (2 - lam) * (1 - (1 - lam) ** (2 * (t + 1)))
                out[start + t] = float(w @ self.sigma_inv_ @ w) / factor
            start += n
        return out

    def fit(self, train_frame: pd.DataFrame) -> "MultivariateResidualMonitor":
        """``train_frame`` = training engines with ``regime_id`` (assigned by a train-fitted RegimeDecoupler)."""
        self.sensors_ = [s for s in self.sensors if s in train_frame.columns and train_frame[s].std() > 1e-6]
        self.sensors_all_ = [s for s in SENSOR_COLUMNS if s in train_frame.columns and train_frame[s].std() > 1e-6]
        self.virtual_ = {}
        cal = None
        for sensor in self.sensors_:
            cross = None
            if self.mode == "cross_sensor":
                cross = [*SETTING_COLUMNS, *[c for c in self.sensors_all_ if c != sensor]]
            vs = VirtualSensor(target_sensor=sensor, mode="exogenous", feature_cols=cross, healthy_cycle_limit=self.healthy_cycle_limit,
                               random_state=self.random_state, estimator=clone(self.estimator) if self.estimator is not None else None,
                               engine_offset=self.engine_offset, offset_window=self.offset_window,
                               calibrate_out_of_window=self.calibrate_out_of_window)  # o T² só usa fit/predict/detectors; o limite da EWMA do sensor isolado sai no mesmo tratamento
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vs.fit(train_frame)
            self.virtual_[sensor] = vs
            rows = vs.calibration_predictions_.set_index(["unit_id", "cycle"])
            flat = rows.reset_index()
            z = self._remove_offset(self._z(vs, flat).to_numpy(), flat)
            col = pd.Series(z, index=rows.index, name=sensor)
            cal = col.to_frame() if cal is None else cal.join(col, how="inner")
        cal = cal.sort_index()
        zc = cal.to_numpy()
        cyc_cal = cal.index.get_level_values("cycle").to_numpy()
        keep = cyc_cal > self._offset_w() if self._out_of_window() else np.ones(len(zc), dtype=bool)
        self.mu_ = zc[keep].mean(axis=0)
        sigma = np.cov(zc[keep], rowvar=False) + 1e-6 * np.eye(zc.shape[1])
        self.sigma_inv_ = np.linalg.inv(sigma)
        t2 = self._t2(zc, cal.index.get_level_values("unit_id").to_numpy())
        self.ucl_ = float(np.quantile(t2[keep], 1 - self.alpha))
        # kept for calibration-noise studies (engine-cluster bootstrap of the UCL, healthy_window_study.py)
        self.calibration_t2_ = pd.DataFrame({"unit_id": cal.index.get_level_values("unit_id").to_numpy(), "t2": t2})[keep].reset_index(drop=True)
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-cycle T2 statistic and breach flag for engines in ``frame`` (needs ``regime_id``)."""
        cols = {}
        index = None
        for sensor, vs in self.virtual_.items():
            pred = vs.predict(frame).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
            cols[sensor] = self.residual_z(vs, pred)
            index = pred[["unit_id", "cycle"]]
        out = index.copy()
        z = np.column_stack([cols[s] for s in self.virtual_])
        out["t2"] = self._t2(z, out["unit_id"].to_numpy())
        out["ucl"] = self.ucl_
        out["breach"] = out["t2"] >= self.ucl_
        return out

    def alarms(self, scored: pd.DataFrame) -> tuple[dict[int, int | None], pd.DataFrame, dict]:
        """First persistent alarm per engine + fleet summary with the same metrics as the univariate SPC."""
        H, k = self.healthy_cycle_limit, self.k
        first, rows = {}, []
        for unit, eng in scored.groupby("unit_id"):
            cyc = eng["cycle"].to_numpy()
            b = eng["breach"].to_numpy(dtype=bool)
            run, alarm = 0, None
            for i in range(len(b)):
                run = run + 1 if (b[i] and cyc[i] > H) else 0
                if run >= k:
                    alarm = int(cyc[i - k + 1])
                    break
            healthy = cyc <= H
            if self._out_of_window():
                healthy = healthy & (cyc > self._offset_w())  # falso alarme só fora da janela que estima o deslocamento
            first[int(unit)] = alarm
            rows.append({"unit_id": int(unit), "eol_cycle": int(cyc.max()), "first_alarm_cycle": alarm,
                         "lead_time_cycles": int(cyc.max()) - alarm if alarm is not None else 0,
                         "healthy_far": float(b[healthy].mean()) if healthy.any() else 0.0,
                         "premature_alarm": bool(b[healthy].any()) if healthy.any() else False,
                         "detected": alarm is not None})
        per = pd.DataFrame(rows)
        det = per.loc[per["detected"]]
        summary = {
            "total_engines": len(per),
            "detection_rate_pct": float(per["detected"].mean() * 100),
            "mean_lead_time_cycles": float(per["lead_time_cycles"].mean()),
            "mean_lead_time_detected_cycles": float(det["lead_time_cycles"].mean()) if len(det) else 0.0,
            "mean_healthy_false_alarm_pct": float(per["healthy_far"].mean() * 100),
            "premature_alarm_pct": float(per["premature_alarm"].mean() * 100),
            "n_sensors": len(self.virtual_),
        }
        return first, per, summary

"""Testes unitários dos estudos complementares da revisão de banca (Exp. A, B, C e D)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm_cmaps.detection.multivariate import MULTIVARIATE_SENSORS, MultivariateResidualMonitor
from pdm_cmaps.domain.schema import SETTING_COLUMNS
from pdm_cmaps.experiments.estudos_complementares import (
    IForestEwmaMonitor,
    PCAEwmaMonitor,
    _strip_endogenous_exog,
    apply_cqr_interval,
    fit_cqr_from_quantiles,
    fit_failure_mode_kmeans,
    flicker_metrics,
    levels_with_hysteresis,
)
from pdm_cmaps.monitoring.alerts import CriticalityPolicy


def test_cqr_interval_calibration_and_monotonicity():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 125, 1000)
    noise = rng.normal(0, 5.0 + 0.08 * x)
    y = np.clip(x + noise, 0.0, 125.0)
    q_lo = x - 1.2 * (5.0 + 0.08 * x)
    q_hi = x + 1.2 * (5.0 + 0.08 * x)
    corr = fit_cqr_from_quantiles(y, q_lo, q_hi, alpha=0.10)
    lo, hi = apply_cqr_interval(q_lo, q_hi, corr, rul_cap=125.0)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    assert np.all(lo <= hi)
    assert np.all((lo >= 0.0) & (hi <= 125.0))
    assert cov >= 0.89


def test_hysteresis_eliminates_oscillations_inside_band():
    policy = CriticalityPolicy(attention=60.0, warning=30.0, critical=15.0, persistence=2)
    # Sequência oscila ao redor do limiar crítico 15 (entre 14 e 18): sem histerese oscila toda hora; com banda 5 fica estável
    rul = np.array([25, 22, 14, 18, 14, 18, 14, 18, 13, 19, 25], dtype=float)
    raw_no_hyst = policy.level_from_rul(rul)
    with_hyst = levels_with_hysteresis(rul, policy, band=5.0)
    assert np.sum(np.diff(with_hyst) < 0) == 1  # só sai de CRITICAL quando passa de 15 + 5 = 20
    assert np.sum(np.diff(raw_no_hyst) < 0) == 4


def test_flicker_metrics_detects_sawtooth_jumps():
    df = pd.DataFrame({
        "unit_id": [1] * 6,
        "cycle": [1, 2, 3, 4, 5, 6],
        "RUL_pred": [40.0, 41.0, 42.0, 40.0, 39.0, 38.0],
        "RUL_lower": [12.0, 35.0, 11.0, 10.0, 36.0, 10.0],
        "level_interval": [3, 1, 3, 3, 1, 3],
    })
    m = flicker_metrics(df, "interval")
    assert m["Lower_jumps_ge15_per_engine"] == 4.0
    assert m["Reversals_per_engine"] == 2.0


def test_pca_ewma_and_iforest_ewma_respond_to_shift():
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(42)
    units_cal = np.repeat(np.arange(1, 21), 50)
    z_cal = rng.normal(0.0, 1.0, size=(len(units_cal), 13))

    pca_mon = PCAEwmaMonitor(lam=0.10, variance_ratio=0.90, alpha=0.0027, random_state=42).fit(z_cal, units_cal)
    forest = IsolationForest(n_estimators=100, random_state=42).fit(z_cal)
    if_mon = IForestEwmaMonitor(forest, lam=0.10, alpha=0.0027).fit(z_cal, units_cal)

    # motor de teste saudável até ciclo 30, depois degrau em 4 sensores
    units_te = np.ones(80, dtype=int)
    z_te = rng.normal(0.0, 1.0, size=(80, 13))
    z_te[30:, :4] += 2.5

    r_pca = pca_mon.score_ratio(z_te, units_te)
    r_if = if_mon.score_ratio(z_te, units_te)
    assert r_pca[:25].mean() < 0.8 and r_pca[-15:].mean() > 1.2
    assert r_if[:25].mean() < 0.8 and r_if[-15:].mean() > 1.0


def test_strip_endogenous_removes_p15_and_epr_but_preserves_monitored_sensors():
    rng = np.random.default_rng(1)
    n_units, T = 12, 80
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, T + 1):
            row = {"unit_id": u, "cycle": c, "regime_id": 0, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0,
                   "T2": 518.67, "P2": 14.62, "P15": 21.61 + rng.normal(0, 0.01), "epr": 1.3 + rng.normal(0, 0.005),
                   "Nf_dmd": 2388.0, "PCNfR_dmd": 100.0}
            for s in MULTIVARIATE_SENSORS:
                row[s] = 100.0 + rng.normal(0, 0.5)
            rows.append(row)
    df = pd.DataFrame(rows)
    stripped = _strip_endogenous_exog(df)
    assert "P15" not in stripped.columns and "epr" not in stripped.columns
    mv = MultivariateResidualMonitor(healthy_cycle_limit=25, random_state=1).fit(stripped)
    assert len(mv.sensors_) == 13
    assert "P15" not in mv.virtual_["T50"].feature_cols and "epr" not in mv.virtual_["T50"].feature_cols


def test_failure_mode_kmeans_recovers_hpc_and_fan_modes():
    rng = np.random.default_rng(9)
    rows = []
    for u in range(1, 21):
        is_fan = u > 10
        for c in range(1, 91):
            prog = max(0.0, (c - 50) / 40.0)
            row = {"unit_id": u, "cycle": c, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
            for s in MULTIVARIATE_SENSORS:
                row[s] = 100.0 + rng.normal(0, 0.2)
            row["T50"] += 5.0 * prog
            row["P30"] += (6.0 if is_fan else -3.0) * prog
            row["phi"] += (5.0 if is_fan else -2.5) * prog
            row["BPR"] += (-0.2 if is_fan else 0.15) * prog
            rows.append(row)
    df = pd.DataFrame(rows)
    _, _, _, z_tr, _ = fit_failure_mode_kmeans(df, n_regimes=1, sensors=MULTIVARIATE_SENSORS, seed=42)
    assert (z_tr.loc[1:10, "modo"] == "HPC").all()
    assert (z_tr.loc[11:20, "modo"] == "Fan").all()

"""Cartas de controle (EWMA, 3σ, envelope), MEWMA e índice de saúde."""


import numpy as np
import pandas as pd
from pdm_cmaps.detection.health_index import HealthIndexFuser
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.detection.spc import EnvelopeAnomalyDetector, StatisticalAnomalyDetector
from pdm_cmaps.detection.virtual_sensor import VirtualSensor

from factories import make_synthetic_cmapss


def test_envelope_anomaly_detector():
    from pdm_cmaps.detection.spc import EnvelopeAnomalyDetector

    detector = EnvelopeAnomalyDetector(k_sigma=2.5, healthy_cycle_limit=30, window_size=4, min_consecutive_breaches=2)
    np.random.seed(42)
    healthy_actual = 1400.0 + np.random.randn(30) * 1.5
    healthy_pred = 1400.0 + np.random.randn(30) * 0.2
    detector.calibrate(healthy_actual, healthy_pred)

    assert detector.sigma_e > 0
    assert detector.mu_amp > 0

    cycles = np.arange(1, 101)
    actual = 1400.0 + np.random.randn(100) * 1.5
    # Add severe thermal runaway / degradation starting at cycle 75
    actual[74:] += np.linspace(4.0, 30.0, 26)
    pred = np.full(100, 1400.0)

    df = pd.DataFrame({"unit_id": 1, "cycle": cycles, "T50": actual, "pred": pred})
    scored_env, summary_env = detector.score_time_series_envelope(df, "T50", "pred")

    assert "envelope_upper" in scored_env.columns
    assert "envelope_lower" in scored_env.columns
    assert summary_env["detected"] is True
    assert summary_env["first_alarm_cycle"] is not None
    assert summary_env["first_alarm_cycle"] >= 70
    assert summary_env["eol_cycle"] == 100
    assert summary_env["lead_time_cycles"] == 100 - summary_env["first_alarm_cycle"]

    scored_geom, summary_geom = detector.score_geometric_envelope(df, "T50")
    assert "geom_upper_hull" in scored_geom.columns
    assert "geom_lower_hull" in scored_geom.columns
    assert summary_geom["detected"] is True
    assert summary_geom["first_alarm_cycle"] is not None


def test_online_alarm_restriction_and_disaggregated_metrics():
    """Verify operational alarm is restricted to online monitoring phase (cycle > H)."""
    detector = StatisticalAnomalyDetector(healthy_cycle_limit=50, threshold_sigma=3.0)
    # Calibrate on healthy residuals
    np.random.seed(42)
    healthy_res = np.random.randn(200) * 0.5
    detector.calibrate(healthy_res)

    # Engine with early spike at cycle 10, but true failure starting at cycle 80
    cycles = np.arange(1, 101)
    res = np.random.randn(100) * 0.5
    res[9] = 10.0  # Spike at cycle 10 (healthy phase, cycle <= 50)
    res[79:] += np.linspace(2.0, 10.0, 21)  # Degraded phase (cycle > 50)

    engine_df = pd.DataFrame({"unit_id": 1, "cycle": cycles, "residual": res})
    scored, summary = detector.score_engine(engine_df)

    assert "first_alarm_cycle" in summary
    assert "premature_alarm_pct" in summary or "premature_alarm" in summary
    # The operational first alarm must occur during online monitoring (> 50)
    assert summary["first_alarm_cycle"] > 50
    assert summary["detected"] is True
    assert summary["healthy_far"] > 0.0 or summary["healthy_far_pct"] > 0.0


def test_envelope_regime_standardized_corridor():
    """Verify EnvelopeAnomalyDetector handles multi-regime FD004 without corridor collapse."""
    detector = EnvelopeAnomalyDetector(healthy_cycle_limit=30, k_sigma=3.0)
    # Calibrate
    actual = 1400.0 + np.random.randn(50) * 2.0
    pred = 1400.0 + np.random.randn(50) * 0.5
    detector.calibrate(actual, pred)

    # Test with regime-normalized residuals
    cycles = np.arange(1, 80)
    res = np.random.randn(79) * 0.5
    res[50:] += np.linspace(1.0, 8.0, 29)
    df = pd.DataFrame({"unit_id": 1, "cycle": cycles, "residual": res, "T50": 1400.0 + res, "pred": 1400.0})
    scored, summary = detector.score_time_series_envelope(df, "T50", "pred")
    assert summary["detected"] is True
    assert summary["first_alarm_cycle"] > 30


def test_todo1_spc_persistence_eliminates_transient_spikes():
    """TODO-1 & F2: Isolated transient spikes do not trigger first_alarm_cycle with min_consecutive_breaches=3."""
    detector = StatisticalAnomalyDetector(healthy_cycle_limit=50, threshold_sigma=3.0, ewma_alpha=0.10, min_consecutive_breaches=3)
    np.random.seed(42)
    healthy_res = np.random.randn(200) * 0.2
    detector.calibrate(healthy_res)

    cycles = np.arange(1, 101)
    res = np.random.randn(100) * 0.2
    # Single isolated massive transient spike at cycle 65 (cycle > 50)
    res[64] = 15.0

    engine_df = pd.DataFrame({"unit_id": 1, "cycle": cycles, "residual": res})
    scored, summary = detector.score_engine(engine_df)
    # The isolated spike at cycle 65 should NOT trigger an alarm because it lasts only 1 cycle
    assert summary["detected"] is False
    assert summary["first_alarm_cycle"] is None

    # Now add 3 consecutive breaches at cycles 80, 81, 82
    res[79:83] = 12.0
    engine_df2 = pd.DataFrame({"unit_id": 1, "cycle": cycles, "residual": res})
    scored2, summary2 = detector.score_engine(engine_df2)
    assert summary2["detected"] is True
    assert summary2["first_alarm_cycle"] == 80  # Offsets to the start of the 3-cycle sequence


def test_todo2_ewma_alpha_default_and_inertia():
    """TODO-2 & F3: Verify default ewma_alpha is 0.10 across detector classes."""
    det = StatisticalAnomalyDetector()
    assert det.ewma_alpha == 0.10

    vs = VirtualSensor()
    assert vs.ewma_alpha == 0.10
    assert vs.min_consecutive_breaches == 3


def test_todo3_geometric_envelope_decoupled_multiregime():
    """TODO-3 & F1: Geometric envelope over decoupled signal restores lead time in multi-regime jumps."""
    detector = EnvelopeAnomalyDetector(healthy_cycle_limit=30, k_sigma=3.0, min_consecutive_breaches=2)
    # Calibrate on healthy residual variance
    actual = 1400.0 + np.random.randn(50) * 0.5
    pred = np.full(50, 1400.0)
    detector.calibrate(actual, pred)

    # Multi-regime jumps: BPR oscillates between 8.4 and 10.8 due to altitude jumps
    cycles = np.arange(1, 101)
    altitude_jump = np.tile([0.0, 2.4], 50)
    base_bpr = 8.4 + altitude_jump
    # Add real mechanical degradation starting at cycle 75
    deg = np.zeros(100)
    deg[74:] = np.linspace(0.5, 3.0, 26)
    actual_bpr = base_bpr + deg
    # Virtual sensor predicts the altitude jump accurately
    pred_bpr = base_bpr

    df = pd.DataFrame({
        "unit_id": 179,
        "cycle": cycles,
        "BPR": actual_bpr,
        "prediction": pred_bpr,
        "residual": actual_bpr - pred_bpr,
    })

    # Decoupled mode detects the true degradation
    scored, summary = detector.score_geometric_envelope(df, "BPR", use_decoupled_signal=True)
    assert summary["detected"] is True
    assert summary["first_alarm_cycle"] is not None
    assert summary["first_alarm_cycle"] >= 74
    assert summary["lead_time_cycles"] > 0
    assert summary["decoupled_mode"] == "residual"


def test_todo5_health_index_fuser():
    """TODO-5 & F7: HealthIndexFuser synthesizes multi-sensor degradation into monotonic [1.0 -> 0.0] HI."""
    df = make_synthetic_cmapss(n_engines=8, seed=42)
    decoupler = RegimeDecoupler(n_clusters=6, random_state=42).fit(df)
    df_trans = decoupler.transform(df)
    decoupler.fit_sensor_scaler(df_trans, ["T50", "P30", "T24"])
    df_norm = decoupler.standardize_sensors(df_trans, ["T50", "P30", "T24"])

    fuser = HealthIndexFuser(healthy_cycle_limit=30, ewma_alpha=0.10, random_state=42)
    scored, per_engine, summary = fuser.evaluate(df_norm, df_norm)

    assert "health_index" in scored.columns
    assert (scored["health_index"] >= 0.0).all() and (scored["health_index"] <= 1.0).all()

    # Verify healthy phase has high HI and end-of-life has lower HI
    for _, eng in scored.groupby("unit_id"):
        healthy_hi = eng.loc[eng["cycle"] <= 30, "health_index"].mean()
        eol_hi = eng.tail(5)["health_index"].mean()
        assert healthy_hi > eol_hi


def test_multivariate_monitor_alarms_on_drift_not_on_healthy_engines():
    import numpy as np
    import pandas as pd
    from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor

    rng = np.random.default_rng(3)
    rows = []
    for unit in range(1, 13):
        drift = unit > 8  # engines 9-12 degrade after cycle 70
        for cycle in range(1, 121):
            base = {"unit_id": unit, "cycle": cycle, "regime_id": 0, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0,
                    "T2": 518.67, "P2": 14.62, "P15": 21.6, "epr": 1.3, "Nf_dmd": 2388.0, "PCNfR_dmd": 100.0}
            shift = 0.08 * max(0, cycle - 70) if drift else 0.0
            for j, s in enumerate(("T24", "T30", "T50", "P30")):
                base[s] = 100.0 + j + rng.normal(0, 1.0) + (shift if s in ("T50", "P30") else 0.0)
            rows.append(base)
    df = pd.DataFrame(rows)
    train = df.loc[df["unit_id"] <= 8]
    test = df.loc[df["unit_id"] > 4]
    # engine_offset=False: aqui se testa o limite NOMINAL. Com o deslocamento ligado (padrão do protocolo principal) o limite calibrado em janelas já
    # centradas fica baixo e motores saudáveis alarmam (3 de 4 neste exemplo); a MEWMA com o deslocamento só vale a falso alarme igualado.
    mon = MultivariateResidualMonitor(sensors=("T24", "T30", "T50", "P30"), healthy_cycle_limit=50, engine_offset=False).fit(train)
    first, per, summary = mon.alarms(mon.score(test))
    assert all(first[u] is not None for u in (9, 10, 11, 12))
    assert sum(first[u] is not None for u in (5, 6, 7, 8)) <= 1


def test_first_persistent_requires_k_consecutive_breaches_after_window():
    import numpy as np
    from pdm_cmaps.reporting.plots import _first_persistent

    cycles = np.arange(1, 11)
    mask = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert _first_persistent(mask, cycles, k=2, after=0) == 1
    assert _first_persistent(mask, cycles, k=2, after=2) == 7
    assert _first_persistent(mask, cycles, k=4, after=0) is None


def test_envelope_runs_ignore_isolated_breaches_and_alert_on_persistent_ones():
    from pdm_cmaps.experiments import envelope_ignored_cases as eic

    cyc = np.arange(1, 121)
    breach = np.zeros(120, bool)
    breach[[10, 60, 90]] = True        # cycle 11 inside the calibration window; 61 and 91 isolated after it
    breach[100:103] = True              # persistent run (3 cycles) -> alert
    r = pd.DataFrame({"Subset": "FD001", "unit_id": 1, "cycle": cyc, "breach": breach, "excess_sigma": np.where(breach, 0.5, 0.0), "RUL": 125.0})
    runs = eic.breach_runs(r, persist=2, h=50)
    assert runs["length"].tolist() == [1, 1, 1, 3]
    assert runs["monitored"].tolist() == [False, True, True, True]                       # first one is in the calibration window
    assert runs["alert"].tolist() == [False, False, False, True]                         # only the persistent run alerts
    assert runs.loc[runs["alert"], "alert_cycle"].iloc[0] == 101 + 1                     # issued when the 2nd breach arrives (run starts at cycle 101)
    summ = eic.summarize(r.assign(T50=0.0), runs, persist=2, h=50)
    assert summ.loc[0, "runs_ignored"] == 2 and summ.loc[0, "runs_alert"] == 1 and summ.loc[0, "alert_delay_cycles_by_rule"] == 1

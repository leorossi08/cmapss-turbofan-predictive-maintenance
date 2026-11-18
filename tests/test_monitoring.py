"""Camada de monitoramento: conformal, criticidade, alertas, custo e relatórios."""

import numpy as np
import pandas as pd

from factories import _toy_timeline


def test_conformal_interval_reaches_nominal_coverage_on_heldout_data():
    """[CALIBRATION] Split-conformal coverage must be close to nominal on fresh data."""
    from pdm_cmaps.monitoring.uncertainty import ConformalRULInterval, interval_metrics

    rng = np.random.default_rng(1)

    def draw(n):
        y = rng.uniform(0, 125, n)
        pred = y + rng.normal(0, 3 + 0.15 * y)  # heteroscedastic: larger error early in life
        return y, np.clip(pred, 0, 125)

    y_cal, p_cal = draw(6000)
    y_te, p_te = draw(6000)
    lo, hi = ConformalRULInterval(alpha=0.10).fit(y_cal, p_cal).predict_interval(p_te)
    metrics = interval_metrics(y_te, lo, hi)
    assert 0.86 <= metrics["coverage"] <= 0.94
    assert np.all(lo >= 0) and np.all(hi <= 125) and np.all(lo <= hi)


def test_debounce_is_causal_and_requires_persistence():
    from pdm_cmaps.monitoring.alerts import debounce

    raw = np.array([0, 3, 0, 2, 2, 2, 3])
    out = debounce(raw, 2)
    assert list(out) == [0, 0, 0, 0, 2, 2, 2]
    # the output at t depends only on raw[:t+1]
    for t in range(len(raw)):
        assert debounce(raw[: t + 1], 2)[-1] == out[t]


def test_anomaly_escalates_only_when_interval_already_shows_risk():
    from pdm_cmaps.monitoring.alerts import (
        NORMAL,
        CriticalityPolicy,
        build_alert_timeline,
    )

    # healthy engine (RUL high) with an SPC alarm must NOT create a maintenance-level alert
    frame = pd.DataFrame({
        "unit_id": 1, "cycle": range(1, 11), "RUL_true": 120.0, "RUL_pred": 120.0, "RUL_lower": 100.0, "RUL_upper": 125.0,
    })
    tl = build_alert_timeline(frame, CriticalityPolicy(), {1: 3})
    assert (tl["level_interval+anomaly"] == NORMAL).all()
    assert tl.loc[tl["cycle"] >= 3, "anomaly_active"].all()


def test_strategy_evaluation_reports_missed_critical_and_false_alarm_rates():
    from pdm_cmaps.monitoring.alerts import evaluate_strategies

    tl, policy = _toy_timeline()
    table = evaluate_strategies(tl, policy)
    assert set(table["Strategy"]) == {"point", "interval", "interval+anomaly"}
    # the conservative lower bound must not miss more critical cycles than the point estimate
    point = table.set_index("Strategy").loc["point", "Missed_critical_rate"]
    interval = table.set_index("Strategy").loc["interval", "Missed_critical_rate"]
    assert interval <= point


def test_rendered_message_contains_only_numbers_from_facts():
    """The message is a fixed template over the facts: every number it prints must come from the facts."""
    import re

    from pdm_cmaps.monitoring.reports import build_facts, render_report

    tl, policy = _toy_timeline()
    facts = build_facts(tl.loc[tl["cycle"] <= 70], policy)
    text = render_report(facts)
    allowed = {facts["unit_id"], facts["cycle"], facts["rul_estimate"], facts["rul_lower"], facts["rul_upper"],
               facts["interval_coverage_pct"], facts["trend_cycles_per_cycle"], facts["trend_window_cycles"]}
    if facts["cycles_since_anomaly"] is not None:
        allowed.add(facts["cycles_since_anomaly"])
    for token in re.findall(r"-?\d+(?:\.\d+)?", text):
        assert float(token) in {float(a) for a in allowed}, token
    assert facts["level_label_pt"] in text and facts["recommended_action_pt"] in text


def test_facts_use_only_past_rows():
    from pdm_cmaps.monitoring.reports import build_facts

    tl, policy = _toy_timeline()
    past = tl.loc[tl["cycle"] <= 50]
    assert build_facts(past, policy)["cycle"] == 50
    assert build_facts(tl, policy)["cycle"] == int(tl["cycle"].max())


def test_mondrian_groups_fall_back_to_rul_bins_for_small_cells():
    from pdm_cmaps.monitoring.uncertainty import ConformalRULInterval

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 125, 3000)
    y = p + rng.normal(0, 10, p.size)
    groups = np.where(np.arange(p.size) < 2990, 0, 1)  # group 1 has only 10 rows -> falls back
    model = ConformalRULInterval(alpha=0.10).fit(y, p, groups=groups)
    assert all(key[0] == 0 for key in model.group_offsets_)
    lo_g, hi_g = model.predict_interval(p[:5], groups=np.ones(5, dtype=int))
    lo, hi = model.predict_interval(p[:5])
    assert np.allclose(lo_g, lo) and np.allclose(hi_g, hi)


def test_normalized_conformal_keeps_coverage_and_adapts_width():
    from pdm_cmaps.monitoring.uncertainty import (
        NormalizedConformalRULInterval,
        interval_metrics,
    )

    rng = np.random.default_rng(1)
    n = 6000
    p = rng.uniform(20, 110, n)
    spread = rng.uniform(1, 10, n)
    y = p + rng.normal(0, spread)
    cal, te = slice(0, 4000), slice(4000, n)
    model = NormalizedConformalRULInterval(alpha=0.10).fit(y[cal], p[cal], spread[cal])
    lo, hi = model.predict_interval(p[te], spread[te])
    assert interval_metrics(y[te], lo, hi)["coverage"] >= 0.87
    width = hi - lo
    assert width[spread[te] > 8].mean() > 1.5 * width[spread[te] < 3].mean()  # floor = median spread damps the ratio


def test_causal_local_spread_uses_only_the_past_of_each_engine():
    from pdm_cmaps.monitoring.uncertainty import causal_local_spread

    pred = np.array([10.0, 10.0, 10.0, 50.0, 0.0, 0.0])
    units = np.array([1, 1, 1, 1, 2, 2])
    out = causal_local_spread(pred, units, window=3)
    assert out[2] == 0.0          # three equal values so far
    assert out[3] > 0.0           # the jump enters only at its own cycle
    assert out[4] == 0.0 and out[5] == 0.0  # engine 2 does not see engine 1


def test_extra_alarm_maps_add_strategies():
    from pdm_cmaps.monitoring.alerts import evaluate_strategies, strategy_names

    tl, policy = _toy_timeline()
    from pdm_cmaps.monitoring.alerts import build_alert_timeline
    base = tl[["unit_id", "cycle", "RUL_true", "RUL_pred", "RUL_lower", "RUL_upper"]]
    tl2 = build_alert_timeline(base, policy, {}, {"anomaly_mv": {7: 1}})
    assert "interval+anomaly_mv" in strategy_names(tl2)
    table = evaluate_strategies(tl2, policy)
    assert set(table["Strategy"]) == {"point", "interval", "interval+anomaly", "interval+anomaly_mv"}


def test_maintenance_cost_counts_failures_and_lead_time():
    from pdm_cmaps.monitoring.alerts import CRITICAL, NORMAL, maintenance_cost

    # one engine, EOL at cycle 100, strategy reaches CRITICAL at cycle 95
    tl = pd.DataFrame({"unit_id": 1, "cycle": range(1, 101)})
    tl["level_point"] = [CRITICAL if c >= 95 else NORMAL for c in tl["cycle"]]
    planned = maintenance_cost(tl, "point", cost_ratio=10, lead=0)
    assert planned["Failures"] == 0 and planned["Mean_unused_cycles"] == 5
    late = maintenance_cost(tl, "point", cost_ratio=10, lead=10)  # executed at 105 > EOL -> failure
    assert late["Failures"] == 1
    assert late["Relative_to_oracle"] > planned["Relative_to_oracle"]

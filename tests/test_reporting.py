"""Figuras (matplotlib e Plotly)."""

from pathlib import Path

import numpy as np
import pandas as pd

from factories import _toy_timeline


def skipped_ids(skipped):
    return [c for c, _ in skipped]


def test_todo7_todo8_plots_execute_without_error(tmp_path: Path):
    """TODO-7 & TODO-8: Test plots execute cleanly with multi-regime discrete points and physical units."""
    from pdm_cmaps.reporting.plots import (
        plot_failure_detection,
        plot_time_series_envelope_detection,
    )
    cycles = np.arange(1, 40)
    trace = pd.DataFrame({
        "unit_id": 1,
        "cycle": cycles,
        "regime_id": cycles % 6,
        "BPR": 8.4 + (cycles % 6) * 0.4,
        "prediction": 8.4 + (cycles % 6) * 0.4,
        "ewma_abs_z": np.linspace(0.5, 3.5, 39),
        "ewma_ucl": np.full(39, 3.0),
        "envelope_upper": np.full(39, 11.5),
        "envelope_lower": np.full(39, 8.0),
        "out_of_envelope": cycles > 35,
        "breach_distance": np.maximum(0, cycles - 35) * 0.1,
    })
    engines = pd.DataFrame([{
        "unit_id": 1,
        "eol_cycle": 39,
        "first_alarm_cycle": 36,
        "lead_time_cycles": 3,
        "detected": True,
    }])
    summary = {
        "unit_id": 1,
        "eol_cycle": 39,
        "first_alarm_cycle": 36,
        "lead_time_cycles": 3,
        "envelope_margin": 1.5,
        "in_envelope_pct": 90.0,
    }

    p1 = plot_failure_detection(trace, engines, tmp_path / "fail.png", healthy_cycle_limit=20, sensor_name="BPR")
    assert p1.exists()

    p2 = plot_time_series_envelope_detection(trace, summary, tmp_path / "env.png", sensor_name="BPR", healthy_cycle_limit=20)
    assert p2.exists()


def test_monitoring_figures_render_and_follow_style_tokens(tmp_path):
    """Every thesis figure function writes a PNG; viz tokens keep the validated categorical order."""
    import numpy as np
    import pandas as pd
    from pdm_cmaps.reporting import monitoring_figures as mf
    from pdm_cmaps.reporting.viz_style import SERIES, STATUS

    assert SERIES == ("#2a78d6", "#eb6834", "#1baf7a")  # validated all-pairs (see viz_style.py)
    assert len(STATUS) == 4

    tl, _ = _toy_timeline()
    mf.plot_rul_trajectories(tl, [7], tmp_path / "traj.png")
    trace = tl.assign(z_score=np.linspace(-1, 5, len(tl)), ewma_abs_z=np.linspace(0.5, 3, len(tl)), ewma_ucl=1.2)
    mf.plot_residual_control_chart(trace, 7, 45, tmp_path / "control.png", 10)
    sens = pd.DataFrame({
        "Strategy": ["point", "interval", "interval+anomaly"] * 2, "Policy": ["critical<=10"] * 3 + ["critical<=15"] * 3,
        "False_alarm_rate_normal": [0.0, 0.01, 0.01, 0.0, 0.02, 0.02], "Missed_critical_rate": [0.03, 0.0, 0.0, 0.02, 0.0, 0.0],
    })
    mf.plot_strategy_tradeoff(sens, tmp_path / "tradeoff.png")
    mf.plot_strategy_tradeoff_repeated(sens.assign(Subset="FD001", Seed=1), tmp_path / "repeated.png")
    ci = pd.DataFrame({"Model": ["A", "B"], "R2_Observed": [0.8, 0.7], "R2_CI_Lower": [0.7, 0.6], "R2_CI_Upper": [0.85, 0.78]})
    mf.plot_model_ci(ci, tmp_path / "ci.png")
    mf.plot_regime_variance(pd.DataFrame({"Subset": ["FD001", "FD002"], "Regimes": [1, 6], "Between_regime_pct": [0.0, 99.6]}), tmp_path / "regime.png")
    assert len(list(tmp_path.glob("*.png"))) == 6


def test_plotly_hub_builds_index_skips_missing_and_writes_explorer(tmp_path):
    from pdm_cmaps.reporting import plotly_hub as ph

    res, out = tmp_path / "results", tmp_path / "out"
    (res / "monitoring_repeated" / "tables").mkdir(parents=True)
    pd.DataFrame({"Subset": ["FD001", "FD002"], "Regimes": [1, 6], "Between_regime_pct_mean": [0.0, 99.5], "Between_regime_pct_std": [0.0, 0.02],
                  "OOS_R2_mean": [-0.02, 0.99], "Runs": [3, 3]}).to_csv(res / "monitoring_repeated" / "tables" / "regime_variance_by_subset.csv", index=False)
    built, skipped = ph.build(ph.Ctx(results=res, out=out), only=["t01", "t02"])
    assert [c.id for c in built] == ["t01"] and [c.id for c in skipped_ids(skipped)] == ["t02"]
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "t01_" in index and "indisponível" in index and (out / "assets" / "plotly.min.js").stat().st_size > 1_000_000
    page = (out / "tcc" / next(c.filename for c in built)).read_text(encoding="utf-8")
    assert "../assets/plotly.min.js" in page and "Índice dos gráficos" in page
    sets = {"FD004": {"names": ["A"], "units": {"7": {"x": [1, 2], "truth": [125.0, 124.0], "series": {"A": [120.0, 118.0]}, "markers": []}}}}
    ph.explorer_page(out / "tcc" / "x.html", "T", "S", sets, {"scatter": True, "defaultOn": ["A"]})
    text = (out / "tcc" / "x.html").read_text(encoding="utf-8")
    assert '"FD004"' in text and "__DATA__" not in text and "__CONFIG__" not in text

"""Monitoring layer: RUL + uncertainty + criticality + alerts + grounded reports, and its figures.

This module only ORCHESTRATES the layer:

  held-out engines -> RUL model (+ conformal interval) -> criticality strategies -> alert events
                   -> structured facts -> deterministic report text

Every table written here answers one research question (RQ-A ... RQ-E):
  (RQ-E: SPC detection summary is recomputed here per seed, so repeated runs need no earlier stage.)
  rul_uncertainty_metrics.csv        RQ-B  does the interval reach its nominal coverage?
  criticality_strategy_comparison.csv  RQ-C  which criticality rule misses fewer critical states?
  criticality_policy_sensitivity.csv RQ-C  do conclusions survive other threshold choices?
  monitoring_repeated_summary.csv    repeated seeds x subsets, mean and std of all of the above
[LEAKAGE] The conformal calibration uses out-of-fold predictions on TRAINING engines only; all
metrics here are computed on the held-out engines of the same seeded engine split as pipeline.run().
"""

from __future__ import annotations

from pdm_cmaps.config import DEFAULT_RESULTS_DIR

from pathlib import Path

import numpy as np
import pandas as pd

from pdm_cmaps.monitoring.alerts import CRITICAL, LEVEL_NAMES, WARNING, CriticalityPolicy, build_alert_timeline, cost_table, evaluate_strategies, generate_alert_events
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE, DEFAULT_RUL_CAP
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.features import CausalFeatureEngineer
from pdm_cmaps.domain.rul import compute_piecewise_rul
from pdm_cmaps.data.splits import split_engine_units
from pdm_cmaps.modeling.registry import get_model_pool, grouped_oof
from pdm_cmaps.reporting.monitoring_figures import plot_model_ci, plot_regime_variance, plot_residual_control_chart, plot_rul_trajectories, plot_strategy_tradeoff, plot_strategy_tradeoff_repeated
from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.monitoring.reports import build_facts, render_report
from pdm_cmaps.monitoring.uncertainty import ConformalRULInterval, NormalizedConformalRULInterval, causal_local_spread, interval_metrics

# Limite do T² da MEWMA nos motores nativos (H = 50): "synthetic" desde 2026-11-18 (decisão do autor, post hoc; ver docs/protocolo-primario.md).
# É um argumento explícito aqui porque o padrão da CLASSE continua "empirical": os estudos de injeção (Seção 5.5 e a parte de injeção da Seção 5.4) usam o limite empírico.
MEWMA_CALIBRATION_MONITORING = "synthetic"

# ============================================================================
# orchestration
# ============================================================================

def _policy_grid(base: CriticalityPolicy) -> dict[str, CriticalityPolicy]:
    """Sensitivity grid for [ASSUMPTION] thresholds: critical level at 10/15/20 cycles (x2, x4 for the others)."""
    grid = {}
    for crit in (10.0, 15.0, 20.0):
        grid[f"critical<={int(crit)}"] = CriticalityPolicy(attention=4 * crit, warning=2 * crit, critical=crit, persistence=base.persistence)
    return grid


def compute_fleet_alarms(train_df: pd.DataFrame, test_df: pd.DataFrame, subset: str, seed: int, healthy_cycle_limit: int):
    """SPC stage for T50: regime decoupling + healthy-calibrated virtual sensor + EWMA alarms on test engines.

    Same configuration as pipeline.run(), so with the same seed the alarms are identical.
    """
    n_regimes = 6 if subset.upper() in {"FD002", "FD004"} else 1
    regimes = RegimeDecoupler(n_clusters=n_regimes, random_state=seed).fit(train_df)
    virtual = VirtualSensor(target_sensor="T50", mode="exogenous", ewma_alpha=0.10, min_consecutive_breaches=2,
                            healthy_cycle_limit=healthy_cycle_limit, random_state=seed)
    trace, per_engine, summary = virtual.evaluate(regimes.transform(train_df), regimes.transform(test_df))
    alarms = {int(r.unit_id): (None if pd.isna(r.first_alarm_cycle) else int(r.first_alarm_cycle)) for r in per_engine.itertuples()}
    return alarms, trace, summary


def compute_monitoring(
    subset: str,
    seed: int = DEFAULT_RANDOM_STATE,
    healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
    alpha: float = 0.10,
    policy: CriticalityPolicy | None = None,
    loader: CMapssLoader | None = None,
    mewma_calibration: str = MEWMA_CALIBRATION_MONITORING,
) -> dict:
    """Pure computation of the whole layer for one (subset, seed); no files are written."""
    policy = policy or CriticalityPolicy()
    loader = loader or CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    train_df = frame.loc[frame["unit_id"].isin(train_units)]
    test_df = frame.loc[frame["unit_id"].isin(test_units)]
    sensors = loader.informative_sensors(train_df)
    base_features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]

    # RUL model with causal temporal features (best family in the benchmark)
    engineer = CausalFeatureEngineer(sensors, windows=(5, 10)).fit(train_df, healthy_cycle_limit)
    train_fe, test_fe = engineer.transform(train_df), engineer.transform(test_df)
    features = [*base_features, *[c for c in train_fe.columns if c not in train_df.columns]]
    estimator = get_model_pool(seed)["LightGBM"]

    # [CALIBRATION][LEAKAGE] conformal calibration on OOF predictions of TRAINING engines only
    _, oof = grouped_oof(train_fe, features, estimator=estimator, n_splits=5, random_state=seed)
    conformal = ConformalRULInterval(alpha=alpha).fit(train_fe["RUL"], np.clip(oof, 0, DEFAULT_RUL_CAP))
    from sklearn.base import clone
    model = clone(estimator).fit(train_fe[features].to_numpy(dtype=float), train_fe["RUL"].to_numpy(dtype=float))
    pred = np.clip(model.predict(test_fe[features].to_numpy(dtype=float)), 0, DEFAULT_RUL_CAP)
    lower, upper = conformal.predict_interval(pred)
    timeline_in = pd.DataFrame({
        "unit_id": test_fe["unit_id"].to_numpy(), "cycle": test_fe["cycle"].to_numpy(),
        "RUL_true": test_fe["RUL"].to_numpy(), "RUL_pred": pred, "RUL_lower": lower, "RUL_upper": upper,
    })
    metrics = interval_metrics(timeline_in["RUL_true"], lower, upper)

    # RQ-B extension: conditional-coverage variants, all calibrated on the same OOF predictions
    n_regimes = 6 if subset.upper() in {"FD002", "FD004"} else 1
    regimes = RegimeDecoupler(n_clusters=n_regimes, random_state=seed).fit(train_df)
    reg_tr = regimes.transform(train_fe)["regime_id"].to_numpy()
    reg_te = regimes.transform(test_fe)["regime_id"].to_numpy()
    oof_c = np.clip(oof, 0, DEFAULT_RUL_CAP)
    spread_tr = causal_local_spread(oof_c, train_fe["unit_id"].to_numpy())
    spread_te = causal_local_spread(pred, test_fe["unit_id"].to_numpy())
    variants = {
        "Mondrian RUL (padrão)": (lower, upper),
        "Mondrian RUL × regime": ConformalRULInterval(alpha=alpha).fit(train_fe["RUL"], oof_c, groups=reg_tr).predict_interval(pred, groups=reg_te),
        "Normalizado (dispersão local)": NormalizedConformalRULInterval(alpha=alpha).fit(train_fe["RUL"], oof_c, spread_tr).predict_interval(pred, spread_te),
    }
    unc_variants = []
    for name, (lo_v, hi_v) in variants.items():
        ins = (timeline_in["RUL_true"] >= lo_v) & (timeline_in["RUL_true"] <= hi_v)
        pe = ins.groupby(timeline_in["unit_id"]).mean()
        unc_variants.append({"Subset": subset, "Seed": seed, "Method": name, **interval_metrics(timeline_in["RUL_true"], lo_v, hi_v),
                             "Per_engine_coverage_p10": float(pe.quantile(0.10)), "Per_engine_coverage_min": float(pe.min())})
    inside = (timeline_in["RUL_true"] >= lower) & (timeline_in["RUL_true"] <= upper)
    per_engine_cov = inside.groupby(timeline_in["unit_id"]).mean()
    rul_metrics = {
        "RMSE": float(np.sqrt(np.mean((timeline_in["RUL_true"] - pred) ** 2))),
        "MAE": float(np.mean(np.abs(timeline_in["RUL_true"] - pred))),
    }
    unc = {
        "Subset": subset, "Seed": seed, "Nominal_coverage": 1 - alpha, **metrics, **rul_metrics,
        "Per_engine_coverage_p10": float(per_engine_cov.quantile(0.10)),
        "Per_engine_coverage_median": float(per_engine_cov.median()),
        "Calibration_rows": int(len(train_fe)), "Test_engines": int(len(test_units)),
    }

    # SPC stage -> anomaly flags (RQ-E): univariate T50 chart and multivariate MEWMA over 13 sensors
    alarms, trace, spc = compute_fleet_alarms(train_df, test_df, subset, seed, healthy_cycle_limit)
    mv = MultivariateResidualMonitor(healthy_cycle_limit=healthy_cycle_limit, random_state=seed, calibration=mewma_calibration).fit(regimes.transform(train_df))
    mv_alarms, _, mv_summary = mv.alarms(mv.score(regimes.transform(test_df)))
    extra = {"anomaly_mv": mv_alarms}

    # RQ-C: criticality strategies, sensitivity to the threshold policy, and decision cost per policy
    timeline = build_alert_timeline(timeline_in, policy, alarms, extra)
    comparison = evaluate_strategies(timeline, policy)
    sens, costs, timelines = [], [], []
    for name, pol in _policy_grid(policy).items():
        tl = build_alert_timeline(timeline_in, pol, alarms, extra)
        sens.append(evaluate_strategies(tl, pol).assign(Policy=name))
        costs.append(cost_table(tl).assign(Policy=name))
        keep = ["unit_id", "cycle", "RUL_true", "RUL_pred", "RUL_lower", "RUL_upper", *[c for c in tl.columns if c.startswith("level_")]]
        timelines.append(tl[keep].assign(Policy=name))
    return {
        "timeline": timeline, "uncertainty": unc, "strategies": comparison,
        "sensitivity": pd.concat(sens, ignore_index=True), "events": generate_alert_events(timeline, "interval+anomaly"),
        "alarms": alarms, "trace": trace, "spc_summary": spc, "test_units": [int(u) for u in test_units], "policy": policy,
        "uncertainty_variants": pd.DataFrame(unc_variants), "spc_mv_summary": mv_summary, "mv_alarms": mv_alarms,
        "costs": pd.concat(costs, ignore_index=True), "timelines": pd.concat(timelines, ignore_index=True),
    }


def run_monitoring_layer(
    subset: str,
    output_dir: Path,
    seed: int = DEFAULT_RANDOM_STATE,
    healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
    alpha: float = 0.10,
    policy: CriticalityPolicy | None = None,
    sample_engine_id: int | None = 179,
    max_report_events: int = 60,
    loader: CMapssLoader | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute the layer for one subset/seed and write its tables, example reports and figures."""
    table_dir, figure_dir = Path(output_dir) / "tables", Path(output_dir) / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"[monitoring] RUL + uncertainty + criticality + alerts for {subset} (alpha={alpha})", flush=True)
    res = compute_monitoring(subset, seed, healthy_cycle_limit, alpha, policy, loader)
    policy, timeline, alarms = res["policy"], res["timeline"], res["alarms"]

    unc = pd.DataFrame([res["uncertainty"]])
    unc.to_csv(table_dir / "rul_uncertainty_metrics.csv", index=False)
    pd.DataFrame([res["spc_summary"]]).to_csv(table_dir / "spc_detection_summary_t50.csv", index=False)
    timeline.to_csv(table_dir / "alert_timeline.csv", index=False)
    res["strategies"].to_csv(table_dir / "criticality_strategy_comparison.csv", index=False)
    res["sensitivity"].to_csv(table_dir / "criticality_policy_sensitivity.csv", index=False)
    events = res["events"]
    events.to_csv(table_dir / "alert_events.csv", index=False)
    u = res["uncertainty"]
    print(f"  -> interval coverage {u['coverage']:.3f} (nominal {1 - alpha:.2f}) | critical-zone {u['coverage_critical_zone']:.3f} | width {u['mean_width']:.1f}", flush=True)
    print(res["strategies"][["Strategy", "Missed_critical_rate", "False_alarm_rate_normal", "Median_warning_lead_time"]].round(3).to_string(index=False), flush=True)

    # alert messages (facts -> deterministic text)
    warn_events = events.loc[events["to_level"].isin([LEVEL_NAMES[WARNING], LEVEL_NAMES[CRITICAL]])]
    if len(warn_events) > max_report_events:
        warn_events = warn_events.sample(max_report_events, random_state=seed)
    messages = []
    for ev in warn_events.to_dict("records"):
        history = timeline.loc[(timeline["unit_id"] == ev["unit_id"]) & (timeline["cycle"] <= ev["cycle"])]
        facts = build_facts(history, policy, coverage_pct=100 * (1 - alpha))
        messages.append((ev, facts, render_report(facts)))
    if messages:
        pd.DataFrame([{**f, "message": m} for _, f, m in messages]).drop(columns=["thresholds"]).to_csv(table_dir / "alert_messages.csv", index=False)
        (table_dir / "example_alert_messages.md").write_text(
            "# Example alert messages\n\n" + "\n\n".join(f"- {m}" for _, _, m in messages[:3]) + "\n", encoding="utf-8")

    # figures (residual / RUL scale; same sample engine across detection figures)
    test_units = res["test_units"]
    sample = sample_engine_id if sample_engine_id in set(test_units) else int(sorted(test_units)[0])
    plot_residual_control_chart(res["trace"], sample, alarms.get(sample), figure_dir / "fig_residual_control_chart.png", healthy_cycle_limit)
    eol = timeline.groupby("unit_id")["cycle"].max().sort_values()
    picks = [int(eol.index[int(q * (len(eol) - 1))]) for q in (0.1, 0.4, 0.7, 0.95)]
    plot_rul_trajectories(timeline, list(dict.fromkeys(picks)), figure_dir / "fig_rul_trajectories_intervals.png", alpha=alpha)
    plot_strategy_tradeoff(res["sensitivity"], figure_dir / "fig_criticality_strategy_tradeoff.png")
    ci_path = table_dir / "thesis_bootstrap_ci.csv"
    if ci_path.exists():
        plot_model_ci(pd.read_csv(ci_path), figure_dir / "fig_model_comparison_ci.png")
    return {"uncertainty": unc, "strategies": res["strategies"], "sensitivity": res["sensitivity"], "events": events}


def compute_regime_variance(subsets, seeds, loader: CMapssLoader | None = None) -> pd.DataFrame:
    """Share of T50 variance explained by flight regime on held-out engines (needs only KMeans, no RUL model)."""
    loader = loader or CMapssLoader()
    rows = []
    for subset in subsets:
        frame = loader.load(subset)
        k = 6 if subset.upper() in {"FD002", "FD004"} else 1
        for seed in seeds:
            train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
            train_df, test_df = frame.loc[frame["unit_id"].isin(train_units)], frame.loc[frame["unit_id"].isin(test_units)]
            decoupler = RegimeDecoupler(n_clusters=k, random_state=seed).fit(train_df)
            v = RegimeDecoupler.variance_decomposition(decoupler.transform(test_df), "T50", reference_frame=decoupler.transform(train_df))
            rows.append({"Subset": subset, "Seed": seed, "Regimes": k, "Between_regime_pct": v["pct_between_regime"],
                         "Within_regime_pct": v["pct_within_regime"], "OOS_R2_regime_mean": v["oos_r2"]})
    return pd.DataFrame(rows)


def run_regime_variance(subsets, seeds, output_dir: Path) -> pd.DataFrame:
    out = Path(output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    table = compute_regime_variance(subsets, seeds)
    table.to_csv(out / "tables" / "regime_variance_by_subset_raw.csv", index=False)
    summary = table.groupby(["Subset", "Regimes"], as_index=False).agg(
        Between_regime_pct_mean=("Between_regime_pct", "mean"), Between_regime_pct_std=("Between_regime_pct", "std"),
        OOS_R2_mean=("OOS_R2_regime_mean", "mean"), Runs=("Seed", "size"))
    summary.to_csv(out / "tables" / "regime_variance_by_subset.csv", index=False)
    plot_regime_variance(table, out / "figures" / "fig_regime_variance_by_subset.png")
    return summary


RUN_PARTS = ("uncertainty", "uncertainty_variants", "spc", "spc_mv", "sensitivity", "costs")


def _run_dir(out: Path, subset: str, seed: int) -> Path:
    return out / "runs" / f"{subset}_seed{seed}"


def run_one(subset: str, seed: int, out: Path, alpha: float, healthy_cycle_limit: int, mewma_calibration: str = MEWMA_CALIBRATION_MONITORING) -> None:
    """Compute one (subset, seed) and write its tables at once, so a crash never loses finished runs."""
    d = _run_dir(out, subset, seed)
    if all((d / f"{part}.csv").exists() for part in RUN_PARTS):
        print(f"[repeat] {subset} seed={seed} already done, skipping", flush=True)
        return
    print(f"[repeat] {subset} seed={seed}", flush=True)
    res = compute_monitoring(subset, seed, healthy_cycle_limit, alpha, mewma_calibration=mewma_calibration)
    d.mkdir(parents=True, exist_ok=True)
    tag = {"Subset": subset, "Seed": seed}
    tables = {
        "uncertainty": pd.DataFrame([res["uncertainty"]]),
        "uncertainty_variants": res["uncertainty_variants"],
        "spc": pd.DataFrame([{**tag, **res["spc_summary"]}]),
        "spc_mv": pd.DataFrame([{**tag, **res["spc_mv_summary"]}]),
        "sensitivity": res["sensitivity"].assign(**tag),
        "costs": res["costs"].assign(**tag),
    }
    res["timelines"].assign(**tag).to_csv(d / "timelines.csv.gz", index=False)
    for part in reversed(RUN_PARTS):  # "uncertainty" written last marks the run complete
        tables[part].to_csv(d / f"{part}.csv", index=False)
    print(f"[repeat] {subset} seed={seed} saved", flush=True)


def _mean_std(df, keys, cols):
    agg = df.groupby(keys)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["Runs"] = df.groupby(keys).size()
    return agg.reset_index()


def aggregate_repeated(out: Path) -> pd.DataFrame:
    """Read every finished run under ``out/runs`` and write the raw + mean/std tables and the figure."""
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    loaded = {part: [] for part in RUN_PARTS}
    for d in sorted((out / "runs").glob("*_seed*")):
        if all((d / f"{part}.csv").exists() for part in RUN_PARTS):
            for part in RUN_PARTS:
                loaded[part].append(pd.read_csv(d / f"{part}.csv"))
    frames = {part: pd.concat(v, ignore_index=True) for part, v in loaded.items() if v}
    unc, strat, spc = frames["uncertainty"], frames["sensitivity"], frames["spc"]
    unc.to_csv(out / "tables" / "monitoring_repeated_uncertainty_raw.csv", index=False)
    strat.to_csv(out / "tables" / "monitoring_repeated_strategies_raw.csv", index=False)
    spc.to_csv(out / "tables" / "monitoring_repeated_spc_raw.csv", index=False)
    unc_cols = ["coverage", "coverage_critical_zone", "mean_width", "Per_engine_coverage_p10", "RMSE", "MAE"]
    _mean_std(unc, ["Subset"], unc_cols).to_csv(out / "tables" / "monitoring_repeated_uncertainty_summary.csv", index=False)
    strat_cols = ["Missed_critical_rate", "False_alarm_rate_normal", "Within_one_level", "Median_warning_lead_time", "Engines_never_alerted", "Engines_premature_alert"]
    strat_summary = _mean_std(strat, ["Subset", "Policy", "Strategy"], strat_cols)
    strat_summary.to_csv(out / "tables" / "monitoring_repeated_summary.csv", index=False)
    spc_cols = ["detection_rate_pct", "mean_lead_time_cycles", "mean_healthy_false_alarm_pct", "premature_alarm_pct"]
    _mean_std(spc, ["Subset"], spc_cols).to_csv(out / "tables" / "monitoring_repeated_spc_summary.csv", index=False)
    if "spc_mv" in frames:
        frames["spc_mv"].to_csv(out / "tables" / "monitoring_repeated_spc_mv_raw.csv", index=False)
        _mean_std(frames["spc_mv"], ["Subset"], spc_cols).to_csv(out / "tables" / "monitoring_repeated_spc_mv_summary.csv", index=False)
    if "uncertainty_variants" in frames:
        v = frames["uncertainty_variants"]
        v.to_csv(out / "tables" / "monitoring_repeated_uncertainty_variants_raw.csv", index=False)
        _mean_std(v, ["Subset", "Method"], ["coverage", "coverage_critical_zone", "mean_width", "Per_engine_coverage_p10", "Per_engine_coverage_min"]).to_csv(
            out / "tables" / "monitoring_repeated_uncertainty_variants_summary.csv", index=False)
    if "costs" in frames:
        c = frames["costs"]
        c.to_csv(out / "tables" / "monitoring_repeated_costs_raw.csv", index=False)
        _mean_std(c, ["Subset", "Policy", "Strategy", "Cost_ratio", "Lead"], ["Relative_to_oracle", "Failures", "Mean_unused_cycles"]).to_csv(
            out / "tables" / "monitoring_repeated_costs_summary.csv", index=False)
    plot_strategy_tradeoff_repeated(strat, out / "figures" / "fig_criticality_tradeoff_repeated.png")
    return strat_summary


def run_repeated_experiments(subsets, seeds, output_dir: Path, alpha: float = 0.10, healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT, jobs: int = 1,
                            mewma_calibration: str = MEWMA_CALIBRATION_MONITORING) -> pd.DataFrame:
    """Repeat the layer over subsets x seeds (each run saved as it finishes) and report mean +- std."""
    out = Path(output_dir)
    jobs_list = [(subset, seed) for subset in subsets for seed in seeds]
    if jobs > 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=jobs)(delayed(run_one)(subset, seed, out, alpha, healthy_cycle_limit, mewma_calibration) for subset, seed in jobs_list)
    else:
        for subset, seed in jobs_list:
            run_one(subset, seed, out, alpha, healthy_cycle_limit, mewma_calibration)
    return aggregate_repeated(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Repeat the monitoring layer over subsets and seeds (mean +- std).")
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--jobs", type=int, default=3, help="parallel (subset, seed) runs")
    parser.add_argument("--regime-variance-only", action="store_true", help="only the T50 regime-variance table and figure (fast)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "monitoring_repeated")
    parser.add_argument("--mewma-calibration", choices=("empirical", "synthetic"), default=MEWMA_CALIBRATION_MONITORING,
                        help="limite do T² da MEWMA (padrão do protocolo para o monitoramento: synthetic; empirical reproduz a rodada anterior)")
    parser.add_argument("--aggregate-only", action="store_true", help="rebuild the summary tables from finished runs")
    args = parser.parse_args(argv)
    if args.aggregate_only:
        print(aggregate_repeated(args.output_dir).round(3).to_string(index=False))
        return 0
    if args.regime_variance_only:
        print(run_regime_variance(args.subsets, args.seeds, args.output_dir).round(3).to_string(index=False))
        return 0
    summary = run_repeated_experiments(args.subsets, args.seeds, args.output_dir, jobs=args.jobs, mewma_calibration=args.mewma_calibration)
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

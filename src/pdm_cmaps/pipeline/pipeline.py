"""Pipeline stages: virtual sensor / SPC / RUL benchmark, additional models and statistical inference.

This file is the RECIPE that calls everything else.

The 5 phases of PdMPipeline.execute() (stages hand data to each other through tables/*.csv):
  1. run()                 load -> split by engine -> regimes -> virtual sensor + SPC alarms ->
                           sensor sweep -> health index -> RUL benchmark + grouped CV
  2. run_auxiliary_spc_horizon   same as 1 with a longer healthy window (H=100) as a
                           sensitivity check; isolated in a scratch folder.
  3. train_additional_models_and_stats   SVM/CNN/LSTM/Transformer, critical-zone table,
                           OFFICIAL test benchmark, hypothesis tests, bootstrap CIs.
  4. monitoring layer      conformal RUL intervals, criticality levels, alert events, grounded
                           reports + faithfulness audit (monitoring/layer.py, alerts.py, reports.py).
  5. figure generation     thesis figures (FD004 only).

Two evaluation protocols exist on purpose:
  [HOLDOUT]  75/25 engine split of the training file, every cycle scored. Good for comparing
             models with each other; optimistic in absolute terms.
  [OFFICIAL] fit on all training engines, score ONLY the last cycle of each truncated test
             engine vs RUL_FDxxx.txt. Use this one when comparing with the literature."""

from __future__ import annotations

from pdm_cmaps.pipeline.extra_models import build_extra_models, fit_predict

from pdm_cmaps.analysis.stats import ThesisHypothesisAnalyzer
from pdm_cmaps.reporting.plots import figure_1_noise_vs_damage
from pdm_cmaps.reporting.plots import figure_2_physical_cascade
from pdm_cmaps.reporting.plots import figure_3_autocorrelation_evidence
from pdm_cmaps.reporting.plots import figure_4_fleet_persistence_tradeoff
from pdm_cmaps.reporting.plots import plot_actual_vs_predicted
from pdm_cmaps.reporting.plots import plot_geometric_envelope_detection

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shutil

from pdm_cmaps.config import DEFAULT_RESULTS_DIR
from pdm_cmaps.data.features import CausalFeatureEngineer
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.splits import partition_healthy_and_lifecycle, split_engine_units
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.detection.spc import EnvelopeAnomalyDetector
from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.domain.rul import compute_piecewise_rul, evaluate_rul_predictions
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE, DEFAULT_RUL_CAP
from pdm_cmaps.modeling.benchmark import RULBenchmark
from pdm_cmaps.modeling.minirocket import MiniRocketRULModel
from pdm_cmaps.modeling.registry import get_model_pool
from pdm_cmaps.modeling.torch_models import PyTorchRNNModel
from pdm_cmaps.pipeline.config import PipelineConfig
from pdm_cmaps.progress import ProgressTracker




PRIMARY_TARGET_SENSORS = ("T50", "P30", "BPR", "Nc")

def run(
    subset: str,
    output_dir: Path,
    seed: int,
    include_lstm: bool = False,
    include_minirocket: bool = False,
    healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
    target_sensors: tuple[str, ...] = PRIMARY_TARGET_SENSORS,
    sample_engine_id: int | None = 179,
    min_consecutive_breaches: int = 2,
    loader: CMapssLoader | None = None,
    decoupler: RegimeDecoupler | None = None,
    benchmark: RULBenchmark | None = None,
) -> dict[str, pd.DataFrame]:
    """Execute the guide's core analyses and write their tables/figures.

    ``loader``, ``decoupler`` and ``benchmark`` are optional injected components (defaults are built from ``subset``/``seed``).
    """
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    print("\n=======================================================================", flush=True)
    print(f"[maintenance] STARTING PdM PIPELINE: Subset={subset} | Seed={seed} | Healthy limit={healthy_cycle_limit} cycles", flush=True)
    print("=======================================================================\n", flush=True)

    # STEP 1 -- [LEAKAGE] split by engine first; everything learned later (regimes, scalers,
    # sensor filter, thresholds, models) is fit on the training engines only.
    print(f"[maintenance] [STEP 1/7] Loading {subset} data and partitioning disjoint engine holdout...", flush=True)
    loader = loader or CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    # Informative-sensor filter is learned on TRAINING engines only (no test-fleet information).
    sensors = loader.informative_sensors(frame.loc[frame["unit_id"].isin(train_units)])
    parts = partition_healthy_and_lifecycle(frame, train_units, test_units, healthy_cycle_limit)
    print(f"  -> Total rows: {len(frame)} | Engines: {frame['unit_id'].nunique()} (Train: {len(train_units)}, Test: {len(test_units)})", flush=True)

    # STEP 2 -- unsupervised regime discovery. FD002/FD004 were simulated with 6 operating
    # conditions, FD001/FD003 with 1 (so decoupling is a no-op there and H1 is skipped).
    n_regimes = 6 if subset in {"FD002", "FD004"} else 1
    print(f"[maintenance] [STEP 2/7] Decoupling operating flight regimes ({n_regimes} clusters)...", flush=True)
    regimes = (decoupler or RegimeDecoupler(n_clusters=n_regimes, random_state=seed)).fit(parts["full_lifecycle_train"])
    train_full = regimes.transform(parts["full_lifecycle_train"])
    test_full = regimes.transform(parts["full_lifecycle_test"])
    healthy_train = regimes.transform(parts["healthy_train"])
    healthy_test = regimes.transform(parts["healthy_test"])

    valid_targets = [s for s in target_sensors if s in sensors]
    if not valid_targets:
        valid_targets = ["T50"] if "T50" in sensors else [sensors[0]]

    target_unit_id = sample_engine_id if (sample_engine_id is not None and sample_engine_id in test_units) else int(test_units[0])
    print(f"  -> Primary sample engine for trace visualization: Engine #{target_unit_id}", flush=True)


    # STEPS 3-5 -- for each target sensor: (a) how much of its variance is regime? (b) fit the
    # healthy-state virtual sensor and turn residuals into SPC alarms (c) draw envelopes.
    print(f"[maintenance] [STEPS 3-5/7] Profiling Virtual Sensors & Envelopes across targets: {', '.join(valid_targets)}...", flush=True)
    for target_sensor in valid_targets:
        s_lower = target_sensor.lower()
        print(f"\n  [sensor: {target_sensor}] Decoupling variance and fitting Virtual Sensor...", flush=True)

        # 1. Variance Decomposition
        variance = RegimeDecoupler.variance_decomposition(test_full, target_sensor, reference_frame=train_full)
        pd.DataFrame([variance]).drop(columns=["clusters"]).to_csv(table_dir / f"{s_lower}_variance_decomposition.csv", index=False)
        if target_sensor == "T50":
            pd.DataFrame([variance]).drop(columns=["clusters"]).to_csv(table_dir / "t50_variance_decomposition.csv", index=False)

        between_pct = variance.get("Between_regime_pct", variance.get("pct_between_regime", 0.0))
        within_pct = variance.get("Within_regime_pct", variance.get("pct_within_regime", 100.0))
        base_r2 = variance.get("regime_baseline_r2", between_pct / 100.0 if between_pct else 0.0)
        print(f"    -> {target_sensor} variance: Between regimes = {between_pct:.1f}%, Within regime = {within_pct:.1f}% (Baseline R² = {base_r2:.4f})", flush=True)

        # 2. Virtual Sensor fitting & Fleet SPC (Strictly Exogenous features, ewma_alpha=0.10)
        virtual_model = VirtualSensor(
            target_sensor=target_sensor,
            mode="exogenous",
            ewma_alpha=0.10,
            min_consecutive_breaches=min_consecutive_breaches,
            healthy_cycle_limit=healthy_cycle_limit,
            random_state=seed,
        )
        residual_trace, anomalies, anomaly_summary = virtual_model.evaluate(healthy_train, test_full)
        h_suffix = f"_h{healthy_cycle_limit}"
        residual_trace.to_csv(table_dir / f"virtual_sensor_residuals_{s_lower}.csv", index=False)
        residual_trace.to_csv(table_dir / f"virtual_sensor_residuals_{s_lower}{h_suffix}.csv", index=False)
        anomalies.to_csv(table_dir / f"fleet_anomalies_{s_lower}.csv", index=False)
        anomalies.to_csv(table_dir / f"fleet_anomalies_{s_lower}{h_suffix}.csv", index=False)
        pd.DataFrame([anomaly_summary]).to_csv(table_dir / f"fleet_anomaly_summary_{s_lower}.csv", index=False)
        pd.DataFrame([anomaly_summary]).to_csv(table_dir / f"fleet_anomaly_summary_{s_lower}{h_suffix}.csv", index=False)
        if target_sensor == "T50":
            residual_trace.to_csv(table_dir / "virtual_sensor_residuals.csv", index=False)
            anomalies.to_csv(table_dir / "fleet_anomalies.csv", index=False)
            pd.DataFrame([anomaly_summary]).to_csv(table_dir / "fleet_anomaly_summary.csv", index=False)

        print(f"    -> Virtual Sensor {target_sensor}: Fleet detection = {anomaly_summary['detection_rate_pct']:.1f}% | Mean lead time = {anomaly_summary['mean_lead_time_cycles']:.1f} cycles", flush=True)

        # 4. Envelope Anomaly Detection for target_unit_id
        envelope_detector = EnvelopeAnomalyDetector(
            k_sigma=3.0,
            healthy_cycle_limit=healthy_cycle_limit,
            window_size=5,
            min_consecutive_breaches=min_consecutive_breaches,
        )
        healthy_res_rows = virtual_model.calibration_predictions_
        if healthy_res_rows is None:
            raise RuntimeError("Virtual sensor did not produce OOF healthy calibration predictions")
        envelope_detector.calibrate(healthy_res_rows[target_sensor], healthy_res_rows["prediction"])

        sample_trace = residual_trace.loc[residual_trace["unit_id"] == target_unit_id].copy()
        scored_env, env_summary = envelope_detector.score_time_series_envelope(
            sample_trace, actual_col=target_sensor, pred_col="prediction",
        )
        scored_geom, geom_summary = envelope_detector.score_geometric_envelope(
            sample_trace, signal_col=target_sensor, use_decoupled_signal=True,
        )

        scored_env.to_csv(table_dir / f"envelope_scored_trace_{s_lower}.csv", index=False)
        scored_env.to_csv(table_dir / f"envelope_scored_trace_{s_lower}{h_suffix}.csv", index=False)
        pd.DataFrame([env_summary]).to_csv(table_dir / f"envelope_summary_{s_lower}.csv", index=False)
        pd.DataFrame([env_summary]).to_csv(table_dir / f"envelope_summary_{s_lower}{h_suffix}.csv", index=False)
        pd.DataFrame([geom_summary]).to_csv(table_dir / f"geometric_envelope_summary_{s_lower}.csv", index=False)
        pd.DataFrame([geom_summary]).to_csv(table_dir / f"geometric_envelope_summary_{s_lower}{h_suffix}.csv", index=False)
        if target_sensor == "T50":
            scored_env.to_csv(table_dir / "envelope_scored_trace.csv", index=False)
            scored_env.to_csv(table_dir / f"envelope_scored_trace{h_suffix}.csv", index=False)
            pd.DataFrame([env_summary]).to_csv(table_dir / "envelope_summary.csv", index=False)
            pd.DataFrame([env_summary]).to_csv(table_dir / f"envelope_summary{h_suffix}.csv", index=False)
            pd.DataFrame([geom_summary]).to_csv(table_dir / "geometric_envelope_summary.csv", index=False)
            pd.DataFrame([geom_summary]).to_csv(table_dir / f"geometric_envelope_summary{h_suffix}.csv", index=False)

        plot_geometric_envelope_detection(
            scored_geom,
            geom_summary,
            figure_dir / f"{s_lower}_geometric_envelope_detection.png",
            title=f"Envelope Geométrico Digital (Hull): {target_sensor}",
            sensor_name=target_sensor,
            healthy_cycle_limit=healthy_cycle_limit,
        )
        if target_sensor == "T50":
            plot_geometric_envelope_detection(
                scored_geom,
                geom_summary,
                figure_dir / "t50_geometric_envelope_detection.png",
                title="Envelope Geométrico de Sinal Digital (Rolling Peak/Valley Hull)",
                sensor_name="T50",
                healthy_cycle_limit=healthy_cycle_limit,
            )
        print(f"    -> Time-series envelope ({target_sensor}): Alarm at cycle {env_summary['first_alarm_cycle']} | EOL: {env_summary['eol_cycle']} | Lead time: {env_summary['lead_time_cycles']} cycles", flush=True)

        # 5. Held-out healthy regime validation
        healthy_prediction = virtual_model.predict(healthy_test)
        regime_rows = []
        for regime_id, rows in healthy_prediction.groupby("regime_id"):
            truth, prediction = rows[target_sensor].to_numpy(), rows["prediction"].to_numpy()
            regime_rows.append({
                "Regime_ID": int(regime_id), "Test_rows": len(rows),
                "R2": float(r2_score(truth, prediction)),
                "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
                "MAE": float(mean_absolute_error(truth, prediction)),
            })
        regime_scores = pd.DataFrame(regime_rows)
        regime_scores.to_csv(table_dir / f"healthy_{s_lower}_by_regime.csv", index=False)
        regime_scores.to_csv(table_dir / f"healthy_{s_lower}_by_regime{h_suffix}.csv", index=False)
        if target_sensor == "T50":
            regime_scores.to_csv(table_dir / "healthy_t50_by_regime.csv", index=False)
            regime_scores.to_csv(table_dir / f"healthy_t50_by_regime{h_suffix}.csv", index=False)

    # STEP 7 -- supervised RUL benchmark. [BASELINE] two naive baselines are included so every
    # model's gain is measured against "no information". Grouped CV (below) gives a spread
    # across engine folds; the holdout gives the per-engine predictions used by stats.py.
    print("[maintenance] [STEP 7/7] Benchmarking RUL models on held-out engines (CatBoost, LightGBM, XGBoost, etc.)...", flush=True)

    rul_features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
    benchmark = benchmark or RULBenchmark(random_state=seed)
    rul_train = parts["full_lifecycle_train"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_train"]))
    rul_test = parts["full_lifecycle_test"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_test"]))
    holdout, predictions = benchmark.benchmark_train_test(rul_train, rul_test, rul_features)

    # Representation-matched GBDT/tree baselines: same causal temporal context that sequence models receive.
    # Removes the "single row vs 15-cycle window" confound from the GBDT-vs-deep comparison (RQ2).
    engineer = CausalFeatureEngineer(sensors, windows=(5, 10)).fit(rul_train, healthy_cycle_limit)
    train_fe = engineer.transform(rul_train)
    test_fe = engineer.transform(rul_test)
    if not np.allclose(test_fe["RUL"].to_numpy(), predictions["y_true"]):
        raise AssertionError("Temporal-feature frame is not row-aligned with the shared holdout")
    fe_features = [*rul_features, *[c for c in train_fe.columns if c not in rul_train.columns]]
    temporal_pool = {
        f"{name} (+temporal)": estimator
        for name, estimator in get_model_pool(seed).items()
        if name in {"XGBoost", "LightGBM", "CatBoost", "Extra Trees"}
    }
    fe_holdout, fe_predictions = RULBenchmark(random_state=seed, model_pool=temporal_pool, verbose=False).benchmark_train_test(
        train_fe, test_fe, fe_features,
    )
    fe_holdout = fe_holdout.loc[~fe_holdout["Model"].str.startswith("Baseline")]
    holdout = pd.concat([holdout, fe_holdout], ignore_index=True)
    for name in temporal_pool:
        predictions[name] = fe_predictions[name]
    holdout.to_csv(table_dir / "rul_holdout_benchmark.csv", index=False)
    best_model = str(holdout.loc[holdout["R2"].idxmax(), "Model"])
    plot_actual_vs_predicted(
        predictions["y_true"], predictions[best_model],
        figure_dir / "rul_best_model_actual_vs_predicted.png",
        f"RUL holdout: {best_model}", "RUL (cycles)",
    )
    test_prediction_frame = parts["full_lifecycle_test"][["unit_id", "cycle"]].reset_index(drop=True)
    test_prediction_frame["RUL_actual"] = predictions["y_true"]
    for model_name, values in predictions.items():
        if model_name != "y_true":
            test_prediction_frame[f"RUL_predicted_{model_name}"] = values

    cv_frame = frame.loc[frame["unit_id"].isin(train_units)].copy()
    cv = benchmark.cross_validate(cv_frame, rul_features, n_splits=5)
    table_dir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(table_dir / "rul_grouped_cross_validation.csv", index=False)

    tables = {
        "regime_variance": pd.DataFrame([variance]),
        "virtual_sensor_regimes": regime_scores,
        "anomaly_summary": pd.DataFrame([anomaly_summary]),
        "rul_holdout": holdout,
        "rul_grouped_cv": cv,
    }
    if include_lstm:
        lstm_frame = frame.copy()
        lstm_train = lstm_frame.loc[lstm_frame["unit_id"].isin(train_units)]
        lstm_test = lstm_frame.loc[lstm_frame["unit_id"].isin(test_units)]
        lstm_features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
        # The legacy PyTorchLSTMModel stopped on TRAINING loss and trained only 4 epochs; the shared
        # sequence base class early-stops on a held-out set of engines, so use it instead.
        model = PyTorchRNNModel(seq_length=15, cell_type="lstm", random_state=seed)
        model.fit(lstm_train, lstm_features, "RUL")
        truth = lstm_test["RUL"].to_numpy(dtype=float)
        estimate = model.predict(lstm_test)
        lstm_results = pd.DataFrame([{
            "Model": "PyTorch LSTM",
            **evaluate_rul_predictions(truth, estimate),
            "Fit_Time_s": round(model.fit_time, 3),
        }])
        lstm_results.to_csv(table_dir / "rul_lstm_holdout.csv", index=False)
        tables["rul_lstm"] = lstm_results

    if include_minirocket:
        print("[maintenance] Fitting MiniRocket + RidgeCV and benchmarking RUL on held-out engines", flush=True)

        mr_train = parts["full_lifecycle_train"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_train"]))
        mr_test = parts["full_lifecycle_test"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_test"]))
        mr_features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
        mr_model = MiniRocketRULModel(seq_length=15, num_kernels=1000, random_state=seed)
        mr_model.fit(mr_train, mr_features, "RUL")
        mr_truth, mr_estimate = mr_model.predict(mr_test)
        if not np.allclose(mr_truth, predictions["y_true"]):
            raise AssertionError("MiniRocket predictions are not aligned with the shared holdout rows")
        predictions["MiniRocket + RidgeCV"] = mr_estimate
        test_prediction_frame["RUL_predicted_MiniRocket + RidgeCV"] = mr_estimate
        mr_results = pd.DataFrame([{
            "Model": "MiniRocket + RidgeCV",
            **evaluate_rul_predictions(mr_truth, mr_estimate),
            "Fit_Time_s": round(mr_model.fit_time, 3),
        }])
        combined_holdout = pd.concat([holdout, mr_results], ignore_index=True)
        combined_holdout.to_csv(table_dir / "rul_holdout_benchmark.csv", index=False)
        tables["rul_minirocket"] = mr_results
    test_prediction_frame.to_csv(table_dir / "rul_holdout_predictions.csv", index=False)
    return tables

def run_guide_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD002", "FD003", "FD004"), default="FD004")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "maintenance_guide")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--include-lstm", action="store_true", help="Run an additional sequence-model holdout comparison")
    parser.add_argument("--include-minirocket", action="store_true", help="Run an additional MiniRocket sequence-model holdout comparison")
    parser.add_argument("--healthy-cycle-limit", type=int, default=DEFAULT_HEALTHY_CYCLE_LIMIT, help="Healthy cycle limit used for virtual sensor calibration (default: 50)")
    parser.add_argument("--target-sensors", nargs="+", default=list(PRIMARY_TARGET_SENSORS), help="Target sensors to evaluate for virtual sensor and envelope detection (default: T50 P30 BPR Nc)")
    parser.add_argument("--min-consecutive-breaches", type=int, default=2, help="Minimum consecutive breaches required to trigger an envelope alarm (default: 2, Western Electric rule)")
    parser.add_argument("--unit-id", type=int, default=179, help="Engine unit ID to use for sample traces (default: 179)")
    args = parser.parse_args()
    tables = run(
        args.subset,
        args.output_dir,
        args.seed,
        args.include_lstm,
        args.include_minirocket,
        args.healthy_cycle_limit,
        target_sensors=tuple(args.target_sensors),
        sample_engine_id=args.unit_id,
        min_consecutive_breaches=args.min_consecutive_breaches,
    )
    print("\n=======================================================================", flush=True)
    print("[maintenance] PIPELINE EXECUTION COMPLETE! Summary of generated tables:", flush=True)
    print("=======================================================================", flush=True)
    for name, table in tables.items():
        print(f"  - Table: {name:<28} ({len(table)} rows)", flush=True)
    print(f"\n[maintenance] Tables written to: {(args.output_dir / 'tables').resolve()}", flush=True)
    print(f"[maintenance] Figures written to: {(args.output_dir / 'figures').resolve()}\n", flush=True)

# Benchmark names that already denote one of the extra models (legacy aliases included).
_EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "Support Vector Machine (SVM)": ("Support Vector Machine (SVM)", "SVM"),
    "PyTorch CNN": ("PyTorch CNN", "1D CNN"),
    "PyTorch RNN (LSTM)": ("PyTorch RNN (LSTM)", "RNN (LSTM)", "PyTorch LSTM"),
    "PyTorch Transformer": ("PyTorch Transformer", "Transformer"),
}

def _official_test_benchmark(
    subset: str,
    full_train: pd.DataFrame,
    sensors: list[str],
    features: list[str],
    model_names: list[str],
    seed: int,
    healthy_cycle_limit: int,
) -> pd.DataFrame | None:
    """Score every named model on the official C-MAPSS test protocol (last cycle per engine vs RUL_FDxxx.txt)."""
    loader = CMapssLoader()
    if not ((loader.data_dir / f"test_{subset}.txt").is_file() and (loader.data_dir / f"RUL_{subset}.txt").is_file()):
        return None
    off_test, true_rul = loader.load_test(subset)
    y_true = np.clip(true_rul.to_numpy(dtype=float), 0.0, float(DEFAULT_RUL_CAP))
    n_units = off_test["unit_id"].nunique()
    unit_keys = off_test["unit_id"].to_numpy()

    def last(values) -> np.ndarray:
        return pd.Series(np.asarray(values, dtype=float)).groupby(unit_keys).last().sort_index().to_numpy()

    pool = get_model_pool(seed)
    extras = build_extra_models(seed)
    y_mean = float(full_train["RUL"].mean())
    cycle_map = full_train.groupby("cycle")["RUL"].mean()
    temporal_suffix = " (+temporal)"
    fe_cache: dict[str, object] = {}
    rows = []
    for name in model_names:
        if name == "Baseline (Mean RUL)":
            pred = np.full(n_units, y_mean)
        elif name == "Baseline (Mean RUL by cycle)":
            last_cycle = off_test.groupby("unit_id")["cycle"].max().sort_index()
            pred = last_cycle.map(cycle_map).fillna(y_mean).to_numpy(dtype=float)
        elif name in extras:
            m_type, model = extras[name]
            pred = last(fit_predict(m_type, model, full_train, features, off_test))
        elif name in pool:
            pred = last(fit_predict("sklearn", pool[name], full_train, features, off_test))
        elif name.endswith(temporal_suffix) and name[: -len(temporal_suffix)] in pool:
            if not fe_cache:
                engineer = CausalFeatureEngineer(sensors, windows=(5, 10)).fit(full_train, healthy_cycle_limit)
                fe_train = engineer.transform(full_train)
                fe_cache.update(
                    train=fe_train,
                    test=engineer.transform(off_test),
                    features=[*features, *[c for c in fe_train.columns if c not in full_train.columns]],
                )
            pred = last(fit_predict(
                "sklearn", pool[name[: -len(temporal_suffix)]], fe_cache["train"], fe_cache["features"], fe_cache["test"],
            ))
        elif name == "MiniRocket + RidgeCV":
            model = MiniRocketRULModel(seq_length=15, num_kernels=1000, random_state=seed)
            pred = last(fit_predict("deep", model, full_train, features, off_test))
        else:
            print(f"[WARN] No official-test recipe for model '{name}'; omitted (no placeholder predictions are fabricated).", flush=True)
            continue
        if len(pred) != len(y_true):
            raise AssertionError(f"{name}: {len(pred)} predictions for {len(y_true)} official test engines")
        rows.append({"Model": name, "N_engines": len(y_true), **evaluate_rul_predictions(y_true, pred)})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)

def train_additional_models_and_stats(
    output_dir: Path | None = None,
    subset: str = "FD004",
    seed: int = DEFAULT_RANDOM_STATE,
    healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
) -> int:
    results_dir = output_dir if output_dir is not None else DEFAULT_RESULTS_DIR / f"maintenance_guide_{subset.lower()}"
    table_dir = results_dir / "tables"
    figure_dir = results_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = table_dir / "rul_holdout_benchmark.csv"
    predictions_path = table_dir / "rul_holdout_predictions.csv"

    if not benchmark_path.exists() or not predictions_path.exists():
        print(f"[ERROR] Baseline tables not found at {table_dir}. Please run pipeline first.")
        return 1

    benchmark_df = pd.read_csv(benchmark_path)
    predictions_df = pd.read_csv(predictions_path)

    existing_models = set(benchmark_df["Model"].unique())
    print(f"[INFO] Existing models in benchmark ({len(existing_models)}):")
    for m in existing_models:
        print(f"  - {m}")

    models_to_train = {}
    
    for extra_name, spec in build_extra_models(seed).items():
        if not any(alias in existing_models for alias in _EXTRA_ALIASES[extra_name]):
            models_to_train[extra_name] = spec

    # Data preparation is unconditional: the official-test evaluation below needs it even when no model is trained here.
    print(f"\n[INFO] Loading {subset} dataset...")
    loader = CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    sensors = loader.informative_sensors(frame.loc[frame["unit_id"].isin(train_units)])
    parts = partition_healthy_and_lifecycle(frame, train_units, test_units, healthy_cycle_limit)

    train_frame = parts["full_lifecycle_train"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_train"]))
    test_frame = parts["full_lifecycle_test"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_test"]))
    features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
    y_test = test_frame["RUL"].to_numpy(dtype=float)

    if not models_to_train:
        print("\n[INFO] All supplementary models (SVM, CNN, RNN, Transformer) are already in the benchmark!")
    else:
        print(f"\n[INFO] Models to train ({len(models_to_train)}):")
        for m in models_to_train:
            print(f"  + {m}")

        new_benchmark_rows = []

        for name, (m_type, model) in models_to_train.items():
            print(f"\n[TRAINING] Fitting '{name}' ({m_type})...", flush=True)
            started = time.perf_counter()
            y_pred = fit_predict(m_type, model, train_frame, features, test_frame)

            fit_time = time.perf_counter() - started
            metrics = evaluate_rul_predictions(y_test, y_pred)
            print(f"  -> '{name}' fit in {fit_time:.2f}s | R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f}", flush=True)

            new_benchmark_rows.append({
                "Model": name,
                **metrics,
                "Fit_Time_s": round(fit_time, 3),
            })
            predictions_df[f"RUL_predicted_{name}"] = y_pred

        # Append to benchmark_df
        updated_benchmark = pd.concat([benchmark_df, pd.DataFrame(new_benchmark_rows)], ignore_index=True)
        updated_benchmark.to_csv(benchmark_path, index=False)
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\n[SUCCESS] Updated {benchmark_path.name} and {predictions_path.name} with all models.")


    # =========================================================================
    # Critical Zone & Censored Test Evaluations
    # =========================================================================
    pred_df = pd.read_csv(predictions_path)
    y_true_all = pred_df["RUL_actual"].to_numpy(dtype=float)
    model_pred_cols = [c for c in pred_df.columns if c.startswith("RUL_predicted_")]

    # Critical Zone (RUL < 50)
    crit_mask = y_true_all < 50.0
    crit_rows = []
    for col in model_pred_cols:
        m_name = col.replace("RUL_predicted_", "")
        crit_metrics = evaluate_rul_predictions(y_true_all[crit_mask], pred_df[col].to_numpy(dtype=float)[crit_mask])
        crit_rows.append({"Model": m_name, "N_cycles": int(np.sum(crit_mask)), **crit_metrics})
    crit_df = pd.DataFrame(crit_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    crit_path = table_dir / "rul_critical_zone_benchmark.csv"
    crit_df.to_csv(crit_path, index=False)
    print(f"[SUCCESS] Critical zone benchmark exported to {crit_path.name}")

    # Official Censored Benchmark (test_{subset}.txt + RUL_{subset}.txt)
    # Official benchmark protocol: train on ALL training engines, score the LAST observed cycle of each
    # truncated test engine against RUL_{subset}.txt (capped at the same RUL ceiling used for training).
    full_train = frame.assign(RUL=compute_piecewise_rul(frame))
    print("\n[INFO] Official test protocol: refitting every benchmarked model on all training engines...", flush=True)
    censored_df = _official_test_benchmark(
        subset, full_train, sensors, features,
        [c.replace("RUL_predicted_", "") for c in model_pred_cols], seed, healthy_cycle_limit,
    )
    if censored_df is not None:
        censored_path = table_dir / "rul_censored_test_benchmark.csv"
        censored_df.to_csv(censored_path, index=False)
        print(f"[SUCCESS] Official censored benchmark exported to {censored_path.name}")
    else:
        print(f"[WARN] test_{subset}.txt / RUL_{subset}.txt not found; official-test benchmark skipped.")

    # =========================================================================
    # Run Thesis Statistical Hypothesis Analysis & p-values
    # =========================================================================
    print("\n" + "=" * 78)
    print("[THESIS ANALYSIS] Running rigorous statistical hypothesis testing with p-values...")
    print("=" * 78)

    analyzer = ThesisHypothesisAnalyzer(random_state=DEFAULT_RANDOM_STATE)

    # 1. Regime variance test (Engine-Block Repeated Measures Friedman)
    t50_res_path = table_dir / "virtual_sensor_residuals_t50.csv"
    if t50_res_path.exists():
        t50_df = pd.read_csv(t50_res_path)
        if "regime_id" in t50_df.columns and "T50" in t50_df.columns:
            if t50_df["regime_id"].nunique() > 1:
                res_h1 = analyzer.test_regime_variance_dominance(t50_df, sensor="T50", unit_col="unit_id")
                print(f"  [H1] Regime Block Friedman: stat={res_h1.test_statistic:.1f}, p={res_h1.p_value:.2e} -> {res_h1.conclusion}")
            else:
                print(f"  [H1] Single-regime dataset ({subset}): skipped (H1 tests variance across >=2 flight regimes).")

        if "residual" in t50_df.columns:
            res_h2 = analyzer.test_physical_degradation_shift(t50_df, healthy_cutoff=healthy_cycle_limit, unit_col="unit_id")
            print(f"  [H2] Degradation Shift (n={t50_df['unit_id'].nunique()}): stat={res_h2.test_statistic:.2f}, p={res_h2.p_value:.2e} -> {res_h2.conclusion}")

            # Fleet-wide Ljung-Box: per-engine tests on the healthy window, combined with Fisher's method
            try:
                res_h3 = analyzer.test_fleet_residual_autocorrelation(
                    t50_df, healthy_cutoff=healthy_cycle_limit, phase_label="frota_saudavel",
                )
                print(f"  [H3] Fleet Healthy Autocorr: Fisher p={res_h3.p_value:.2e} -> {res_h3.conclusion}")
            except ValueError as exc:
                print(f"  [H3] skipped: {exc}")

    # Model superiority tests vs baseline
    y_true = pred_df["RUL_actual"].to_numpy()
    # [BASELINE] Test against the informative "mean RUL by cycle" baseline: every model trivially beats the global mean.
    _base_col = "RUL_predicted_Baseline (Mean RUL by cycle)"
    baseline_pred = pred_df[_base_col].to_numpy() if _base_col in pred_df.columns else np.full(len(y_true), y_true.mean())

    model_cols = [c for c in pred_df.columns if c.startswith("RUL_predicted_") and "Baseline" not in c]
    for col in model_cols:
        m_name = col.replace("RUL_predicted_", "")
        res_m = analyzer.test_model_superiority_vs_baseline(y_true, pred_df[col].to_numpy(), baseline_pred, m_name, pred_df["unit_id"].to_numpy())
        print(f"  [H4] Superiority of {m_name:<26} vs Baseline: W={res_m.test_statistic:.1f}, p={res_m.p_value:.2e}")

    # IEEE CIM 2026 RQ2: GBDT (XGBoost) vs Deep models
    if "RUL_predicted_XGBoost" in pred_df.columns:
        xgb_pred = pred_df["RUL_predicted_XGBoost"].to_numpy()
        for deep_name in ["PyTorch Transformer", "PyTorch RNN (LSTM)", "PyTorch CNN", "MLP", "Support Vector Machine (SVM)"]:
            deep_col = f"RUL_predicted_{deep_name}"
            if deep_col in pred_df.columns:
                res_comp = analyzer.test_gbdt_vs_sequence_model_parity(y_true, xgb_pred, pred_df[deep_col].to_numpy(), deep_name, pred_df["unit_id"].to_numpy())
                print(f"  [H5] IEEE RQ2 XGBoost vs {deep_name:<22}: W={res_comp.test_statistic:.1f}, p={res_comp.p_value:.2e} -> {res_comp.conclusion}")

    # RQ2 with matched representation: XGBoost given the same causal temporal context vs deep sequence models
    if "RUL_predicted_XGBoost (+temporal)" in pred_df.columns:
        xgb_t_pred = pred_df["RUL_predicted_XGBoost (+temporal)"].to_numpy()
        for deep_name in ["PyTorch Transformer", "PyTorch RNN (LSTM)", "PyTorch CNN"]:
            deep_col = f"RUL_predicted_{deep_name}"
            if deep_col in pred_df.columns:
                res_comp = analyzer.test_gbdt_vs_sequence_model_parity(y_true, xgb_t_pred, pred_df[deep_col].to_numpy(), deep_name, pred_df["unit_id"].to_numpy())
                res_comp.hypothesis_id += "_TEMPORAL"
                res_comp.description += " (XGBoost com features temporais causais)"
                print(f"  [H5b] XGBoost(+temporal) vs {deep_name:<22}: W={res_comp.test_statistic:.1f}, p={res_comp.p_value:.2e}")

    # Export hypothesis summary
    hyp_summary, hyp_path = analyzer.export_results(table_dir)
    print(f"\n[SUCCESS] Hypothesis test summary exported to {hyp_path.name} ({len(hyp_summary)} tests).")

    # Compute 95% Bootstrap Confidence Intervals
    print("\n[INFO] Computing 95% Bootstrap Confidence Intervals (2,000 fleet resamples)...", flush=True)
    m_dict = {c.replace("RUL_predicted_", ""): c for c in model_cols}
    boot_ci = analyzer.compute_bootstrap_confidence_intervals(pred_df, m_dict, n_bootstraps=2000)
    boot_ci_path = table_dir / "thesis_bootstrap_ci.csv"
    boot_ci.to_csv(boot_ci_path, index=False)
    print(f"[SUCCESS] Bootstrap 95% CI exported to {boot_ci_path.name} ({len(boot_ci)} models).")
    print(boot_ci[["Model", "R2_Observed", "R2_Mean", "R2_CI_Lower", "R2_CI_Upper", "RMSE_Observed", "RMSE_Mean", "RMSE_CI_Lower", "RMSE_CI_Upper"]].to_string(index=False))

    return 0

class PdMPipeline:
    """Main object-oriented orchestrator for the PdM C-MAPSS analytical pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        loader: CMapssLoader | None = None,
        decoupler: RegimeDecoupler | None = None,
        benchmark: RULBenchmark | None = None,
    ) -> None:
        self.config = config
        self.output_dir = config.resolved_output_dir
        self.table_dir = self.output_dir / "tables"
        self.figure_dir = self.output_dir / "figures"

        self.loader = loader or CMapssLoader()
        n_clusters = 6 if config.subset in {"FD002", "FD004"} else 1
        self.decoupler = decoupler or RegimeDecoupler(n_clusters=n_clusters, random_state=config.seed)
        self.benchmark = benchmark or RULBenchmark(random_state=config.seed)

    def prepare_directories(self) -> None:
        """Create target directories, cleaning prior outputs if requested."""
        if self.config.clean_output and self.output_dir.exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def run_core_workflow(self) -> dict[str, pd.DataFrame]:
        """Execute core virtual-sensor, SPC anomaly detection and RUL benchmark."""
        return run(
            subset=self.config.subset,
            output_dir=self.output_dir,
            seed=self.config.seed,
            include_lstm=self.config.include_lstm,
            include_minirocket=self.config.include_minirocket,
            healthy_cycle_limit=self.config.healthy_cycle_limit,
            sample_engine_id=self.config.unit_id,
            loader=self.loader,
            decoupler=self.decoupler,
            benchmark=self.benchmark,
        )

    def run_auxiliary_spc_horizon(self, horizon: int = 100) -> None:
        """Run auxiliary SPC calibration with a different healthy window limit.

        Runs in an isolated scratch directory so the primary (H = config) outputs are never overwritten;
        only the horizon-suffixed ``*_h{horizon}`` tables are copied back next to the primary tables.
        """
        if self.config.healthy_cycle_limit == horizon:
            return
        aux_dir = self.output_dir / f"_aux_h{horizon}"
        run(
            subset=self.config.subset,
            output_dir=aux_dir,
            seed=self.config.seed,
            include_lstm=False,
            include_minirocket=False,
            healthy_cycle_limit=horizon,
            sample_engine_id=self.config.unit_id,
            loader=self.loader,
        )
        for source in (aux_dir / "tables").glob(f"*_h{horizon}.csv"):
            shutil.copy2(source, self.table_dir / source.name)
        shutil.rmtree(aux_dir, ignore_errors=True)

    def run_additional_models_and_statistics(self) -> None:
        """Train supplementary models (SVM, CNN, RNN, Transformer) and run hypothesis testing."""
        train_additional_models_and_stats(
            output_dir=self.output_dir,
            subset=self.config.subset,
            seed=self.config.seed,
            healthy_cycle_limit=self.config.healthy_cycle_limit,
        )

    def run_monitoring_layer(self) -> None:
        """RUL uncertainty, criticality levels, alert events and grounded reports (see monitoring/layer.py)."""
        from pdm_cmaps.monitoring.layer import run_monitoring_layer

        run_monitoring_layer(
            subset=self.config.subset,
            output_dir=self.output_dir,
            seed=self.config.seed,
            healthy_cycle_limit=self.config.healthy_cycle_limit,
            sample_engine_id=self.config.unit_id,
            loader=self.loader,
        )

    def run_figure_generation(self) -> None:
        """Render publication-quality thesis diagnostic figures."""
        if self.config.subset == "FD004":
            figure_1_noise_vs_damage(self.output_dir)
            figure_2_physical_cascade(self.output_dir)
            figure_3_autocorrelation_evidence(self.output_dir)
            figure_4_fleet_persistence_tradeoff(self.output_dir)
        else:
            print(f"[figures] Thesis diagnostic figures 1-4 are specific to FD004 case study (skipped for {self.config.subset}).")

    def execute(self) -> int:
        """Execute the end-to-end pipeline in ordered stages."""
        t_start = time.perf_counter()
        sep = "=" * 78
        print(f"\n{sep}")
        print(f"[PdM PIPELINE] Initializing End-to-End Pipeline ({self.config.subset})")
        print(f"{sep}")
        print("  Configuration:")
        print(f"    - Subset: {self.config.subset}")
        print(f"    - Seed: {self.config.seed}")
        print(f"    - Healthy window limit (H): {self.config.healthy_cycle_limit}")
        print(f"    - Target unit ID: {self.config.unit_id}")
        print(f"    - MiniRocket: {self.config.include_minirocket}")
        print(f"    - LSTM: {self.config.include_lstm}")
        print(f"    - Output: {self.output_dir}")

        # Phase 0: Setup
        self.prepare_directories()
        run_aux = self.config.healthy_cycle_limit != 100
        progress = ProgressTracker(5 if run_aux else 4)  # prints "[i/N] fase ... | decorrido | restante ~"

        # Phase 1: Core PdM Pipeline (Regimes, Virtual Sensor, Envelopes, Baseline Models)
        progress.start("fase 1: pipeline analítico (regimes, sensor virtual, envelope, modelos-base)")
        self.run_core_workflow()
        progress.finish("fase 1")

        # Phase 2: Auxiliary SPC Horizon (H = 100)
        if run_aux:
            progress.start("fase 2: horizonte SPC auxiliar (H = 100)")
            self.run_auxiliary_spc_horizon(horizon=100)
            progress.finish("fase 2")

        # Phase 3: Additional Models & Hypothesis Testing
        progress.start("fase 3: modelos adicionais e testes estatísticos")
        self.run_additional_models_and_statistics()
        progress.finish("fase 3")

        # Phase 4: monitoring layer (uncertainty, criticality, alerts, grounded reports)
        progress.start("fase 4: camada de monitoramento (incerteza, criticidade, alertas, relatórios)")
        self.run_monitoring_layer()
        progress.finish("fase 4")

        # Phase 5: Thesis Figures
        progress.start("fase 5: figuras")
        self.run_figure_generation()
        progress.finish("fase 5")

        elapsed = time.perf_counter() - t_start
        print(f"\n{sep}")
        print(f"[PdM PIPELINE] Completed successfully in {elapsed:.1f}s")
        print(f"  - Figures: {self.figure_dir}")
        print(f"  - Tables:  {self.table_dir}")
        print(f"{sep}\n")
        return 0

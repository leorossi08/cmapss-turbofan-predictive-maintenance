"""Repeated RUL benchmark (RQ-A) over subsets x seeds, in both evaluation protocols.

Uses the same models and recipes as pipeline.py.

WHAT   For every subset and seed, fits the main model families and scores them in
         holdout   75/25 engine split of train_FDxxx (the seed changes the split AND the model init);
         official  refit on all train_FDxxx engines, last cycle of test_FDxxx vs RUL_FDxxx (the seed
                   changes only the model initialisation, so its spread is pure training variance).
       Each (subset, seed, protocol, model) row is appended to a CSV as soon as it is computed, so the run
       can be stopped and resumed.
WHY    pipeline.py benchmarks one seed. "Model A is competitive with model B" needs a spread over seeds.
CONCEPTS
  [GROUPED-CV][LEAKAGE] same engine-level split and train-only fitting as pipeline.py.
  [PSEUDOREPLICATION] the per-engine MAE (holdout) or per-engine error (official) is stored so that
                 paired tests compare engines, not cycles.

Run from the repo root (long, mostly the deep models; nothing else heavy at the same time):
    .venv/bin/python -m pdm_cmaps.experiments.benchmark_repeated --output-dir results/benchmark_repeated
    .venv/bin/python -m pdm_cmaps.experiments.benchmark_repeated --output-dir results/benchmark_repeated --aggregate-only
"""

from __future__ import annotations

from pdm_cmaps.config import DEFAULT_RESULTS_DIR

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RUL_CAP
from pdm_cmaps.data.features import CausalFeatureEngineer
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.domain.rul import compute_piecewise_rul, evaluate_rul_predictions
from pdm_cmaps.data.splits import split_engine_units
from pdm_cmaps.modeling.minirocket import MiniRocketRULModel
from pdm_cmaps.modeling.registry import get_model_pool
from pdm_cmaps.pipeline.extra_models import build_extra_models, fit_predict

TABULAR = ("Ridge", "Random Forest", "XGBoost", "XGBoost (NASA Loss)", "LightGBM", "CatBoost", "MLP")
TEMPORAL = ("XGBoost", "LightGBM")
DEEP = ("PyTorch CNN", "PyTorch RNN (LSTM)", "PyTorch Transformer")


def _models(seed: int) -> list[tuple[str, str, object, bool]]:
    """(name, kind, estimator, uses temporal features) in a fixed order."""
    pool = get_model_pool(seed)
    extra = build_extra_models(seed)
    out = [(name, "sklearn", pool[name], False) for name in TABULAR]
    out += [(f"{name} (+temporal)", "sklearn", get_model_pool(seed)[name], True) for name in TEMPORAL]
    out += [(name, *extra[name], False) for name in DEEP]
    out.append(("MiniRocket + RidgeCV", "deep", MiniRocketRULModel(seq_length=15, num_kernels=1000, random_state=seed), False))
    return out


def _prepare(frame: pd.DataFrame, train_df: pd.DataFrame, eval_df: pd.DataFrame, sensors: list[str]):
    base = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
    eng = CausalFeatureEngineer(sensors, windows=(5, 10)).fit(train_df, DEFAULT_HEALTHY_CYCLE_LIMIT)
    tr_fe, ev_fe = eng.transform(train_df), eng.transform(eval_df)
    temporal = [*base, *[c for c in tr_fe.columns if c not in train_df.columns]]
    return base, temporal, tr_fe, ev_fe


def run(subsets, seeds, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows_path, err_path = out / "benchmark_rows.csv", out / "benchmark_engine_errors.csv"
    done = set()
    if rows_path.exists():
        prev = pd.read_csv(rows_path)
        done = set(zip(prev["Subset"], prev["Seed"], prev["Protocol"], prev["Model"]))
    loader = CMapssLoader()
    for subset in subsets:
        frame = loader.load(subset)
        frame["RUL"] = compute_piecewise_rul(frame)
        off_test, true_rul = loader.load_test(subset)
        y_off = np.clip(true_rul.to_numpy(dtype=float), 0.0, float(DEFAULT_RUL_CAP))
        for seed in seeds:
            tr_units, te_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
            train_df = frame.loc[frame["unit_id"].isin(tr_units)].reset_index(drop=True)
            test_df = frame.loc[frame["unit_id"].isin(te_units)].reset_index(drop=True)
            sensors = loader.informative_sensors(train_df)
            setups = {
                "holdout": (train_df, test_df, *_prepare(frame, train_df, test_df, sensors)),
                "official": (frame, off_test, *_prepare(frame, frame, off_test, sensors)),
            }
            for protocol, (tr, ev, base, temporal, tr_fe, ev_fe) in setups.items():
                for name, kind, model, use_temporal in _models(seed):
                    if (subset, seed, protocol, name) in done:
                        continue
                    t0 = time.perf_counter()
                    if use_temporal:
                        pred = fit_predict(kind, model, tr_fe, temporal, ev_fe)
                        units = ev_fe["unit_id"].to_numpy()
                    else:
                        pred = fit_predict(kind, model, tr, base, ev)
                        units = ev["unit_id"].to_numpy()
                    if protocol == "holdout":
                        truth = (ev_fe if use_temporal else ev)["RUL"].to_numpy(dtype=float)
                        metrics = evaluate_rul_predictions(truth, pred)
                        err = pd.DataFrame({"unit_id": units, "abs_err": np.abs(truth - pred)}).groupby("unit_id")["abs_err"].mean()
                    else:
                        last = pd.Series(pred).groupby(units).last().sort_index()
                        metrics = evaluate_rul_predictions(y_off, last.to_numpy())
                        err = pd.Series(np.abs(y_off - last.to_numpy()), index=last.index)
                    row = {"Subset": subset, "Seed": seed, "Protocol": protocol, "Model": name, **metrics,
                           "Fit_s": round(time.perf_counter() - t0, 1)}
                    pd.DataFrame([row]).to_csv(rows_path, mode="a", header=not rows_path.exists(), index=False)
                    e = err.rename("abs_err").reset_index().rename(columns={"index": "unit_id"})
                    e.assign(Subset=subset, Seed=seed, Protocol=protocol, Model=name).to_csv(
                        err_path, mode="a", header=not err_path.exists(), index=False)
                    print(f"[bench] {subset} seed={seed} {protocol:8s} {name:24s} RMSE {row['RMSE']:.2f} ({row['Fit_s']} s)", flush=True)


def aggregate(out: Path) -> pd.DataFrame:
    rows = pd.read_csv(out / "benchmark_rows.csv")
    summary = rows.groupby(["Protocol", "Subset", "Model"]).agg(
        R2_mean=("R2", "mean"), R2_std=("R2", "std"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        Mean_NASA_mean=("Mean_NASA_Score", "mean"), Runs=("Seed", "size")).reset_index()
    rows["Rank"] = rows.groupby(["Protocol", "Subset", "Seed"])["RMSE"].rank()
    ranks = rows.groupby(["Protocol", "Subset", "Model"]).agg(Rank_mean=("Rank", "mean"), Top3_runs=("Rank", lambda r: int((r <= 3).sum()))).reset_index()
    summary = summary.merge(ranks, on=["Protocol", "Subset", "Model"]).sort_values(["Protocol", "Subset", "RMSE_mean"])
    summary.to_csv(out / "benchmark_repeated_summary.csv", index=False)

    # paired Wilcoxon per seed, best tabular (XGBoost +temporal) vs each deep model, on per-engine errors;
    # p-values combined over seeds with Fisher's method (seeds are independent splits in the holdout protocol)
    err = pd.read_csv(out / "benchmark_engine_errors.csv")
    tests = []
    ref = "XGBoost (+temporal)"
    for (protocol, subset), g in err.groupby(["Protocol", "Subset"]):
        for other in [*DEEP, "MiniRocket + RidgeCV", "XGBoost", "LightGBM (+temporal)"]:
            ps, diffs = [], []
            for seed, gs in g.groupby("Seed"):
                a = gs.loc[gs["Model"] == ref].set_index("unit_id")["abs_err"]
                b = gs.loc[gs["Model"] == other].set_index("unit_id")["abs_err"]
                both = a.index.intersection(b.index)
                if len(both) < 5:
                    continue
                ps.append(stats.wilcoxon(a[both], b[both]).pvalue)
                diffs.append(float((b[both] - a[both]).median()))
            if ps:
                fisher = stats.combine_pvalues(ps, method="fisher").pvalue
                tests.append({"Protocol": protocol, "Subset": subset, "Comparison": f"{ref} vs {other}", "Seeds": len(ps),
                              "Median_err_diff_other_minus_ref": float(np.mean(diffs)), "p_per_seed_max": float(max(ps)),
                              "p_fisher": float(fisher)})
    t = pd.DataFrame(tests)
    t.to_csv(out / "benchmark_repeated_tests.csv", index=False)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "benchmark_repeated")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.aggregate_only:
        run(args.subsets, args.seeds, args.output_dir)
    summary = aggregate(args.output_dir)
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

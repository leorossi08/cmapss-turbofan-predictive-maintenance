"""Healthy-window sensitivity study: how many initial cycles should calibrate the SPC baseline?

WHAT: repeats the T50 virtual-sensor EWMA chart and the 13-residual MEWMA chart for several
healthy windows H (50, 75, 100 and a relative window), over subsets x seeds, on the held-out
engines of the same seeded engine split as monitoring/layer.py / pipeline.run().

WHY: a longer window should give less noisy control limits (more calibration rows) and fewer false
alarms, but (i) monitoring only starts after cycle H (blind period) and (ii) the calibration set
starts to include early degradation, which inflates sigma / the UCL and delays detection.

[ASSUMPTION] "Relative" window: 25 % of each engine's own life would use the end of life of the
engine being monitored (unknown online). The causal version used here is
    H_rel = round(0.25 x median life of the TRAINING engines)   (one integer per subset and seed).

Metrics that do NOT depend on H (the detector's own `healthy_far` / `premature_alarm` are measured
on cycles <= H, so they change window and denominator with H and cannot be compared across H):
  * early_alarm_pct     first alarm at a cycle whose true RUL (EOL - cycle) >= 125 (piecewise cap)
  * far_ref_pct         per-cycle breach rate on monitored cycles with RUL >= 125
  * far_common_pct      same, restricted to cycles > 100 so every H is scored on the SAME cycles
  * detection, lead time (EOL - first alarm), blind period (= H), engines with EOL <= H + k
  * contamination_pct   % of training calibration rows (cycle <= H) with true RUL < 125
  * calibration noise   engine-cluster bootstrap CV of the T50 per-regime residual sigma and of
                        the MEWMA UCL (quantile of OOF healthy T2, resampled by engine)

[LEAKAGE] everything is fit on training engines only; RUL_true is used only for scoring.
[GROUPED-CV] bootstrap resamples whole engines.

Run:  .venv/bin/python -m pdm_cmaps.experiments.healthy_window_study            (4 subsets x 3 seeds x 4 windows)
      .venv/bin/python -m pdm_cmaps.experiments.healthy_window_study --aggregate-only
Output: results/healthy_window_study/{runs,tables,figures}
"""

from __future__ import annotations

from pdm_cmaps.config import DEFAULT_RESULTS_DIR

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm_cmaps.domain.schema import DEFAULT_RUL_CAP
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.splits import split_engine_units
from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.reporting import viz_style

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "healthy_window_study"
FIXED_WINDOWS = (50, 75, 100)
REL_FRACTION = 0.25
COMMON_START = 100  # far_common_pct is scored only on cycles > this (largest fixed H)
N_BOOT = 200


def _windows(train_df: pd.DataFrame) -> dict[str, int]:
    life = train_df.groupby("unit_id")["cycle"].max()
    h_rel = int(round(REL_FRACTION * float(life.median())))
    return {**{f"H={h}": h for h in FIXED_WINDOWS}, f"H=25% vida (={h_rel})": h_rel}


def _cluster_boot_cv(values_by_unit: pd.DataFrame, stat, rng: np.random.Generator) -> tuple[float, float]:
    """Engine-cluster bootstrap of ``stat(frame)``; returns (point estimate, coefficient of variation %)."""
    units = values_by_unit["unit_id"].unique()
    groups = {u: g for u, g in values_by_unit.groupby("unit_id")}
    point = float(stat(values_by_unit))
    reps = []
    for _ in range(N_BOOT):
        pick = rng.choice(units, size=units.size, replace=True)
        reps.append(stat(pd.concat([groups[u] for u in pick], ignore_index=True)))
    reps = np.asarray(reps, dtype=float)
    return point, float(100 * reps.std(ddof=1) / abs(reps.mean())) if reps.mean() else float("nan")


def _reference_metrics(per_cycle: pd.DataFrame, first_alarm: dict, H: int, k: int) -> dict:
    """H-independent false-alarm metrics. ``per_cycle``: unit_id, cycle, breach, rul_true."""
    pc = per_cycle.copy()
    mon = pc["cycle"] > H
    ref = mon & (pc["rul_true"] >= DEFAULT_RUL_CAP)
    common = (pc["cycle"] > COMMON_START) & (pc["rul_true"] >= DEFAULT_RUL_CAP)
    eol = pc.groupby("unit_id")["cycle"].max()
    early = [a is not None and (eol[u] - a) >= DEFAULT_RUL_CAP for u, a in first_alarm.items()]
    engines_with_ref = int(pc.loc[ref, "unit_id"].nunique())
    return {
        "early_alarm_pct": 100 * float(np.mean(early)),
        "far_ref_pct": 100 * float(pc.loc[ref, "breach"].mean()) if ref.any() else float("nan"),
        "far_common_pct": 100 * float(pc.loc[common, "breach"].mean()) if common.any() else float("nan"),
        "engines_with_ref_cycles": engines_with_ref,
        "engines_unmonitorable": int((eol <= H + k).sum()),
        "monitored_cycles_pct": 100 * float(mon.mean()),
    }


def run_one(subset: str, seed: int, out: Path, mewma_offset: bool = True) -> pd.DataFrame:
    """All windows for one (subset, seed); writes runs/<subset>_seed<seed>.csv and returns it."""
    path = out / "runs" / f"{subset}_seed{seed}.csv"
    if path.exists():
        return pd.read_csv(path)
    loader = CMapssLoader()
    frame = loader.load(subset)
    frame["rul_true"] = frame.groupby("unit_id")["cycle"].transform("max") - frame["cycle"]
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    train_df = frame.loc[frame["unit_id"].isin(train_units)]
    test_df = frame.loc[frame["unit_id"].isin(test_units)]
    n_regimes = 6 if subset.upper() in {"FD002", "FD004"} else 1
    regimes = RegimeDecoupler(n_clusters=n_regimes, random_state=seed).fit(train_df)
    tr, te = regimes.transform(train_df), regimes.transform(test_df)
    rul_lookup = te.set_index(["unit_id", "cycle"])["rul_true"]
    rng = np.random.default_rng(seed)
    rows = []
    for label, H in _windows(train_df).items():
        print(f"[window] {subset} seed={seed} {label}", flush=True)
        calib = tr.loc[tr["cycle"] <= H]
        base = {"Subset": subset, "Seed": seed, "Window": label, "H": H,
                "calibration_rows": int(len(calib)),
                "contamination_pct": 100 * float((calib["rul_true"] < DEFAULT_RUL_CAP).mean())}

        # --- T50 univariate chart (same config as monitoring.compute_fleet_alarms) ---
        vs = VirtualSensor(target_sensor="T50", mode="exogenous", ewma_alpha=0.10, min_consecutive_breaches=2,
                           healthy_cycle_limit=H, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace, per_engine, summary = vs.evaluate(tr, te)
        alarms = {int(r.unit_id): (None if pd.isna(r.first_alarm_cycle) else int(r.first_alarm_cycle)) for r in per_engine.itertuples()}
        breach = (trace["ewma_abs_z"] >= trace["ewma_ucl"]) & (trace["z_score"].abs() >= vs.threshold_sigma)
        pc = trace[["unit_id", "cycle"]].assign(breach=breach.to_numpy())
        pc["rul_true"] = rul_lookup.reindex(pd.MultiIndex.from_frame(pc[["unit_id", "cycle"]])).to_numpy()
        cal = vs.calibration_predictions_[["unit_id", "regime_id", "residual"]]
        sigma, sigma_cv = _cluster_boot_cv(cal, lambda f: f.groupby("regime_id")["residual"].std().mean(), rng)
        rows.append({**base, "Detector": "T50 EWMA", **summary, **_reference_metrics(pc, alarms, H, 2),
                     "calib_stat": sigma, "calib_stat_name": "sigma_residuo_T50", "calib_cv_pct": sigma_cv})

        # --- MEWMA on 13 residuals ---
        mv = MultivariateResidualMonitor(healthy_cycle_limit=H, random_state=seed, engine_offset=mewma_offset).fit(tr)
        scored = mv.score(te)
        mv_alarms, _, mv_summary = mv.alarms(scored)
        pc = scored[["unit_id", "cycle", "breach"]].copy()
        pc["rul_true"] = rul_lookup.reindex(pd.MultiIndex.from_frame(pc[["unit_id", "cycle"]])).to_numpy()
        ucl, ucl_cv = _cluster_boot_cv(mv.calibration_t2_, lambda f: np.quantile(f["t2"], 1 - mv.alpha), rng)
        mv_summary.pop("n_sensors", None)
        rows.append({**base, "Detector": "MEWMA (13 resíduos)", **mv_summary, **_reference_metrics(pc, mv_alarms, H, mv.k),
                     "calib_stat": ucl, "calib_stat_name": "UCL_MEWMA", "calib_cv_pct": ucl_cv})
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


METRICS = ["H", "calibration_rows", "contamination_pct", "calib_stat", "calib_cv_pct", "detection_rate_pct",
           "mean_lead_time_detected_cycles", "early_alarm_pct", "far_ref_pct", "far_common_pct",
           "mean_healthy_false_alarm_pct", "premature_alarm_pct", "engines_unmonitorable", "monitored_cycles_pct"]


def aggregate(out: Path) -> pd.DataFrame:
    runs = pd.concat([pd.read_csv(p) for p in sorted((out / "runs").glob("*.csv"))], ignore_index=True)
    runs["Window_kind"] = runs["Window"].str.replace(r" \(=\d+\)", "", regex=True)
    g = runs.groupby(["Subset", "Detector", "Window_kind"], sort=False)[METRICS]
    summary = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    summary["n_seeds"] = g.size().to_numpy()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    runs.to_csv(tables / "healthy_window_runs.csv", index=False)
    summary.to_csv(tables / "healthy_window_summary.csv", index=False)
    plot_tradeoff(summary, out / "figures" / "fig_healthy_window_tradeoff.png")
    return summary


def plot_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    """Rows = detector, columns = metric; one line per subset, x = window (ordered by mean H)."""
    viz_style.apply_style()
    panels = [("calib_cv_pct", "Ruído de calibração\n(CV bootstrap, %)"),
              ("contamination_pct", "Contaminação da calibração\n(% linhas com RUL < 125)"),
              ("far_common_pct", "Falso alarme por ciclo\n(ciclos > 100 e RUL ≥ 125, %)"),
              ("early_alarm_pct", "Motores com alarme precoce\n(1º alarme com RUL ≥ 125, %)"),
              ("mean_lead_time_detected_cycles", "Antecedência média\n(ciclos antes da falha)"),
              ("detection_rate_pct", "Taxa de detecção (%)")]
    detectors = list(dict.fromkeys(summary["Detector"]))
    subsets = list(dict.fromkeys(summary["Subset"]))
    colors = dict(zip(subsets, [*viz_style.SERIES, viz_style.INK_2]))
    markers = dict(zip(subsets, ["o", "s", "^", "D"]))
    fig, axes = plt.subplots(len(detectors), len(panels), figsize=(4.0 * len(panels), 3.6 * len(detectors)), squeeze=False)
    for i, det in enumerate(detectors):
        for j, (col, title) in enumerate(panels):
            ax = axes[i, j]
            for sub in subsets:
                d = summary[(summary["Detector"] == det) & (summary["Subset"] == sub)].sort_values("H_mean")
                ax.errorbar(d["H_mean"], d[f"{col}_mean"], yerr=d[f"{col}_std"], color=colors[sub], marker=markers[sub],
                            lw=viz_style.LINE_W, ms=6, capsize=3, label=sub)
            viz_style.style_axes(ax)
            ax.set_title(title if i == 0 else "", fontsize=10)
            ax.set_xticks(sorted(set(FIXED_WINDOWS)))
            if j == 0:
                ax.set_ylabel(det, fontsize=10, fontweight="bold")
            if i == len(detectors) - 1:
                ax.set_xlabel("Janela saudável H (ciclos)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Janela saudável: ruído de calibração × contaminação × falso alarme × antecedência "
                 "(média ± dp de 3 seeds; ponto não inteiro em x = janela relativa 25 % da vida mediana)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    summary.to_csv(path.with_suffix(".csv"), index=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=None, help=f"padrão: {OUT_DEFAULT} (protocolo principal, com o deslocamento do motor em todos os detectores); --no-mewma-offset grava em {OUT_DEFAULT}_offset_so_t50")
    p.add_argument("--mewma-offset", action=argparse.BooleanOptionalAction, default=True, help="(padrão ligado desde 2026-11-18; --no-mewma-offset reproduz o protocolo anterior, só o T50) deslocamento do motor (janela = H de cada linha) também na MEWMA; o T50 EWMA já o usa por padrão; falso alarme fora da janela: far_ref_pct e far_common_pct")
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args(argv)
    if a.output_dir is None:
        a.output_dir = OUT_DEFAULT if a.mewma_offset else OUT_DEFAULT.with_name(OUT_DEFAULT.name + "_offset_so_t50")
    if not a.aggregate_only:
        jobs = [(s, seed) for s in a.subsets for seed in a.seeds]
        if a.jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=a.jobs)(delayed(run_one)(s, seed, a.output_dir, a.mewma_offset) for s, seed in jobs)
        else:
            for s, seed in jobs:
                run_one(s, seed, a.output_dir, a.mewma_offset)
    summary = aggregate(a.output_dir)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["Subset", "Detector", "Window_kind", "H_mean", "contamination_pct_mean", "calib_cv_pct_mean",
                "far_common_pct_mean", "early_alarm_pct_mean", "mean_lead_time_detected_cycles_mean", "detection_rate_pct_mean"]
        print(summary[cols].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

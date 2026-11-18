"""Hard, numerous, randomly drawn faults: does a learning model beat the regime mean as virtual sensor?

WHY THIS FILE EXISTS: with the simple faults of fault_detection_study.py the "regime mean, no ML" baseline already
detects most faults, and the learning models differ little. Two structural reasons, not a lack of fault difficulty:
  (1) the inputs of the exogenous virtual sensor (settings + ambient channels) are nearly constant inside a regime,
      so there is little for a model to learn (measured: the learning models cut the healthy residual sd by only
      ~5-25 % relative to the regime mean, median ~10 %);
  (2) a bigger fault is easier for EVERY model; difficulty alone does not create a gap between models.
So the fault family is made richer in the ways that can legitimately separate models, not "harder until ML wins":
  * many faults per engine, drawn by a documented random process (seeded, NOT tuned to any estimator);
  * shapes: step, ramp, saturating, intermittent (on/off bursts), plus variance increase for isolated sensors;
  * magnitude log-uniform from nearly undetectable to moderate (so power is measured over a continuum);
  * two families: "processo" (coherent multi-sensor faults along the real end-of-life signature, on 3, 6 or 13
    sensors) and "sensor" (one sensor only, random sign: an instrument-type fault);
  * condition-dependent faults: in half of the multi-regime faults the gain differs per operating regime;
  * two information modes for the virtual sensor: "exogenous" (thesis detector) and "cross_sensor" (also sees the
    other sensors; a consistency check where a learning model has real information to use).

PRE-REGISTERED HYPOTHESES AND DECISION RULE (written before the first run; results are reported whatever they are)
  H1 exogenous mode, processo faults: learning models give a small positive power gain over the regime mean.
     Expectation from the residual ratio (~0.9): at most a few points.
  H2 cross_sensor mode, sensor faults: learning models beat the regime mean clearly (they predict a sensor from the
     others, an isolated bias breaks that relation); processo faults are partly ABSORBED (coherent shifts keep the
     relation), so cross_sensor may lose to exogenous there.
  Decision: an advantage is claimed only if the 95 % cluster-bootstrap CI (clusters = recipient engines) of the paired
  mean power difference vs the regime mean excludes 0. No estimator hyperparameter search; no fault re-drawing.

[LEAKAGE] estimators fit on TRAINING engines (healthy window H); faults are injected into TEST-engine bootstraps and
reach the charts as raw columns; the answer key (family, shape, magnitude, t0, sensors) is used only in scoring.
[ASSUMPTION] fault shapes/magnitudes are plausible, not simulated physics; the end-of-life signature (real data)
fixes the physical direction of "processo" faults. False-alarm budget 0.2 per 100 monitored cycles, matched per chart
on independent healthy controls; single-sensor charts have that budget EACH (a 13-chart panel would multiply it).

Run:  .venv/bin/python -m pdm_cmaps.experiments.fault_hard_study [--subsets FD004 FD002 FD001] [--jobs 3]     /  --aggregate-only
Output: results/falhas_dificeis/{runs,tabelas,figuras}
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
from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor
from pdm_cmaps.experiments.fault_detection_study import BUDGETS, HORIZON_CTRL, MIN_POOL, POST_DET, alarm_matrix, bootstrap_series, control_far
from pdm_cmaps.experiments.fault_injection import INJECT_SENSORS, World, blind_columns, inject, load_world
from pdm_cmaps.experiments.fault_model_study import BASELINE, ESTIMATORS, MEWMA, make_estimator, model_scores
from pdm_cmaps.reporting import viz_style

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "falhas_dificeis"
DEFAULT_ESTIMATORS = ["Extra Trees (padrão)", "LightGBM", "Ridge", "MLP", BASELINE]
MODES = ("exogenous", "cross_sensor")
SHAPES = ("degrau", "rampa", "saturante", "intermitente")
T0_RANGE = (10, 60)             # t0 = H + U[10, 60]
HORIZON_EXTRA = T0_RANGE[1] + POST_DET
PROC_MAG = (0.03, 0.40)         # fraction of the end-of-life signature
SENS_MAG = (0.5, 4.0)           # healthy sigma units
BINS = {"processo": [0.03, 0.06, 0.12, 0.25, 0.40001], "sensor": [0.5, 1.0, 2.0, 4.00001]}
BOOT = 2000


# ============================================================================
# fault family (documented random process)
# ============================================================================

def draw_fault(world: World, rng: np.random.Generator, H: int) -> dict:
    k_all = len(INJECT_SENSORS)
    family = "processo" if rng.random() < 0.5 else "sensor"
    spec = {"family": family, "t0": H + int(rng.integers(T0_RANGE[0], T0_RANGE[1] + 1)), "L": int(rng.integers(20, 81)), "period": int(rng.integers(8, 21)),
            "weights": np.zeros(k_all), "gain": None, "shape": str(rng.choice(SHAPES))}
    if family == "processo":
        k = int(rng.choice([3, 6, k_all]))
        idx = np.sort(rng.choice(k_all, size=k, replace=False))
        spec["weights"][idx] = world.signature[idx]
        spec["mag"] = float(np.exp(rng.uniform(np.log(PROC_MAG[0]), np.log(PROC_MAG[1]))))
    else:
        j = int(rng.integers(0, k_all))
        idx = np.array([j])
        spec["weights"][j] = float(rng.choice([-1.0, 1.0]))
        spec["mag"] = float(np.exp(rng.uniform(np.log(SENS_MAG[0]), np.log(SENS_MAG[1]))))
        if rng.random() < 0.2:
            spec["shape"] = "variancia"
    spec["sensors"] = idx.tolist()
    if world.regimes.n_clusters > 1 and rng.random() < 0.5:
        spec["gain"] = rng.uniform(0.1, 1.5, size=world.regimes.n_clusters)
    return spec


def fault_dz(spec: dict, base: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Standardized-unit perturbation (rows x INJECT_SENSORS) for one fault spec."""
    c, t0 = base["cycle"].to_numpy(), spec["t0"]
    on = c >= t0
    shape = spec["shape"]
    if shape == "degrau":
        g = on.astype(float)
    elif shape == "rampa":
        g = np.clip((c - t0 + 1) / spec["L"], 0, 1) * on
    elif shape == "saturante":
        g = (1 - np.exp(-(c - t0 + 1) / (spec["L"] / 3))) * on
    elif shape == "intermitente":
        g = (((c - t0) // spec["period"]) % 2 == 0) * on
    elif shape == "variancia":
        out = np.zeros((len(base), len(INJECT_SENSORS)))
        j = spec["sensors"][0]
        out[on, j] = rng.normal(0.0, spec["mag"], int(on.sum()))
        return out
    else:
        raise ValueError(shape)
    gain = spec["gain"][base["regime_id"].to_numpy(dtype=int)] if spec["gain"] is not None else 1.0
    return spec["mag"] * (g * gain)[:, None] * spec["weights"][None, :]


def build_series(world: World, seed: int, n_ctrl: int, n_reps: int, n_faults: int, max_engines: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    H = world.H
    engines = {int(u): g.sort_values("cycle") for u, g in world.test.groupby("unit_id")}
    recipients = [u for u in sorted(engines) if (engines[u]["rul_true"] >= DEFAULT_RUL_CAP).sum() >= MIN_POOL]
    if max_engines:
        recipients = recipients[:max_engines]
    horizon = H + HORIZON_EXTRA
    frames, meta, bid = [], [], 0
    for u in recipients:
        for _ in range(n_ctrl):
            bid += 1
            frames.append(blind_columns(bootstrap_series(engines[u], HORIZON_CTRL, rng), bid))
            meta.append({"blind_id": bid, "recipient": u, "role": "controle", "family": "", "shape": "", "mag": np.nan, "k": 0, "sensors": "", "gain": False, "t0": 0})
        for _ in range(n_reps):
            base = bootstrap_series(engines[u], horizon, rng)
            for _ in range(n_faults):
                spec = draw_fault(world, rng, H)
                bid += 1
                frames.append(blind_columns(inject(world, base, fault_dz(spec, base, rng)), bid))
                meta.append({"blind_id": bid, "recipient": u, "role": "falha", "family": spec["family"], "shape": spec["shape"], "mag": spec["mag"], "k": len(spec["sensors"]),
                             "sensors": ",".join(INJECT_SENSORS[j] for j in spec["sensors"]), "gain": spec["gain"] is not None, "t0": spec["t0"]})
    blind = pd.concat(frames, ignore_index=True).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return blind, pd.DataFrame(meta)


# ============================================================================
# scoring per (mode, estimator)
# ============================================================================

def _matched(scores: pd.DataFrame, meta_idx: pd.DataFrame, H: int, budget: float):
    ids, mat, last = alarm_matrix(scores, np.maximum(meta_idx["t0"], H + 1))
    m = meta_idx.loc[ids]
    ctrl = (m["role"] == "controle").to_numpy()
    far, _ = control_far(mat[ctrl], last[ctrl], H)
    ok = np.flatnonzero(far <= budget)
    if not ok.size:
        return None
    i = int(ok[0])
    fault = ~ctrl
    alarm = mat[fault, i]
    t0 = m.loc[fault, "t0"].to_numpy()
    return pd.DataFrame({"blind_id": ids[fault], "detected": ~np.isnan(alarm), "delay": alarm - t0}), float(far[i])


def score_estimator(world: World, tr: pd.DataFrame, te: pd.DataFrame, meta: pd.DataFrame, est: str, mode: str, seed: int, budget: float = BUDGETS[0],
                    mewma_offset: bool = True) -> pd.DataFrame:
    H = world.H
    ctrl_ids = set(meta.loc[meta["role"] == "controle", "blind_id"].astype(int))
    mv = MultivariateResidualMonitor(healthy_cycle_limit=H, random_state=seed, estimator=make_estimator(est, seed), mode=mode, engine_offset=mewma_offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mv.fit(tr)
        scores, _ = model_scores(mv, te, ctrl_ids, H)
    midx = meta.set_index("blind_id")
    rows = []
    res = _matched(scores[MEWMA], midx, H, budget)
    if res is not None:
        rows.append(res[0].assign(Chart="MEWMA", far_chart=res[1]))
    for j, sensor in enumerate(INJECT_SENSORS):
        if sensor not in scores:
            continue
        res = _matched(scores[sensor], midx, H, budget)
        if res is None:
            continue
        per, far = res
        mine = midx.loc[per["blind_id"], "sensors"].to_numpy() == sensor  # isolated faults on exactly this sensor
        fam = midx.loc[per["blind_id"], "family"].to_numpy() == "sensor"
        rows.append(per[mine & fam].assign(Chart="sensor afetado", far_chart=far))
    return pd.concat(rows, ignore_index=True).assign(Mode=mode, Estimator=est)


def run_one(subset: str, seed: int, H: int, out: Path, n_ctrl: int, n_reps: int, n_faults: int, max_engines: int | None, estimators: list[str],
            mewma_offset: bool = True) -> None:
    tag = f"{subset}_seed{seed}_H{H}"
    path = out / "runs" / f"{tag}_series.csv"
    if path.exists():
        return
    world = load_world(subset, seed, H, loader=CMapssLoader())
    blind, meta = build_series(world, seed, n_ctrl, n_reps, n_faults, max_engines)
    print(f"[{tag}] {int((meta['role'] == 'falha').sum())} faults, {int((meta['role'] == 'controle').sum())} controls", flush=True)
    tr, te = world.regimes.transform(world.train), world.regimes.transform(blind)
    parts = []
    for mode in MODES:
        for est in estimators:
            if mode == "cross_sensor" and est == BASELINE:
                continue  # the regime mean ignores its inputs: identical in both modes
            print(f"[{tag}] {mode} / {est}", flush=True)
            parts.append(score_estimator(world, tr, te, meta, est, mode, seed, mewma_offset=mewma_offset))
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.assign(Subset=subset, Seed=seed, H=H).to_csv(out / "runs" / f"{tag}_manifesto.csv", index=False)
    pd.concat(parts, ignore_index=True).assign(Subset=subset, Seed=seed, H=H).to_csv(path, index=False)


# ============================================================================
# aggregation: power by family/magnitude and paired difference vs the regime mean (cluster bootstrap)
# ============================================================================

def _load(out: Path, kind: str) -> pd.DataFrame:
    files = sorted((out / "runs").glob(f"*_{kind}.csv"))
    if not files:
        raise SystemExit(f"nenhum resultado em {out / 'runs'}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def paired_delta(series: pd.DataFrame, manifest: pd.DataFrame, rng: np.random.Generator, n_boot: int = BOOT) -> pd.DataFrame:
    """Mean paired (estimator - regime mean) detection difference in points, 95 % CI by bootstrap over recipient engines."""
    base = series[series["Estimator"] == BASELINE][["Subset", "Seed", "blind_id", "Chart", "detected"]].rename(columns={"detected": "base"})
    d = series[series["Estimator"] != BASELINE].merge(base, on=["Subset", "Seed", "blind_id", "Chart"], how="inner")
    d = d.merge(manifest[["Subset", "Seed", "blind_id", "recipient", "family", "mag"]], on=["Subset", "Seed", "blind_id"])
    d["diff"] = 100.0 * (d["detected"].astype(float) - d["base"].astype(float))
    d["cluster"] = d["Subset"] + "|" + d["Seed"].astype(str) + "|" + d["recipient"].astype(str)
    rows = []
    for keys, g in d.groupby(["Mode", "Estimator", "Chart", "family"]):
        cl = g.groupby("cluster")["diff"].agg(["sum", "count"]).to_numpy()
        idx = rng.integers(0, len(cl), (n_boot, len(cl)))
        boot = cl[idx, 0].sum(axis=1) / cl[idx, 1].sum(axis=1)
        rows.append({"Mode": keys[0], "Estimator": keys[1], "Chart": keys[2], "family": keys[3], "n_faults": len(g), "n_clusters": len(cl),
                     "delta_pp": float(g["diff"].mean()), "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(rows)


def power_by_bin(series: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    d = series.merge(manifest[["Subset", "Seed", "blind_id", "family", "mag", "shape", "gain"]], on=["Subset", "Seed", "blind_id"])
    parts = []
    for fam, edges in BINS.items():
        x = d[d["family"] == fam].copy()
        x["bin"] = pd.cut(x["mag"], edges, include_lowest=True)
        x["mag_mid"] = x["bin"].apply(lambda b: float(np.sqrt(b.left * b.right)) if pd.notna(b) else np.nan)
        parts.append(x)
    x = pd.concat(parts)
    return x.groupby(["Mode", "Estimator", "Chart", "family", "mag_mid"])["detected"].agg(power_pct=lambda s: 100 * s.mean(), n="size").reset_index()


def aggregate(out: Path, seed: int = 0) -> dict[str, pd.DataFrame]:
    tabs = out / "tabelas"
    tabs.mkdir(parents=True, exist_ok=True)
    series, manifest = _load(out, "series"), _load(out, "manifesto")
    delta = paired_delta(series, manifest, np.random.default_rng(seed))
    byb = power_by_bin(series, manifest)
    by_shape = series.merge(manifest[["Subset", "Seed", "blind_id", "family", "shape"]], on=["Subset", "Seed", "blind_id"]).groupby(
        ["Mode", "Estimator", "Chart", "family", "shape"])["detected"].agg(power_pct=lambda s: 100 * s.mean(), n="size").reset_index()
    delta.to_csv(tabs / "diferenca_vs_media_regime.csv", index=False)
    byb.to_csv(tabs / "poder_por_magnitude.csv", index=False)
    by_shape.to_csv(tabs / "poder_por_forma.csv", index=False)
    return {"delta": delta, "bin": byb, "shape": by_shape}


def fig_delta(delta: pd.DataFrame, path: Path) -> None:
    viz_style.apply_style()
    cases = [(c, f) for c in ("MEWMA", "sensor afetado") for f in ("processo", "sensor") if not delta[(delta["Chart"] == c) & (delta["family"] == f)].empty]
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.2), squeeze=False, sharex=True)
    colors = {"exogenous": viz_style.SERIES[0], "cross_sensor": viz_style.SERIES[1]}
    for ax, (chart, fam) in zip(axes[0], cases):
        d = delta[(delta["Chart"] == chart) & (delta["family"] == fam)].sort_values(["Mode", "Estimator"])
        labels = [f"{e.replace(' (padrão)', '')} · {m.replace('exogenous', 'exóg.').replace('cross_sensor', 'cross')}" for e, m in zip(d["Estimator"], d["Mode"])]
        y = np.arange(len(d))
        ax.barh(y, d["delta_pp"], xerr=[d["delta_pp"] - d["ci_lo"], d["ci_hi"] - d["delta_pp"]], color=[colors[m] for m in d["Mode"]], height=0.65, capsize=2)
        ax.axvline(0, color=viz_style.INK, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{chart} · falhas de {fam}", fontsize=10)
        ax.set_xlabel("poder − poder da média do regime (pontos)", fontsize=8)
        viz_style.style_axes(ax, grid="x")
    fig.suptitle("Ganho de detecção sobre a média do regime (sem ML), IC 95 % por bootstrap de motores. Azul = exógeno, laranja = cross-sensor", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_power(byb: pd.DataFrame, path: Path) -> None:
    viz_style.apply_style()
    cases = [(c, f) for c in ("MEWMA", "sensor afetado") for f in ("processo", "sensor") if not byb[(byb["Chart"] == c) & (byb["family"] == f)].empty]
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 3.9), squeeze=False, sharey=True)
    ests = list(dict.fromkeys(byb["Estimator"]))
    pal = dict(zip(ests, [*viz_style.SERIES, viz_style.STATUS[3], viz_style.INK_2, viz_style.MUTED]))
    for ax, (chart, fam) in zip(axes[0], cases):
        for (est, mode), g in byb[(byb["Chart"] == chart) & (byb["family"] == fam)].groupby(["Estimator", "Mode"]):
            g = g.sort_values("mag_mid")
            ax.plot(g["mag_mid"], g["power_pct"], marker="o", ms=4, lw=viz_style.LINE_W, color=pal[est], ls="-" if mode == "exogenous" else "--", label=f"{est.replace(' (padrão)', '')} · {mode}")
        ax.set_xscale("log")
        ax.set_ylim(-3, 103)
        ax.set_title(f"{chart} · {fam}", fontsize=10)
        ax.set_xlabel("magnitude (fração da assinatura)" if fam == "processo" else "magnitude (σ)", fontsize=8)
        viz_style.style_axes(ax)
    axes[0, 0].set_ylabel("poder (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.suptitle("Poder de detecção ao longo de faltas sorteadas (falso alarme ≤ 0,2 por 100 ciclos); cheio = exógeno, tracejado = cross-sensor", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--H", type=int, default=25)
    p.add_argument("--subsets", nargs="+", default=["FD004", "FD002", "FD001"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    p.add_argument("--estimators", nargs="+", default=DEFAULT_ESTIMATORS, choices=ESTIMATORS)
    p.add_argument("--n-ctrl", type=int, default=4)
    p.add_argument("--n-reps", type=int, default=2)
    p.add_argument("--n-faults", type=int, default=12, help="random faults per engine and replicate")
    p.add_argument("--max-engines", type=int, default=None)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=None, help=f"padrão: {OUT_DEFAULT} (protocolo principal, com o deslocamento do motor em todos os detectores); --no-mewma-offset grava em {OUT_DEFAULT}_offset_so_t50")
    p.add_argument("--mewma-offset", action=argparse.BooleanOptionalAction, default=True, help="(padrão ligado desde 2026-11-18; --no-mewma-offset reproduz o protocolo anterior, só o T50) deslocamento do motor (por motor e regime) removido do z de todos os sensores, no T² e nos gráficos por sensor")
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args(argv)
    if a.output_dir is None:
        a.output_dir = OUT_DEFAULT if a.mewma_offset else OUT_DEFAULT.with_name(OUT_DEFAULT.name + "_offset_so_t50")
    if not a.aggregate_only:
        jobs = [(s, seed) for s in a.subsets for seed in a.seeds]
        args = lambda s, seed: (s, seed, a.H, a.output_dir, a.n_ctrl, a.n_reps, a.n_faults, a.max_engines, a.estimators, a.mewma_offset)  # noqa: E731
        if a.jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=a.jobs)(delayed(run_one)(*args(s, seed)) for s, seed in jobs)
        else:
            for s, seed in jobs:
                run_one(*args(s, seed))
    res = aggregate(a.output_dir)
    fig_delta(res["delta"], a.output_dir / "figuras" / "fig_ganho_vs_media_regime.png")
    fig_power(res["bin"], a.output_dir / "figuras" / "fig_poder_faltas_sorteadas.png")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res["delta"].round(2).sort_values(["Chart", "family", "Mode", "Estimator"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Early-detection study with injected faults: power, delay and false alarms at MATCHED false-alarm rate.

WHAT: injects faults with a known start t0 into healthy surrogate engines (see fault_injection.py) and
scores four detectors on the same blind series:
    T50 bruto (sem regime)        EWMA of |z| on raw T50, global healthy mean/sd (ablation)
    T50 EWMA (com regime)         the thesis detector: regime-conditioned virtual-sensor residual (default 3sigma gate)
    T50 EWMA (com regime, sem portão)  same statistic and limit without the gate: like-for-like with the raw baseline
    MEWMA bruto (sem regime)      MEWMA on the 13 raw sensors, globally standardized (ablation)
    MEWMA (13 resíduos)           the thesis detector: 13 regime-conditioned residuals
WHY: with 99.5 % of the T50 variance between regimes (FD002/FD004), a detector that does not remove the
regime should need much larger faults. The injected fault (start known only to us) turns that claim
into a measurable power curve. FD001/FD003 have one regime, so they are the natural control (the raw and
regime detectors should coincide there).

WHAT IS MEASURED (no RUL anywhere)
  * operating curves: every detector's threshold is swept (alarm when ratio = statistic / limit >= s for
    k=2 consecutive cycles after H; s = 1 is the default rule). Power vs false-alarm rate, not one point.
  * false alarms: many independent healthy controls; FAR = alarms per 100 monitored cycles and
    ARL0 = monitored cycles per alarm (lower bound when no alarm occurred).
  * matched comparison: for each detector the smallest s whose control FAR <= budget (0.2 or 1 per 100
    cycles) is used, so detectors are compared at the SAME false-alarm price.
  * delay = first alarm at cycle >= t0 minus t0; a fault series is monitored from t0 on, and false alarms
    before t0 are priced by the controls (so the power denominator is constant and curves are monotone).
  * minimum detectable magnitude: smallest magnitude with power >= 80 % (per seed; median over seeds).
  * adopted setting: H = 25, t0 = H+10, budget 0.2 per 100 cycles, seeds {42, 7, 123}. Sweeps (H in 25/35/50 and
    50/75/100) show power falling slowly as H grows at equal false-alarm rate; H = 25 is nominally the best but
    only by ~1.6 points on average over 12 paired cells (not significant); delay is unchanged. H < 25 was not tested.

[LEAKAGE] detectors are fit on TRAINING engines only; the faulty / control series come from TEST engines
and reach the detectors as raw C-MAPSS columns (no label, scenario or t0); the answer key is a separate
table used only by the scoring functions. [CAUSALITY] injection touches cycles >= t0 only; a control and
its faulty twin share the same bootstrap base. [ASSUMPTION] healthy surrogate = i.i.d. bootstrap of the
engine's own cycles with true RUL >= 125 (healthy; regimes and noise are i.i.d. per cycle in C-MAPSS, the
engine-specific offset is kept). The raw baselines never use the raw-3sigma gate (it would make them
unreachable under regime variance); the regime T50 chart is reported with the default gate AND without it
(like-for-like), so the effect of the regime is not confounded with the gate.

Run:  .venv/bin/python -m pdm_cmaps.experiments.fault_detection_study [--subsets FD004] [--jobs 4]  /  --aggregate-only
Output: results/injecao_deteccao/{runs,tabelas,figuras}
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
from pdm_cmaps.detection.spc import StatisticalAnomalyDetector
from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.experiments.fault_injection import SCENARIOS, World, blind_columns, delta_z, inject, load_world
from pdm_cmaps.reporting import viz_style

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "injecao_deteccao"
ADOPTED_H = 25                 # window used by the figures (sweep 25/35/50 and 50/75/100: see module docstring)
DET_RAW_T50 = "T50 bruto (sem regime)"
DET_T50 = "T50 EWMA (com regime)"
DET_T50_NOGATE = "T50 EWMA (com regime, sem portão)"
DET_RAW_MV = "MEWMA bruto (sem regime)"
DET_MV = "MEWMA (13 resíduos, com regime)"
DETECTORS = (DET_RAW_T50, DET_T50, DET_T50_NOGATE, DET_RAW_MV, DET_MV)
BUDGETS = (0.2,)                # best/adopted budget: false alarms per 100 monitored cycles (--budgets 0.2 1.0 for more)
S_GRID = np.unique(np.append(np.geomspace(0.3, 4.0, 40), 1.0))
K_CONSEC = 2
HORIZON_CTRL = 300
POST_DET = 80
LATE_T0_OFFSET = 60
LATE_KINDS = (("degrau", (0.25, 0.5)), ("rampa", (0.25, 0.5)))
MIN_POOL = 30
POWER_TARGET = 80.0


# ============================================================================
# series generation (healthy bootstrap + injected faults)
# ============================================================================

def bootstrap_series(engine: pd.DataFrame, horizon: int, rng: np.random.Generator) -> pd.DataFrame | None:
    """i.i.d. bootstrap of the engine's healthy cycles (true RUL >= cap), renumbered 1..horizon."""
    pool = engine.loc[engine["rul_true"] >= DEFAULT_RUL_CAP]
    if len(pool) < MIN_POOL:
        return None
    base = pool.iloc[rng.integers(0, len(pool), horizon)].reset_index(drop=True).copy()
    base["cycle"] = np.arange(1, horizon + 1)
    return base


def build_series(world: World, seed: int, n_ctrl: int, n_rep: int, max_engines: int | None = None, late_t0: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blind frame (raw schema, unit_id = blind id) and the answer key."""
    rng = np.random.default_rng(seed)
    H = world.H
    t0s = (H + 10, H + LATE_T0_OFFSET) if late_t0 else (H + 10,)
    engines = {int(u): g.sort_values("cycle") for u, g in world.test.groupby("unit_id")}
    recipients = [u for u in sorted(engines) if (engines[u]["rul_true"] >= DEFAULT_RUL_CAP).sum() >= MIN_POOL]
    if max_engines:
        recipients = recipients[:max_engines]
    frames, meta, bid = [], [], 0
    for u in recipients:
        for _ in range(n_ctrl):
            base = bootstrap_series(engines[u], HORIZON_CTRL, rng)
            bid += 1
            frames.append(blind_columns(base, bid))
            meta.append({"blind_id": bid, "recipient": u, "role": "controle", "kind": "", "magnitude": np.nan, "t0": 0, "T": HORIZON_CTRL})
        for t0 in t0s:
            horizon = t0 + POST_DET
            grid = SCENARIOS if t0 == t0s[0] else dict(LATE_KINDS)
            for _ in range(n_rep):
                base = bootstrap_series(engines[u], horizon, rng)
                for kind, mags in grid.items():
                    if kind == "um_regime" and world.regimes.n_clusters < 2:
                        continue
                    for m in mags:
                        bid += 1
                        frames.append(blind_columns(inject(world, base, delta_z(world, base, kind, m, t0, rng)), bid))
                        meta.append({"blind_id": bid, "recipient": u, "role": "falha", "kind": kind, "magnitude": m, "t0": t0, "T": horizon})
    blind = pd.concat(frames, ignore_index=True).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return blind, pd.DataFrame(meta)


# ============================================================================
# detector scores: ratio = statistic / limit per cycle (blind frame only)
# ============================================================================

def _stack(ids: np.ndarray, cycles: np.ndarray, ratio: np.ndarray, gate: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"blind_id": ids, "cycle": cycles, "ratio": ratio, "gate": gate})


def _engine_level(x: np.ndarray, cycles: np.ndarray, H: int) -> np.ndarray:
    """Média do próprio motor nos primeiros H ciclos (0 se houver menos de 3), para tirar o nível do motor de um sinal bruto."""
    w = cycles <= H
    return x - (x[w].mean(axis=0) if w.sum() >= 3 else 0.0)


def raw_t50_scores(world: World, blind: pd.DataFrame, engine_offset: bool = False) -> pd.DataFrame:
    """EWMA of |z| on raw T50 (global healthy mean/sd); no regime removal, no raw-3sigma gate. ``engine_offset`` tira o nível de cada motor."""
    hh = world.train.loc[world.train["cycle"] <= world.H]
    healthy = hh["T50"].to_numpy(dtype=float)
    if engine_offset:
        healthy = hh["T50"] - hh.groupby("unit_id")["T50"].transform("mean")
        healthy = healthy.to_numpy(dtype=float)
    det = StatisticalAnomalyDetector(threshold_sigma=3.0, ewma_alpha=0.10, healthy_cycle_limit=world.H,
                                     min_consecutive_breaches=K_CONSEC).calibrate(healthy)
    parts = []
    for uid, g in blind.groupby("unit_id", sort=True):
        x = g["T50"].to_numpy(dtype=float)
        if engine_offset:
            x = _engine_level(x, g["cycle"].to_numpy(), world.H)
        e = det.ewma_abs_z(det.compute_z_scores(x))
        parts.append(_stack(np.full(len(g), uid), g["cycle"].to_numpy(), e / det.ewma_upper_limits(len(e)), np.ones(len(g), bool)))
    return pd.concat(parts, ignore_index=True)


def raw_mewma_scores(world: World, blind: pd.DataFrame, sensors: list[str], lam: float = 0.10, alpha: float = 0.0027, engine_offset: bool = False) -> pd.DataFrame:
    """MEWMA T2 on globally standardized raw sensors; limit = empirical (1 - alpha) quantile of healthy T2."""
    mon = MultivariateResidualMonitor(sensors=sensors, lam=lam, alpha=alpha, healthy_cycle_limit=world.H)
    h = world.train.loc[world.train["cycle"] <= world.H].sort_values(["unit_id", "cycle"])
    mean, sd = h[sensors].mean(), h[sensors].std().clip(lower=1e-8)
    zdf = (h[sensors] - mean) / sd
    if engine_offset:
        zdf = zdf - zdf.groupby(h["unit_id"].to_numpy()).transform("mean")
    zh = zdf.to_numpy()
    mon.mu_ = zh.mean(axis=0)
    mon.sigma_inv_ = np.linalg.inv(np.cov(zh, rowvar=False) + 1e-6 * np.eye(zh.shape[1]))
    ucl = float(np.quantile(mon._t2(zh, h["unit_id"].to_numpy()), 1 - alpha))
    b = blind.sort_values(["unit_id", "cycle"])
    zb = ((b[sensors] - mean) / sd)
    if engine_offset:
        for uid, idx in b.groupby("unit_id").groups.items():
            part = zb.loc[idx].to_numpy()
            zb.loc[idx] = _engine_level(part, b.loc[idx, "cycle"].to_numpy(), world.H)
    t2 = mon._t2(zb.to_numpy(), b["unit_id"].to_numpy())
    return _stack(b["unit_id"].to_numpy(), b["cycle"].to_numpy(), t2 / ucl, np.ones(len(b), bool))


def regime_scores(world: World, blind: pd.DataFrame, seed: int, engine_offset: bool = True, mewma_offset: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """The thesis detectors: regime-conditioned T50 EWMA (with and without the raw-3sigma gate) and the 13-residual MEWMA."""
    tr, te = world.regimes.transform(world.train), world.regimes.transform(blind)
    vs = VirtualSensor(target_sensor="T50", mode="exogenous", ewma_alpha=0.10, min_consecutive_breaches=K_CONSEC,
                       healthy_cycle_limit=world.H, random_state=seed, engine_offset=engine_offset)
    mv = MultivariateResidualMonitor(healthy_cycle_limit=world.H, random_state=seed, engine_offset=mewma_offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace, _, _ = vs.evaluate(tr, te)
        mv.fit(tr)
        sc = mv.score(te)
    t50 = _stack(trace["unit_id"].to_numpy(), trace["cycle"].to_numpy(), (trace["ewma_abs_z"] / trace["ewma_ucl"]).to_numpy(),
                 (trace["z_score"].abs() >= vs.threshold_sigma).to_numpy())
    nogate = t50.assign(gate=True)  # same statistic and limit, only the raw-3sigma gate removed (like-for-like with the raw baseline)
    mvs = _stack(sc["unit_id"].to_numpy(), sc["cycle"].to_numpy(), (sc["t2"] / sc["ucl"]).to_numpy(), np.ones(len(sc), bool))
    return t50, nogate, mvs, list(mv.sensors_)


def score_all(world: World, blind: pd.DataFrame, seed: int, engine_offset: bool = True, mewma_offset: bool = True) -> dict[str, pd.DataFrame]:
    """Receives ONLY the blind frame (raw schema). ``mewma_offset`` dá o deslocamento do motor também à MEWMA e aos dois detectores brutos
    (para que a ablação do regime continue limpa: todos os detectores recebem o mesmo tratamento)."""
    t50, nogate, mvs, sensors = regime_scores(world, blind, seed, engine_offset, mewma_offset)
    return {DET_RAW_T50: raw_t50_scores(world, blind, mewma_offset), DET_T50: t50, DET_T50_NOGATE: nogate,
            DET_RAW_MV: raw_mewma_scores(world, blind, sensors, engine_offset=mewma_offset), DET_MV: mvs}


# ============================================================================
# alarms over a threshold grid, operating curves, matched comparison
# ============================================================================

def alarm_cycles(ratio: np.ndarray, gate: np.ndarray, cycles: np.ndarray, min_cycle: int, s_grid: np.ndarray = S_GRID, k: int = K_CONSEC) -> np.ndarray:
    """First alarm cycle (start of the first run of ``k`` breaches at cycles >= ``min_cycle``) per threshold s; NaN = none."""
    out = np.full(len(s_grid), np.nan)
    n = len(ratio)
    if n < k:
        return out
    breach = (ratio[:, None] >= s_grid[None, :]) & gate[:, None] & (cycles >= min_cycle)[:, None]
    run = breach[k - 1:].copy()
    for j in range(1, k):
        run &= breach[k - 1 - j:n - j]
    has = run.any(axis=0)
    first = run.argmax(axis=0)
    out[has] = cycles[first[has]]
    return out


def alarm_matrix(scores: pd.DataFrame, min_cycle_by_id: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(blind ids, alarm matrix [series x s], last cycle per series). Monitoring of a series starts at its min cycle."""
    scores = scores.sort_values(["blind_id", "cycle"])
    ids, starts = np.unique(scores["blind_id"].to_numpy(), return_index=True)
    ratio, gate, cyc = scores["ratio"].to_numpy(), scores["gate"].to_numpy(), scores["cycle"].to_numpy()
    bounds = np.append(starts, len(scores))
    mins = min_cycle_by_id.loc[ids].to_numpy()
    mat = np.vstack([alarm_cycles(ratio[a:b], gate[a:b], cyc[a:b], int(mc)) for (a, b), mc in zip(zip(bounds[:-1], bounds[1:]), mins)])
    return ids, mat, cyc[bounds[1:] - 1]


def control_far(alarms: np.ndarray, last: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    """False alarms per 100 monitored cycles and ARL0 for each s (events / exposure until first alarm)."""
    has = ~np.isnan(alarms)
    exposure = np.where(has, alarms - H, (last - H)[:, None]).sum(axis=0)
    events = has.sum(axis=0)
    with np.errstate(divide="ignore"):
        return 100 * events / exposure, np.where(events > 0, exposure / np.maximum(events, 1), np.inf)


def analyse_detector(name: str, scores: pd.DataFrame, meta: pd.DataFrame, H: int, budgets: tuple = BUDGETS) -> dict[str, pd.DataFrame]:
    indexed = meta.set_index("blind_id")
    # controls are monitored from H+1; faulty series from t0 (alarms before t0 are false alarms, counted in the controls)
    ids, mat, last = alarm_matrix(scores, np.maximum(indexed["t0"], H + 1))
    m = indexed.loc[ids]
    ctrl = (m["role"] == "controle").to_numpy()
    far, arl0 = control_far(mat[ctrl], last[ctrl], H)
    curves, matched, delays = [], [], []
    stars = {b: (int(np.flatnonzero(far <= b)[0]) if (far <= b).any() else None) for b in budgets}
    fault = m[~ctrl].assign(row=np.flatnonzero(~ctrl))
    for (kind, mag, t0), g in fault.groupby(["kind", "magnitude", "t0"]):
        a = mat[g["row"].to_numpy()]
        det = ~np.isnan(a)
        elig = np.ones_like(det)
        delay = np.where(det, a - t0, np.nan)
        n_elig = elig.sum(axis=0)
        power = 100 * det.sum(axis=0) / n_elig
        dmed = np.array([np.nanmedian(delay[:, i]) if det[:, i].any() else np.nan for i in range(len(S_GRID))])
        dp90 = np.array([np.nanpercentile(delay[:, i], 90) if det[:, i].any() else np.nan for i in range(len(S_GRID))])
        curves.append(pd.DataFrame({"Detector": name, "kind": kind, "magnitude": mag, "t0": t0, "s": S_GRID, "far_ctrl_per100": far,
                                    "arl0": arl0, "n_elig": n_elig, "power_pct": power, "delay_med": dmed, "delay_p90": dp90}))
        for b, i in stars.items():
            row = {"Detector": name, "kind": kind, "magnitude": mag, "t0": t0, "budget": b, "s_star": np.nan, "far_ctrl_per100": np.nan,
                   "arl0": np.nan, "n_elig": 0, "power_pct": np.nan, "delay_med": np.nan, "delay_p90": np.nan}
            if i is not None:
                row.update(s_star=S_GRID[i], far_ctrl_per100=far[i], arl0=arl0[i], n_elig=int(n_elig[i]), power_pct=power[i],
                           delay_med=dmed[i], delay_p90=dp90[i])
                keep = elig[:, i]
                delays.append(pd.DataFrame({"Detector": name, "kind": kind, "magnitude": mag, "t0": t0, "budget": b, "delay": delay[keep, i]}))
            matched.append(row)
    return {"curvas": pd.concat(curves, ignore_index=True), "pareado": pd.DataFrame(matched),
            "atrasos": pd.concat(delays, ignore_index=True) if delays else pd.DataFrame()}


def run_one(subset: str, seed: int, H: int, out: Path, n_ctrl: int = 6, n_rep: int = 2, max_engines: int | None = None,
            budgets: tuple = BUDGETS, late_t0: bool = False, engine_offset: bool = True, mewma_offset: bool = True) -> None:
    run_dir = out / "runs"
    tag = f"{subset}_seed{seed}_H{H}"
    if all((run_dir / f"{tag}_{k}.csv").exists() for k in ("curvas", "pareado", "atrasos")):
        return
    world = load_world(subset, seed, H, loader=CMapssLoader())
    blind, meta = build_series(world, seed, n_ctrl, n_rep, max_engines, late_t0)
    print(f"[{tag}] {meta['blind_id'].nunique()} series ({(meta['role'] == 'controle').sum()} controls)", flush=True)
    scores = score_all(world, blind, seed, engine_offset, mewma_offset)
    parts = {"curvas": [], "pareado": [], "atrasos": []}
    for name, sc in scores.items():
        res = analyse_detector(name, sc, meta, H, budgets)
        for k in parts:
            parts[k].append(res[k])
    run_dir.mkdir(parents=True, exist_ok=True)
    for k, v in parts.items():
        pd.concat(v, ignore_index=True).assign(Subset=subset, Seed=seed, H=H).to_csv(run_dir / f"{tag}_{k}.csv", index=False)


# ============================================================================
# aggregation, tables, figures
# ============================================================================

def _load(out: Path, kind: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in sorted((out / "runs").glob(f"*_{kind}.csv"))], ignore_index=True)


def min_detectable(pareado: pd.DataFrame) -> pd.DataFrame:
    """Smallest magnitude with power >= 80 % per (subset, seed, H, detector, kind, t0, budget)."""
    ok = pareado[pareado["power_pct"] >= POWER_TARGET]
    keys = ["Subset", "Seed", "H", "Detector", "kind", "t0", "budget"]
    mdm = ok.groupby(keys)["magnitude"].min().rename("mdm").reset_index()
    allk = pareado[keys].drop_duplicates().merge(mdm, on=keys, how="left")
    g = allk.groupby(["Subset", "H", "Detector", "kind", "t0", "budget"])
    return g.agg(mdm_mediana=("mdm", "median"), seeds_que_atingem=("mdm", lambda s: int(s.notna().sum())), n_seeds=("mdm", "size")).reset_index()


def aggregate(out: Path) -> dict[str, pd.DataFrame]:
    tabs = out / "tabelas"
    tabs.mkdir(parents=True, exist_ok=True)
    par, cur = _load(out, "pareado"), _load(out, "curvas")
    keys = ["Subset", "H", "Detector", "kind", "magnitude", "t0", "budget"]
    g = par.groupby(keys)[["s_star", "far_ctrl_per100", "arl0", "power_pct", "delay_med", "delay_p90"]]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    summ["n_seeds"] = g.size().to_numpy()
    mdm = min_detectable(par)
    default = cur[np.isclose(cur["s"], 1.0)].groupby(["Subset", "H", "Detector", "kind", "magnitude", "t0"])[
        ["far_ctrl_per100", "arl0", "power_pct", "delay_med"]].mean().reset_index()
    summ.to_csv(tabs / "pareado_far_resumo.csv", index=False)
    mdm.to_csv(tabs / "magnitude_minima_detectavel.csv", index=False)
    default.to_csv(tabs / "regra_padrao_s1.csv", index=False)
    return {"resumo": summ, "mdm": mdm, "padrao": default, "curvas": cur, "atrasos": _load(out, "atrasos")}


STYLE = {DET_RAW_T50: (viz_style.SERIES[0], "--"), DET_T50: (viz_style.SERIES[0], "-"), DET_T50_NOGATE: (viz_style.SERIES[2], "-"),
         DET_RAW_MV: (viz_style.SERIES[1], "--"), DET_MV: (viz_style.SERIES[1], "-")}


def fig_ablation(summ: pd.DataFrame, path: Path, H: int = ADOPTED_H, budget: float = 0.2, subsets=("FD001", "FD002", "FD004")) -> None:
    viz_style.apply_style()
    kinds = list(SCENARIOS)
    subsets = [s for s in subsets if s in set(summ["Subset"])]
    fig, axes = plt.subplots(len(subsets), len(kinds), figsize=(3.3 * len(kinds), 2.8 * len(subsets)), squeeze=False, sharey=True)
    base = summ[(summ["H"] == H) & (summ["budget"] == budget) & (summ["t0"] == H + 10)]
    for i, sub in enumerate(subsets):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            for det in DETECTORS:
                d = base[(base["Subset"] == sub) & (base["kind"] == kind) & (base["Detector"] == det)].sort_values("magnitude")
                if d.empty:
                    continue
                col, ls = STYLE[det]
                ax.errorbar(d["magnitude"], d["power_pct_mean"], yerr=d["power_pct_std"].fillna(0), color=col, ls=ls, marker="o", ms=4,
                            lw=viz_style.LINE_W, capsize=2, label=det)
            viz_style.style_axes(ax)
            ax.set_ylim(-3, 103)
            if i == 0:
                ax.set_title(kind.replace("_", " "), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{sub}\npoder (%)", fontsize=9)
            if i == len(subsets) - 1:
                ax.set_xlabel("magnitude" + (" (× assinatura de fim de vida)" if kind in ("degrau", "rampa", "um_regime") else " (σ)"), fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Ablação do desacoplamento de regime: poder de detecção com o MESMO falso alarme "
                 f"({budget:g} por 100 ciclos), H = {H}. Tracejado = sem regime, cheio = com regime (média ± dp de 3 seeds)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_operating(cur: pd.DataFrame, path: Path, subset: str = "FD004", H: int = ADOPTED_H) -> None:
    viz_style.apply_style()
    cases = [("degrau", 0.25), ("rampa", 0.25), ("degrau", 0.5)]
    fig, axes = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 3.8), squeeze=False, sharey=True)
    for ax, (kind, mag) in zip(axes[0], cases):
        for det in DETECTORS:
            d = cur[(cur["Subset"] == subset) & (cur["H"] == H) & (cur["kind"] == kind) & np.isclose(cur["magnitude"], mag)
                    & (cur["t0"] == H + 10) & (cur["Detector"] == det)]
            m = d.groupby("s")[["far_ctrl_per100", "power_pct"]].mean().sort_index()
            col, ls = STYLE[det]
            ax.plot(m["far_ctrl_per100"].clip(lower=1e-3), m["power_pct"], color=col, ls=ls, lw=viz_style.LINE_W, label=det)
            one = m.loc[np.isclose(m.index, 1.0)]
            ax.scatter(one["far_ctrl_per100"].clip(lower=1e-3), one["power_pct"], color=col, s=40, zorder=5, edgecolors=viz_style.SURFACE)
        for b in BUDGETS:
            ax.axvline(b, color=viz_style.MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        viz_style.style_axes(ax, grid="both")
        ax.set_title(f"{subset} · {kind} {mag:g}", fontsize=10)
        ax.set_xlabel("falsos alarmes por 100 ciclos (controles)")
    axes[0, 0].set_ylabel("poder de detecção (%)")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle("Curvas operacionais (limite varrido; ponto = regra padrão s = 1; pontilhado = orçamentos de falso alarme)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_cumulative(delays: pd.DataFrame, path: Path, subset: str = "FD004", H: int = ADOPTED_H, budget: float = 0.2) -> None:
    viz_style.apply_style()
    cases = [("degrau", 0.25), ("degrau", 0.5), ("rampa", 0.5)]
    fig, axes = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 3.6), squeeze=False, sharey=True)
    grid = np.arange(0, POST_DET + 1)
    for ax, (kind, mag) in zip(axes[0], cases):
        for det in DETECTORS:
            d = delays[(delays["Subset"] == subset) & (delays["H"] == H) & (delays["kind"] == kind) & np.isclose(delays["magnitude"], mag)
                       & (delays["t0"] == H + 10) & (delays["budget"] == budget) & (delays["Detector"] == det)]
            if d.empty:
                continue
            x = d["delay"].to_numpy()
            col, ls = STYLE[det]
            ax.step(grid, [100 * np.mean(x <= g) if len(x) else 0 for g in grid], where="post", color=col, ls=ls, lw=viz_style.LINE_W, label=det)
        viz_style.style_axes(ax)
        ax.set_title(f"{subset} · {kind} {mag:g}", fontsize=10)
        ax.set_xlabel("ciclos após o início da falha (t0)")
    axes[0, 0].set_ylabel("séries detectadas até o atraso (%)")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle(f"Detecção acumulada por atraso (falso alarme igualado a {budget:g} por 100 ciclos; sem detecção = censurado no fim da série)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_window(summ: pd.DataFrame, path: Path, budget: float = 0.2, kind: str = "degrau", mag: float = 0.25) -> None:
    viz_style.apply_style()
    subsets = sorted(set(summ["Subset"]))
    fig, axes = plt.subplots(2, len(subsets), figsize=(3.4 * len(subsets), 6.0), squeeze=False)
    for j, sub in enumerate(subsets):
        for det in DETECTORS:
            d = summ[(summ["Subset"] == sub) & (summ["Detector"] == det) & (summ["kind"] == kind) & np.isclose(summ["magnitude"], mag)
                     & (summ["budget"] == budget) & (summ["t0"] == summ["H"] + 10)].sort_values("H")
            col, ls = STYLE[det]
            axes[0, j].errorbar(d["H"], d["power_pct_mean"], yerr=d["power_pct_std"].fillna(0), color=col, ls=ls, marker="o", ms=4,
                                lw=viz_style.LINE_W, capsize=2, label=det)
            axes[1, j].errorbar(d["H"], d["delay_med_mean"], yerr=d["delay_med_std"].fillna(0), color=col, ls=ls, marker="o", ms=4,
                                lw=viz_style.LINE_W, capsize=2)
        axes[0, j].set_title(sub, fontsize=10)
        axes[0, j].set_ylim(-3, 103)
        axes[1, j].set_xlabel("janela saudável H (ciclos)")
        for i in range(2):
            viz_style.style_axes(axes[i, j])
            axes[i, j].set_xticks([50, 75, 100])
    axes[0, 0].set_ylabel("poder (%)")
    axes[1, 0].set_ylabel("atraso mediano (ciclos)")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.suptitle(f"Efeito da janela saudável H na detecção ({kind} {mag:g}, falso alarme igualado a {budget:g} por 100 ciclos)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_figures(res: dict[str, pd.DataFrame], out: Path) -> None:
    fig_dir = out / "figuras"
    fig_ablation(res["resumo"], fig_dir / "fig_ablacao_regime.png")
    fig_operating(res["curvas"], fig_dir / "fig_curvas_operacionais.png")
    fig_cumulative(res["atrasos"], fig_dir / "fig_deteccao_acumulada.png")
    if res["resumo"]["H"].nunique() > 1:
        fig_window(res["resumo"], fig_dir / "fig_janela_poder_atraso.png")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    p.add_argument("--windows", nargs="+", type=int, default=[25], help="healthy windows H; 25 is the adopted value (25 35 50 reproduces the sweep)")
    p.add_argument("--budgets", nargs="+", type=float, default=list(BUDGETS), help="false alarms per 100 cycles")
    p.add_argument("--late-t0", action="store_true", help="also inject at t0 = H + 60 (robustness)")
    p.add_argument("--n-ctrl", type=int, default=6, help="independent healthy controls per recipient engine")
    p.add_argument("--n-rep", type=int, default=2, help="independent bootstrap bases per recipient and t0")
    p.add_argument("--max-engines", type=int, default=None)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=None, help=f"padrão: {OUT_DEFAULT} (protocolo principal: deslocamento do motor em todos os detectores); --no-mewma-offset grava em {OUT_DEFAULT}_offset_so_t50 e --no-engine-offset em {OUT_DEFAULT}_sem_offset")
    p.add_argument("--engine-offset", action=argparse.BooleanOptionalAction, default=True, help="T50 EWMA com o deslocamento de cada motor (média do z nos primeiros H ciclos) removido; padrão ligado; --no-engine-offset desliga também a MEWMA e os brutos (rodada sem nenhuma correção)")
    p.add_argument("--mewma-offset", action=argparse.BooleanOptionalAction, default=True, help="deslocamento do motor também na MEWMA com regime e nos dois detectores brutos (todos os detectores com o mesmo tratamento); padrão ligado (protocolo principal desde 2026-11-18); --no-mewma-offset reproduz o protocolo anterior (só o T50)")
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args(argv)
    if not a.engine_offset:
        a.mewma_offset = False  # sem a correção no T50, a rodada de referência não tem correção em nenhum detector
    if a.output_dir is None:
        suffix = "" if (a.engine_offset and a.mewma_offset) else ("_offset_so_t50" if a.engine_offset else "_sem_offset")
        a.output_dir = OUT_DEFAULT.with_name(OUT_DEFAULT.name + suffix)
    if not a.aggregate_only:
        jobs = [(s, seed, h) for s in a.subsets for seed in a.seeds for h in a.windows]
        if a.jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=a.jobs)(delayed(run_one)(s, seed, h, a.output_dir, a.n_ctrl, a.n_rep, a.max_engines, tuple(a.budgets), a.late_t0, a.engine_offset, a.mewma_offset) for s, seed, h in jobs)
        else:
            for s, seed, h in jobs:
                run_one(s, seed, h, a.output_dir, a.n_ctrl, a.n_rep, a.max_engines, tuple(a.budgets), a.late_t0, a.engine_offset, a.mewma_offset)
    res = aggregate(a.output_dir)
    make_figures(res, a.output_dir)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res["mdm"][(res["mdm"]["H"] == 50) & (res["mdm"]["budget"] == 0.2) & (res["mdm"]["t0"] == 60)].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

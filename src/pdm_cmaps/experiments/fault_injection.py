"""Blind fault-injection benchmark: do the detectors and RUL models react to a fault only WE know about?

WHAT: takes held-out (test) engines, replaces them by a healthy surrogate, injects a fault at a cycle
t0 chosen by us, and scores the unchanged detectors / RUL models, which only ever see sensor columns.
Three stages that talk ONLY through files, so the "answer" cannot leak into the models:

  generate : writes dados_cegos/*.csv (schema = raw C-MAPSS columns, nothing else) and the sealed
             answer key selado/manifesto_*.csv (+ its sha256 printed next to the blind data)
  detect   : reads ONLY dados_cegos/; fits detectors/models on TRAINING engines; writes alarms/predictions
  score    : reads the sealed manifest + the stage-2 outputs; writes tables and figures

TWO KINDS OF FAULT (they answer different questions)
  1. SYNTHETIC (detector power): step / ramp / one-regime / variance / sensor-bias, with magnitude as a
     fraction of the fleet's own end-of-life signature. The true RUL is not defined, so this only scores
     detection (power, delay) against a paired un-injected control (the SAME surrogate engine).
  2. TRANSPLANT (RUL models): the last N cycles of a DONOR test engine, in regime-standardized sensor
     units, are added to a recipient's healthy trajectory from t0 on. The donor's remaining life gives a
     true RUL by construction: label(c) = min(EOL - c, 125) with EOL = t0 + N - 1.

[LEAKAGE] detectors/models are fit on training engines only; donors and recipients are test engines;
the blind files hold no RUL, no scenario, no t0. [CAUSALITY] injection touches cycles >= t0 only, so a
control and its faulty twin are identical before t0 (checked by the tests).
[ASSUMPTION] healthy surrogate: the recipient's real cycles with true RUL >= 125 (natural degradation is
negligible there), extended by resampling its own healthy rows when the horizon is longer; this keeps
the settings/regime process but drops its temporal autocorrelation beyond the real prefix.
[ASSUMPTION] only the 13 MEWMA sensors are modified (never the exogenous settings/ambient channels).
Fault shapes are plausible, not physically simulated: "signature" = mean regime-standardized sensor
deviation of the last 10 cycles of the training engines.

Run:  .venv/bin/python -m pdm_cmaps.experiments.fault_injection --subset FD004 [--include-deep] [--stage all|generate|detect|score]
Output: results/injecao_de_falhas/<subset>_seed<seed>/{dados_cegos,selado,saidas,tabelas,figuras}
"""

from __future__ import annotations

from pdm_cmaps.config import DEFAULT_RESULTS_DIR

import argparse
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm_cmaps.domain.schema import COLUMNS, DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RUL_CAP
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.features import CausalFeatureEngineer
from pdm_cmaps.domain.rul import compute_piecewise_rul
from pdm_cmaps.data.splits import split_engine_units
from pdm_cmaps.detection.virtual_sensor import CANONICAL_EXOGENOUS_FEATURES, VirtualSensor
from pdm_cmaps.detection.multivariate import MULTIVARIATE_SENSORS, MultivariateResidualMonitor
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.reporting import viz_style

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "injecao_de_falhas"
INJECT_SENSORS = tuple(s for s in MULTIVARIATE_SENSORS if s not in CANONICAL_EXOGENOUS_FEATURES)
EOL_WINDOW = 10          # cycles used for the end-of-life signature
POST_CYCLES = 60         # observation window after t0 for the synthetic faults
RAMP_LEN = 40            # cycles to reach full magnitude for ramp-type faults
TRANSPLANT_N = 80        # donor cycles transplanted (= true remaining life at t0)
BIAS_SENSOR = "T50"
# magnitude = fraction of the EOL signature (signature kinds) or sigma units (variance / bias)
SCENARIOS = {
    "degrau": (0.05, 0.10, 0.25, 0.50),
    "rampa": (0.05, 0.10, 0.25, 0.50),
    "um_regime": (0.10, 0.25, 0.50, 1.00),
    "variancia": (0.5, 1.0, 2.0),
    "vies_sensor": (1.0, 2.0, 4.0),
}
DEFAULT_MODELS = ["Ridge", "LightGBM", "XGBoost", "LightGBM (+temporal)", "XGBoost (+temporal)"]
DEEP_MODELS = ["PyTorch CNN", "PyTorch RNN (LSTM)", "PyTorch Transformer"]
DETECTORS = ("T50 EWMA", "MEWMA (13 resíduos)")


# ============================================================================
# world: split, regimes, standardization, signature
# ============================================================================

@dataclass
class World:
    subset: str
    seed: int
    H: int
    train: pd.DataFrame
    test: pd.DataFrame
    regimes: RegimeDecoupler
    ref_mean: pd.DataFrame   # regime_id x INJECT_SENSORS (healthy training window)
    ref_std: pd.DataFrame
    signature: np.ndarray    # EOL deviation in regime-standardized units, one per INJECT_SENSORS
    main_regime: int


def load_world(subset: str, seed: int, H: int, loader: CMapssLoader | None = None) -> World:
    loader = loader or CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    frame["rul_true"] = frame.groupby("unit_id")["cycle"].transform("max") - frame["cycle"]
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    train = frame.loc[frame["unit_id"].isin(train_units)]
    test = frame.loc[frame["unit_id"].isin(test_units)]
    n_regimes = 6 if subset.upper() in {"FD002", "FD004"} else 1
    regimes = RegimeDecoupler(n_clusters=n_regimes, random_state=seed).fit(train)
    tr, te = regimes.transform(train), regimes.transform(test)
    healthy = tr.loc[tr["cycle"] <= H]
    ref_mean = healthy.groupby("regime_id")[list(INJECT_SENSORS)].mean()
    ref_std = healthy.groupby("regime_id")[list(INJECT_SENSORS)].std().fillna(1.0).clip(lower=1e-8)
    world = World(subset, seed, H, tr, te, regimes, ref_mean, ref_std, np.zeros(len(INJECT_SENSORS)), 0)
    z = standardize(world, tr)
    base = z[(tr["cycle"] <= H).to_numpy()].mean(axis=0)
    end = z[(tr["rul_true"] <= EOL_WINDOW).to_numpy()].mean(axis=0)
    world.signature = end - base
    world.main_regime = int(tr["regime_id"].value_counts().idxmax())
    return world


def _lookup(table: pd.DataFrame, regime_id: np.ndarray) -> np.ndarray:
    fallback = table.mean()
    return table.reindex(regime_id).fillna(fallback).to_numpy(dtype=float)


def standardize(world: World, frame: pd.DataFrame) -> np.ndarray:
    """Regime-standardized INJECT_SENSORS (rows x sensors) using the healthy TRAINING window."""
    rid = frame["regime_id"].to_numpy(dtype=int)
    x = frame[list(INJECT_SENSORS)].to_numpy(dtype=float)
    return (x - _lookup(world.ref_mean, rid)) / _lookup(world.ref_std, rid)


# ============================================================================
# healthy surrogate + fault shapes
# ============================================================================

def healthy_surrogate(engine: pd.DataFrame, horizon: int, H: int, rng: np.random.Generator) -> pd.DataFrame | None:
    """Healthy trajectory of ``horizon`` cycles built from the engine's own cycles with true RUL >= cap."""
    healthy = engine.loc[engine["rul_true"] >= DEFAULT_RUL_CAP].sort_values("cycle")
    if len(healthy) < H + 10:
        return None
    if len(healthy) >= horizon:
        base = healthy.iloc[:horizon]
    else:
        pool = healthy.iloc[H // 2:]
        extra = pool.iloc[rng.integers(0, len(pool), horizon - len(healthy))]
        base = pd.concat([healthy, extra], ignore_index=True)
    base = base.reset_index(drop=True).copy()
    base["cycle"] = np.arange(1, horizon + 1)
    return base


def fault_profile(kind: str, cycles: np.ndarray, t0: int, ramp_len: int = RAMP_LEN) -> np.ndarray:
    """0 before t0; 1 after (degrau) or a linear ramp reaching 1 after ``ramp_len`` cycles."""
    on = cycles >= t0
    if kind == "degrau":
        return on.astype(float)
    return np.clip((cycles - t0 + 1) / ramp_len, 0.0, 1.0) * on


def delta_z(world: World, base: pd.DataFrame, kind: str, magnitude: float, t0: int, rng: np.random.Generator) -> np.ndarray:
    """Fault added to the INJECT_SENSORS, in regime-standardized units (rows x sensors)."""
    cycles = base["cycle"].to_numpy()
    n, k = len(base), len(INJECT_SENSORS)
    if kind in ("degrau", "rampa", "um_regime"):
        g = fault_profile("degrau" if kind == "degrau" else "rampa", cycles, t0)
        if kind == "um_regime":
            g = g * (base["regime_id"].to_numpy() == world.main_regime)
        return magnitude * g[:, None] * world.signature[None, :]
    out = np.zeros((n, k))
    on = (cycles >= t0)
    if kind == "variancia":
        out[on] = rng.normal(0.0, magnitude, (int(on.sum()), k))
    elif kind == "vies_sensor":
        out[on, INJECT_SENSORS.index(BIAS_SENSOR)] = magnitude
    else:
        raise ValueError(f"unknown fault kind {kind!r}")
    return out


def inject(world: World, base: pd.DataFrame, dz: np.ndarray) -> pd.DataFrame:
    """Return ``base`` with ``dz`` (standardized units) added to the INJECT_SENSORS in raw units."""
    out = base.copy()
    sd = _lookup(world.ref_std, base["regime_id"].to_numpy(dtype=int))
    out[list(INJECT_SENSORS)] = base[list(INJECT_SENSORS)].to_numpy(dtype=float) + dz * sd
    return out


def donor_tail(world: World, donor: pd.DataFrame, n_cycles: int) -> np.ndarray | None:
    """Last ``n_cycles`` of a donor in standardized units, minus the donor's own healthy-window mean."""
    donor = donor.sort_values("cycle")
    if len(donor) < n_cycles + world.H:
        return None
    z = standardize(world, donor)
    own_base = z[(donor["cycle"] <= world.H).to_numpy()].mean(axis=0)
    return z[-n_cycles:] - own_base


def blind_columns(frame: pd.DataFrame, unit_id: int) -> pd.DataFrame:
    """Only the raw C-MAPSS schema: no RUL, no regime, no scenario."""
    return frame.assign(unit_id=unit_id)[COLUMNS]


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ============================================================================
# stage 1: generate
# ============================================================================

def generate(world: World, run_dir: Path, max_engines: int | None = None) -> dict:
    rng = np.random.default_rng(world.seed)
    t0 = world.H + 10
    horizon_det = t0 + POST_CYCLES
    horizon_rul = t0 + TRANSPLANT_N - 1  # EOL = last transplanted cycle
    engines = {int(u): g.sort_values("cycle") for u, g in world.test.groupby("unit_id")}
    recipients = [u for u in sorted(engines) if (engines[u]["rul_true"] >= DEFAULT_RUL_CAP).sum() >= world.H + 10]
    if max_engines:
        recipients = recipients[:max_engines]
    det_frames, det_meta, rul_frames, rul_meta = [], [], [], []
    det_id = rul_id = 0
    for u in recipients:
        base = healthy_surrogate(engines[u], horizon_det, world.H, rng)
        det_id += 1
        det_frames.append(blind_columns(base, det_id))
        det_meta.append({"blind_id": det_id, "recipient": u, "role": "controle", "kind": "", "magnitude": np.nan, "t0": t0})
        for kind, mags in SCENARIOS.items():
            if kind == "um_regime" and world.regimes.n_clusters < 2:
                continue
            for m in mags:
                det_id += 1
                det_frames.append(blind_columns(inject(world, base, delta_z(world, base, kind, m, t0, rng)), det_id))
                det_meta.append({"blind_id": det_id, "recipient": u, "role": "falha", "kind": kind, "magnitude": m, "t0": t0})
        base_r = healthy_surrogate(engines[u], horizon_rul, world.H, rng)
        donors = [d for d in sorted(engines) if d != u and len(engines[d]) >= TRANSPLANT_N + world.H]
        if not donors:
            continue
        donor = int(rng.choice(donors))
        tail = donor_tail(world, engines[donor], TRANSPLANT_N)
        dz = np.zeros((horizon_rul, len(INJECT_SENSORS)))
        dz[t0 - 1:] = tail
        rul_id += 1
        rul_frames.append(blind_columns(base_r, rul_id))
        rul_meta.append({"blind_id": rul_id, "recipient": u, "donor": -1, "role": "controle", "t0": t0, "eol_cycle": horizon_rul})
        rul_id += 1
        rul_frames.append(blind_columns(inject(world, base_r, dz), rul_id))
        rul_meta.append({"blind_id": rul_id, "recipient": u, "donor": donor, "role": "transplante", "t0": t0, "eol_cycle": horizon_rul})
    blind_dir, sealed_dir = run_dir / "dados_cegos", run_dir / "selado"
    blind_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(det_frames, ignore_index=True).to_csv(blind_dir / "cego_deteccao.csv", index=False)
    pd.concat(rul_frames, ignore_index=True).to_csv(blind_dir / "cego_rul.csv", index=False)
    pd.DataFrame(det_meta).to_csv(sealed_dir / "manifesto_deteccao.csv", index=False)
    pd.DataFrame(rul_meta).to_csv(sealed_dir / "manifesto_rul.csv", index=False)
    digest = {p.name: manifest_sha256(p) for p in sorted(sealed_dir.glob("manifesto_*.csv"))}
    (blind_dir / "manifesto_sha256.txt").write_text("".join(f"{h}  {n}\n" for n, h in digest.items()))
    print(f"[generate] {len(recipients)} recipients | {det_id} detector series | {rul_id} RUL series | t0={t0}", flush=True)
    return {"recipients": len(recipients), "t0": t0}


# ============================================================================
# stage 2: detect / predict (reads ONLY dados_cegos/)
# ============================================================================

def detect(world: World, run_dir: Path) -> pd.DataFrame:
    blind = pd.read_csv(run_dir / "dados_cegos" / "cego_deteccao.csv")
    tr, te = world.regimes.transform(world.train), world.regimes.transform(blind)
    vs = VirtualSensor(target_sensor="T50", mode="exogenous", ewma_alpha=0.10, min_consecutive_breaches=2,
                       healthy_cycle_limit=world.H, random_state=world.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, per_engine, _ = vs.evaluate(tr, te)
        # engine_offset=False de propósito: este estudo reporta a detecção no limite NOMINAL (sem falso alarme igualado), e a MEWMA com o deslocamento do motor
        # tem limite nominal mal calibrado (falso alarme nos controles de até 86 % em H = 25); só vale a falso alarme igualado (fault_detection_study).
        mv = MultivariateResidualMonitor(healthy_cycle_limit=world.H, random_state=world.seed, engine_offset=False).fit(tr)
        mv_alarms, _, _ = mv.alarms(mv.score(te))
    t50 = {int(r.unit_id): r.first_alarm_cycle for r in per_engine.itertuples()}
    out = pd.DataFrame({"blind_id": sorted(t50)})
    out[DETECTORS[0]] = out["blind_id"].map(t50)
    out[DETECTORS[1]] = out["blind_id"].map(mv_alarms)
    saidas = run_dir / "saidas"
    saidas.mkdir(parents=True, exist_ok=True)
    out.to_csv(saidas / "alarmes.csv", index=False)
    return out


def _model_recipes(names: list[str], seed: int):
    from pdm_cmaps.pipeline.extra_models import build_extra_models
    from pdm_cmaps.modeling.minirocket import MiniRocketRULModel
    from pdm_cmaps.modeling.registry import get_model_pool

    pool, extras = get_model_pool(seed), build_extra_models(seed)
    suffix = " (+temporal)"
    for name in names:
        if name in extras:
            yield name, extras[name][0], extras[name][1], False
        elif name in pool:
            yield name, "sklearn", pool[name], False
        elif name.endswith(suffix) and name[: -len(suffix)] in pool:
            yield name, "sklearn", pool[name[: -len(suffix)]], True
        elif name == "MiniRocket + RidgeCV":
            yield name, "deep", MiniRocketRULModel(seq_length=15, num_kernels=1000, random_state=seed), False
        else:
            print(f"[WARN] no recipe for model {name!r}; omitted", flush=True)


def predict_rul(world: World, run_dir: Path, names: list[str]) -> pd.DataFrame:
    from pdm_cmaps.pipeline.extra_models import fit_predict

    blind = pd.read_csv(run_dir / "dados_cegos" / "cego_rul.csv").sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    loader = CMapssLoader()
    sensors = loader.informative_sensors(world.train)
    features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
    engineer = CausalFeatureEngineer(sensors, windows=(5, 10)).fit(world.train, world.H)
    fe_train, fe_blind = engineer.transform(world.train), engineer.transform(blind)
    fe_features = [*features, *[c for c in fe_train.columns if c not in world.train.columns]]
    out = blind[["unit_id", "cycle"]].rename(columns={"unit_id": "blind_id"}).copy()
    for name, m_type, model, temporal in _model_recipes(names, world.seed):
        print(f"[predict] {name}", flush=True)
        tr_f, ev_f, cols = (fe_train, fe_blind, fe_features) if temporal else (world.train, blind, features)
        pred = fit_predict(m_type, model, tr_f, cols, ev_f)
        out[name] = np.clip(pred, 0.0, float(DEFAULT_RUL_CAP))
    saidas = run_dir / "saidas"
    saidas.mkdir(parents=True, exist_ok=True)
    out.to_csv(saidas / "predicoes_rul.csv", index=False)
    return out


# ============================================================================
# stage 3: score (reads the sealed manifest)
# ============================================================================

def score_detection(manifest: pd.DataFrame, alarms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-series outcome and per (kind, magnitude, detector) summary. Pre-t0 alarms are false alarms."""
    rows = []
    for det in DETECTORS:
        merged = manifest.merge(alarms[["blind_id", det]], on="blind_id").rename(columns={det: "alarm"})
        merged["Detector"] = det
        rows.append(merged)
    per = pd.concat(rows, ignore_index=True)
    per["has_alarm"] = per["alarm"].notna()
    per["false_alarm"] = per["has_alarm"] & (per["alarm"] < per["t0"])
    per["detected"] = per["has_alarm"] & (per["alarm"] >= per["t0"])
    per["delay"] = np.where(per["detected"], per["alarm"] - per["t0"], np.nan)
    ctrl = per[per["role"] == "controle"].groupby("Detector")["has_alarm"].mean().mul(100).rename("far_controle_pct")
    faults = per[per["role"] == "falha"]
    elig = faults[~faults["false_alarm"]]  # a series already alarmed before t0 cannot show its own detection
    g = elig.groupby(["kind", "magnitude", "Detector"])
    summary = g.agg(n=("detected", "size"), poder_pct=("detected", lambda s: 100 * s.mean()),
                    atraso_mediano=("delay", "median"), atraso_p90=("delay", lambda s: s.quantile(0.9))).reset_index()
    pre = faults.groupby(["kind", "magnitude", "Detector"])["false_alarm"].mean().mul(100).rename("pre_t0_alarme_pct").reset_index()
    summary = summary.merge(pre, on=["kind", "magnitude", "Detector"]).merge(ctrl.reset_index(), on="Detector")
    return per, summary


def _first_run_below(values: np.ndarray, cycles: np.ndarray, threshold: float, after: int, window: int = 3) -> int | None:
    smooth = pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()
    hit = np.flatnonzero((smooth <= threshold) & (cycles > after))
    return int(cycles[hit[0]]) if hit.size else None


def score_rul(manifest: pd.DataFrame, preds: pd.DataFrame, H: int, alert_rul: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = [c for c in preds.columns if c not in ("blind_id", "cycle")]
    rows = []
    for meta in manifest.itertuples():
        eng = preds[preds["blind_id"] == meta.blind_id].sort_values("cycle")
        cyc = eng["cycle"].to_numpy()
        truth = np.minimum(meta.eol_cycle - cyc, DEFAULT_RUL_CAP)
        fault = cyc >= meta.t0
        for model in models:
            p = eng[model].to_numpy()
            alert = _first_run_below(p, cyc, alert_rul, H)
            row = {"blind_id": meta.blind_id, "recipient": meta.recipient, "role": meta.role, "Model": model,
                   "alert_cycle": alert, "alert_before_t0": alert is not None and alert < meta.t0,
                   "alert_lead": (meta.eol_cycle - alert) if alert is not None and alert >= meta.t0 else np.nan,
                   "pred_min": float(p[cyc > H].min())}
            if meta.role == "transplante":
                row["RMSE_falha"] = float(np.sqrt(np.mean((p[fault] - truth[fault]) ** 2)))
                row["Bias_falha"] = float(np.mean(p[fault] - truth[fault]))
                for cp in (60, 30, 10):
                    j = np.flatnonzero(truth == cp)
                    row[f"pred_em_RUL{cp}"] = float(p[j[0]]) if j.size else np.nan
            rows.append(row)
    per = pd.DataFrame(rows)
    ctrl = per[per["role"] == "controle"].groupby("Model")["alert_cycle"].apply(lambda s: 100 * s.notna().mean()).rename("alerta_falso_controle_pct")
    tp = per[per["role"] == "transplante"]
    summary = tp.groupby("Model").agg(
        n=("blind_id", "size"), RMSE_falha=("RMSE_falha", "mean"), Bias_falha=("Bias_falha", "mean"),
        pred_RUL60=("pred_em_RUL60", "mean"), pred_RUL30=("pred_em_RUL30", "mean"), pred_RUL10=("pred_em_RUL10", "mean"),
        alerta_pct=("alert_lead", lambda s: 100 * s.notna().mean()), antecedencia_mediana=("alert_lead", "median"),
        alerta_antes_t0_pct=("alert_before_t0", lambda s: 100 * s.mean()),
    ).join(ctrl).reset_index().sort_values("RMSE_falha")
    return per, summary


# ============================================================================
# figures
# ============================================================================

def plot_detection_power(summary: pd.DataFrame, path: Path) -> None:
    viz_style.apply_style()
    kinds = [k for k in SCENARIOS if k in set(summary["kind"])]
    fig, axes = plt.subplots(2, len(kinds), figsize=(3.6 * len(kinds), 6.2), squeeze=False)
    colors = dict(zip(DETECTORS, viz_style.SERIES[:2]))
    for j, kind in enumerate(kinds):
        for det in DETECTORS:
            d = summary[(summary["kind"] == kind) & (summary["Detector"] == det)].sort_values("magnitude")
            axes[0, j].plot(d["magnitude"], d["poder_pct"], marker="o", color=colors[det], lw=viz_style.LINE_W, label=det)
            axes[1, j].plot(d["magnitude"], d["atraso_mediano"], marker="o", color=colors[det], lw=viz_style.LINE_W)
        unit = "× assinatura de fim de vida" if kind in ("degrau", "rampa", "um_regime") else "σ"
        axes[0, j].set_title(f"{kind.replace('_', ' ')}", fontsize=10)
        axes[1, j].set_xlabel(f"magnitude ({unit})", fontsize=8)
        axes[0, j].set_ylim(-3, 103)
        for i in range(2):
            viz_style.style_axes(axes[i, j])
    axes[0, 0].set_ylabel("poder de detecção (%)")
    axes[1, 0].set_ylabel("atraso mediano após t0 (ciclos)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    far = summary.drop_duplicates("Detector").set_index("Detector")["far_controle_pct"]
    fig.suptitle("Falhas injetadas (t0 conhecido só por nós): poder e atraso de detecção. Falso alarme nos controles: "
                 + " · ".join(f"{d} {far[d]:.1f} %" for d in DETECTORS), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    summary.to_csv(path.with_suffix(".csv"), index=False)


NOMES_RUL_PT = {
    "XGBoost (NASA Loss)": "XGBoost (perda NASA)",
    "Support Vector Machine (SVM)": "SVM linear",
    "MiniRocket + RidgeCV": "MiniRocket + Ridge",
    "PyTorch RNN (LSTM)": "LSTM (PyTorch)",
    "PyTorch CNN": "CNN (PyTorch)",
    "PyTorch Transformer": "Transformer (PyTorch)",
}


def plot_rul_transplant(manifest: pd.DataFrame, preds: pd.DataFrame, path: Path, recipient: int | None = None) -> None:
    viz_style.apply_style()
    models = [c for c in preds.columns if c not in ("blind_id", "cycle")]
    recipient = recipient if recipient is not None else int(manifest["recipient"].iloc[0])
    pair = manifest[manifest["recipient"] == recipient].set_index("role")
    f_id, c_id = int(pair.loc["transplante", "blind_id"]), int(pair.loc["controle", "blind_id"])
    t0, eol = int(pair.loc["transplante", "t0"]), int(pair.loc["transplante", "eol_cycle"])
    fault, ctrl = (preds[preds["blind_id"] == i].sort_values("cycle") for i in (f_id, c_id))
    # grade de 3 colunas com uma célula livre para a legenda (5 modelos -> 2 x 3), legível na largura da página
    ncol = min(len(models) + 1, 3)
    nrow = int(np.ceil((len(models) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), sharex=True, sharey=True, squeeze=False)
    cyc = fault["cycle"].to_numpy()
    truth = np.minimum(eol - cyc, DEFAULT_RUL_CAP)
    for ax, model in zip(axes.flat, models):
        ax.plot(cyc, truth, color=viz_style.INK, lw=1.6, label="RUL verdadeira (por construção)")
        ax.plot(ctrl["cycle"], ctrl[model], color=viz_style.MUTED, lw=1.4, ls="--", label="controle (sem falha)")
        ax.plot(cyc, fault[model], color=viz_style.SERIES[0], lw=1.6, label="previsto com falha")
        ax.axvline(t0, color=viz_style.STATUS[3], lw=1.4, ls=":")
        ax.set_title(NOMES_RUL_PT.get(model, model), fontsize=11, loc="left")
        ax.tick_params(labelsize=9.5)
        viz_style.style_axes(ax)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=viz_style.STATUS[3], lw=1.4, ls=":"))
    l.append(f"início da falha (t0 = {t0},\noculto dos modelos)")
    list(axes.flat)[len(models)].legend(h, l, frameon=False, fontsize=10, loc="center")
    fig.suptitle(f"Transplante de degradação no motor receptor {recipient}", fontsize=12, x=0.01, ha="left")
    fig.supxlabel("ciclo", fontsize=11)
    fig.supylabel("RUL (ciclos)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def score(world: World, run_dir: Path, alert_rul: float) -> dict[str, pd.DataFrame]:
    sealed, out, tabs, figs = run_dir / "selado", run_dir / "saidas", run_dir / "tabelas", run_dir / "figuras"
    tabs.mkdir(parents=True, exist_ok=True)
    expected = (run_dir / "dados_cegos" / "manifesto_sha256.txt").read_text().split()
    for h, name in zip(expected[0::2], expected[1::2]):
        if manifest_sha256(sealed / name) != h:
            raise RuntimeError(f"sealed manifest {name} changed after generation")
    res = {}
    det_manifest = pd.read_csv(sealed / "manifesto_deteccao.csv")
    per, summ = score_detection(det_manifest, pd.read_csv(out / "alarmes.csv"))
    per.to_csv(tabs / "deteccao_por_serie.csv", index=False)
    summ.to_csv(tabs / "deteccao_resumo.csv", index=False)
    plot_detection_power(summ, figs / "fig_poder_deteccao.png")
    res["deteccao"] = summ
    rul_path = out / "predicoes_rul.csv"
    if rul_path.exists():
        rul_manifest, preds = pd.read_csv(sealed / "manifesto_rul.csv"), pd.read_csv(rul_path)
        per_r, summ_r = score_rul(rul_manifest, preds, world.H, alert_rul)
        per_r.to_csv(tabs / "rul_por_serie.csv", index=False)
        summ_r.to_csv(tabs / "rul_resumo.csv", index=False)
        plot_rul_transplant(rul_manifest, preds, figs / "fig_rul_transplante.png")
        res["rul"] = summ_r
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--subset", default="FD004")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--healthy-cycle-limit", type=int, default=DEFAULT_HEALTHY_CYCLE_LIMIT)
    p.add_argument("--stage", choices=["all", "generate", "detect", "score"], default="all")
    p.add_argument("--include-deep", action="store_true", help="also CNN / LSTM / Transformer / MiniRocket (slow)")
    p.add_argument("--alert-rul", type=float, default=60.0, help="a model 'alerts' when its 3-cycle mean RUL <= this")
    p.add_argument("--max-engines", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    a = p.parse_args(argv)
    subset = a.subset.upper()
    run_dir = a.output_dir / f"{subset}_seed{a.seed}"
    world = load_world(subset, a.seed, a.healthy_cycle_limit)
    if a.stage in ("all", "generate"):
        generate(world, run_dir, a.max_engines)
    if a.stage in ("all", "detect"):
        detect(world, run_dir)
        predict_rul(world, run_dir, DEFAULT_MODELS + (DEEP_MODELS if a.include_deep else []))
    if a.stage in ("all", "score"):
        res = score(world, run_dir, a.alert_rul)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(res["deteccao"].round(2).to_string(index=False))
            if "rul" in res:
                print(res["rul"].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

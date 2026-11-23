"""Predicted vs real RUL for every model and every held-out engine, with maintenance markers.

WHAT: reads `results/maintenance_guide_<subset>/tables/rul_holdout_predictions.csv` (written by
pipeline.run(): seed-42 75/25 engine split, every cycle of each held-out engine, all models, H=50)
and writes one figure per (model, engine), one all-models grid per engine and one pooled
predicted-vs-real scatter per model:

  results/rul_predicoes/<subset>/
      modelos/<model>/motor_XXX.png          real vs predicted RUL along the engine life
      modelos/<model>/_dispersao.png         pooled predicted vs real (all held-out engines)
      motores/motor_XXX.png                  all models for one engine (small multiples)
      manutencao/motor_XXX.png               same grid, only engines with a maintenance event / candidate
      metricas_por_motor.csv                 RMSE / MAE per (model, engine)
      candidatos_descontinuidade.csv         step-drop scan of the core speed Nc on ALL engines

WHY: the holdout tables only give fleet averages; per-engine traces show WHERE a model fails
(early plateau, late life, a reset). C-MAPSS is run-to-failure: no maintenance is recorded. The one
verified exception is FD004 engine 179, whose core (Nc, T30, T50, phi, P30) drops back to the
healthy level at cycle 242 while the fan side (Nf, T24) does not move
(found by scan_discontinuities below and confirmed by inspecting the five core channels).

[ASSUMPTION] Candidate scan: Nc standardized per regime with the cycles <= 50 of all engines;
drop(t) = mean(z[t:t+10]) - mean(z[t-10:t]); an engine is a CANDIDATE when min drop < -3 sigma.
Candidates are drawn as grey dotted lines and are NOT called maintenance.
[CAUSALITY] the RUL label is piecewise from the final failure (cap 125); it does NOT reset at a
maintenance event, so after a reset the "real" curve is the label convention, not physics.

Run:  .venv/bin/python -m pdm_cmaps.experiments.rul_prediction_plots [--subsets FD004] [--jobs 8]
"""

from __future__ import annotations

from pdm_cmaps.config import DEFAULT_RESULTS_DIR

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.reporting import viz_style

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "rul_predicoes"
VERIFIED_MAINTENANCE = {("FD004", 179): 242}  # core reset of engine 179; scan_discontinuities flags it as verified_maintenance
STEP_WINDOW = 10
STEP_SIGMA = 3.0
REAL_COLOR, PRED_COLOR, EVENT_COLOR, CAND_COLOR = viz_style.INK, viz_style.SERIES[0], viz_style.STATUS[3], viz_style.MUTED


def slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def scan_discontinuities(subset: str, loader: CMapssLoader | None = None) -> pd.DataFrame:
    """Largest sustained drop of the regime-standardized core speed Nc per engine (all train engines)."""
    frame = (loader or CMapssLoader()).load(subset)
    k = 6 if subset.upper() in {"FD002", "FD004"} else 1
    d = RegimeDecoupler(n_clusters=k, random_state=42).fit(frame).transform(frame)
    ref = d.loc[d["cycle"] <= 50].groupby("regime_id")["Nc"].agg(["mean", "std"])
    d["z"] = (d["Nc"] - d["regime_id"].map(ref["mean"])) / d["regime_id"].map(ref["std"])
    rows = []
    for unit, g in d.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        z, c, n = g["z"].to_numpy(), g["cycle"].to_numpy(), len(g)
        cs = np.concatenate([[0.0], np.cumsum(z)])
        t = np.arange(STEP_WINDOW, n - STEP_WINDOW + 1)
        if t.size == 0:
            continue
        after = (cs[t + STEP_WINDOW] - cs[t]) / STEP_WINDOW
        before = (cs[t] - cs[t - STEP_WINDOW]) / STEP_WINDOW
        i = int(np.argmin(after - before))
        rows.append({"Subset": subset, "unit_id": int(unit), "cycle": int(c[t[i]]), "eol_cycle": int(c.max()),
                     "Nc_z_before": float(before[i]), "Nc_z_after": float(after[i]), "drop_sigma": float(after[i] - before[i]),
                     "candidate": bool(after[i] - before[i] < -STEP_SIGMA),
                     "verified_maintenance": VERIFIED_MAINTENANCE.get((subset, int(unit))) == int(c[t[i]])})
    return pd.DataFrame(rows).sort_values("drop_sigma").reset_index(drop=True)


def _events(subset: str, unit: int, cands: pd.DataFrame) -> list[tuple[int, str]]:
    ev = []
    if (subset, unit) in VERIFIED_MAINTENANCE:
        ev.append((VERIFIED_MAINTENANCE[(subset, unit)], "verified"))
    for r in cands.loc[(cands["unit_id"] == unit) & cands["candidate"] & ~cands["verified_maintenance"]].itertuples():
        ev.append((int(r.cycle), "candidate"))
    return ev


def _draw_events(ax, events, label: bool) -> None:
    for cyc, kind in events:
        if kind == "verified":
            ax.axvline(cyc, color=EVENT_COLOR, lw=1.6, ls="--", zorder=1)
            if label:
                ax.annotate(f"manutenção / reset\ndo núcleo (ciclo {cyc})", xy=(cyc, 1), xycoords=("data", "axes fraction"),
                            xytext=(4, -4), textcoords="offset points", va="top", fontsize=8, color=viz_style.INK)
        else:
            ax.axvline(cyc, color=CAND_COLOR, lw=1.2, ls=":", zorder=1)
            if label:
                ax.annotate(f"queda brusca de Nc (ciclo {cyc})\ncandidato, não verificado", xy=(cyc, 1),
                            xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
                            va="top", fontsize=8, color=viz_style.INK_2)


def _engine_figure(eng: pd.DataFrame, model: str, subset: str, unit: int, events, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(eng["cycle"], eng["RUL_actual"], color=REAL_COLOR, lw=viz_style.LINE_W, label="RUL real (rótulo, teto 125)")
    ax.plot(eng["cycle"], eng["pred"], color=PRED_COLOR, lw=viz_style.LINE_W, label="RUL predito")
    _draw_events(ax, events, label=True)
    err = eng["pred"] - eng["RUL_actual"]
    rmse, mae = float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))
    viz_style.style_axes(ax)
    ax.set_xlabel("Ciclo")
    ax.set_ylabel("RUL (ciclos)")
    ax.set_ylim(bottom=min(0, float(eng["pred"].min()) - 5))
    ax.set_title(f"{subset} · motor {unit} · {model}\nRMSE {rmse:.1f} · MAE {mae:.1f} ciclos (holdout, seed 42, H=50)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


NOMES_PT = {
    "XGBoost (NASA Loss)": "XGBoost (perda NASA)",
    "Support Vector Machine (SVM)": "SVM linear",
    "MiniRocket + RidgeCV": "MiniRocket + Ridge",
    "PyTorch RNN (LSTM)": "LSTM (PyTorch)",
    "PyTorch CNN": "CNN (PyTorch)",
    "PyTorch Transformer": "Transformer (PyTorch)",
}


def _grid_figure(eng: pd.DataFrame, models: list[str], subset: str, unit: int, events, path: Path) -> None:
    ncol = 5
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow), sharex=True, sharey=True, squeeze=False)
    ymax = float(eng["RUL_actual"].max())
    for ax, model in zip(axes.flat, models):
        pred = eng[f"RUL_predicted_{model}"]
        rmse = float(np.sqrt(np.mean((pred - eng["RUL_actual"]) ** 2)))
        ax.plot(eng["cycle"], eng["RUL_actual"], color=REAL_COLOR, lw=1.5)
        ax.plot(eng["cycle"], pred, color=PRED_COLOR, lw=1.5)
        _draw_events(ax, events, label=False)
        for cyc, kind in events:
            if kind == "verified":  # o rótulo vai em todos os painéis: a figura é lida reduzida, e a legenda do título é pequena
                ax.text(cyc + 8, ymax * 1.13, f"descontinuidade (ciclo {cyc})", ha="left", va="center", fontsize=8.5, color=EVENT_COLOR, fontweight="bold")
        viz_style.style_axes(ax)
        label = NOMES_PT.get(model, model)
        ax.set_title(f"{label}\n" + f"RMSE {rmse:.1f}".replace(".", ","), fontsize=9.5)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    note = ""
    if any(k == "verified" for _, k in events):
        note = " · linha vermelha tracejada = descontinuidade no ciclo 242"
    elif events:
        note = " · linha cinza pontilhada = queda brusca de Nc (candidato, não verificado)"
    fig.suptitle(f"{subset} · motor {unit}: RUL real (preto) × prevista (azul), todos os modelos{note}", fontsize=13)
    fig.supxlabel("Ciclo", fontsize=11)
    fig.supylabel("RUL (ciclos)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _scatter_figure(df: pd.DataFrame, model: str, subset: str, path: Path) -> None:
    from sklearn.metrics import r2_score

    truth, pred = df["RUL_actual"].to_numpy(), df[f"RUL_predicted_{model}"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(truth, pred, s=6, alpha=0.2, color=PRED_COLOR, edgecolors="none", label="todos os ciclos, motores do holdout")
    for (sub, unit), cyc in VERIFIED_MAINTENANCE.items():
        if sub != subset:
            continue
        m = df["unit_id"] == unit
        if m.any():
            after = m & (df["cycle"] >= cyc)
            ax.scatter(truth[m & ~after], pred[m & ~after], s=14, color=viz_style.SERIES[1], label=f"motor {unit}, antes do ciclo {cyc}")
            ax.scatter(truth[after], pred[after], s=14, color=EVENT_COLOR, label=f"motor {unit}, após reset (ciclo ≥ {cyc})")
    lo, hi = float(min(truth.min(), pred.min())), float(max(truth.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color=viz_style.INK_2, ls="--", lw=1.2, label="predição perfeita")
    viz_style.style_axes(ax, grid="both")
    ax.set_xlabel("RUL real (ciclos)")
    ax.set_ylabel("RUL predito (ciclos)")
    ax.set_title(f"{subset} · {model}\nR² holdout = {r2_score(truth, pred):.3f} · n = {len(truth)} ciclos", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _model_job(df: pd.DataFrame, model: str, subset: str, cands: pd.DataFrame, out: Path) -> list[dict]:
    viz_style.apply_style()
    folder = out / "modelos" / slug(model)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for unit, eng in df.groupby("unit_id"):
        eng = eng.sort_values("cycle").assign(pred=lambda e: e[f"RUL_predicted_{model}"])
        _engine_figure(eng, model, subset, int(unit), _events(subset, int(unit), cands), folder / f"motor_{int(unit):03d}.png")
        err = eng["pred"] - eng["RUL_actual"]
        rows.append({"Subset": subset, "Model": model, "unit_id": int(unit), "n_cycles": len(eng),
                     "RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err))), "Bias": float(err.mean())})
    _scatter_figure(df, model, subset, folder / "_dispersao.png")
    return rows


def _engine_job(eng: pd.DataFrame, models, subset: str, unit: int, events, out: Path) -> None:
    viz_style.apply_style()
    _grid_figure(eng.sort_values("cycle"), models, subset, unit, events, out / "motores" / f"motor_{unit:03d}.png")
    if events:
        _grid_figure(eng.sort_values("cycle"), models, subset, unit, events, out / "manutencao" / f"motor_{unit:03d}.png")


def run_subset(subset: str, out_root: Path, jobs: int) -> pd.DataFrame:
    from joblib import Parallel, delayed

    src = DEFAULT_RESULTS_DIR / f"maintenance_guide_{subset.lower()}" / "tables" / "rul_holdout_predictions.csv"
    df = pd.read_csv(src)
    models = [c.replace("RUL_predicted_", "") for c in df.columns if c.startswith("RUL_predicted_")]
    out = out_root / subset
    for sub in ("modelos", "motores", "manutencao"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cands = scan_discontinuities(subset)
    cands.to_csv(out / "candidatos_descontinuidade.csv", index=False)
    print(f"[{subset}] {df['unit_id'].nunique()} holdout engines x {len(models)} models; "
          f"{int(cands['candidate'].sum())} Nc-drop candidates in the whole fleet", flush=True)
    per_model = Parallel(n_jobs=jobs)(delayed(_model_job)(df, m, subset, cands, out) for m in models)
    Parallel(n_jobs=jobs)(delayed(_engine_job)(eng, models, subset, int(u), _events(subset, int(u), cands), out)
                          for u, eng in df.groupby("unit_id"))
    metrics = pd.DataFrame([r for rows in per_model for r in rows])
    metrics.to_csv(out / "metricas_por_motor.csv", index=False)
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    a = p.parse_args(argv)
    for s in a.subsets:
        run_subset(s.upper(), a.output_dir, a.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

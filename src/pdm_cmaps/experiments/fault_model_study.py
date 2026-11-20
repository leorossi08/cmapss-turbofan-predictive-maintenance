"""Which learning model should be the virtual sensor? Detection of injected faults per sensor and per estimator.

WHAT: the detectors compare each sensor (T50, Nc, P30, ...) with what a model expects from the operating
conditions. Here the model inside every virtual sensor is swapped (same engines, same injected faults, same
detector rule) and the power / delay of each chart is measured at a MATCHED false-alarm rate:
    Extra Trees (padrão) · Random Forest · LightGBM · XGBoost · Ridge · MLP · Média do regime (sem ML)
Charts scored per estimator: one EWMA chart per sensor (13 sensors, regime-conditioned, no raw-3sigma gate)
and the 13-residual MEWMA built from the same residuals.
WHY: virtual-sensor inputs are settings + near-constant ambient channels, so within a regime the model is close
to a regime mean. The "Média do regime (sem ML)" row tests that directly: if the learning models
do not beat it, the gain of the detector comes from the regime conditioning, not from the learner.

[LEAKAGE] estimators fit on TRAINING engines (healthy window H) only; injected series come from TEST engines
and reach the charts as raw columns; the answer key is used only by the scoring functions.
[ASSUMPTION] estimators are used with the repo defaults (no hyperparameter search); same series for all
estimators within a (subset, seed); false-alarm budget 0.2 per 100 monitored cycles.

Run:  .venv/bin/python -m pdm_cmaps.experiments.fault_model_study --H 50 [--subsets FD004] [--jobs 3]   /  --aggregate-only
Output: results/injecao_modelos/{runs,tabelas,figuras}
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
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor

from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor
from pdm_cmaps.experiments.fault_detection_study import BUDGETS, analyse_detector, build_series
from pdm_cmaps.experiments.fault_injection import World, load_world
from pdm_cmaps.modeling.registry import get_model
from pdm_cmaps.reporting import viz_style

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "injecao_modelos"
MEWMA = "MEWMA (13 resíduos)"
DIAGNOSTIC_ESTIMATORS = ["Mistura ET 25%", "Mistura ET 50%", "Extra Trees regularizado"]  # só para o diagnóstico do FD003 (ruído de estimação)
DEFAULT_ESTIMATORS = ["Extra Trees (padrão)", "Random Forest", "LightGBM", "XGBoost", "Ridge", "MLP", "Média do regime (sem ML)"]  # os do estudo principal
ESTIMATORS = [*DEFAULT_ESTIMATORS, *DIAGNOSTIC_ESTIMATORS]  # os aceitos na linha de comando
BASELINE = "Média do regime (sem ML)"


class BlendWithMean(BaseEstimator, RegressorMixin):
    """weight * base(x) + (1 - weight) * mean(y): encolhe um modelo aprendido na direção da média do regime (diagnóstico de ruído de estimação)."""

    def __init__(self, base=None, weight: float = 0.5):
        self.base = base
        self.weight = weight

    def fit(self, X, y):
        self.base_ = clone(self.base).fit(X, y)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return self.weight * np.asarray(self.base_.predict(X), dtype=float) + (1.0 - self.weight) * self.mean_
KEY_FAULTS = (("degrau", 0.25), ("rampa", 0.25), ("variancia", 1.0), ("vies_sensor", 2.0))


def make_estimator(name: str, seed: int):
    """Fresh, untuned estimator with the repo defaults (same family/size as the benchmark pool)."""
    if name == "Extra Trees (padrão)":
        return get_model("extra_trees", seed, n_estimators=60, max_depth=12, n_jobs=1)
    if name in ("Mistura ET 25%", "Mistura ET 50%"):
        return BlendWithMean(get_model("extra_trees", seed, n_estimators=60, max_depth=12, n_jobs=1), 0.25 if "25" in name else 0.5)
    if name == "Extra Trees regularizado":
        return get_model("extra_trees", seed, n_estimators=60, max_depth=12, min_samples_leaf=100, n_jobs=1)
    if name == "Random Forest":
        return get_model("rf", seed, n_estimators=60, max_depth=12, n_jobs=1)
    if name == "LightGBM":
        return get_model("lightgbm", seed, n_estimators=80, max_depth=6, n_jobs=1)
    if name == "XGBoost":
        return get_model("xgboost", seed, n_estimators=80, max_depth=6, n_jobs=1)
    if name == "Ridge":
        return get_model("ridge", seed, alpha=10)
    if name == "MLP":
        return get_model("mlp", seed, hidden_layer_sizes=(64, 32), max_iter=180)
    if name == BASELINE:
        return DummyRegressor(strategy="mean")  # ignores the inputs: the regime mean
    raise ValueError(name)


def model_scores(mv: MultivariateResidualMonitor, te: pd.DataFrame, ids_control: set[int], H: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Per-sensor EWMA ratios (regime-conditioned, no gate), the MEWMA ratio and the healthy residual sd on controls."""
    scores, zcols, resid = {}, {}, []
    index = None
    for sensor, vs in mv.virtual_.items():
        pred = vs.predict(te).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        index = pred[["unit_id", "cycle"]]
        zcols[sensor] = mv.residual_z(vs, pred)  # entra no T² (com o deslocamento do motor removido, se o monitor o usa)
        parts = []
        for uid, g in pred.assign(z=mv.chart_z(vs, pred)).groupby("unit_id", sort=True):
            e = vs.ewma_detector.ewma_abs_z(g["z"].to_numpy())
            parts.append(pd.DataFrame({"blind_id": uid, "cycle": g["cycle"].to_numpy(), "ratio": e / vs.ewma_detector.ewma_upper_limits(len(e)), "gate": True}))
        scores[sensor] = pd.concat(parts, ignore_index=True)
        ctrl = pred[pred["unit_id"].isin(ids_control) & (pred["cycle"] > H)]
        resid.append({"Detector": sensor, "resid_sd": float(ctrl["residual"].std())})
    out = index.copy()
    zmat = np.column_stack([zcols[s] for s in mv.virtual_])
    t2 = mv._t2(zmat, out["unit_id"].to_numpy())
    scores[MEWMA] = pd.DataFrame({"blind_id": out["unit_id"].to_numpy(), "cycle": out["cycle"].to_numpy(), "ratio": t2 / mv.ucl_, "gate": True})
    return scores, pd.DataFrame(resid)


def run_one(subset: str, seed: int, H: int, out: Path, n_ctrl: int = 6, n_rep: int = 2, max_engines: int | None = None, estimators: list[str] | None = None,
            mewma_offset: bool = True) -> None:
    tag = f"{subset}_seed{seed}_H{H}"
    path = out / "runs" / f"{tag}_pareado.csv"
    if path.exists():
        return
    world: World = load_world(subset, seed, H, loader=CMapssLoader())
    blind, meta = build_series(world, seed, n_ctrl, n_rep, max_engines)
    tr, te = world.regimes.transform(world.train), world.regimes.transform(blind)
    ctrl_ids = set(meta.loc[meta["role"] == "controle", "blind_id"].astype(int))
    rows, resid_rows = [], []
    for est in estimators or DEFAULT_ESTIMATORS:
        print(f"[{tag}] {est}", flush=True)
        mv = MultivariateResidualMonitor(healthy_cycle_limit=H, random_state=seed, estimator=make_estimator(est, seed), engine_offset=mewma_offset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.fit(tr)
            scores, resid = model_scores(mv, te, ctrl_ids, H)
        resid_rows.append(resid.assign(Estimator=est))
        for name, sc in scores.items():
            rows.append(analyse_detector(name, sc, meta, H, BUDGETS)["pareado"].assign(Estimator=est))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(resid_rows).assign(Subset=subset, Seed=seed, H=H).to_csv(out / "runs" / f"{tag}_residuos.csv", index=False)
    pd.concat(rows).assign(Subset=subset, Seed=seed, H=H).to_csv(path, index=False)


def _load(out: Path, kind: str) -> pd.DataFrame:
    files = sorted((out / "runs").glob(f"*_{kind}.csv"))
    if not files:
        raise SystemExit(f"nenhum resultado em {out / 'runs'}: rode sem --aggregate-only")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def aggregate(out: Path) -> dict[str, pd.DataFrame]:
    tabs = out / "tabelas"
    tabs.mkdir(parents=True, exist_ok=True)
    par, res = _load(out, "pareado"), _load(out, "residuos")
    keys = ["Subset", "H", "Estimator", "Detector", "kind", "magnitude", "budget"]
    g = par.groupby(keys)[["power_pct", "delay_med", "far_ctrl_per100"]]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    base = res[res["Estimator"] == BASELINE].set_index(["Subset", "Seed", "H", "Detector"])["resid_sd"]
    res = res.join(base.rename("resid_sd_baseline"), on=["Subset", "Seed", "H", "Detector"])
    res["resid_sd_ratio"] = res["resid_sd"] / res["resid_sd_baseline"]
    rsum = res.groupby(["Subset", "H", "Estimator", "Detector"])["resid_sd_ratio"].mean().reset_index()
    ok = par[par["power_pct"] >= 80.0]
    mdm_keys = ["Subset", "Seed", "H", "Estimator", "Detector", "kind", "budget"]
    mdm = ok.groupby(mdm_keys)["magnitude"].min().rename("mdm").reset_index()
    mdm = par[mdm_keys].drop_duplicates().merge(mdm, on=mdm_keys, how="left").groupby(["Subset", "H", "Estimator", "Detector", "kind", "budget"])["mdm"].median().reset_index()
    summ.to_csv(tabs / "modelos_pareado_resumo.csv", index=False)
    rsum.to_csv(tabs / "modelos_residuo_vs_media_regime.csv", index=False)
    mdm.to_csv(tabs / "modelos_magnitude_minima.csv", index=False)
    return {"resumo": summ, "residuo": rsum, "mdm": mdm}


def fig_mewma_by_estimator(summ: pd.DataFrame, path: Path, H: int) -> None:
    viz_style.apply_style()
    subsets = sorted(set(summ["Subset"]))
    fig, axes = plt.subplots(len(KEY_FAULTS), len(subsets), figsize=(3.6 * len(subsets), 2.7 * len(KEY_FAULTS)), squeeze=False, sharey=True)
    for i, (kind, mag) in enumerate(KEY_FAULTS):
        for j, sub in enumerate(subsets):
            d = summ[(summ["Subset"] == sub) & (summ["Detector"] == MEWMA) & (summ["kind"] == kind) & np.isclose(summ["magnitude"], mag) & (summ["H"] == H)].set_index("Estimator").reindex(DEFAULT_ESTIMATORS)
            colors = [viz_style.INK_2 if e == BASELINE else viz_style.SERIES[1] for e in DEFAULT_ESTIMATORS]
            axes[i, j].barh(range(len(DEFAULT_ESTIMATORS)), d["power_pct_mean"], xerr=d["power_pct_std"].fillna(0), color=colors, height=0.65, capsize=2)
            axes[i, j].set_yticks(range(len(DEFAULT_ESTIMATORS)))
            axes[i, j].set_yticklabels(DEFAULT_ESTIMATORS if j == 0 else [], fontsize=8)
            axes[i, j].invert_yaxis()
            axes[i, j].set_xlim(0, 105)
            viz_style.style_axes(axes[i, j], grid="x")
            if i == 0:
                axes[i, j].set_title(sub, fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(f"{kind} {mag:g}", fontsize=9)
            if i == len(KEY_FAULTS) - 1:
                axes[i, j].set_xlabel("poder de detecção (%)", fontsize=8)
    fig.suptitle(f"MEWMA com cada modelo como sensor virtual (H = {H}, falso alarme ≤ 0,2 por 100 ciclos; cinza = média do regime, sem ML)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_sensor_heatmap(summ: pd.DataFrame, path: Path, H: int, kind: str = "degrau", mag: float = 0.25) -> None:
    viz_style.apply_style()
    subsets = sorted(set(summ["Subset"]))
    fig, axes = plt.subplots(1, len(subsets), figsize=(4.6 * len(subsets), 5.2), squeeze=False)
    for ax, sub in zip(axes[0], subsets):
        d = summ[(summ["Subset"] == sub) & (summ["kind"] == kind) & np.isclose(summ["magnitude"], mag) & (summ["H"] == H) & (summ["Detector"] != MEWMA)]
        p = d.pivot(index="Detector", columns="Estimator", values="power_pct_mean").reindex(columns=DEFAULT_ESTIMATORS)
        im = ax.imshow(p.values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(DEFAULT_ESTIMATORS)))
        ax.set_xticklabels([e.replace(" (padrão)", "").replace(" (sem ML)", "") for e in DEFAULT_ESTIMATORS], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(p.index)))
        ax.set_yticklabels(p.index, fontsize=8)
        for (r, c), v in np.ndenumerate(p.values):
            if np.isfinite(v):
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else viz_style.INK)
        ax.set_title(sub, fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6, label="poder (%)")
    fig.suptitle(f"Poder de detecção por sensor e modelo ({kind} {mag:g}, H = {H}, falso alarme ≤ 0,2 por 100 ciclos)", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_residual_ratio(rsum: pd.DataFrame, path: Path, H: int) -> None:
    viz_style.apply_style()
    d = rsum[(rsum["H"] == H) & (rsum["Detector"] != MEWMA)].groupby(["Subset", "Estimator"])["resid_sd_ratio"].median().unstack("Estimator").reindex(columns=DEFAULT_ESTIMATORS)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.8 / len(d.index)
    for k, sub in enumerate(d.index):
        ax.bar(np.arange(len(DEFAULT_ESTIMATORS)) + k * w, d.loc[sub], w, label=sub, color=[*viz_style.SERIES, viz_style.INK_2][k % 4])
    ax.axhline(1.0, color=viz_style.INK, lw=1)
    ax.set_xticks(np.arange(len(DEFAULT_ESTIMATORS)) + 0.4 - w / 2)
    ax.set_xticklabels([e.replace(" (padrão)", "").replace(" (sem ML)", "") for e in DEFAULT_ESTIMATORS], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("desvio do resíduo saudável / o da média do regime\n(mediana dos sensores; < 1 = modelo melhor que a média)")
    viz_style.style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"O modelo prevê o sensor melhor que a média do regime? (controles saudáveis, H = {H})", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_figures(res: dict[str, pd.DataFrame], out: Path, H: int) -> None:
    fig_mewma_by_estimator(res["resumo"], out / "figuras" / "fig_mewma_por_modelo.png", H)
    fig_sensor_heatmap(res["resumo"], out / "figuras" / "fig_sensor_x_modelo.png", H)
    fig_residual_ratio(res["residuo"], out / "figuras" / "fig_residuo_vs_media_regime.png", H)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--H", type=int, required=True, help="healthy window (use the best one from fault_detection_study)")
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    p.add_argument("--estimators", nargs="+", default=None, choices=ESTIMATORS)
    p.add_argument("--n-ctrl", type=int, default=6)
    p.add_argument("--n-rep", type=int, default=2)
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
        if a.jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=a.jobs)(delayed(run_one)(s, seed, a.H, a.output_dir, a.n_ctrl, a.n_rep, a.max_engines, a.estimators, a.mewma_offset) for s, seed in jobs)
        else:
            for s, seed in jobs:
                run_one(s, seed, a.H, a.output_dir, a.n_ctrl, a.n_rep, a.max_engines, a.estimators, a.mewma_offset)
    res = aggregate(a.output_dir)
    make_figures(res, a.output_dir, a.H)
    s = res["resumo"]
    top = s[(s["H"] == a.H) & (s["Detector"] == MEWMA) & (s["kind"] == "degrau") & np.isclose(s["magnitude"], 0.25)]
    with pd.option_context("display.width", 200):
        print(top.pivot(index="Estimator", columns="Subset", values="power_pct_mean").reindex(DEFAULT_ESTIMATORS).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Virtual sensor for EVERY informative sensor, EVERY model and EVERY held-out engine (feeds the dashboards).

WHAT: the pipeline fits one virtual sensor (default ExtraTrees) for four primary sensors and, historically,
saved the multi-model comparison only for engine 179. This study repeats the same protocol for all
informative sensors and a pool of regressors, on the same engine split, regimes and healthy window as the
pipeline, and writes the full trace of the held-out engines so any dashboard filter (dataset, engine,
sensor, model) has data.

[LEAKAGE] same discipline as the pipeline: engine split first; sensor filter, regimes and every regressor
are fit on TRAINING engines (healthy window only); held-out engines are only scored.
[ASSUMPTION] the regressors use the exogenous inputs (settings + ambient channels), exactly like the
pipeline's virtual sensor, so the comparison isolates the regressor, not the feature set.

Run:  .venv/bin/python -m pdm_cmaps.experiments.virtual_sensor_models [--subset FD004 | --all] [--models ...]
Output: results/virtual_sensor_models/<subset>/{trace.csv.gz, per_engine_alarms.csv, model_scores.csv}
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from pdm_cmaps.config import DEFAULT_RESULTS_DIR
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.splits import partition_healthy_and_lifecycle, split_engine_units
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.detection.virtual_sensor_metrics import virtual_sensor_r2
from pdm_cmaps.domain.rul import compute_piecewise_rul
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT
from pdm_cmaps.modeling.registry import get_model

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "virtual_sensor_models"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# display name -> registry key and kwargs (small, untuned, same spirit as the RUL pool)
MODELS = {
    # n_jobs=1: os dados são pequenos (janela saudável); com várias threads o custo é só sincronização
    "Extra Trees": ("extra_trees", {"n_estimators": 60, "max_depth": 12, "n_jobs": 1}),
    "Random Forest": ("random_forest", {"n_estimators": 60, "max_depth": 12, "n_jobs": 1}),
    "XGBoost": ("xgboost", {"n_estimators": 80, "max_depth": 6, "n_jobs": 1}),
    "LightGBM": ("lightgbm", {"n_estimators": 80, "max_depth": 6, "verbose": -1, "n_jobs": 1}),
    "CatBoost": ("catboost", {"iterations": 100, "depth": 6, "n_jobs": 1}),
    "HistGradientBoosting": ("gradient_boosting", {"max_iter": 80, "max_depth": 6}),
    "Ridge": ("ridge", {"alpha": 10}),
}
TRACE_COLS = ["unit_id", "cycle", "regime_id", "actual", "prediction", "z_score", "ewma_abs_z", "ewma_ucl"]


class SafeRegressor(BaseEstimator, RegressorMixin):
    """Envolve um regressor e devolve a constante quando o alvo é constante (alguns sensores não variam dentro de um regime;
    CatBoost e outros recusam ajustar). Fora disso, é o próprio regressor."""

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.const_ = float(y[0]) if y.size and np.ptp(y) == 0 else None
        self.model_ = None if self.const_ is not None else clone(self.base).fit(X, y)
        return self

    def predict(self, X):
        n = len(X)
        return np.full(n, self.const_) if self.const_ is not None else np.asarray(self.model_.predict(X), dtype=float)


def run_subset(subset: str, out_dir: Path, models: list[str], seed: int = 42, H: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
               min_consecutive: int = 2, engine_offset: bool = True) -> None:
    t0 = time.time()
    loader = CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    sensors = loader.informative_sensors(frame.loc[frame["unit_id"].isin(train_units)])
    parts = partition_healthy_and_lifecycle(frame, train_units, test_units, H)
    n_regimes = 6 if subset in {"FD002", "FD004"} else 1
    regimes = RegimeDecoupler(n_clusters=n_regimes, random_state=seed).fit(parts["full_lifecycle_train"])
    healthy_train = regimes.transform(parts["healthy_train"])
    test_full = regimes.transform(parts["full_lifecycle_test"])
    print(f"[{subset}] {len(test_units)} held-out engines | {len(sensors)} sensors | {len(models)} models", flush=True)

    traces, alarms, scores = [], [], []
    for sensor in sensors:
        for name in models:
            key, kw = MODELS[name]
            vs = VirtualSensor(target_sensor=sensor, mode="exogenous", ewma_alpha=0.10, healthy_cycle_limit=H,
                               min_consecutive_breaches=min_consecutive, random_state=seed,
                               estimator=SafeRegressor(get_model(key, seed, **kw)),
                               engine_offset=engine_offset, offset_window=H // 2 if engine_offset else None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scored, per_engine, summary = vs.evaluate(healthy_train, test_full)
            t = scored.rename(columns={sensor: "actual"})[TRACE_COLS].copy()
            t.insert(0, "sensor", sensor)
            t.insert(1, "model", name)
            traces.append(t)
            per_engine.insert(0, "sensor", sensor)
            per_engine.insert(1, "model", name)
            alarms.append(per_engine)
            resid = (scored["residual"].to_numpy())
            healthy = scored["cycle"].to_numpy() <= H
            scores.append({"sensor": sensor, "model": name, "rmse_holdout": float(np.sqrt(np.mean(resid ** 2))),
                           "rmse_healthy_window": float(np.sqrt(np.mean(resid[healthy] ** 2))) if healthy.any() else np.nan,
                           **{k: v for k, v in summary.items()}})
        print(f"  [{subset}] {sensor} ok ({time.time() - t0:.0f}s)", flush=True)

    target = out_dir / subset
    target.mkdir(parents=True, exist_ok=True)
    pd.concat(traces, ignore_index=True).round({"actual": 3, "prediction": 3, "z_score": 3, "ewma_abs_z": 4, "ewma_ucl": 4}).to_csv(
        target / "trace.csv.gz", index=False)
    pd.concat(alarms, ignore_index=True).to_csv(target / "per_engine_alarms.csv", index=False)
    pd.DataFrame(scores).to_csv(target / "model_scores.csv", index=False)
    add_r2_columns(target, H)
    print(f"[{subset}] done in {time.time() - t0:.0f}s -> {target}", flush=True)


R2_COLUMNS = ["r2_holdout", "r2_healthy", "r2_late_model", "r2_late_motor_base", "r2_late_model_offset", "pct_between_engine"]


def add_r2_columns(target: Path, H: int = DEFAULT_HEALTHY_CYCLE_LIMIT) -> pd.DataFrame:
    """Calcula as métricas de R² (ver detection/virtual_sensor_metrics.py) a partir do trace gravado e grava em model_scores.csv.

    Não reajusta nenhum modelo: serve também para atualizar resultados antigos (`--recompute-r2`)."""
    scores = pd.read_csv(target / "model_scores.csv").drop(columns=R2_COLUMNS, errors="ignore")
    r2 = virtual_sensor_r2(pd.read_csv(target / "trace.csv.gz"), H)
    out = scores.merge(r2, on=["sensor", "model"], how="left")
    out.to_csv(target / "model_scores.csv", index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=SUBSETS, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None, help=f"padrão: {OUT_DEFAULT} (ou {OUT_DEFAULT}_sem_offset com --no-engine-offset)")
    ap.add_argument("--engine-offset", action=argparse.BooleanOptionalAction, default=True, help="remove o deslocamento de cada motor (média do z nos ciclos 1 a H//2) do resíduo; padrão ligado; --no-engine-offset reproduz o comportamento anterior")
    ap.add_argument("--recompute-r2", action="store_true", help="só recalcula as colunas de R² de model_scores.csv a partir do trace existente (não reajusta modelos)")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = OUT_DEFAULT if a.engine_offset else OUT_DEFAULT.with_name(OUT_DEFAULT.name + "_sem_offset")
    for s in (SUBSETS if a.all or a.subset is None else (a.subset,)):
        if a.recompute_r2:
            add_r2_columns(a.out / s)
            print(f"[{s}] R² recalculado", flush=True)
        else:
            run_subset(s, a.out, a.models, a.seed, engine_offset=a.engine_offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

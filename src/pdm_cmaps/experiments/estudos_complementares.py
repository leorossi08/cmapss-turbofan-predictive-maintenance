"""Estudos complementares da revisão de banca (Exp. A, B, C e D).

Exp. A — Incerteza conformal e custo da decisão com o modelo de RUL ajustado (FD004):
    Compara LightGBM (referência global w=10) contra XGBoost e CatBoost (ajustados por CV agrupada:
    padronização por regime + janela 50), cruzando Mondrian em 3 tercis de RUL prevista, CQR contínuo
    (Conformalized Quantile Regression sobre quantis 0,05/0,95 fora da amostra) e histerese na transição
    entre níveis (banda de 5 ciclos). Quantifica largura, cobertura, saltos "dente de serra",
    oscilações de nível por motor e custo relativo ao oráculo.

Exp. B — Ablação de exogeneidade estrita (P15 e epr) nos sensores virtuais e na MEWMA (FD001–FD004):
    Compara a especificação canônica contra a estritamente externa (removendo P15 e epr antes do ajuste
    e da previsão) no R² dentro do regime (ciclos 26–50, H=50, 13 sensores) e no poder de detecção a
    falso alarme igualado (H=25, 3 sementes).

Exp. C — Baselines multivariados com memória temporal vs. MEWMA (FD001–FD004):
    Avalia PCA-T²/SPE (com suavização EWMA λ=0,10) e Isolation Forest (com suavização EWMA λ=0,10)
    contra MEWMA e Isolation Forest pontual sobre as MESMAS séries cegas e o mesmo orçamento de falso
    alarme (0,2 por 100 ciclos), com bootstrap em cluster por motor receptor (4000 réplicas) e Holm.

Exp. D — Assinaturas de fim de vida e detecção nativa separadas por modo de falha (FD003 e FD004):
    Separa os modos HPC e fan apenas nos motores de treino com KMeans (K=2) sobre os desvios
    padronizados por regime de fim de vida (13 sensores), relata as assinaturas físicas/padronizadas
    por modo e estratifica a detecção nativa (T50 EWMA e MEWMA sintética, H=50) por modo nos motores
    de teste do holdout 75/25.

Execução:
    .venv/bin/python -m pdm_cmaps.experiments.estudos_complementares --study all --jobs 4
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold

from pdm_cmaps.config import DEFAULT_RESULTS_DIR
from pdm_cmaps.data.features import CausalFeatureEngineer
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.splits import split_engine_units
from pdm_cmaps.detection.multivariate import MULTIVARIATE_SENSORS, MultivariateResidualMonitor
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.detection.virtual_sensor_metrics import virtual_sensor_r2
from pdm_cmaps.domain.rul import compute_piecewise_rul
from pdm_cmaps.domain.schema import (
    DEFAULT_HEALTHY_CYCLE_LIMIT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RUL_CAP,
    SETTING_COLUMNS,
)
from pdm_cmaps.protocol import TEST_RATIO
from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.experiments.fault_detection_study import (
    ADOPTED_H,
    BUDGETS,
    DET_MV,
    S_GRID,
    alarm_matrix,
    analyse_detector,
    bootstrap_series,
    build_series,
    iforest_ratio,
)
from pdm_cmaps.experiments.fault_injection import (
    INJECT_SENSORS,
    World,
    blind_columns,
    delta_z,
    inject,
    load_world,
    standardize,
)
from pdm_cmaps.experiments.poder_bootstrap import N_BOOT, SEED_BOOT, holm
from pdm_cmaps.experiments.selecao_hiperparametros import montar_dados, montar_modelo
from pdm_cmaps.modeling.registry import get_model_pool, grouped_oof
from pdm_cmaps.monitoring.alerts import (
    CRITICAL,
    CriticalityPolicy,
    build_alert_timeline,
    cost_table,
    debounce,
    evaluate_strategies,
    strategy_names,
)
from pdm_cmaps.monitoring.layer import MEWMA_CALIBRATION_MONITORING, _policy_grid, compute_fleet_alarms
from pdm_cmaps.monitoring.uncertainty import ConformalRULInterval, interval_metrics
from pdm_cmaps.pipeline.extra_models import fit_predict

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "estudos_complementares"
TABLES_DEFAULT = DEFAULT_RESULTS_DIR.parent / "tcc-writing" / "tables"
PARAMS_DEFAULT = DEFAULT_RESULTS_DIR / "hiperparametros"
SUBSETS_ALL = ("FD001", "FD002", "FD003", "FD004")

DET_IFOREST_POINT = "Isolation Forest (pontual)"
DET_IFOREST_EWMA = "Isolation Forest (EWMA)"
DET_PCA_EWMA = "PCA-T²/SPE (EWMA)"
MEM_DETECTORS = (DET_MV, DET_PCA_EWMA, DET_IFOREST_EWMA, DET_IFOREST_POINT)

CONTRASTS_MEM = {
    "MEWMA com regime - PCA-T²/SPE (EWMA)": (DET_MV, DET_PCA_EWMA),
    "MEWMA com regime - Isolation Forest (EWMA)": (DET_MV, DET_IFOREST_EWMA),
    "Isolation Forest (EWMA) - Isolation Forest (pontual)": (DET_IFOREST_EWMA, DET_IFOREST_POINT),
}
FAMILIES_MEM = {
    label: (("degrau", 0.10), ("degrau", 0.05), ("rampa", 0.10))
    for label in CONTRASTS_MEM
}

ENDOGENOUS_EXOGENOUS_LEAK_CANDIDATES = ("P15", "epr")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ============================================================================
# Shared mathematical helpers (CQR, hysteresis, EWMA 1D/ND, PCA-T2/SPE EWMA)
# ============================================================================


def fit_cqr_from_quantiles(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Finite-sample conformal correction Q_{1-alpha} for CQR (Romano et al., 2019)."""
    y = np.asarray(y_cal, dtype=float)
    lo = np.minimum(np.asarray(q_lo_cal, dtype=float), np.asarray(q_hi_cal, dtype=float))
    hi = np.maximum(np.asarray(q_lo_cal, dtype=float), np.asarray(q_hi_cal, dtype=float))
    if y.shape != lo.shape or y.shape != hi.shape or y.size < 20:
        raise ValueError("need matching calibration arrays with >= 20 points for CQR")
    scores = np.maximum(lo - y, y - hi)
    n = y.size
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    return float(np.quantile(scores, level, method="higher"))


def apply_cqr_interval(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    correction: float,
    rul_cap: float = DEFAULT_RUL_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Build non-crossing CQR prediction interval clipped to [0, rul_cap]."""
    lo_raw = np.minimum(np.asarray(q_lo, dtype=float), np.asarray(q_hi, dtype=float))
    hi_raw = np.maximum(np.asarray(q_lo, dtype=float), np.asarray(q_hi, dtype=float))
    lo = np.clip(lo_raw - correction, 0.0, rul_cap)
    hi = np.clip(hi_raw + correction, 0.0, rul_cap)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def levels_with_hysteresis(
    rul: np.ndarray,
    policy: CriticalityPolicy,
    band: float = 5.0,
) -> np.ndarray:
    """Raw criticality levels with Schmitt-trigger de-escalation hysteresis (band cycles)."""
    r = np.asarray(rul, dtype=float)
    raw = policy.level_from_rul(r)
    thresh = (np.inf, float(policy.attention), float(policy.warning), float(policy.critical))
    out = np.empty(len(raw), dtype=int)
    if len(raw) == 0:
        return out
    cur = int(raw[0])
    for t in range(len(raw)):
        cand = int(raw[t])
        if cand > cur:
            cur = cand
        else:
            while cur > 0 and r[t] > thresh[cur] + band:
                cur -= 1
        out[t] = cur
    return out


def apply_hysteresis_to_timeline(
    frame: pd.DataFrame,
    policy: CriticalityPolicy,
    first_alarm_cycle: dict[int, int | None],
    extra_alarms: dict[str, dict[int, int | None]],
    band: float = 5.0,
) -> pd.DataFrame:
    """Build criticality timeline where interval-based levels use hysteresis on de-escalation."""
    parts = []
    for unit_id, eng in frame.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=False):
        eng = eng.copy()
        eng["anomaly_active"] = (
            eng["cycle"] >= first_alarm_cycle[int(unit_id)]
            if first_alarm_cycle.get(int(unit_id)) is not None
            else pd.Series(False, index=eng.index)
        )
        eng["level_true"] = policy.level_from_rul(eng["RUL_true"])
        raw_point = policy.level_from_rul(eng["RUL_pred"])
        raw_int = levels_with_hysteresis(eng["RUL_lower"].to_numpy(), policy, band=band)
        raw = {
            "point": raw_point,
            "interval": raw_int,
            "interval+anomaly": np.minimum(raw_int + eng["anomaly_active"].to_numpy(dtype=int) * (raw_int > 0), CRITICAL),
        }
        for name, alarm_map in extra_alarms.items():
            ac = alarm_map.get(int(unit_id))
            active = eng["cycle"] >= ac if ac is not None else pd.Series(False, index=eng.index)
            eng[f"{name}_active"] = active
            raw[f"interval+{name}"] = np.minimum(raw_int + np.asarray(active, dtype=int) * (raw_int > 0), CRITICAL)
        for name, lvls in raw.items():
            eng[f"level_{name}"] = debounce(lvls, policy.persistence)
        parts.append(eng)
    return pd.concat(parts, ignore_index=True)


def flicker_metrics(timeline: pd.DataFrame, strategy: str) -> dict[str, float]:
    """Compute per-engine level transitions, reversals, and conformal sawtooth jumps."""
    col = f"level_{strategy}"
    transitions, reversals, jumps = [], [], []
    for _, eng in timeline.groupby("unit_id", sort=False):
        eng = eng.sort_values("cycle")
        lvl = eng[col].to_numpy(dtype=int)
        lo = eng["RUL_lower"].to_numpy(dtype=float)
        pred = eng["RUL_pred"].to_numpy(dtype=float)
        if len(lvl) > 1:
            dl = np.diff(lvl)
            transitions.append(int((dl != 0).sum()))
            reversals.append(int((dl < 0).sum()))
            jumps.append(int(((np.abs(np.diff(lo)) >= 15.0) & (np.abs(np.diff(pred)) <= 5.0)).sum()))
        else:
            transitions.append(0)
            reversals.append(0)
            jumps.append(0)
    return {
        "Transitions_per_engine": float(np.mean(transitions)),
        "Reversals_per_engine": float(np.mean(reversals)),
        "Lower_jumps_ge15_per_engine": float(np.mean(jumps)),
    }


def ewma_smooth_matrix(z: np.ndarray, units: np.ndarray, mu: np.ndarray, lam: float = 0.10) -> np.ndarray:
    """Causal per-engine EWMA recursion standardized by finite-sample factor sqrt(lam/(2-lam)*(1-(1-lam)^(2t)))."""
    z = np.asarray(z, dtype=float)
    units = np.asarray(units)
    out = np.empty_like(z, dtype=float)
    factor_cache: dict[int, float] = {}
    start = 0
    for _, n in pd.Series(units).groupby(units, sort=False).size().items():
        w = np.zeros(z.shape[1], dtype=float)
        for t in range(n):
            w = lam * (z[start + t] - mu) + (1.0 - lam) * w
            if (t + 1) not in factor_cache:
                factor_cache[t + 1] = float(np.sqrt(lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2 * (t + 1)))))
            out[start + t] = w / factor_cache[t + 1]
        start += n
    return out


class PCAEwmaMonitor:
    """PCA T² + SPE joint monitor evaluated on causally EWMA-smoothed residuals."""

    def __init__(self, lam: float = 0.10, variance_ratio: float = 0.90, alpha: float = 0.0027, random_state: int = 42):
        self.lam = float(lam)
        self.variance_ratio = float(variance_ratio)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def fit(self, z_cal: np.ndarray, units_cal: np.ndarray) -> "PCAEwmaMonitor":
        zc = np.asarray(z_cal, dtype=float)
        p = zc.shape[1]
        self.mu_ = zc.mean(axis=0)
        pca = PCA(n_components=None, random_state=self.random_state).fit(zc - self.mu_)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, self.variance_ratio) + 1)
        self.n_components_ = max(1, min(k, p - 1))
        self.components_ = pca.components_[: self.n_components_]  # (k, p)
        self.eigvals_ = np.maximum(pca.explained_variance_[: self.n_components_], 1e-8)

        w_cal = ewma_smooth_matrix(zc, units_cal, self.mu_, lam=self.lam)
        t2_cal, spe_cal = self._t2_and_spe(w_cal)
        self.tau_t2_ = max(float(np.quantile(t2_cal, 1.0 - self.alpha)), 1e-8)
        self.tau_spe_ = max(float(np.quantile(spe_cal, 1.0 - self.alpha)), 1e-8)
        comb = t2_cal / self.tau_t2_ + spe_cal / self.tau_spe_
        self.tau_comb_ = max(float(np.quantile(comb, 1.0 - self.alpha)), 1e-8)
        return self

    def _t2_and_spe(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj = w @ self.components_.T
        t2 = np.sum((proj ** 2) / self.eigvals_, axis=1)
        spe = np.clip(np.sum(w ** 2, axis=1) - np.sum(proj ** 2, axis=1), 0.0, None)
        return t2, spe

    def score_ratio(self, z: np.ndarray, units: np.ndarray) -> np.ndarray:
        w = ewma_smooth_matrix(z, units, self.mu_, lam=self.lam)
        t2, spe = self._t2_and_spe(w)
        return (t2 / self.tau_t2_ + spe / self.tau_spe_) / self.tau_comb_


class IForestEwmaMonitor:
    """Isolation Forest point anomaly scores smoothed per engine with causal EWMA."""

    def __init__(self, forest: IsolationForest, lam: float = 0.10, alpha: float = 0.0027):
        self.forest = forest
        self.lam = float(lam)
        self.alpha = float(alpha)

    def fit(self, z_cal: np.ndarray, units_cal: np.ndarray) -> "IForestEwmaMonitor":
        raw_cal = (-self.forest.score_samples(z_cal)).reshape(-1, 1)
        self.mu_ = raw_cal.mean(axis=0)
        smoothed = ewma_smooth_matrix(raw_cal, units_cal, self.mu_, lam=self.lam).ravel()
        self.ucl_ = max(float(np.quantile(smoothed, 1.0 - self.alpha)), 1e-8)
        return self

    def score_ratio(self, z: np.ndarray, units: np.ndarray) -> np.ndarray:
        raw = (-self.forest.score_samples(z)).reshape(-1, 1)
        smoothed = ewma_smooth_matrix(raw, units, self.mu_, lam=self.lam).ravel()
        return np.clip(smoothed / self.ucl_, 0.0, None)


# ============================================================================
# Exp. A: Tuned RUL + CQR + Hysteresis (FD004)
# ============================================================================


def _make_quantile_estimator(model_name: str, q: float, params: dict, seed: int, n_jobs: int):
    h = dict(params.get("hyper", {}))
    if model_name == "LightGBM (ref.)":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=80, max_depth=6,
            learning_rate=0.08, random_state=seed, n_jobs=n_jobs, verbose=-1,
        )
    if model_name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q, tree_method="hist",
            n_estimators=h.get("n_estimators", 300), max_depth=h.get("max_depth", 6),
            learning_rate=h.get("learning_rate", 0.05), subsample=h.get("subsample", 0.8),
            colsample_bytree=h.get("colsample_bytree", 0.8), min_child_weight=h.get("min_child_weight", 3),
            reg_lambda=h.get("reg_lambda", 1.0), random_state=seed, n_jobs=n_jobs,
        )
    if model_name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}",
            iterations=h.get("iterations", 400), depth=h.get("depth", 5),
            learning_rate=h.get("learning_rate", 0.08), l2_leaf_reg=h.get("l2_leaf_reg", 5.0),
            random_seed=seed, thread_count=n_jobs, verbose=0,
        )
    raise ValueError(f"Unsupported quantile estimator for {model_name!r}")


def _oof_and_test_predictions(
    model_label: str,
    model_key: str | None,
    params: dict | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensors: list[str],
    seed: int,
    n_jobs: int,
) -> dict:
    """Zero-leakage 5-fold GroupKFold OOF + full-train test predictions for point + 0.05/0.95 quantiles."""
    if model_label == "LightGBM (ref.)":
        engineer = CausalFeatureEngineer(sensors, windows=(5, 10)).fit(train_df, DEFAULT_HEALTHY_CYCLE_LIMIT)
        tr_fe = engineer.transform(train_df)
        te_fe = engineer.transform(test_df)
        base_features = [*sensors, *SETTING_COLUMNS, "cycle"]
        features = [*base_features, *[c for c in tr_fe.columns if c not in train_df.columns]]
        est_point = clone(get_model_pool(seed)["LightGBM"])
        if hasattr(est_point, "set_params"):
            est_point.set_params(n_jobs=n_jobs)
        _, oof_point = grouped_oof(tr_fe, features, estimator=est_point, n_splits=5, random_state=seed)
        _, oof_q05 = grouped_oof(tr_fe, features, estimator=_make_quantile_estimator("LightGBM (ref.)", 0.05, {}, seed, n_jobs), n_splits=5, random_state=seed)
        _, oof_q95 = grouped_oof(tr_fe, features, estimator=_make_quantile_estimator("LightGBM (ref.)", 0.95, {}, seed, n_jobs), n_splits=5, random_state=seed)

        X_tr, y_tr = tr_fe[features].to_numpy(dtype=float), tr_fe["RUL"].to_numpy(dtype=float)
        X_te = te_fe[features].to_numpy(dtype=float)
        pred_point = clone(est_point).fit(X_tr, y_tr).predict(X_te)
        pred_q05 = _make_quantile_estimator("LightGBM (ref.)", 0.05, {}, seed, n_jobs).fit(X_tr, y_tr).predict(X_te)
        pred_q95 = _make_quantile_estimator("LightGBM (ref.)", 0.95, {}, seed, n_jobs).fit(X_tr, y_tr).predict(X_te)
        return {
            "tr_meta": tr_fe[["unit_id", "cycle", "RUL"]].reset_index(drop=True),
            "te_meta": te_fe[["unit_id", "cycle", "RUL"]].reset_index(drop=True),
            "oof_point": np.clip(oof_point, 0.0, DEFAULT_RUL_CAP),
            "oof_q05": np.clip(oof_q05, 0.0, DEFAULT_RUL_CAP),
            "oof_q95": np.clip(oof_q95, 0.0, DEFAULT_RUL_CAP),
            "pred_point": np.clip(pred_point, 0.0, DEFAULT_RUL_CAP),
            "pred_q05": np.clip(pred_q05, 0.0, DEFAULT_RUL_CAP),
            "pred_q95": np.clip(pred_q95, 0.0, DEFAULT_RUL_CAP),
        }

    assert params is not None and model_key is not None
    oof_parts = []
    splitter = GroupKFold(n_splits=5)
    for tr_idx, va_idx in splitter.split(train_df, groups=train_df["unit_id"]):
        tr_fold = train_df.iloc[tr_idx].reset_index(drop=True)
        va_fold = train_df.iloc[va_idx].reset_index(drop=True)
        tr_f, va_f, cols_f = montar_dados(tr_fold, va_fold, sensors, 6, params)
        kind, est = montar_modelo(model_key, params, seed, device="cpu", n_jobs=n_jobs)
        pred_f = fit_predict(kind, est, tr_f, cols_f, va_f)

        X_f, y_f = tr_f[cols_f].to_numpy(dtype=float), tr_f["RUL"].to_numpy(dtype=float)
        Xv_f = va_f[cols_f].to_numpy(dtype=float)
        q05_f = _make_quantile_estimator(model_key, 0.05, params, seed, n_jobs).fit(X_f, y_f).predict(Xv_f)
        q95_f = _make_quantile_estimator(model_key, 0.95, params, seed, n_jobs).fit(X_f, y_f).predict(Xv_f)
        part = va_f[["unit_id", "cycle", "RUL"]].copy()
        part["oof_point"] = np.clip(pred_f, 0.0, DEFAULT_RUL_CAP)
        part["oof_q05"] = np.clip(q05_f, 0.0, DEFAULT_RUL_CAP)
        part["oof_q95"] = np.clip(q95_f, 0.0, DEFAULT_RUL_CAP)
        oof_parts.append(part)

    oof_df = pd.concat(oof_parts, ignore_index=True).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    tr_full, te_full, cols = montar_dados(train_df, test_df, sensors, 6, params)
    assert (oof_df["unit_id"].to_numpy() == tr_full["unit_id"].to_numpy()).all()
    assert (oof_df["cycle"].to_numpy() == tr_full["cycle"].to_numpy()).all()

    kind, est_full = montar_modelo(model_key, params, seed, device="cpu", n_jobs=n_jobs)
    pred_point = fit_predict(kind, est_full, tr_full, cols, te_full)
    X_tr, y_tr = tr_full[cols].to_numpy(dtype=float), tr_full["RUL"].to_numpy(dtype=float)
    X_te = te_full[cols].to_numpy(dtype=float)
    pred_q05 = _make_quantile_estimator(model_key, 0.05, params, seed, n_jobs).fit(X_tr, y_tr).predict(X_te)
    pred_q95 = _make_quantile_estimator(model_key, 0.95, params, seed, n_jobs).fit(X_tr, y_tr).predict(X_te)
    return {
        "tr_meta": oof_df[["unit_id", "cycle", "RUL"]].reset_index(drop=True),
        "te_meta": te_full[["unit_id", "cycle", "RUL"]].reset_index(drop=True),
        "oof_point": oof_df["oof_point"].to_numpy(dtype=float),
        "oof_q05": oof_df["oof_q05"].to_numpy(dtype=float),
        "oof_q95": oof_df["oof_q95"].to_numpy(dtype=float),
        "pred_point": np.clip(pred_point, 0.0, DEFAULT_RUL_CAP),
        "pred_q05": np.clip(pred_q05, 0.0, DEFAULT_RUL_CAP),
        "pred_q95": np.clip(pred_q95, 0.0, DEFAULT_RUL_CAP),
    }


def run_exp_a(seeds: list[int], out: Path, tables: Path, params_dir: Path, n_jobs: int) -> None:
    print("\n=== Exp. A: Modelo ajustado + CQR + histerese no monitoramento (FD004) ===", flush=True)
    loader = CMapssLoader()
    frame = loader.load("FD004")
    frame["RUL"] = compute_piecewise_rul(frame)
    params_xgb = json.loads((params_dir / "FD004" / "XGBoost" / "melhores.json").read_text())["params"]
    params_cb = json.loads((params_dir / "FD004" / "CatBoost" / "melhores.json").read_text())["params"]
    specs = (
        ("LightGBM (ref.)", None, None),
        ("XGBoost (ajustado)", "XGBoost", params_xgb),
        ("CatBoost (ajustado)", "CatBoost", params_cb),
    )
    base_policy = CriticalityPolicy()
    policies = _policy_grid(base_policy)

    summary_rows, cost_rows, traj_parts = [], [], []
    for seed in seeds:
        t0 = time.perf_counter()
        train_units, test_units = split_engine_units(frame, test_ratio=TEST_RATIO, random_state=seed)
        train_df = frame.loc[frame["unit_id"].isin(train_units)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        test_df = frame.loc[frame["unit_id"].isin(test_units)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        sensors = loader.informative_sensors(train_df)

        regimes = RegimeDecoupler(n_clusters=6, random_state=seed).fit(train_df)
        alarms, _, _ = compute_fleet_alarms(train_df, test_df, "FD004", seed, DEFAULT_HEALTHY_CYCLE_LIMIT)
        mv = MultivariateResidualMonitor(
            healthy_cycle_limit=DEFAULT_HEALTHY_CYCLE_LIMIT, random_state=seed, calibration=MEWMA_CALIBRATION_MONITORING,
        ).fit(regimes.transform(train_df))
        mv_alarms, _, _ = mv.alarms(mv.score(regimes.transform(test_df)))
        extra_alarms = {"anomaly_mv": mv_alarms}

        for model_label, model_key, params in specs:
            bundle = _oof_and_test_predictions(model_label, model_key, params, train_df, test_df, sensors, seed, n_jobs)
            y_cal = bundle["tr_meta"]["RUL"].to_numpy(dtype=float)
            y_true = bundle["te_meta"]["RUL"].to_numpy(dtype=float)
            pred = bundle["pred_point"]

            mon = ConformalRULInterval(alpha=0.10, n_bins=3).fit(y_cal, bundle["oof_point"])
            lo_mon, hi_mon = mon.predict_interval(pred)

            q_corr = fit_cqr_from_quantiles(y_cal, bundle["oof_q05"], bundle["oof_q95"], alpha=0.10)
            lo_cqr, hi_cqr = apply_cqr_interval(bundle["pred_q05"], bundle["pred_q95"], q_corr)

            variants = (
                ("Mondrian 3 tercis", lo_mon, hi_mon, False),
                ("Mondrian + histerese", lo_mon, hi_mon, True),
                ("CQR contínuo", lo_cqr, hi_cqr, False),
                ("CQR + histerese", lo_cqr, hi_cqr, True),
            )
            rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - pred)))

            for method_name, lo, hi, use_hyst in variants:
                tl_in = pd.DataFrame({
                    "unit_id": bundle["te_meta"]["unit_id"].to_numpy(),
                    "cycle": bundle["te_meta"]["cycle"].to_numpy(),
                    "RUL_true": y_true, "RUL_pred": pred, "RUL_lower": lo, "RUL_upper": hi,
                })
                im = interval_metrics(y_true, lo, hi, critical_threshold=50.0)
                inside = (y_true >= lo) & (y_true <= hi)
                p10 = float(pd.Series(inside).groupby(tl_in["unit_id"]).mean().quantile(0.10))

                for pol_name, pol in policies.items():
                    tl = (
                        apply_hysteresis_to_timeline(tl_in, pol, alarms, extra_alarms, band=5.0)
                        if use_hyst
                        else build_alert_timeline(tl_in, pol, alarms, extra_alarms)
                    )
                    ct = cost_table(tl, trigger=CRITICAL).assign(
                        Subset="FD004", Seed=seed, Model=model_label, Conformal_method=method_name, Policy=pol_name,
                    )
                    cost_rows.append(ct)
                    if pol_name == "critical<=15":
                        ev_strat = evaluate_strategies(tl, pol).set_index("Strategy")
                        for strat in strategy_names(tl):
                            fm = flicker_metrics(tl, strat)
                            sr = ev_strat.loc[strat]
                            summary_rows.append({
                                "Subset": "FD004", "Seed": seed, "Model": model_label,
                                "Conformal_method": method_name, "Strategy": strat,
                                **im, "Per_engine_coverage_p10": p10, "RMSE": rmse, "MAE": mae,
                                "Alert_F1": float(sr["Alert_F1"]),
                                "Missed_critical_rate": float(sr["Missed_critical_rate"]),
                                "False_alarm_rate_normal": float(sr["False_alarm_rate_normal"]),
                                "Median_warning_lead_time": float(sr["Median_warning_lead_time"]),
                                **fm,
                            })
                        if seed == 42 and method_name in {"Mondrian 3 tercis", "CQR contínuo", "CQR + histerese"}:
                            traj_parts.append(
                                tl[["unit_id", "cycle", "RUL_true", "RUL_pred", "RUL_lower", "RUL_upper",
                                    "level_point", "level_interval", "level_interval+anomaly", "level_interval+anomaly_mv"]]
                                .assign(Model=model_label, Conformal_method=method_name)
                            )
        print(f"[Exp. A] seed={seed} concluída em {time.perf_counter() - t0:.1f} s", flush=True)

    raw_res = pd.DataFrame(summary_rows)
    raw_cost = pd.concat(cost_rows, ignore_index=True)
    exp_dir = out / "exp_a"
    exp_dir.mkdir(parents=True, exist_ok=True)
    raw_res.to_csv(exp_dir / "resumo_por_semente.csv", index=False)
    raw_cost.to_csv(exp_dir / "custos_por_semente.csv", index=False)
    if traj_parts:
        pd.concat(traj_parts, ignore_index=True).to_csv(out / "exp_a_trajetorias_seed42.csv", index=False)

    num_cols = [
        "coverage", "coverage_critical_zone", "mean_width", "mean_width_critical_zone",
        "Per_engine_coverage_p10", "RMSE", "MAE", "Alert_F1", "Missed_critical_rate",
        "False_alarm_rate_normal", "Median_warning_lead_time",
        "Transitions_per_engine", "Reversals_per_engine", "Lower_jumps_ge15_per_engine",
    ]
    g_res = raw_res.groupby(["Subset", "Model", "Conformal_method", "Strategy"])[num_cols]
    summ_res = g_res.mean().add_suffix("_mean").join(g_res.std().add_suffix("_std")).reset_index()
    summ_res["Runs"] = g_res.size().to_numpy()

    g_cost = raw_cost.groupby(["Subset", "Model", "Conformal_method", "Policy", "Strategy", "Cost_ratio", "Lead"])[
        ["Relative_to_oracle", "Failures", "Mean_unused_cycles"]
    ]
    summ_cost = g_cost.mean().add_suffix("_mean").join(g_cost.std().add_suffix("_std")).reset_index()
    summ_cost["Runs"] = g_cost.size().to_numpy()

    tables.mkdir(parents=True, exist_ok=True)
    summ_res.to_csv(tables / "conformal_modelo_ajustado_resumo.csv", index=False)
    summ_cost.to_csv(tables / "conformal_modelo_ajustado_custos.csv", index=False)

    ref_row = summ_res[
        (summ_res["Model"] == "LightGBM (ref.)")
        & (summ_res["Conformal_method"] == "Mondrian 3 tercis")
        & (summ_res["Strategy"] == "interval")
    ].iloc[0]
    ref_csv = tables / "monitoring_repeated_uncertainty_summary.csv"
    if ref_csv.exists():
        old_fd4 = pd.read_csv(ref_csv).query("Subset == 'FD004'").iloc[0]
        diff = abs(ref_row["RMSE_mean"] - old_fd4["RMSE_mean"])
        print(f"[Exp. A] conferência LightGBM ref FD004 RMSE={ref_row['RMSE_mean']:.4f} vs versionado={old_fd4['RMSE_mean']:.4f} (Δ={diff:.5f})", flush=True)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summ_res[summ_res["Strategy"] == "interval"][[
            "Model", "Conformal_method", "RMSE_mean", "coverage_critical_zone_mean",
            "mean_width_critical_zone_mean", "Transitions_per_engine_mean", "Reversals_per_engine_mean",
            "Lower_jumps_ge15_per_engine_mean",
        ]].round(3).to_string(index=False), flush=True)


# ============================================================================
# Exp. B: Strict exogeneity ablation (removing P15 and epr)
# ============================================================================


def _strip_endogenous_exog(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in ENDOGENOUS_EXOGENOUS_LEAK_CANDIDATES if c in df.columns]
    return df.drop(columns=drop)


def _exp_b_r2_one(subset: str) -> list[dict]:
    loader = CMapssLoader()
    frame = loader.load(subset)
    n_regimes = 6 if subset in ("FD002", "FD004") else 1
    train_units, test_units = split_engine_units(frame, test_ratio=TEST_RATIO, random_state=42)
    tr_base = frame.loc[frame["unit_id"].isin(train_units)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    te_base = frame.loc[frame["unit_id"].isin(test_units)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    reg = RegimeDecoupler(n_clusters=n_regimes, random_state=42).fit(tr_base)
    tr_r, te_r = reg.transform(tr_base), reg.transform(te_base)

    rows = []
    for var, tr_in, te_in in (
        ("canonico", tr_r, te_r),
        ("estritamente_externo", _strip_endogenous_exog(tr_r), _strip_endogenous_exog(te_r)),
    ):
        mv = MultivariateResidualMonitor(
            healthy_cycle_limit=DEFAULT_HEALTHY_CYCLE_LIMIT, random_state=42, engine_offset=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.fit(tr_in)
        assert len(mv.sensors_) == 13, f"{subset} {var}: n_sensores={len(mv.sensors_)}"
        if var == "estritamente_externo":
            assert not any(c in mv.virtual_["T50"].feature_cols for c in ENDOGENOUS_EXOGENOUS_LEAK_CANDIDATES)
        parts = []
        for sensor, vs in mv.virtual_.items():
            p = vs.predict(te_in)
            parts.append(pd.DataFrame({
                "sensor": sensor, "model": "Extra Trees",
                "unit_id": p["unit_id"].to_numpy(), "cycle": p["cycle"].to_numpy(),
                "actual": p[sensor].to_numpy(dtype=float), "prediction": p["prediction"].to_numpy(dtype=float),
                "regime_id": p["regime_id"].to_numpy(),
            }))
        trace = pd.concat(parts, ignore_index=True)
        r2_tab = virtual_sensor_r2(trace, healthy_limit=DEFAULT_HEALTHY_CYCLE_LIMIT).set_index("sensor")
        rows.append({
            "Subset": subset, "variante": var, "n_sensores": len(mv.sensors_),
            "r2_bruto_mediana": float(r2_tab["r2_late_model"].median()),
            "r2_dentro_regime_mediana": float(r2_tab["r2_late_within_regime"].median()),
            "teto_ruido_mediana": float(r2_tab["noise_ceiling_late"].median()),
            "fracao_teto_mediana": float(r2_tab["frac_ceiling"].median()),
            "r2_T50_dentro_regime": float(r2_tab.loc["T50", "r2_late_within_regime"]),
            "r2_P30_dentro_regime": float(r2_tab.loc["P30", "r2_late_within_regime"]),
            "r2_BPR_dentro_regime": float(r2_tab.loc["BPR", "r2_late_within_regime"]),
        })
    return rows


def _exp_b_power_one(subset: str, seed: int) -> list[pd.DataFrame]:
    loader = CMapssLoader()
    world = load_world(subset, seed, ADOPTED_H, loader=loader)
    blind, meta = build_series(world, seed, 6, 2, None, False)
    tr = world.regimes.transform(world.train)
    te = world.regimes.transform(blind)
    out = []
    for var, tr_in, te_in in (
        ("canonico", tr, te),
        ("estritamente_externo", _strip_endogenous_exog(tr), _strip_endogenous_exog(te)),
    ):
        mv = MultivariateResidualMonitor(healthy_cycle_limit=world.H, random_state=seed, engine_offset=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.fit(tr_in)
            sc = mv.score(te_in)
        scores = pd.DataFrame({
            "blind_id": sc["unit_id"].to_numpy(), "cycle": sc["cycle"].to_numpy(),
            "ratio": (sc["t2"] / sc["ucl"]).to_numpy(), "gate": np.ones(len(sc), dtype=bool),
        })
        res = analyse_detector(f"MEWMA ({var})", scores, meta, ADOPTED_H, BUDGETS)["pareado"]
        out.append(res.assign(Subset=subset, Seed=seed, variante=var))
    return out


def run_exp_b(seeds: list[int], out: Path, tables: Path, n_jobs: int = 2) -> None:
    print("\n=== Exp. B: Ablação de exogeneidade estrita (sem P15 e epr) ===", flush=True)
    from joblib import Parallel, delayed

    r2_nested = Parallel(n_jobs=min(n_jobs, len(SUBSETS_ALL)))(delayed(_exp_b_r2_one)(s) for s in SUBSETS_ALL)
    r2_rows = [row for part in r2_nested for row in part]

    jobs = [(s, seed) for s in SUBSETS_ALL for seed in seeds]
    pod_nested = Parallel(n_jobs=n_jobs)(delayed(_exp_b_power_one)(s, seed) for s, seed in jobs)
    pod_runs = [df for part in pod_nested for df in part]

    r2_df = pd.DataFrame(r2_rows)
    pod_raw = pd.concat(pod_runs, ignore_index=True)
    g = pod_raw.groupby(["Subset", "variante", "kind", "magnitude"])
    pod_summ = g["power_pct"].agg(["mean", "std"]).rename(columns={"mean": "power_pct_mean", "std": "power_pct_std"})
    pod_summ["delay_med_mean"] = g["delay_med"].mean()
    pod_summ["far_ctrl_per100_mean"] = g["far_ctrl_per100"].mean()
    pod_summ["n_seeds"] = g.size()
    pod_summ = pod_summ.reset_index()

    exp_dir = out / "exp_b"
    exp_dir.mkdir(parents=True, exist_ok=True)
    pod_raw.to_csv(exp_dir / "poder_por_semente.csv", index=False)
    r2_df.to_csv(tables / "ablativo_exogeneidade_r2.csv", index=False)
    pod_summ.to_csv(tables / "ablativo_exogeneidade_poder.csv", index=False)
    print(r2_df.round(4).to_string(index=False), flush=True)
    deg = pod_summ[(pod_summ["kind"] == "degrau") & (pod_summ["magnitude"] == 0.10)]
    print(deg.round(3).to_string(index=False), flush=True)


# ============================================================================
# Exp. C: Memory multivariate baselines + cluster bootstrap + Holm
# ============================================================================


def _extract_residuals(mv: MultivariateResidualMonitor, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (unit_id, cycle, z_matrix[n, 13]) sorted by (unit_id, cycle)."""
    cols, index = {}, None
    for sensor, v in mv.virtual_.items():
        pred = v.predict(frame).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        cols[sensor] = mv.residual_z(v, pred)
        index = pred[["unit_id", "cycle"]]
    assert index is not None
    z = np.column_stack([cols[s] for s in mv.virtual_])
    return index["unit_id"].to_numpy(), index["cycle"].to_numpy(), z


def _bootstrap_memory_runs(npz_paths: list[Path], n_boot: int, boot_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster bootstrap over recipient engines using the same procedure as poder_bootstrap.py."""
    runs_data = []
    for path in npz_paths:
        npz = np.load(path)
        meta = pd.read_csv(path.with_name(path.stem + "_manifesto.csv"))
        subset, seed, H = str(meta["Subset"].iloc[0]), int(meta["Seed"].iloc[0]), int(meta["H"].iloc[0])
        midx = meta.set_index("blind_id")
        det = {}
        for name in MEM_DETECTORS:
            sl = _slug(name)
            ids, mat, last = npz[f"{sl}__ids"], npz[f"{sl}__mat"], npz[f"{sl}__last"]
            m = midx.loc[ids]
            ctrl = (m["role"] == "controle").to_numpy()
            has = ~np.isnan(mat)
            expo = np.where(has, mat - H, (last - H)[:, None])
            det[name] = {
                "ctrl_has": has[ctrl], "ctrl_expo": expo[ctrl], "fault_has": has[~ctrl],
                "ctrl_rec": m["recipient"].to_numpy()[ctrl], "fault_rec": m["recipient"].to_numpy()[~ctrl],
                "groups": m.loc[~ctrl, ["kind", "magnitude", "t0"]].reset_index(drop=True),
            }
        ref = det[DET_MV]
        recipients = np.unique(np.concatenate([ref["ctrl_rec"], ref["fault_rec"]]))
        code = {r: i for i, r in enumerate(recipients)}
        keys = ref["groups"].astype(str).agg("|".join, axis=1)
        group_keys = pd.unique(keys)
        gcode = {k: i for i, k in enumerate(group_keys)}
        group_table = ref["groups"].assign(_k=keys).drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
        for d in det.values():
            d["ctrl_code"] = np.array([code[r] for r in d["ctrl_rec"]])
            d["fault_code"] = np.array([code[r] for r in d["fault_rec"]])
            d["gcode"] = np.array([gcode[k] for k in d["groups"].astype(str).agg("|".join, axis=1)])
        runs_data.append({
            "subset": subset, "seed": seed, "H": H, "det": det,
            "recipients": recipients, "group_keys": group_keys, "group_table": group_table,
        })

    def _run_power(r: dict, name: str, counts: np.ndarray, budget: float = BUDGETS[0]) -> np.ndarray:
        d = r["det"][name]
        wc = counts[d["ctrl_code"]].astype(float)
        events, exposure = wc @ d["ctrl_has"], wc @ d["ctrl_expo"]
        with np.errstate(divide="ignore", invalid="ignore"):
            far = 100.0 * events / exposure
        ok = np.flatnonzero(far <= budget)
        n_groups = len(r["group_keys"])
        if not ok.size:
            return np.full(n_groups, np.nan)
        wf = counts[d["fault_code"]].astype(float)
        detected = d["fault_has"][:, ok[0]].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 100.0 * np.bincount(d["gcode"], wf * detected, n_groups) / np.bincount(d["gcode"], wf, n_groups)

    rng = np.random.default_rng(boot_seed)
    by_subset: dict[str, list[dict]] = {}
    for r in runs_data:
        by_subset.setdefault(r["subset"], []).append(r)

    power_rows, contrast_rows = [], []
    didx = {n: i for i, n in enumerate(MEM_DETECTORS)}
    for subset, rs in sorted(by_subset.items()):
        ref_keys = list(rs[0]["group_keys"])
        assert all(list(r["group_keys"]) == ref_keys for r in rs)
        boot = np.full((n_boot, len(MEM_DETECTORS), len(ref_keys)), np.nan)
        for b in range(n_boot):
            per_seed = []
            for r in rs:
                counts = np.bincount(rng.integers(0, len(r["recipients"]), len(r["recipients"])), minlength=len(r["recipients"]))
                per_seed.append(np.vstack([_run_power(r, n, counts) for n in MEM_DETECTORS]))
            with np.errstate(all="ignore"):
                boot[b] = np.nanmean(np.stack(per_seed), axis=0)
        table = rs[0]["group_table"]
        pts = {n: np.nanmean(np.vstack([_run_power(r, n, np.ones(len(r["recipients"]))) for r in rs]), axis=0) for n in MEM_DETECTORS}
        for g, cell in table.iterrows():
            for n in MEM_DETECTORS:
                lo, hi = np.nanpercentile(boot[:, didx[n], g], [2.5, 97.5])
                power_rows.append({
                    "Subset": subset, "Detector": n, "kind": cell["kind"],
                    "magnitude": cell["magnitude"], "t0": cell["t0"],
                    "poder_pct": float(pts[n][g]), "ic_lo": float(lo), "ic_hi": float(hi), "n_seeds": len(rs),
                })
            for label, (a, c) in CONTRASTS_MEM.items():
                d = boot[:, didx[a], g] - boot[:, didx[c], g]
                d = d[np.isfinite(d)]
                if not d.size:
                    continue
                p = float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean())))
                p = max(p, 1.0 / (len(d) + 1))
                contrast_rows.append({
                    "Subset": subset, "Contraste": label, "kind": cell["kind"],
                    "magnitude": cell["magnitude"], "t0": cell["t0"],
                    "diferenca_pp": float(pts[a][g] - pts[c][g]),
                    "ic_lo": float(np.percentile(d, 2.5)), "ic_hi": float(np.percentile(d, 97.5)),
                    "p_boot": p,
                })

    power_df, contr_df = pd.DataFrame(power_rows), pd.DataFrame(contrast_rows)
    contr_df["familia"] = False
    contr_df["p_holm"] = np.nan
    for label, cells in FAMILIES_MEM.items():
        sel = (
            contr_df["Contraste"].eq(label)
            & contr_df[["kind", "magnitude"]].apply(tuple, axis=1).isin(list(cells))
            & (contr_df["t0"] == ADOPTED_H + 10)
        )
        contr_df.loc[sel, "familia"] = True
        if sel.any():
            contr_df.loc[sel, "p_holm"] = holm(contr_df.loc[sel, "p_boot"].to_numpy())
    return power_df, contr_df


def _exp_c_one(subset: str, seed: int, alarm_dir: Path) -> tuple[list[pd.DataFrame], Path]:
    t0 = time.perf_counter()
    world = load_world(subset, seed, ADOPTED_H, loader=CMapssLoader())
    blind, meta = build_series(world, seed, 6, 2, None, False)
    tr, te = world.regimes.transform(world.train), world.regimes.transform(blind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mv = MultivariateResidualMonitor(healthy_cycle_limit=world.H, random_state=seed, engine_offset=True).fit(tr)
        uid_te, cyc_te, z_te = _extract_residuals(mv, te)

    mv_ratio = mv._t2(z_te, uid_te) / mv.ucl_
    z_cal_df = mv.calibration_z_
    z_cal = z_cal_df.to_numpy()
    units_cal = z_cal_df.index.get_level_values("unit_id").to_numpy()

    pca_mon = PCAEwmaMonitor(lam=0.10, variance_ratio=0.90, alpha=0.0027, random_state=seed).fit(z_cal, units_cal)
    pca_ratio = pca_mon.score_ratio(z_te, uid_te)

    forest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(z_cal)
    if_pt_ratio = iforest_ratio(forest, z_cal, z_te)
    if_ewma_ratio = IForestEwmaMonitor(forest, lam=0.10, alpha=0.0027).fit(z_cal, units_cal).score_ratio(z_te, uid_te)

    ones = np.ones(len(uid_te), dtype=bool)
    det_scores = {
        DET_MV: pd.DataFrame({"blind_id": uid_te, "cycle": cyc_te, "ratio": mv_ratio, "gate": ones}),
        DET_PCA_EWMA: pd.DataFrame({"blind_id": uid_te, "cycle": cyc_te, "ratio": pca_ratio, "gate": ones}),
        DET_IFOREST_EWMA: pd.DataFrame({"blind_id": uid_te, "cycle": cyc_te, "ratio": if_ewma_ratio, "gate": ones}),
        DET_IFOREST_POINT: pd.DataFrame({"blind_id": uid_te, "cycle": cyc_te, "ratio": if_pt_ratio, "gate": ones}),
    }

    midx = meta.set_index("blind_id")
    arrays: dict[str, np.ndarray] = {}
    pareado = []
    for name, sc in det_scores.items():
        pareado.append(analyse_detector(name, sc, meta, ADOPTED_H, BUDGETS)["pareado"].assign(Subset=subset, Seed=seed))
        ids, mat, last = alarm_matrix(sc, np.maximum(midx["t0"], ADOPTED_H + 1))
        sl = _slug(name)
        arrays[f"{sl}__ids"], arrays[f"{sl}__mat"], arrays[f"{sl}__last"] = ids, mat.astype(np.float32), last

    npz_path = alarm_dir / f"{subset}_seed{seed}_H{ADOPTED_H}.npz"
    meta.assign(Subset=subset, Seed=seed, H=ADOPTED_H).to_csv(npz_path.with_name(npz_path.stem + "_manifesto.csv"), index=False)
    np.savez_compressed(npz_path, **arrays)
    print(f"[Exp. C] {subset} seed={seed} ({time.perf_counter() - t0:.1f} s)", flush=True)
    return pareado, npz_path


def run_exp_c(seeds: list[int], out: Path, tables: Path, n_boot: int, n_jobs: int = 2) -> None:
    print("\n=== Exp. C: Baselines multivariados com memória temporal (PCA-T²/SPE EWMA, IForest EWMA) ===", flush=True)
    from joblib import Parallel, delayed

    exp_dir = out / "exp_c"
    alarm_dir = exp_dir / "alarmes"
    alarm_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, seed) for s in SUBSETS_ALL for seed in seeds]
    res_list = Parallel(n_jobs=n_jobs)(delayed(_exp_c_one)(s, seed, alarm_dir) for s, seed in jobs)
    pareado_parts = [df for par, _ in res_list for df in par]
    npz_paths = [p for _, p in res_list]

    par_all = pd.concat(pareado_parts, ignore_index=True)
    g = par_all.groupby(["Subset", "Detector", "kind", "magnitude"])
    comp = g["power_pct"].agg(["mean", "std"]).rename(columns={"mean": "power_pct_mean", "std": "power_pct_std"})
    comp["delay_med_mean"] = g["delay_med"].mean()
    comp["far_ctrl_per100_mean"] = g["far_ctrl_per100"].mean()
    comp["n_seeds"] = g.size()
    comp = comp.reset_index()

    power_boot, contr_boot = _bootstrap_memory_runs(npz_paths, n_boot=n_boot, boot_seed=SEED_BOOT)
    comp.to_csv(tables / "detectores_memoria_comparacao.csv", index=False)
    power_boot.round(3).to_csv(exp_dir / "poder_memoria_ic.csv", index=False)
    contr_boot.round(4).to_csv(tables / "contrastes_memoria_ic.csv", index=False)

    ver_cmp = tables / "detectores_comparacao.csv"
    if ver_cmp.exists():
        old_mv = pd.read_csv(ver_cmp).query("Detector == @DET_MV")
        new_mv = comp.query("Detector == @DET_MV")
        merged = new_mv.merge(old_mv, on=["Subset", "Detector", "kind", "magnitude"], suffixes=("_new", "_old"))
        print(f"[Exp. C] conferência MEWMA vs versionado: max |Δpoder| = {(merged['power_pct_mean_new'] - merged['power_pct_mean_old']).abs().max():.6f} pp", flush=True)

    with pd.option_context("display.width", 240, "display.max_columns", 20):
        print(contr_boot[contr_boot["familia"]].round(4).to_string(index=False), flush=True)


# ============================================================================
# Exp. D: Separated EOL failure modes & native detection per mode (FD003/4)
# ============================================================================


def _engine_eol_vectors(
    frame: pd.DataFrame,
    regimes: RegimeDecoupler,
    sensors: tuple[str, ...],
    healthy_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
    tail_cycles: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-engine standardized (and regime-adjusted physical) EOL shift vs. first healthy cycles."""
    rf = regimes.transform(frame)
    norm_df = regimes.standardize_sensors(rf, list(sensors), suffix="_z")
    reg_healthy = rf.loc[rf["cycle"] <= healthy_limit].groupby("regime_id")[list(sensors)].mean()

    z_rows, phys_rows = [], []
    for unit_id, eng in norm_df.sort_values("cycle").groupby("unit_id", sort=True):
        early = eng.loc[eng["cycle"] <= healthy_limit]
        tail = eng.tail(tail_cycles)
        # Regime-adjusted physical delta: compare each tail cycle against engine's healthy mean in same regime (or global regime healthy mean)
        eng_reg_means = early.groupby("regime_id")[list(sensors)].mean()

        zr = {"unit_id": int(unit_id)}
        pr = {"unit_id": int(unit_id)}
        for s in sensors:
            zr[s] = float(tail[f"{s}_z"].mean() - early[f"{s}_z"].mean())
            base_vals = [
                float(eng_reg_means.loc[r, s]) if (r in eng_reg_means.index) else float(reg_healthy.loc[r, s])
                for r in tail["regime_id"].to_numpy(dtype=int)
            ]
            pr[s] = float(np.mean(tail[s].to_numpy(dtype=float) - np.asarray(base_vals)))
        z_rows.append(zr)
        phys_rows.append(pr)
    return pd.DataFrame(z_rows).set_index("unit_id"), pd.DataFrame(phys_rows).set_index("unit_id")


def fit_failure_mode_kmeans(
    train_df: pd.DataFrame,
    n_regimes: int,
    sensors: tuple[str, ...] = MULTIVARIATE_SENSORS,
    seed: int = 42,
    healthy_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
) -> tuple[RegimeDecoupler, KMeans, dict[int, str], pd.DataFrame, pd.DataFrame]:
    """Fit KMeans(K=2) ONLY on training engines' standardized EOL shifts; label higher-ΔP30 cluster as Fan."""
    reg = RegimeDecoupler(n_clusters=n_regimes, random_state=seed).fit(train_df)
    tr_r = reg.transform(train_df)
    reg.fit_sensor_scaler(tr_r.loc[tr_r["cycle"] <= healthy_limit], list(sensors))
    z_tr, phys_tr = _engine_eol_vectors(train_df, reg, sensors, healthy_limit=healthy_limit)

    km = KMeans(n_clusters=2, n_init=20, random_state=seed).fit(z_tr[list(sensors)].to_numpy(dtype=float))
    c0_p30 = float(z_tr.loc[km.labels_ == 0, "P30"].mean())
    c1_p30 = float(z_tr.loc[km.labels_ == 1, "P30"].mean())
    fan_cluster = 1 if c1_p30 >= c0_p30 else 0
    label_map = {fan_cluster: "Fan", 1 - fan_cluster: "HPC"}
    z_tr = z_tr.assign(modo=[label_map[int(c)] for c in km.labels_])
    phys_tr = phys_tr.assign(modo=z_tr["modo"])
    return reg, km, label_map, z_tr, phys_tr


def _world_mode_signatures(world: World, seed: int) -> dict[str, np.ndarray]:
    """Train-only KMeans(K=2) EOL signatures (media, HPC, Fan) in world.signature units."""
    tr = world.train
    z_tr = standardize(world, tr)
    h_mask = (tr["cycle"] <= world.H).to_numpy()
    e_mask = (tr["rul_true"] <= 10).to_numpy()
    unit_ids = sorted(tr["unit_id"].unique())
    vecs = []
    for u in unit_ids:
        um = (tr["unit_id"] == u).to_numpy()
        vecs.append(z_tr[um & e_mask].mean(axis=0) - z_tr[um & h_mask].mean(axis=0))
    V = np.vstack(vecs)
    km = KMeans(n_clusters=2, n_init=20, random_state=seed).fit(V)
    p30_idx = list(INJECT_SENSORS).index("P30")
    fan_cluster = 1 if V[km.labels_ == 1, p30_idx].mean() >= V[km.labels_ == 0, p30_idx].mean() else 0
    return {
        "media": world.signature,
        "HPC": V[km.labels_ == (1 - fan_cluster)].mean(axis=0),
        "Fan": V[km.labels_ == fan_cluster].mean(axis=0),
    }


def _exp_d_injection_modes(seeds: list[int], tables: Path, exp_dir: Path) -> None:
    """Compare degrau (0.05, 0.10, 0.25) power using mixed vs pure-HPC vs pure-Fan signatures."""
    from dataclasses import replace

    loader = CMapssLoader()
    mags = (0.05, 0.10, 0.25)
    t0_inj = ADOPTED_H + 10
    horizon = t0_inj + 80
    rows = []

    for subset in ("FD003", "FD004"):
        for seed in seeds:
            world = load_world(subset, seed, ADOPTED_H, loader=loader)
            sigs = _world_mode_signatures(world, seed)
            rng = np.random.default_rng(seed)
            engines = {int(u): g.sort_values("cycle") for u, g in world.test.groupby("unit_id")}
            recipients = [u for u in sorted(engines) if (engines[u]["rul_true"] >= DEFAULT_RUL_CAP).sum() >= 30]

            ctrl_frames, ctrl_meta, bid = [], [], 0
            for u in recipients:
                for _ in range(6):
                    base = bootstrap_series(engines[u], 300, rng)
                    bid += 1
                    ctrl_frames.append(blind_columns(base, bid))
                    ctrl_meta.append({"blind_id": bid, "recipient": u, "role": "controle", "kind": "", "magnitude": np.nan, "t0": 0, "T": 300})
            ctrl_df = pd.concat(ctrl_frames, ignore_index=True)

            tr = world.regimes.transform(world.train)
            te_ctrl = world.regimes.transform(ctrl_df)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vs_t50 = VirtualSensor(
                    target_sensor="T50", mode="exogenous", ewma_alpha=0.10,
                    min_consecutive_breaches=2, healthy_cycle_limit=world.H, random_state=seed,
                ).fit(tr)
                mv = MultivariateResidualMonitor(healthy_cycle_limit=world.H, random_state=seed, engine_offset=True).fit(tr)
                t50_ctrl_tr, _, _ = vs_t50.evaluate(tr, te_ctrl)
                mv_ctrl_sc = mv.score(te_ctrl)

            for sig_name, sig_vec in sigs.items():
                w_sig = replace(world, signature=sig_vec)
                sig_rng = np.random.default_rng(seed + 1000)
                f_frames, f_meta = [], []
                fbid = bid
                for u in recipients:
                    for _ in range(2):
                        base = bootstrap_series(engines[u], horizon, sig_rng)
                        for m in mags:
                            fbid += 1
                            f_frames.append(blind_columns(inject(w_sig, base, delta_z(w_sig, base, "degrau", m, t0_inj, sig_rng)), fbid))
                            f_meta.append({"blind_id": fbid, "recipient": u, "role": "falha", "kind": "degrau", "magnitude": m, "t0": t0_inj, "T": horizon})
                meta_all = pd.DataFrame([*ctrl_meta, *f_meta])
                f_df = world.regimes.transform(pd.concat(f_frames, ignore_index=True))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t50_f_tr, _, _ = vs_t50.evaluate(tr, f_df)
                    mv_f_sc = mv.score(f_df)

                t50_ctrl = t50_ctrl_tr.sort_values(["unit_id", "cycle"])
                t50_f = t50_f_tr.sort_values(["unit_id", "cycle"])
                sc_t50 = pd.concat([
                    pd.DataFrame({"blind_id": t50_ctrl["unit_id"].to_numpy(), "cycle": t50_ctrl["cycle"].to_numpy(), "ratio": (t50_ctrl["ewma_abs_z"] / t50_ctrl["ewma_ucl"]).to_numpy(), "gate": True}),
                    pd.DataFrame({"blind_id": t50_f["unit_id"].to_numpy(), "cycle": t50_f["cycle"].to_numpy(), "ratio": (t50_f["ewma_abs_z"] / t50_f["ewma_ucl"]).to_numpy(), "gate": True}),
                ], ignore_index=True)
                sc_mv = pd.concat([
                    pd.DataFrame({"blind_id": mv_ctrl_sc["unit_id"].to_numpy(), "cycle": mv_ctrl_sc["cycle"].to_numpy(), "ratio": (mv_ctrl_sc["t2"] / mv_ctrl_sc["ucl"]).to_numpy(), "gate": True}),
                    pd.DataFrame({"blind_id": mv_f_sc["unit_id"].to_numpy(), "cycle": mv_f_sc["cycle"].to_numpy(), "ratio": (mv_f_sc["t2"] / mv_f_sc["ucl"]).to_numpy(), "gate": True}),
                ], ignore_index=True)

                for dname, sc_d in (("T50 EWMA (com regime, sem portão)", sc_t50), (DET_MV, sc_mv)):
                    par = analyse_detector(dname, sc_d, meta_all, ADOPTED_H, BUDGETS)["pareado"]
                    rows.append(par.assign(Subset=subset, Seed=seed, assinatura=sig_name))

    inj_raw = pd.concat(rows, ignore_index=True)
    inj_raw.to_csv(exp_dir / "injecao_assinatura_por_semente.csv", index=False)
    g = inj_raw.groupby(["Subset", "assinatura", "Detector", "magnitude"])
    inj_summ = g["power_pct"].agg(["mean", "std"]).rename(columns={"mean": "power_pct_mean", "std": "power_pct_std"})
    inj_summ["delay_med_mean"] = g["delay_med"].mean()
    inj_summ["far_ctrl_per100_mean"] = g["far_ctrl_per100"].mean()
    inj_summ["n_seeds"] = g.size()
    inj_summ = inj_summ.reset_index()
    inj_summ.to_csv(tables / "injecao_assinatura_por_modo.csv", index=False)
    print(inj_summ[inj_summ["magnitude"] == 0.10].round(2).to_string(index=False), flush=True)


def run_exp_d(seeds: list[int], out: Path, tables: Path) -> None:
    print("\n=== Exp. D: Modos de falha separados (HPC vs. Fan) no FD003 e FD004 ===", flush=True)
    loader = CMapssLoader()
    sig_rows, det_rows = [], []

    for subset in ("FD003", "FD004"):
        frame = loader.load(subset)
        n_regimes = 6 if subset == "FD004" else 1
        _, _, _, z_all, phys_all = fit_failure_mode_kmeans(frame, n_regimes, MULTIVARIATE_SENSORS, seed=42)
        for modo in ("HPC", "Fan"):
            zm = z_all.loc[z_all["modo"] == modo]
            pm = phys_all.loc[phys_all["modo"] == modo]
            for s in MULTIVARIATE_SENSORS:
                sig_rows.append({
                    "Subset": subset, "modo": modo, "sensor": s, "n_motores": int(len(zm)),
                    "delta_z_mean": float(zm[s].mean()), "delta_z_std": float(zm[s].std()),
                    "delta_fisico_mean": float(pm[s].mean()), "delta_fisico_std": float(pm[s].std()),
                    "sinal_dominante": "+" if float(zm[s].mean()) >= 0 else "-",
                })

        for seed in seeds:
            train_units, test_units = split_engine_units(frame, test_ratio=TEST_RATIO, random_state=seed)
            train_df = frame.loc[frame["unit_id"].isin(train_units)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
            test_df = frame.loc[frame["unit_id"].isin(test_units)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
            reg, km, label_map, _, _ = fit_failure_mode_kmeans(train_df, n_regimes, MULTIVARIATE_SENSORS, seed=seed)
            z_te, _ = _engine_eol_vectors(test_df, reg, MULTIVARIATE_SENSORS, healthy_limit=DEFAULT_HEALTHY_CYCLE_LIMIT)
            te_modes = pd.Series(
                [label_map[int(c)] for c in km.predict(z_te[list(MULTIVARIATE_SENSORS)].to_numpy(dtype=float))],
                index=z_te.index, name="modo",
            )

            _, per_t50 = _t50_per_engine(train_df, test_df, subset, seed)
            mv = MultivariateResidualMonitor(
                healthy_cycle_limit=DEFAULT_HEALTHY_CYCLE_LIMIT, random_state=seed, calibration=MEWMA_CALIBRATION_MONITORING,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mv.fit(reg.transform(train_df))
                _, per_mv, _ = mv.alarms(mv.score(reg.transform(test_df)))

            for det_name, per in (("T50 EWMA (com regime)", per_t50), ("MEWMA (13 sensores, sintético)", per_mv)):
                merged = per.join(te_modes, on="unit_id")
                for modo in ("HPC", "Fan"):
                    g = merged.loc[merged["modo"] == modo]
                    det = g.loc[g["detected"]]
                    det_rows.append({
                        "Subset": subset, "Seed": seed, "modo": modo, "Detector": det_name,
                        "n_motores": int(len(g)),
                        "deteccao_pct": float(g["detected"].mean() * 100.0) if len(g) else np.nan,
                        "antecedencia_mediana": float(g["lead_time_cycles"].median()) if len(g) else np.nan,
                        "antecedencia_mediana_detectados": float(det["lead_time_cycles"].median()) if len(det) else np.nan,
                        "falso_alarme_saudavel_pct": float(g["healthy_far"].mean() * 100.0) if len(g) else np.nan,
                    })

    sig_df = pd.DataFrame(sig_rows)
    det_raw = pd.DataFrame(det_rows)
    g = det_raw.groupby(["Subset", "modo", "Detector"])
    cols_d = ["deteccao_pct", "antecedencia_mediana", "antecedencia_mediana_detectados", "falso_alarme_saudavel_pct"]
    det_summ = g[cols_d].mean().reset_index()
    g_std = g[["deteccao_pct", "antecedencia_mediana"]].std()
    for col in ("deteccao_pct", "antecedencia_mediana"):
        det_summ[f"{col}_std"] = g_std[col].to_numpy()
    det_summ["n_motores_total"] = g["n_motores"].sum().to_numpy()
    det_summ["n_seeds"] = g.size().to_numpy()

    exp_dir = out / "exp_d"
    exp_dir.mkdir(parents=True, exist_ok=True)
    det_raw.to_csv(exp_dir / "deteccao_nativa_por_semente.csv", index=False)
    sig_df.to_csv(tables / "assinaturas_modo_falha_eol.csv", index=False)
    det_summ.to_csv(tables / "deteccao_nativa_por_modo.csv", index=False)

    print(sig_df[sig_df["sensor"].isin(["T50", "P30", "phi", "BPR", "Nf", "P15"])].round(3).to_string(index=False), flush=True)
    print(det_summ.round(2).to_string(index=False), flush=True)
    _exp_d_injection_modes(seeds, tables, exp_dir)


def _t50_per_engine(train_df: pd.DataFrame, test_df: pd.DataFrame, subset: str, seed: int) -> tuple[dict, pd.DataFrame]:
    n_regimes = 6 if subset in ("FD002", "FD004") else 1
    regimes = RegimeDecoupler(n_clusters=n_regimes, random_state=seed).fit(train_df)
    vs = VirtualSensor(
        target_sensor="T50", mode="exogenous", ewma_alpha=0.10, min_consecutive_breaches=2,
        healthy_cycle_limit=DEFAULT_HEALTHY_CYCLE_LIMIT, random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, per_engine, _ = vs.evaluate(regimes.transform(train_df), regimes.transform(test_df))
    return {int(r.unit_id): (None if pd.isna(r.first_alarm_cycle) else int(r.first_alarm_cycle)) for r in per_engine.itertuples()}, per_engine.copy()


# ============================================================================
# CLI
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", choices=["all", "a", "b", "c", "d"], default="all")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--tables-dir", type=Path, default=TABLES_DEFAULT)
    ap.add_argument("--params-dir", type=Path, default=PARAMS_DEFAULT)
    a = ap.parse_args(argv)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    a.tables_dir.mkdir(parents=True, exist_ok=True)
    if a.study in ("all", "a"):
        run_exp_a(list(a.seeds), a.output_dir, a.tables_dir, a.params_dir, a.jobs)
    if a.study in ("all", "b"):
        run_exp_b(list(a.seeds), a.output_dir, a.tables_dir, a.jobs)
    if a.study in ("all", "c"):
        run_exp_c(list(a.seeds), a.output_dir, a.tables_dir, a.boot, a.jobs)
    if a.study in ("all", "d"):
        run_exp_d(list(a.seeds), a.output_dir, a.tables_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Busca de hiperparâmetros (inclui a janela) por validação cruzada agrupada por motor, só nos motores de treino.

WHAT   Para cada (subconjunto, modelo), um estudo Optuna (TPE, semente fixa) escolhe os hiperparâmetros, o tamanho da janela, se há atributos
       temporais e se os sensores são padronizados por regime. O objetivo é o RMSE (RUL truncada em 125) nos motores de validação de um
       GroupKFold por motor. A janela (`window`) é tamanho máximo dos atributos temporais (5, 10 e `window`) ou `seq_length` das redes.
WHY    Substituir hiperparâmetros fixos e escolhas de janela feitas à mão por uma seleção em validação, sem tocar o teste oficial.
       Regra do erro-padrão (1-SE): entre as tentativas com RMSE médio dentro de 1 erro-padrão (das dobras) do melhor, escolhe-se a de
       MENOR janela (0 = sem atributos temporais), isto é, a mais simples; evita escolher a janela por ruído.
[LEAKAGE] Tudo é ajustado só nos motores de treino de cada dobra (regimes, estatísticas por regime, escalonadores, atributos causais).
          O teste oficial não é lido aqui.

Rodar (da raiz; no Colab, com `--device cuda` para as redes e o XGBoost/CatBoost):
    python -m pdm_cmaps.experiments.selecao_hiperparametros --subsets FD004 --models XGBoost "PyTorch RNN (LSTM)" --output-dir results/hiperparametros
Saída: <output-dir>/<subconjunto>/<modelo>/{melhores.json, tentativas.csv, estudo.db}. Retomável (o estudo é um SQLite).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pdm_cmaps.data.features import CausalFeatureEngineer
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.domain.rul import compute_piecewise_rul
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, SETTING_COLUMNS
from pdm_cmaps.experiments.benchmark_oficial_completo import _regime_frames
from pdm_cmaps.modeling.registry import get_model
from pdm_cmaps.pipeline.extra_models import fit_predict

WINDOWS = (10, 15, 20, 30, 40, 50, 60)
NETS = ("PyTorch RNN (LSTM)", "PyTorch CNN", "PyTorch Transformer")
MODELS = ("Ridge", "Random Forest", "XGBoost", "XGBoost (NASA Loss)", "LightGBM", "CatBoost", "MLP", "MiniRocket + RidgeCV", *NETS)
# tentativas por modelo (escalonadas por --trials-scale); redes e MiniRocket são caras, ficam com menos
BUDGET = {"Ridge": 30, "MLP": 30, "Random Forest": 40, "XGBoost": 60, "XGBoost (NASA Loss)": 60, "LightGBM": 60, "CatBoost": 40,
          "MiniRocket + RidgeCV": 20, "PyTorch RNN (LSTM)": 20, "PyTorch CNN": 20, "PyTorch Transformer": 20}
N_SPLITS = {m: (3 if m in NETS or m.startswith("MiniRocket") else 5) for m in MODELS}


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


# --------------------------------------------------------------------------------------------------------------------
# espaço de busca e construção do modelo
# --------------------------------------------------------------------------------------------------------------------
def sugerir(trial, model: str) -> dict:
    """Hiperparâmetros de uma tentativa. `window` = 0 quando o modelo não usa janela."""
    p: dict = {"representation": trial.suggest_categorical("representation", ["global", "regime"])}
    sequence = model in NETS or model.startswith("MiniRocket")
    if sequence:
        p["window"] = trial.suggest_categorical("window", [w for w in WINDOWS])
        p["temporal"] = False
    else:
        p["temporal"] = trial.suggest_categorical("temporal", [False, True])
        p["window"] = trial.suggest_categorical("window", list(WINDOWS)) if p["temporal"] else 0
    h: dict = {}
    if model == "Ridge":
        h["alpha"] = trial.suggest_float("alpha", 1e-2, 1e3, log=True)
    elif model == "Random Forest":
        h.update(n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50), max_depth=trial.suggest_int("max_depth", 6, 24),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10), max_features=trial.suggest_float("max_features", 0.3, 1.0))
    elif model in ("XGBoost", "XGBoost (NASA Loss)"):
        h.update(n_estimators=trial.suggest_int("n_estimators", 50, 600, step=50), max_depth=trial.suggest_int("max_depth", 3, 10),
                 learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True), subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0), min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True))
    elif model == "LightGBM":
        h.update(n_estimators=trial.suggest_int("n_estimators", 50, 600, step=50), max_depth=trial.suggest_int("max_depth", 3, 12),
                 learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True), num_leaves=trial.suggest_int("num_leaves", 15, 127),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1, colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 100), reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True))
    elif model == "CatBoost":
        h.update(iterations=trial.suggest_int("iterations", 100, 800, step=100), depth=trial.suggest_int("depth", 4, 10),
                 learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True), l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True))
    elif model == "MLP":
        layers = {"64-32": (64, 32), "128-64": (128, 64), "128-64-32": (128, 64, 32), "256-128": (256, 128)}
        h.update(hidden_layer_sizes=layers[trial.suggest_categorical("hidden", list(layers))], alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                 learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True))
    elif model == "MiniRocket + RidgeCV":
        h["num_kernels"] = trial.suggest_categorical("num_kernels", [500, 1000, 2000])
    elif model in NETS:
        h.update(learning_rate=trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True), batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
                 dropout=trial.suggest_float("dropout", 0.0, 0.4))
        if model == "PyTorch RNN (LSTM)":
            h.update(hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128]), num_layers=trial.suggest_int("num_layers", 1, 3))
        elif model == "PyTorch CNN":
            h.update(num_filters=trial.suggest_categorical("num_filters", [32, 64, 128]), kernel_size=trial.suggest_categorical("kernel_size", [3, 5]))
        else:
            h.update(d_model=trial.suggest_categorical("d_model", [32, 64, 128]), nhead=trial.suggest_categorical("nhead", [2, 4]),
                     num_layers=trial.suggest_int("num_layers", 1, 3), dim_feedforward=trial.suggest_categorical("dim_feedforward", [64, 128, 256]))
    else:
        raise ValueError(f"modelo desconhecido: {model!r}")
    p["hyper"] = h
    return p


def montar_modelo(model: str, params: dict, seed: int, device: str = "cpu", n_jobs: int = 2):
    """Modelo novo, com os hiperparâmetros de `params`; (tipo, estimador) como em `pipeline.extra_models.fit_predict`."""
    h = dict(params["hyper"])
    gpu = device != "cpu"
    if model == "Ridge":
        return "sklearn", get_model("ridge", seed, **h)
    if model == "Random Forest":
        return "sklearn", get_model("rf", seed, n_jobs=n_jobs, **h)
    if model == "XGBoost":
        extra = {"device": "cuda", "tree_method": "hist"} if gpu else {}
        return "sklearn", get_model("xgboost", seed, n_jobs=n_jobs, **h, **extra)
    if model == "XGBoost (NASA Loss)":
        return "sklearn", get_model("xgboost_nasa_loss", seed, n_jobs=n_jobs, **h)
    if model == "LightGBM":
        return "sklearn", get_model("lightgbm", seed, n_jobs=n_jobs, **h)
    if model == "CatBoost":
        extra = {"task_type": "GPU"} if gpu else {}
        return "sklearn", get_model("catboost", seed, n_jobs=n_jobs, **h, **extra)
    if model == "MLP":
        return "sklearn", get_model("mlp", seed, **h)
    if model == "MiniRocket + RidgeCV":
        from pdm_cmaps.modeling.minirocket import MiniRocketRULModel
        return "deep", MiniRocketRULModel(seq_length=params["window"], random_state=seed, device=device, **h)
    from pdm_cmaps.modeling.torch_models import PyTorchCNNModel, PyTorchRNNModel, PyTorchTransformerModel
    cls = {"PyTorch RNN (LSTM)": PyTorchRNNModel, "PyTorch CNN": PyTorchCNNModel, "PyTorch Transformer": PyTorchTransformerModel}[model]
    return "deep", cls(seq_length=params["window"], random_state=seed, device=device, **h)


# --------------------------------------------------------------------------------------------------------------------
# dados: representação (global ou por regime) e atributos temporais, ajustados só no treino
# --------------------------------------------------------------------------------------------------------------------
def montar_dados(train: pd.DataFrame, evalf: pd.DataFrame, sensors: list[str], n_regimes: int, params: dict):
    """(treino, avaliação, colunas) prontos para `fit_predict`; tudo ajustado em `train`. A ordem das linhas de `evalf` pode mudar (ordenada por motor e ciclo)."""
    if params["representation"] == "regime":
        tr, ev, cols = _regime_frames(train, evalf, sensors, n_regimes)
        sens = [c for c in cols if c.endswith("_norm")]
    else:
        tr, ev, cols, sens = train, evalf, [*sensors, *SETTING_COLUMNS, "cycle"], sensors
    if params["temporal"]:
        eng = CausalFeatureEngineer(sens, windows=tuple(sorted({5, 10, int(params["window"])}))).fit(tr, DEFAULT_HEALTHY_CYCLE_LIMIT)
        tr_fe, ev_fe = eng.transform(tr), eng.transform(ev)
        cols = [*cols, *[c for c in tr_fe.columns if c not in tr.columns]]
        return tr_fe, ev_fe, cols
    return tr.sort_values(["unit_id", "cycle"]).reset_index(drop=True), ev.sort_values(["unit_id", "cycle"]).reset_index(drop=True), cols


def preparar(subset: str):
    """(quadro de treino com RUL, sensores informativos, nº de regimes) de um subconjunto; só os motores de treino do C-MAPSS."""
    loader = CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    return frame, loader.informative_sensors(frame), (6 if subset in ("FD002", "FD004") else 1)


def validar(frame: pd.DataFrame, sensors: list[str], n_regimes: int, model: str, params: dict, seed: int, n_splits: int, device: str = "cpu",
            n_jobs: int = 2, trial=None) -> list[float]:
    """RMSE (RUL truncada) em cada dobra de um GroupKFold por motor, só nos motores de treino. Poda após cada dobra se `trial` for dado."""
    from sklearn.model_selection import GroupKFold

    import optuna

    scores: list[float] = []
    for k, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=n_splits).split(frame, groups=frame["unit_id"])):
        tr_df, va_df = frame.iloc[tr_idx].reset_index(drop=True), frame.iloc[va_idx].reset_index(drop=True)
        kind, est = montar_modelo(model, params, seed, device)
        tr, ev, cols = montar_dados(tr_df, va_df, sensors, n_regimes, params)
        pred = fit_predict(kind, est, tr, cols, ev)
        scores.append(float(np.sqrt(np.mean((np.asarray(pred, dtype=float) - ev["RUL"].to_numpy(dtype=float)) ** 2))))
        if trial is not None:
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return scores


# --------------------------------------------------------------------------------------------------------------------
# escolha final: regra do erro-padrão
# --------------------------------------------------------------------------------------------------------------------
def escolher_um_se(tentativas: pd.DataFrame) -> pd.Series:
    """Entre as tentativas dentro de 1 erro-padrão do melhor RMSE médio, a de MENOR janela (desempate: menor RMSE médio).

    `tentativas` tem as colunas `rmse_mean`, `rmse_se` e `window` (0 = sem janela). Só entram tentativas completas (sem poda)."""
    d = tentativas.dropna(subset=["rmse_mean"])
    if d.empty:
        raise ValueError("nenhuma tentativa completa")
    best = d.loc[d["rmse_mean"].idxmin()]
    limite = float(best["rmse_mean"] + (best["rmse_se"] if np.isfinite(best["rmse_se"]) else 0.0))
    dentro = d[d["rmse_mean"] <= limite]
    return dentro.sort_values(["window", "rmse_mean"]).iloc[0]


def tabela_de_tentativas(study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        folds = t.user_attrs.get("fold_rmse")
        if t.state.name != "COMPLETE" or not folds:
            continue
        a = np.asarray(folds, dtype=float)
        rows.append({"trial": t.number, "rmse_mean": float(a.mean()), "rmse_se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan"),
                     "window": int(t.user_attrs["params"]["window"]), "params": json.dumps(t.user_attrs["params"], ensure_ascii=False)})
    return pd.DataFrame(rows)


def buscar(subset: str, model: str, out: Path, trials: int | None = None, seed: int = 42, device: str = "cpu", n_jobs: int = 2) -> dict:
    """Roda (ou retoma) o estudo de um (subconjunto, modelo) e grava `melhores.json`."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    target = out / subset / slug(model)
    target.mkdir(parents=True, exist_ok=True)
    frame, sensors, n_regimes = preparar(subset)
    study = optuna.create_study(study_name=f"{subset}_{slug(model)}", storage=f"sqlite:///{target / 'estudo.db'}", load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed), pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    alvo = trials if trials is not None else BUDGET[model]
    feitas = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    t0 = time.time()

    def objetivo(trial):
        params = sugerir(trial, model)
        trial.set_user_attr("params", params)
        scores = validar(frame, sensors, n_regimes, model, params, seed, N_SPLITS[model], device, n_jobs, trial)
        trial.set_user_attr("fold_rmse", scores)
        return float(np.mean(scores))

    if alvo - feitas > 0:
        study.optimize(objetivo, n_trials=alvo - feitas, gc_after_trial=True)
    tent = tabela_de_tentativas(study)
    tent.to_csv(target / "tentativas.csv", index=False)
    escolhida = escolher_um_se(tent)
    melhor_media = tent.loc[tent["rmse_mean"].idxmin()]
    resultado = {"subset": subset, "model": model, "params": json.loads(escolhida["params"]), "cv_rmse": float(escolhida["rmse_mean"]), "cv_se": float(escolhida["rmse_se"]),
                 "melhor_rmse_medio": float(melhor_media["rmse_mean"]), "melhor_trial": int(melhor_media["trial"]), "trial_escolhida": int(escolhida["trial"]),
                 "n_tentativas_completas": int(len(tent)), "n_tentativas_total": int(len(study.trials)), "n_splits": N_SPLITS[model], "seed": seed,
                 "segundos": round(time.time() - t0, 1)}
    (target / "melhores.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2))
    print(f"[{subset}] {model}: CV RMSE {resultado['cv_rmse']:.2f} ± {resultado['cv_se']:.2f} (janela {resultado['params']['window']}, "
          f"{resultado['params']['representation']}), {resultado['n_tentativas_completas']} completas de {resultado['n_tentativas_total']} ({resultado['segundos']:.0f} s)", flush=True)
    return resultado


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--trials", type=int, default=None, help="tentativas por estudo (padrão: orçamento do modelo)")
    ap.add_argument("--trials-scale", type=float, default=1.0, help="multiplica o orçamento padrão de cada modelo (ignorado com --trials)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="cpu ou cuda (só na busca; a refit final do teste oficial é sempre na CPU)")
    ap.add_argument("--n-jobs", type=int, default=2, help="threads dos modelos de árvores")
    ap.add_argument("--output-dir", type=Path, default=Path("results/hiperparametros"))
    a = ap.parse_args(argv)
    for subset in a.subsets:
        for model in a.models:
            n = a.trials if a.trials is not None else max(1, int(round(BUDGET[model] * a.trials_scale)))
            buscar(subset, model, a.output_dir, n, a.seed, a.device, a.n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

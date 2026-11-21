"""Exploração do R² dos sensores do FD001 e do FD003 (branch exp/sensor-virtual-r2-fd001-fd003).

PERGUNTA: que técnicas e modelos levam o R² de prever um sensor perto de 0,90 (ou acima) nos subconjuntos de regime único,
e a que custo para a detecção? O teto do R² é medido antes (`teto_ruido.csv`): sem entrada que acompanhe o ruído do próprio
ciclo, R² ≤ 1 − σ²_ruído/Var(y), e na janela saudável esse teto já fica abaixo de 0,90 na maioria dos sensores.

DUAS TRILHAS, com rótulos e tabelas separados (decisão do autor em 2026-11-18):
  A  exógeno, janela saudável: o sensor virtual VÁLIDO para detecção (entradas = ajustes e ambiente; treino só nos ciclos
     ≤ H). R² na janela tardia (ciclos 26 a 50) e detecção/falso alarme ao lado. Compara o modelo com o teto.
  B1 cross-sensor treinado só na janela saudável (`VirtualSensor(mode="cross_sensor")`): ainda é detector (o resíduo sobe
     quando a relação entre sensores se rompe). R² em todos os ciclos dos motores de teste + detecção.
  B1s igual à B1, mas SEM o sensor "gêmeo" nas entradas (Nf<->NRf, Nc<->NRc são o mesmo sinal corrigido pela temperatura de
     entrada; sem tirá-lo, o R² alto pode ser tautologia e não relação entre sensores). Checagem de validade da B1.
  B2 cross-sensor treinado no CICLO DE VIDA inteiro: regressão pura. O modelo VÊ a degradação, então o resíduo some e NÃO
     serve como detector; só o R² faz sentido (por isso não há coluna de detecção).

[LEAKAGE] divisão por MOTOR (split_engine_units), nunca por linha; o baseline do motor é a média dos primeiros W ciclos dele
(causal); R² de B2 é reportado em todos os ciclos e só depois do ciclo W (onde o baseline já existe). Nenhum atraso/média
móvel do próprio alvo entra como entrada (isso viraria um modelo autorregressivo e mataria a detecção).
[ASSUMPTION] hiperparâmetros fixos e pequenos, sem ajuste no teste; todas as configurações são reportadas (nada de "melhor
de N" sem os N), e cada semente muda a divisão treino/teste.

Run:  OMP_NUM_THREADS=2 .venv/bin/python -m pdm_cmaps.experiments.exploracao_r2 --track B2 --subset FD003
Saída: results/exploracao_r2/<trilha>_<subset>.csv
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pdm_cmaps.config import DEFAULT_RESULTS_DIR
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.splits import partition_healthy_and_lifecycle, split_engine_units
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.detection.virtual_sensor_metrics import r2, virtual_sensor_r2
from pdm_cmaps.domain.rul import compute_piecewise_rul
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, SETTING_COLUMNS
from pdm_cmaps.experiments.virtual_sensor_models import SafeRegressor
from pdm_cmaps.modeling.registry import get_model

OUT = DEFAULT_RESULTS_DIR / "exploracao_r2"
SUBSETS = ("FD001", "FD003")
SEEDS = (42, 7, 123)
SEEDS10 = (42, 7, 123, 1, 2, 3, 4, 5, 6, 8)  # B1 e B1s: 10 divisões, para dar estabilidade aos achados da B1
TWINS = {"Nf": "NRf", "NRf": "Nf", "Nc": "NRc", "NRc": "Nc"}  # mesmo sinal, corrigido pela temperatura de entrada
H = DEFAULT_HEALTHY_CYCLE_LIMIT  # 50
W = H // 2  # janela do baseline do motor (25)

# modelos pequenos e sem ajuste no teste; "chave do registro" e argumentos
MODELS = {
    "Ridge": ("ridge", {"alpha": 10}),
    "Extra Trees": ("extra_trees", {"n_estimators": 100, "min_samples_leaf": 5, "n_jobs": 2}),
    "LightGBM": ("lightgbm", {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "verbose": -1, "n_jobs": 2}),
}


def prepare(subset: str, seed: int):
    """Carrega, divide por motor e devolve (frame, treino, teste, sensores informativos) para a semente."""
    loader = CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    sensors = loader.informative_sensors(frame.loc[frame["unit_id"].isin(train_units)])
    return frame, train_units, test_units, sensors


def engine_baseline(frame: pd.DataFrame, cols: list[str], w: int = W) -> pd.DataFrame:
    """Média dos primeiros w ciclos de cada motor (causal), alinhada linha a linha com `frame`."""
    base = frame.loc[frame["cycle"] <= w].groupby("unit_id")[cols].mean()
    return base.reindex(frame["unit_id"]).set_index(frame.index)


# ---------------------------------------------------------------- Trilhas A e B1 (via VirtualSensor, com detecção)
def run_virtual_sensor_track(track: str, subset: str, seeds=SEEDS) -> pd.DataFrame:
    mode = "exogenous" if track == "A" else "cross_sensor"
    drop_twins = track == "B1s"
    rows = []
    for seed in seeds:
        frame, train_units, test_units, sensors = prepare(subset, seed)
        parts = partition_healthy_and_lifecycle(frame, train_units, test_units, H)
        regimes = RegimeDecoupler(n_clusters=1, random_state=seed).fit(parts["full_lifecycle_train"])
        healthy_train, test_full = regimes.transform(parts["healthy_train"]), regimes.transform(parts["full_lifecycle_test"])
        for sensor in sensors:
            for name in ("Ridge", "Extra Trees"):
                key, kw = MODELS[name]
                feats = None
                if drop_twins:  # mesmas entradas da B1, menos o gêmeo do alvo
                    feats = [c for c in healthy_train.columns if c not in {sensor, TWINS.get(sensor), "unit_id", "cycle", "regime_id", "RUL", "subpart"}]
                vs = VirtualSensor(target_sensor=sensor, mode=mode, feature_cols=feats, ewma_alpha=0.10, healthy_cycle_limit=H, min_consecutive_breaches=2,
                                   random_state=seed, estimator=SafeRegressor(get_model(key, seed, **kw)), engine_offset=True, offset_window=W)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scored, _, summary = vs.evaluate(healthy_train, test_full)
                trace = scored.rename(columns={sensor: "actual"}).assign(sensor=sensor, model=name)
                m = virtual_sensor_r2(trace[["sensor", "model", "unit_id", "cycle", "actual", "prediction"]], H).iloc[0].to_dict()
                rows.append({"track": track, "subset": subset, "seed": seed, "sensor": sensor, "model": name,
                             **{k: m[k] for k in ("r2_holdout", "r2_late_model", "r2_late_motor_base", "r2_late_model_offset", "pct_between_engine")},
                             "detection_rate_pct": summary["detection_rate_pct"], "healthy_false_alarm_pct": summary["mean_healthy_false_alarm_pct"],
                             "premature_alarm_pct": summary["premature_alarm_pct"], "lead_time_detected": summary["mean_lead_time_detected_cycles"]})
        print(f"  [{track} {subset}] seed {seed} ok", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Trilha B2 (regressão pura no ciclo de vida inteiro)
def feature_sets(frame: pd.DataFrame, target: str, inputs: list[str]) -> dict[str, tuple[pd.DataFrame, bool]]:
    """Conjuntos de entradas (todos causais e SEM o alvo). O booleano diz se o alvo é o desvio do baseline do motor."""
    x = frame[inputs]
    base_x = engine_baseline(frame, inputs)
    cent = (x - base_x).add_suffix("_c")
    roll = frame.groupby("unit_id")[inputs].transform(lambda s: s.rolling(5, min_periods=1).mean())
    roll_c = (roll - base_x).add_suffix("_rc")
    return {
        "bruto": (x, False),
        "bruto+centrado": (pd.concat([x, cent], axis=1), False),
        "centrado->desvio": (cent, True),
        "media5_centrada->desvio": (roll_c, True),
    }


def run_lifecycle_track(subset: str, seeds=SEEDS) -> pd.DataFrame:
    t0, rows = time.time(), []
    for seed in seeds:
        frame, train_units, test_units, sensors = prepare(subset, seed)
        tr_mask, te_mask = frame["unit_id"].isin(train_units).to_numpy(), frame["unit_id"].isin(test_units).to_numpy()
        after = (frame["cycle"] > W).to_numpy()
        for sensor in sensors:
            inputs = [c for c in sensors if c != sensor] + SETTING_COLUMNS
            y = frame[sensor].to_numpy(dtype=float)
            base_y = engine_baseline(frame, [sensor])[sensor].to_numpy()
            for fs_name, (X, delta) in feature_sets(frame, sensor, inputs).items():
                target = y - base_y if delta else y
                for name, (key, kw) in MODELS.items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = SafeRegressor(get_model(key, seed, **kw)).fit(X[tr_mask].to_numpy(), target[tr_mask])
                        pred = model.predict(X[te_mask].to_numpy())
                    pred = pred + base_y[te_mask] if delta else pred
                    yt, a = y[te_mask], after[te_mask]
                    rows.append({"track": "B2", "subset": subset, "seed": seed, "sensor": sensor, "features": fs_name, "model": name,
                                 "r2_all_cycles": r2(yt, pred), "r2_after_w": r2(yt[a], pred[a])})
        print(f"  [B2 {subset}] seed {seed} ok ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=("A", "B1", "B1s", "B2"), required=True)
    ap.add_argument("--subset", choices=SUBSETS, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="padrão: 3 sementes (A, B2) ou 10 (B1, B1s)")
    ap.add_argument("--tag", default="", help="sufixo do arquivo de saída (ex.: _10s)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(a.seeds) if a.seeds else (SEEDS10 if a.track in ("B1", "B1s") else SEEDS)
    df = run_lifecycle_track(a.subset, seeds) if a.track == "B2" else run_virtual_sensor_track(a.track, a.subset, seeds)
    f = a.out / f"{a.track}_{a.subset}{a.tag}.csv"
    df.to_csv(f, index=False)
    print(f"gravado {f} ({len(df)} linhas)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

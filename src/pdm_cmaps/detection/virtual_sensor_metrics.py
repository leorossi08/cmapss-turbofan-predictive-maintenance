"""R² do sensor virtual e a régua do deslocamento entre motores.

Por que existe: em subconjunto de regime único as entradas exógenas quase não variam, então o sensor virtual vira a média
dos motores de treino e o R² fica perto de 0 (negativo quando a degradação afasta a leitura). Boa parte da variância que
sobra é o deslocamento de cada motor em relação à média da frota (desgaste inicial e fabricação). Estas métricas separam
as causas sem mudar o detector, que já padroniza o resíduo por motor (μ_e, σ_e da janela saudável).

Janela saudável H é dividida ao meio: os primeiros H//2 ciclos de cada motor calibram, os H//2 seguintes avaliam. Assim o
baseline do motor e o deslocamento são causais (só usam o passado) e comparáveis com o modelo atual na mesma janela.

Colunas devolvidas por (sensor, modelo):
- r2_holdout           R² em todos os ciclos dos motores de teste (negativo é esperado com degradação).
- r2_healthy           R² nos ciclos 1 a H.
- r2_late_model        R² do modelo atual nos ciclos H//2+1 a H.
- r2_late_motor_base   R² de prever a média do próprio motor nos ciclos 1 a H//2 (baseline do motor).
- r2_late_model_offset R² do modelo atual somado ao deslocamento médio do motor nos ciclos 1 a H//2.
- pct_between_engine   % da variância na janela 1 a H que é diferença de média entre motores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRACE_REQUIRED = ("sensor", "model", "unit_id", "cycle", "actual", "prediction")


def r2(y, p) -> float:
    """R² = 1 - SSR/SST; NaN quando a leitura é constante (SST = 0) ou há menos de 2 pontos."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return float("nan")
    sst = float(((y - y.mean()) ** 2).sum())
    return float("nan") if sst <= 0 else 1.0 - float(((y - p) ** 2).sum()) / sst


def _between_engine_pct(window: pd.DataFrame) -> float:
    total = float(((window["actual"] - window["actual"].mean()) ** 2).sum())
    if total <= 0 or window["unit_id"].nunique() < 2:
        return float("nan")
    means = window.groupby("unit_id")["actual"].transform("mean")
    return 100.0 * float(((means - window["actual"].mean()) ** 2).sum()) / total


def _one(group: pd.DataFrame, healthy_limit: int) -> pd.Series:
    cal_end = healthy_limit // 2
    cal, late = group[group["cycle"] <= cal_end], group[(group["cycle"] > cal_end) & (group["cycle"] <= healthy_limit)]
    base = cal.groupby("unit_id")["actual"].mean().rename("base")
    off = (cal["actual"] - cal["prediction"]).groupby(cal["unit_id"]).mean().rename("off")
    late = late.join(base, on="unit_id").join(off, on="unit_id").dropna(subset=["base", "off"])
    return pd.Series({
        "r2_holdout": r2(group["actual"], group["prediction"]),
        "r2_healthy": r2(group.loc[group["cycle"] <= healthy_limit, "actual"], group.loc[group["cycle"] <= healthy_limit, "prediction"]),
        "r2_late_model": r2(late["actual"], late["prediction"]),
        "r2_late_motor_base": r2(late["actual"], late["base"]),
        "r2_late_model_offset": r2(late["actual"], late["prediction"] + late["off"]),
        "pct_between_engine": _between_engine_pct(group[group["cycle"] <= healthy_limit]),
    })


def virtual_sensor_r2(trace: pd.DataFrame, healthy_limit: int) -> pd.DataFrame:
    """Métricas de R² por (sensor, modelo) a partir do trace (colunas sensor, model, unit_id, cycle, actual, prediction)."""
    missing = [c for c in TRACE_REQUIRED if c not in trace.columns]
    if missing:
        raise ValueError(f"trace sem as colunas {missing}")
    rows = []
    for (sensor, model), g in trace.groupby(["sensor", "model"], sort=False):
        rows.append({"sensor": sensor, "model": model, **_one(g, healthy_limit).to_dict()})
    return pd.DataFrame(rows)

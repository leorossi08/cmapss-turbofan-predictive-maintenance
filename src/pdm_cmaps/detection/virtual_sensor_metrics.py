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
- r2_late_within_regime  R² do modelo atual nos ciclos H//2+1 a H com a variância medida DENTRO do regime (SST em torno da média de
                         cada regime). Em regime único é igual a r2_late_model; em multirregime tira o que o regime explica de graça
                         (o R² bruto chega a ~1,00 só porque o regime domina a variância) e fica comparável com o regime único.
- r2_holdout_within_regime  o mesmo, em todos os ciclos dos motores de teste.
- noise_ceiling_late   teto de R² nos ciclos H//2+1 a H: 1 − σ²_ruído / Var(y dentro do regime), com σ² = média de Δy²/2 só entre
                       ciclos CONSECUTIVOS DO MESMO REGIME (o salto de regime não é ruído). Estimado nos motores de teste.
- frac_ceiling         r2_late_within_regime / noise_ceiling_late (NaN se o teto for ≤ 0): quanto do que era possível o modelo pegou.
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


def error_metrics(group: pd.DataFrame) -> tuple[float, float, float]:
    """(MAE, RMSE, NRMSE dentro do regime) de um trace; o NRMSE divide o RMSE pelo desvio dos valores reais em torno da média do próprio regime.

    O ganho sobre a média do regime (1 - MSE / MSE da média do regime) é exatamente o R² dentro do regime (`r2_holdout_within_regime`),
    porque a soma de quadrados em torno da média de cada regime é o erro de quem prevê sempre a média do regime."""
    w = group.dropna(subset=["actual", "prediction"])
    if len(w) < 3:
        return float("nan"), float("nan"), float("nan")
    err = (w["actual"] - w["prediction"]).to_numpy(dtype=float)
    reg = w["regime_id"] if "regime_id" in w else pd.Series(0, index=w.index)
    sd = float(np.sqrt(((w["actual"] - w.groupby(reg)["actual"].transform("mean")) ** 2).mean()))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return float(np.mean(np.abs(err))), rmse, (rmse / sd if sd > 0 else float("nan"))


def _between_engine_pct(window: pd.DataFrame) -> float:
    total = float(((window["actual"] - window["actual"].mean()) ** 2).sum())
    if total <= 0 or window["unit_id"].nunique() < 2:
        return float("nan")
    means = window.groupby("unit_id")["actual"].transform("mean")
    return 100.0 * float(((means - window["actual"].mean()) ** 2).sum()) / total


def _within_regime(window: pd.DataFrame) -> tuple[float, float, float]:
    """(R² dentro do regime, teto de ruído, fração do teto) de uma janela; regime único se não houver regime_id."""
    w = window.dropna(subset=["actual", "prediction"])
    if len(w) < 3:
        return float("nan"), float("nan"), float("nan")
    reg = w["regime_id"] if "regime_id" in w else pd.Series(0, index=w.index)
    sst = float(((w["actual"] - w.groupby(reg)["actual"].transform("mean")) ** 2).sum())
    if sst <= 0:
        return float("nan"), float("nan"), float("nan")
    r2w = 1.0 - float(((w["actual"] - w["prediction"]) ** 2).sum()) / sst
    # ruído: só pares de ciclos consecutivos do mesmo motor E do mesmo regime
    w = w.assign(_reg=reg).sort_values(["unit_id", "cycle"])
    dy = w.groupby("unit_id")["actual"].diff()
    same = (w.groupby("unit_id")["cycle"].diff() == 1) & (w.groupby("unit_id")["_reg"].diff() == 0)
    d = dy[same].dropna()
    if len(d) < 5:
        return r2w, float("nan"), float("nan")
    ceiling = 1.0 - float((d ** 2).mean() / 2.0) / (sst / len(w))
    return r2w, ceiling, (r2w / ceiling if ceiling > 0 else float("nan"))


def _one(group: pd.DataFrame, healthy_limit: int) -> pd.Series:
    cal_end = healthy_limit // 2
    cal, late = group[group["cycle"] <= cal_end], group[(group["cycle"] > cal_end) & (group["cycle"] <= healthy_limit)]
    base = cal.groupby("unit_id")["actual"].mean().rename("base")
    off = (cal["actual"] - cal["prediction"]).groupby(cal["unit_id"]).mean().rename("off")
    late = late.join(base, on="unit_id").join(off, on="unit_id").dropna(subset=["base", "off"])
    r2_w_late, ceiling, frac = _within_regime(late)
    return pd.Series({
        "mae_holdout": error_metrics(group)[0],
        "nrmse_within_regime_holdout": error_metrics(group)[2],
        "r2_holdout": r2(group["actual"], group["prediction"]),
        "r2_healthy": r2(group.loc[group["cycle"] <= healthy_limit, "actual"], group.loc[group["cycle"] <= healthy_limit, "prediction"]),
        "r2_late_model": r2(late["actual"], late["prediction"]),
        "r2_late_motor_base": r2(late["actual"], late["base"]),
        "r2_late_model_offset": r2(late["actual"], late["prediction"] + late["off"]),
        "pct_between_engine": _between_engine_pct(group[group["cycle"] <= healthy_limit]),
        "r2_late_within_regime": r2_w_late,
        "r2_holdout_within_regime": _within_regime(group)[0],
        "noise_ceiling_late": ceiling,
        "frac_ceiling": frac,
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

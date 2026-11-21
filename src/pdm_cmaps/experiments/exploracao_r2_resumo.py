"""Resume a exploração de R² (média das sementes) ao lado do teto de ruído. Lê results/exploracao_r2/{A,B1,B2}_<subset>.csv.

Run:  .venv/bin/python -m pdm_cmaps.experiments.exploracao_r2_resumo
Saída: results/exploracao_r2/resumo_<subset>.csv (uma linha por sensor) e impressão no terminal.
"""
from __future__ import annotations

import pandas as pd

from pdm_cmaps.experiments.exploracao_r2 import OUT, SUBSETS


def resumo(subset: str) -> pd.DataFrame | None:
    teto = pd.read_csv(OUT / "teto_ruido.csv").query("subset == @subset").set_index("sensor")
    out = pd.DataFrame(index=teto.index)
    out["teto_saudavel"] = teto["ceil_late_26_50"].clip(lower=0)
    out["teto_todos"] = teto["ceil_all"]
    for track in ("A", "B1"):
        f = OUT / f"{track}_{subset}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        best = d.groupby(["sensor", "model"]).mean(numeric_only=True).reset_index()
        for model, tag in (("Extra Trees", "ET"), ("Ridge", "Rg")):
            m = best[best.model == model].set_index("sensor")
            if track == "A":
                out[f"A_{tag}_tardio"] = m["r2_late_model_offset"]
                out["A_base_motor"] = m["r2_late_motor_base"]
                out[f"A_{tag}_deteccao"] = m["detection_rate_pct"]
            else:
                out[f"B1_{tag}_tardio"] = m["r2_late_model"]
                out[f"B1_{tag}_todos"] = m["r2_holdout"]
                out[f"B1_{tag}_deteccao"] = m["detection_rate_pct"]
                out[f"B1_{tag}_FA%"] = m["healthy_false_alarm_pct"]
    f = OUT / f"B2_{subset}.csv"
    if f.exists():
        d = pd.read_csv(f)
        g = d.groupby(["sensor", "features", "model"])[["r2_all_cycles", "r2_after_w"]].mean().reset_index()
        out["B2_melhor_todos"] = g.groupby("sensor")["r2_all_cycles"].max()
        top = g.loc[g.groupby("sensor")["r2_all_cycles"].idxmax()].set_index("sensor")
        out["B2_config"] = top["features"] + " / " + top["model"]
        out["B2_ridge_bruto"] = g.query("features == 'bruto' and model == 'Ridge'").set_index("sensor")["r2_all_cycles"]
    out.round(3).to_csv(OUT / f"resumo_{subset}.csv")
    return out


def main() -> int:
    pd.set_option("display.width", 250)
    for s in SUBSETS:
        r = resumo(s)
        print(f"\n=== {s} ===\n{r.round(2).to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def tabela_r2_regime(model: str = "Extra Trees") -> pd.DataFrame:
    """Tabela para a monografia (tcc-writing/tables/sensor_virtual_r2_regime.csv): R² do sensor virtual bruto, dentro do regime, teto de
    ruído e fração do teto, mediana sobre os 15 sensores comuns aos quatro subconjuntos (nos multirregime há canais ambientes extras, como o
    T2, cujo R² é trivial por serem entradas do modelo). Ciclos 26 a 50 dos motores de teste, uma divisão treino-teste (semente 42)."""
    from pdm_cmaps.config import DEFAULT_RESULTS_DIR
    base = DEFAULT_RESULTS_DIR / "virtual_sensor_models"
    scores = {s: pd.read_csv(base / s / "model_scores.csv") for s in SUBSETS_ALL}
    common = set.intersection(*[set(x["sensor"]) for x in scores.values()])
    rows = []
    for s, x in scores.items():
        m = x[(x["model"] == model) & x["sensor"].isin(common)]
        rows.append({"Subset": s, "n_sensores": len(m), "r2_bruto": m["r2_late_model"].median(), "r2_dentro_do_regime": m["r2_late_within_regime"].median(),
                     "teto_de_ruido": m["noise_ceiling_late"].median(), "fracao_do_teto": m["frac_ceiling"].median()})
    out = pd.DataFrame(rows).round(3)
    dest = DEFAULT_RESULTS_DIR.parent / "tcc-writing" / "tables" / "sensor_virtual_r2_regime.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return out


SUBSETS_ALL = ("FD001", "FD002", "FD003", "FD004")


def tabela_exploracao() -> pd.DataFrame:
    """Tabela para a monografia (tcc-writing/tables/sensor_virtual_exploracao_r2.csv): R² por sensor do FD001 e do FD003 em cada trilha.

    A (exógeno), B1 e B1s (outros sensores, treino na janela saudável, sem e com o gêmeo fora): ciclos 26 a 50, Extra Trees, média das
    divisões (A: 3; B1 e B1s: 10). B2 (outros sensores, treino no ciclo de vida inteiro, SEM detecção): todos os ciclos dos motores de teste,
    configuração fixa bruto + centrado com Extra Trees (3 divisões), para não escolher a melhor no teste."""
    from pdm_cmaps.config import DEFAULT_RESULTS_DIR
    rows = []
    teto_all = pd.read_csv(OUT / "teto_ruido.csv")
    for s in ("FD001", "FD003"):
        teto = teto_all[teto_all["subset"] == s].set_index("sensor")
        def trk(f):
            d = pd.read_csv(OUT / f)
            d = d[d["model"] == "Extra Trees"]
            return d.groupby("sensor").agg(r2=("r2_late_model", "mean"), det=("detection_rate_pct", "mean"))
        A, B1, B1s = trk(f"A_{s}.csv"), trk(f"B1_{s}_10s.csv"), trk(f"B1s_{s}_10s.csv")
        b2 = pd.read_csv(OUT / f"B2_{s}.csv").query("features == 'bruto+centrado' and model == 'Extra Trees'").groupby("sensor")["r2_all_cycles"].mean()
        for sn in teto.index:
            rows.append({"Subset": s, "sensor": sn, "teto_ruido_26_50": round(max(float(teto.loc[sn, "ceil_late_26_50"]), 0), 3),
                         "A_r2": A.loc[sn, "r2"], "A_deteccao_pct": A.loc[sn, "det"], "B1_r2": B1.loc[sn, "r2"], "B1_deteccao_pct": B1.loc[sn, "det"],
                         "B1s_r2": B1s.loc[sn, "r2"], "B1s_deteccao_pct": B1s.loc[sn, "det"], "B2_r2_todos_ciclos": b2.loc[sn]})
    out = pd.DataFrame(rows).round(3)
    dest = DEFAULT_RESULTS_DIR.parent / "tcc-writing" / "tables" / "sensor_virtual_exploracao_r2.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return out

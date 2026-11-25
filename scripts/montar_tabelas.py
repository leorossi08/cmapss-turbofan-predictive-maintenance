"""Monta tcc-writing/tables/ a partir de results/ (cópias das tabelas dos estudos e tabelas derivadas das previsões e das execuções).

Uso (da raiz): python scripts/montar_tabelas.py [--diag results/diagnostico_fd003] [--boost results/benchmark_oficial_boosting]
Cópias: as de `pdm_cmaps.cli.update_all.TABLE_COPIES`, mais as do protocolo só-T50, do bootstrap, dos nativos e das falhas sorteadas.
Derivadas: testes pareados por motor do RUL oficial, boosting por regime, largura do conformal na zona crítica, diagnóstico do FD003 e, da busca de
hiperparâmetros, `hiperparametros_escolhidos`, `hiperparametros_janela_por_modelo`, `rul_ajustado_vs_padrao` e `rul_ajustado_metricas_estendidas`.
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdm_cmaps.cli.update_all import TABLE_COPIES  # noqa: E402

RES, TAB = ROOT / "results", ROOT / "tcc-writing" / "tables"
ZONA_CRITICA = 50.0


def _nasa(d):
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def copiar() -> list[str]:
    faltando = []
    pares = {**TABLE_COPIES,
             "contrastes_ic_principal.csv": "injecao_deteccao_ic/principal/contrastes_ic.csv", "poder_ic_principal.csv": "injecao_deteccao_ic/principal/poder_ic.csv",
             "contrastes_ic_so_t50.csv": "injecao_deteccao_ic/so_t50/contrastes_ic.csv", "poder_ic_so_t50.csv": "injecao_deteccao_ic/so_t50/poder_ic.csv",
             "contrastes_ic_extras.csv": "injecao_deteccao_ic/extras/contrastes_ic.csv", "poder_ic_extras.csv": "injecao_deteccao_ic/extras/poder_ic.csv",
             "pareado_far_resumo_offset_so_t50.csv": "injecao_deteccao_offset_so_t50/tabelas/pareado_far_resumo.csv",
             "magnitude_minima_detectavel_offset_so_t50.csv": "injecao_deteccao_offset_so_t50/tabelas/magnitude_minima_detectavel.csv",
             "regra_padrao_s1_offset_so_t50.csv": "injecao_deteccao_offset_so_t50/tabelas/regra_padrao_s1.csv",
             "native_mewma_offset_resumo.csv": "native_mewma_offset/resumo.csv", "native_offset_fora_janela_resumo.csv": "native_offset_fora_janela/resumo.csv",
             "falhas_dificeis_por_subconjunto.csv": "falhas_dificeis/por_subconjunto.csv",
             "rul_oficial_resumo.csv": "benchmark_oficial/resumo_oficial.csv", "rul_oficial_metricas_por_semente.csv": "benchmark_oficial/metricas_por_semente.csv",
             "rul_ajustado_resumo.csv": "benchmark_oficial_tuned/resumo_oficial.csv", "rul_ajustado_metricas_por_semente.csv": "benchmark_oficial_tuned/metricas_por_semente.csv"}
    derivadas = {"margin_check_point_thresholds_raw.csv"}  # montada em `limiares_previsao_pontual`
    for dest, origem in pares.items():
        if dest in derivadas:
            continue
        src = RES / origem
        if src.exists():
            shutil.copy2(src, TAB / dest)
        else:
            faltando.append(f"{dest} <- {origem}")
    diag = [RES / d / "tabelas" / "modelos_pareado_resumo.csv" for d in ("injecao_modelos_diag", "injecao_modelos_diag24")]
    if all(p.exists() for p in diag):
        from pdm_cmaps.experiments.fault_model_study import DIAGNOSTIC_ESTIMATORS  # os estimadores só do diagnóstico do FD003 não entram aqui

        d = pd.concat([pd.read_csv(p) for p in diag])
        d[~d.Estimator.isin(DIAGNOSTIC_ESTIMATORS)].to_csv(TAB / "decomposicao_regime_modelo.csv", index=False)
    else:
        faltando.append("decomposicao_regime_modelo.csv")
    return faltando


def testes_por_motor() -> pd.DataFrame:
    """Erro de cada motor = média sobre as sementes do erro absoluto; diferença (rede - XGBoost com atributos temporais) por motor; Wilcoxon pareado."""
    p = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "benchmark_oficial/*/previsoes_oficial.csv"))]).drop_duplicates(["Subset", "Seed", "Model", "Config", "unit_id"])
    p["ae"] = (p.pred - p.rul_cap).abs()
    e = p.groupby(["Subset", "Model", "Config", "unit_id"]).ae.mean().reset_index()
    ref = e[(e.Model == "XGBoost (+temporal)") & (e.Config == "global_w15")].set_index(["Subset", "unit_id"]).ae
    linhas = []
    for (s, m, c), g in e[e.Model.str.startswith(("PyTorch", "MiniRocket"))].groupby(["Subset", "Model", "Config"]):
        a = g.set_index("unit_id").ae.sort_index()
        r = ref.loc[s].sort_index()
        d = a - r
        linhas.append(dict(Subset=s, Model=m, Config=c, n_motores=len(a), RMSE_rede_menos_xgb_mediana_erro=float(d.median()), p_wilcoxon=float(wilcoxon(a, r).pvalue)))
    return pd.DataFrame(linhas)


def boosting_regime(boost: Path) -> pd.DataFrame:
    linhas = []
    for f in sorted(boost.glob("*/previsoes_oficial.csv")):
        for (s, m, c), g in pd.read_csv(f).groupby(["Subset", "Model", "Config"]):
            d, dr = g.pred - g.rul_cap, g.pred - g.rul_raw
            linhas.append(dict(Subset=s, Model=m, Config=c, RMSE_cap=float(np.sqrt((d ** 2).mean())), RMSE_raw=float(np.sqrt((dr ** 2).mean())), NASA_cap=float(_nasa(d).mean())))
    return pd.DataFrame(linhas)


def limiares_previsao_pontual() -> pd.DataFrame:
    """Custo relativo ao oráculo (razão 10) da previsão pontual (limiar c de 10 a 50) e do limite inferior (c de 10 a 20), com a persistência de dois ciclos."""
    from pdm_cmaps.monitoring.alerts import CRITICAL, NORMAL, debounce, maintenance_cost

    linhas = []
    for f in sorted(glob.glob(str(RES / "monitoring_repeated/runs/*/timelines.csv.gz"))):
        t = pd.read_csv(f)
        s, seed = t.Subset.iloc[0], int(t.Seed.iloc[0])
        t = t[t.Policy == "critical<=10"].sort_values(["unit_id", "cycle"]).reset_index(drop=True)  # a previsão e o limite inferior não dependem da política
        for strat, col, cs in (("point", "RUL_pred", (10, 15, 20, 25, 30, 35, 40, 50)), ("interval", "RUL_lower", (10, 15, 20))):
            for c in cs:
                tl = t[["unit_id", "cycle"]].copy()
                tl["level_x"] = np.concatenate([debounce(np.where(g[col].to_numpy() <= c, CRITICAL, NORMAL), 2) for _, g in t.groupby("unit_id", sort=False)])
                for L in (0, 5, 10):
                    r = maintenance_cost(tl, "x", 10, L)
                    linhas.append(dict(S=s, seed=seed, strat=strat, c=c, L=L, rel=r["Relative_to_oracle"], fail=float(r["Failures"])))
    return pd.DataFrame(linhas)


def largura_conformal() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "monitoring_repeated/runs/*/uncertainty_variants.csv"))])
    m = d[d.Method.str.contains("padrão")]
    g = m.groupby("Subset")[["mean_width", "mean_width_critical_zone"]].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g["Runs"] = m.groupby("Subset").size()
    return g.reset_index().rename(columns={"mean_width_mean": "mean_width_mean", "mean_width_critical_zone_mean": "mean_width_critical_zone_mean"})


def diagnostico(diag: Path) -> pd.DataFrame:
    partes = []
    for h in (25, 50):
        f = diag / f"por_subconjunto_H{h}.csv"
        partes.append(pd.read_csv(f).assign(H=h))
    d = pd.concat(partes)
    return d[(d.Mode == "exogenous") & (d.Chart == "MEWMA")]


def hiperparametros_escolhidos() -> pd.DataFrame:
    """Uma linha por (subconjunto, modelo): janela, representação, RMSE e erro-padrão da validação cruzada, tentativas e os parâmetros do modelo."""
    import json

    linhas = []
    for f in sorted(glob.glob(str(RES / "hiperparametros/FD00*/*/melhores.json"))):
        d = json.loads(Path(f).read_text())
        p = d["params"]
        linhas.append(dict(Subset=d["subset"], Model=d["model"], Janela=p["window"], Temporal=p["temporal"], Representacao=p["representation"],
                           CV_RMSE=d["cv_rmse"], CV_EP=d["cv_se"], Melhor_RMSE_medio=d["melhor_rmse_medio"], Tentativas_completas=d["n_tentativas_completas"],
                           Tentativas_total=d["n_tentativas_total"], Dobras=d["n_splits"], Parametros=json.dumps(p["hyper"], ensure_ascii=False, sort_keys=True)))
    return pd.DataFrame(linhas).sort_values(["Subset", "Model"])


def ajustado_vs_padrao(boost: Path) -> pd.DataFrame:
    """RMSE (truncado e sem truncar) e pontuação NASA média do teste oficial: ajustado (3 sementes) contra o padrão (janela 15, global) e contra a melhor
    configuração anterior (escolhida à mão: janela 30, regime), por modelo. A melhor anterior sai do RMSE truncado entre todas as configurações já rodadas."""
    aj = pd.read_csv(RES / "benchmark_oficial_tuned/resumo_oficial.csv")
    ant = pd.read_csv(RES / "benchmark_oficial/resumo_oficial.csv")
    pool = ant[["Subset", "Model", "Config", "RMSE_cap_mean", "RMSE_raw_mean", "Mean_NASA_cap_mean"]].rename(
        columns={"RMSE_cap_mean": "rmse", "RMSE_raw_mean": "rmse_raw", "Mean_NASA_cap_mean": "nasa"})
    bo = boosting_regime(boost)
    if len(bo):
        bo = bo.assign(Model=bo.Model.str.replace(" (+temporal)", "", regex=False)).rename(columns={"RMSE_cap": "rmse", "RMSE_raw": "rmse_raw", "NASA_cap": "nasa"})
        pool = pd.concat([pool, bo[["Subset", "Model", "Config", "rmse", "rmse_raw", "nasa"]]])
    pad = pool[pool.Config == "global_w15"].set_index(["Subset", "Model"])
    melhor = pool.sort_values("rmse").groupby(["Subset", "Model"]).head(1).set_index(["Subset", "Model"])
    linhas = []
    for _, r in aj.iterrows():
        k = (r.Subset, r.Model)
        linhas.append(dict(Subset=r.Subset, Model=r.Model,
                           Ajustado_RMSE=r.RMSE_cap_mean, Ajustado_RMSE_dp=r.RMSE_cap_std, Ajustado_RMSE_sem_truncar=r.RMSE_raw_mean, Ajustado_NASA_medio=r.Mean_NASA_cap_mean,
                           Padrao_RMSE=pad.rmse.get(k, np.nan), Padrao_RMSE_sem_truncar=pad.rmse_raw.get(k, np.nan), Padrao_NASA_medio=pad.nasa.get(k, np.nan),
                           Melhor_anterior_config=melhor.Config.get(k, ""), Melhor_anterior_RMSE=melhor.rmse.get(k, np.nan), Melhor_anterior_NASA_medio=melhor.nasa.get(k, np.nan),
                           Ganho_sobre_padrao=pad.rmse.get(k, np.nan) - r.RMSE_cap_mean, Ganho_sobre_melhor_anterior=melhor.rmse.get(k, np.nan) - r.RMSE_cap_mean))
    return pd.DataFrame(linhas).sort_values(["Subset", "Ajustado_RMSE"])


def metricas_estendidas_ajustado() -> pd.DataFrame:
    """Métricas além de RMSE/NASA no teste oficial ajustado (média de 3 sementes): viés, erro mediano e P90, acerto em +-10 e +-20 ciclos e, para
    c = 10, 15 e 20 ciclos, precisão, revocação, F1 e PR-AUC do estado crítico (positivo = RUL verdadeira <= c; previsto crítico = previsão <= c)."""
    t = pd.read_csv(RES / "benchmark_oficial_tuned/resumo_oficial.csv")
    cols = ["Subset", "Model", "Bias_cap_mean", "Median_AE_cap_mean", "P90_AE_cap_mean", "Hit_within_10_cap_mean", "Hit_within_20_cap_mean",
            *[f"{m}_c{c}_cap_mean" for c in (10, 15, 20) for m in ("Precision", "Recall", "F1", "PR_AUC")]]
    return t[cols].rename(columns=lambda c: c.replace("_cap_mean", "")).sort_values(["Subset", "Model"])


def janela_por_modelo(h: pd.DataFrame) -> pd.DataFrame:
    """Janela escolhida por modelo nos quatro subconjuntos e quantas vezes bateu no limite do espaço de busca (60 ciclos)."""
    g = h.pivot(index="Model", columns="Subset", values="Janela")
    g["No_limite_60"] = (g[["FD001", "FD002", "FD003", "FD004"]] == 60).sum(axis=1)
    return g.reset_index()


def sensor_virtual_metricas(subsets=("FD001", "FD002", "FD003", "FD004")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MAE, NRMSE (RMSE dividido pelo desvio dentro do regime) e ganho sobre a média do regime (= R² dentro do regime) de cada sensor virtual e modelo,
    recalculados do `trace.csv.gz` de `virtual_sensor_models` (sem refazer os modelos). Devolve (por sensor e modelo, resumo por modelo = mediana dos sensores)."""
    from pdm_cmaps.detection.virtual_sensor_metrics import virtual_sensor_r2
    from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT

    partes = []
    for fd in subsets:
        f = RES / "virtual_sensor_models" / fd / "trace.csv.gz"
        if f.exists():
            partes.append(virtual_sensor_r2(pd.read_csv(f), DEFAULT_HEALTHY_CYCLE_LIMIT).assign(Subset=fd))
    por = pd.concat(partes, ignore_index=True)[["Subset", "sensor", "model", "mae_holdout", "nrmse_within_regime_holdout", "r2_holdout", "r2_holdout_within_regime"]]
    por = por.rename(columns={"r2_holdout_within_regime": "ganho_sobre_media_do_regime"})
    resumo = por.groupby(["Subset", "model"])[["nrmse_within_regime_holdout", "ganho_sobre_media_do_regime", "r2_holdout"]].median().reset_index()
    resumo["n_sensores"] = por.groupby(["Subset", "model"]).size().to_numpy()
    return por, resumo


def alertas_f1() -> pd.DataFrame:
    """Precisão, revocação e F1 de cada estratégia de criticidade, linha a linha nos motores de teste (positivo = nível verdadeiro >= ALERTA; alerta =
    nível previsto >= ALERTA), por política e subconjunto, média das sementes. Recalculado das linhas do tempo (`timelines.csv.gz`)."""
    from pdm_cmaps.monitoring.alerts import CriticalityPolicy, evaluate_strategies

    linhas = []
    for f in sorted(glob.glob(str(RES / "monitoring_repeated/runs/*/timelines.csv.gz"))):
        t = pd.read_csv(f)
        for pol, g in t.groupby("Policy"):
            e = evaluate_strategies(g.drop(columns=["Policy", "Subset", "Seed"]), CriticalityPolicy())
            linhas.append(e[["Strategy", "Alert_precision", "Alert_recall", "Alert_F1"]].assign(Subset=t.Subset.iloc[0], Seed=int(t.Seed.iloc[0]), Policy=pol))
    d = pd.concat(linhas, ignore_index=True)
    return d.groupby(["Subset", "Policy", "Strategy"])[["Alert_precision", "Alert_recall", "Alert_F1"]].mean().reset_index()


def detectores_comparacao() -> pd.DataFrame:
    """Poder e atraso, a falso alarme igualado (0,2 por 100 ciclos, H = 25, início em t0 = 35), da MEWMA com regime e do T50 com regime e sem portão
    (protocolo principal) contra a EWMA clássica de Roberts sobre o resíduo com sinal e o Isolation Forest sobre os 13 resíduos, nas mesmas séries cegas."""
    from pdm_cmaps.experiments.fault_detection_study import DET_MV, DET_T50_NOGATE, EXTRA_DETECTORS

    principal = pd.read_csv(TAB / "pareado_far_resumo.csv")
    extras = pd.read_csv(RES / "injecao_deteccao_extras" / "tabelas" / "pareado_far_resumo.csv")
    d = pd.concat([principal[principal.Detector.isin([DET_MV, DET_T50_NOGATE])], extras[extras.Detector.isin(EXTRA_DETECTORS)]], ignore_index=True)
    d = d[(d.H == 25) & np.isclose(d.budget, 0.2) & (d.t0 == 35) & d.kind.isin(["degrau", "rampa"]) & d.magnitude.round(2).isin([0.05, 0.10, 0.25])]
    cols = ["Subset", "Detector", "kind", "magnitude", "power_pct_mean", "power_pct_std", "delay_med_mean", "far_ctrl_per100_mean", "n_seeds"]
    return d[cols].sort_values(["Subset", "kind", "magnitude", "Detector"]).reset_index(drop=True)


def sem_minirocket() -> None:
    """O MiniRocket saiu da monografia: remove de todas as tabelas as linhas que o citam (os resultados em results/ continuam completos)."""
    for f in sorted(TAB.glob("*.csv")):
        linhas = f.read_text(encoding="utf-8").split("\n")
        manter = [ln for ln in linhas if "ocket" not in ln]
        if len(manter) != len(linhas):
            f.write_text("\n".join(manter), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", type=Path, default=RES / "diagnostico_fd003")
    ap.add_argument("--boost", type=Path, default=RES / "benchmark_oficial_boosting")
    ap.add_argument("--so-metricas-extras", action="store_true", help="só os sensores virtuais (MAE, NRMSE) e o F1 dos alertas, recalculados dos traces e linhas do tempo (Colab)")
    a = ap.parse_args()
    if a.so_metricas_extras:
        TAB.mkdir(parents=True, exist_ok=True)
        por, resumo = sensor_virtual_metricas()
        por.to_csv(TAB / "sensor_virtual_metricas_por_sensor.csv", index=False)
        resumo.to_csv(TAB / "sensor_virtual_metricas_resumo.csv", index=False)
        alertas_f1().to_csv(TAB / "alertas_precisao_f1.csv", index=False)
        return
    TAB.mkdir(parents=True, exist_ok=True)
    faltando = copiar()
    testes_por_motor().to_csv(TAB / "rul_oficial_testes_por_motor.csv", index=False)
    boosting_regime(a.boost).to_csv(TAB / "rul_oficial_boosting_regime.csv", index=False)
    limiares_previsao_pontual().sort_values(["S", "seed", "strat", "c", "L"]).to_csv(TAB / "margin_check_point_thresholds_raw.csv", index=False)
    largura_conformal().to_csv(TAB / "conformal_largura_zona_critica.csv", index=False)
    diagnostico(a.diag).to_csv(TAB / "diagnostico_fd003_ruido_estimacao.csv", index=False)
    h = hiperparametros_escolhidos()
    h.to_csv(TAB / "hiperparametros_escolhidos.csv", index=False)
    janela_por_modelo(h).to_csv(TAB / "hiperparametros_janela_por_modelo.csv", index=False)
    ajustado_vs_padrao(a.boost).to_csv(TAB / "rul_ajustado_vs_padrao.csv", index=False)
    metricas_estendidas_ajustado().to_csv(TAB / "rul_ajustado_metricas_estendidas.csv", index=False)
    if (RES / "injecao_deteccao_extras" / "tabelas" / "pareado_far_resumo.csv").exists():
        detectores_comparacao().to_csv(TAB / "detectores_comparacao.csv", index=False)
    sem_minirocket()
    print(f"{len(list(TAB.glob('*.csv')))} tabelas em {TAB}")
    if faltando:
        print("sem origem:", *faltando, sep="\n  ")


if __name__ == "__main__":
    main()

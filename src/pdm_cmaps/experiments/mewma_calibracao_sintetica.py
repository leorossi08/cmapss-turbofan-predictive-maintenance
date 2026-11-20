"""Calibração do limite do T² da MEWMA com séries saudáveis sintéticas: ela corrige o falso alarme nominal?

PROBLEMA (protocolo principal): o limite é o quantil (1 − α) do T² medido nos próprios ciclos da janela, já com o deslocamento do motor tirado
DELA, onde a média do z é zero por construção. O T² sai subestimado, o limite fica baixo e, depois da janela, o erro de estimar o deslocamento
faz o falso alarme nominal subir (0,20 a 2,64 por 100 ciclos contra 0,01 a 0,06 sem a correção).

TRATAMENTOS (MEWMA de 13 resíduos, mesma divisão por motor e mesmos regimes da camada de monitoramento, H = 50):
    H, limite empírico                  deslocamento nos 50 primeiros ciclos; limite do protocolo principal
    H, limite sintético                 o mesmo ajuste, só o limite muda (calibration="synthetic", W = 50)
    janela 25, empírico dentro          deslocamento nos 25 primeiros ciclos, limite medido nos ciclos 1 a 50
    janela 25, empírico fora            idem, mas μ, Σ e limite só nos ciclos 26 a 50 (a opção que já existia)
    janela 25, limite sintético         W = 25 com o limite sintético

VALIDAÇÃO NÃO CIRCULAR (critério revisado *post hoc*, com o motivo abaixo): os controles independentes (reamostragem iid dos ciclos saudáveis) compartilham a premissa da calibração sintética e
passariam por construção. A validação é feita em DADOS REAIS: nos tratamentos de janela 25, os ciclos 26 a 50 dos motores de teste são saudáveis
pelo protocolo e NÃO entraram na estimativa do deslocamento. Para α em {0,05; 0,01; 0,0027} mede-se a fração desses ciclos acima do limite do
próprio α (curva de calibração): calibrado = fração observada perto de α. Critério fixado ANTES de rodar: razão observado/nominal entre 0,5 e 2
em α = 0,05 e α = 0,01, em cada subconjunto, para o limite sintético; e mais perto de 1 (em escala log) que o melhor limite empírico em ao menos
3 dos 4 subconjuntos. Em α = 0,0027 o número de ciclos é pequeno (poucos eventos esperados): só se reporta, com IC por bootstrap de motores.
REVISÃO POST HOC (2026-11-18, depois do primeiro resultado, FD001 semente 42): o limite sintético pooleia 150 ciclos pós-janela, dominados pelo
regime estacionário, e saiu 55% a 76% acima do empírico, com só 0,3% dos ciclos reais 26 a 50 acima dele em α = 0,05. O diagnóstico mostrou que a
média do T² sintético (21,9) bate com a teoria para ruído i.i.d. (22,9), mas a média real nesses ciclos é 17,0: os ciclos 26 a 50 são o INÍCIO depois
da janela de 25, em que a EWMA (λ = 0,10) ainda carrega a memória da janela (z centrado por construção) e só chega ao estacionário depois de 20 a
30 passos. Validar um transitório contra um limite estacionário reprova um gerador correto. Por isso o estudo reporta DOIS limites sintéticos:
    horizonte   quantil dos 150 ciclos pós-janela: o que o monitoramento longo usa (calibra o falso alarme médio por ciclo ao longo do horizonte)
    casado      quantil só dos mesmos ciclos pós-janela da validação (1 a H − W): compara igual com igual e testa o GERADOR em dados reais
O critério (razão observado/nominal entre 0,5 e 2 em α = 0,05 e 0,01, em cada subconjunto, e mais perto de 1 que o melhor empírico em ao menos 3 de 4)
passa a valer para o limite CASADO; o limite do horizonte é reportado e se espera mais folgado nesses ciclos iniciais. A mudança foi feita
depois de ver o resultado e está registrada aqui por isso.

Nos tratamentos de janela H os ciclos 26 a 50 ESTÃO dentro da janela do deslocamento (centrados nela): esses números não validam nada e não entram.

O QUE ISTO NÃO RESOLVE: calibrar o limite conserta o falso alarme nominal. A antecedência nativa continua sem ser verdade-terreno, porque RUL ≥ 125
não é um rótulo de saúde (a degradação inicial real e o falso alarme se misturam depois da janela).

Run:  .venv/bin/python -m pdm_cmaps.experiments.mewma_calibracao_sintetica [--subsets FD001 ...] [--seeds 42 7 123]
Saída: results/mewma_calibracao_sintetica/{runs/*.csv,motores/*.csv,por_semente.csv,curva_calibracao.csv}
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pdm_cmaps.config import DEFAULT_RESULTS_DIR
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.splits import split_engine_units
from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.experiments.native_offset_fora_janela import (CTRL_ID0, H_NATIVE, W_SPLIT, _control_far, _controls, _late_metrics)

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "mewma_calibracao_sintetica"
ALPHAS = (0.05, 0.01, 0.0027)
# (rótulo, offset_window, calibrate_out_of_window, qual limite: "empirical" | "synthetic", vale como validação em dados reais?)
TREATMENTS = (
    ("H, limite empírico", None, False, "empirical", False),
    ("H, limite sintético", None, False, "synthetic", False),
    ("janela 25, empírico dentro", W_SPLIT, False, "empirical", True),
    ("janela 25, empírico fora", W_SPLIT, True, "empirical", True),
    ("janela 25, limite sintético", W_SPLIT, False, "synthetic", True),
)
FITS = ((None, False), (W_SPLIT, False), (W_SPLIT, True))  # (offset_window, calibrate_out_of_window): três ajustes cobrem os cinco tratamentos


def _ucl_for(mv: MultivariateResidualMonitor, which: str, alpha: float, n_post: int | None = None) -> float:
    """Limite para `alpha`. `n_post` (só no sintético) restringe aos primeiros n ciclos pós-janela (limite "casado")."""
    if which == "synthetic":
        sample = mv.synthetic_t2_ if n_post is None else mv.synthetic_t2_cycles_[:, :n_post].ravel()
        return float(np.quantile(sample, 1 - alpha))
    return float(np.quantile(mv.calibration_t2_["t2"], 1 - alpha))


def run_one(subset: str, seed: int, out: Path, H: int = H_NATIVE) -> pd.DataFrame:
    path = out / "runs" / f"{subset}_seed{seed}.csv"
    if path.exists():
        return pd.read_csv(path)
    frame = CMapssLoader().load(subset)
    frame["rul_true"] = frame.groupby("unit_id")["cycle"].transform("max") - frame["cycle"]
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    train_df, test_df = frame.loc[frame["unit_id"].isin(train_units)], frame.loc[frame["unit_id"].isin(test_units)]
    regimes = RegimeDecoupler(n_clusters=6 if subset.upper() in {"FD002", "FD004"} else 1, random_state=seed).fit(train_df)
    tr, te = regimes.transform(train_df), regimes.transform(test_df)
    rul = te.set_index(["unit_id", "cycle"])["rul_true"]
    eol = te.groupby("unit_id")["cycle"].max()
    ctrl = _controls(test_df, regimes, seed)
    te_all = pd.concat([te, ctrl], ignore_index=True)
    rows, per_engine = [], []
    for window, out_of_window in FITS:
        mv = MultivariateResidualMonitor(healthy_cycle_limit=H, random_state=seed, engine_offset=True, offset_window=window,
                                         calibrate_out_of_window=out_of_window,
                                         calibration="synthetic" if (window, out_of_window) != (W_SPLIT, True) else "empirical")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.fit(tr)
            scored_all = mv.score(te_all)
        real = scored_all["unit_id"] < CTRL_ID0
        scored, c = scored_all[real].copy(), scored_all[~real].sort_values(["unit_id", "cycle"])
        t2_ctrl = c["t2"].to_numpy()
        for label, w, oow, which, valid in TREATMENTS:
            if (w, oow) != (window, out_of_window):
                continue
            if which == "synthetic" and mv.synthetic_t2_ is None:
                continue
            ucl = _ucl_for(mv, which, mv.alpha)
            scored["ucl"], scored["breach"] = ucl, scored["t2"].to_numpy() >= ucl
            mv.ucl_ = ucl  # alarms() usa o limite do monitor
            alarms, _, summ = mv.alarms(scored)
            pc = scored[["unit_id", "cycle", "breach"]].copy()
            pc["rul_true"] = rul.reindex(pd.MultiIndex.from_frame(pc[["unit_id", "cycle"]])).to_numpy()
            summ.pop("n_sensors", None)
            far_ctrl = _control_far(t2_ctrl / ucl, np.ones(len(c), bool), c["unit_id"].to_numpy(), c["cycle"].to_numpy(), H)
            row = {"Subset": subset, "Seed": seed, "Tratamento": label, "valida_em_dados_reais": valid, "ucl": round(ucl, 3),
                   "ucl_ratio_vs_empirico": round(ucl / _ucl_for(mv, "empirical", mv.alpha), 3), **summ, **_late_metrics(pc, alarms, H, eol), "far_ctrl_s1": far_ctrl}
            # curva de calibração nos ciclos reais 26 a 50 (limite do próprio alpha)
            win = (scored["cycle"] > W_SPLIT) & (scored["cycle"] <= H)
            limits = [("horizonte", None)] + ([("casado", H - W_SPLIT)] if which == "synthetic" else [])
            for a in ALPHAS:
                for lim, n_post in limits:
                    u = _ucl_for(mv, which, a, n_post)
                    hit = scored.loc[win, "t2"].to_numpy() >= u
                    row[f"obs_{lim}_a{a}"] = float(hit.mean())
                    g = scored.loc[win, ["unit_id"]].assign(h=hit).groupby("unit_id")["h"].agg(["sum", "size"])
                    per_engine += [{"Subset": subset, "Seed": seed, "Tratamento": label, "limite": lim, "alpha": a, "unit_id": int(u_), "hits": int(r["sum"]),
                                    "cycles": int(r["size"])} for u_, r in g.iterrows()]
            rows.append(row)
        print(f"[{subset} seed {seed}] ajuste janela={window} fora={out_of_window} ok", flush=True)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    (out / "motores").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_engine).to_csv(out / "motores" / f"{subset}_seed{seed}.csv", index=False)
    return df


def calibration_curve(out: Path, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Taxa observada de ciclos 26 a 50 acima do limite de cada alpha, por subconjunto e tratamento, com IC por bootstrap de motores
    (o cluster é o motor de teste de cada divisão; as sementes são pooladas). Só os tratamentos que validam em dados reais."""
    runs = pd.concat([pd.read_csv(p) for p in sorted((out / "runs").glob("*.csv"))], ignore_index=True)
    runs.to_csv(out / "por_semente.csv", index=False)
    eng = pd.concat([pd.read_csv(p) for p in sorted((out / "motores").glob("*.csv"))], ignore_index=True)
    valid = set(runs.loc[runs["valida_em_dados_reais"], "Tratamento"])
    rng = np.random.default_rng(seed)
    rows = []
    for (sub, trat, lim, a), g in eng[eng["Tratamento"].isin(valid)].groupby(["Subset", "Tratamento", "limite", "alpha"]):
        h, n = g["hits"].to_numpy(float), g["cycles"].to_numpy(float)
        obs = h.sum() / n.sum()
        idx = rng.integers(0, len(h), size=(n_boot, len(h)))
        boot = h[idx].sum(axis=1) / n[idx].sum(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({"Subset": sub, "Tratamento": trat, "limite": lim, "alpha": a, "observado": obs, "razao_obs_nominal": obs / a, "ic95_lo": lo, "ic95_hi": hi,
                     "eventos": int(h.sum()), "ciclos": int(n.sum())})
    res = pd.DataFrame(rows).round(5)
    res.to_csv(out / "curva_calibracao.csv", index=False)
    g = runs.groupby(["Subset", "Tratamento"], sort=False).agg(
        limite_ucl=("ucl", "mean"), razao_ucl_vs_empirico=("ucl_ratio_vs_empirico", "mean"), deteccao_pct=("detection_rate_pct", "mean"),
        antecedencia_ciclos=("mean_lead_time_detected_cycles", "mean"), falsos_alarmes_100ciclos_controles=("far_ctrl_s1", "mean"),
        far_ref_pct=("far_ref_pct", "mean"), alarme_prematuro_pct=("early_alarm_pct", "mean"), sementes=("Seed", "nunique")).round(3).reset_index()
    g.to_csv(out / "resumo.csv", index=False)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.aggregate_only:
        jobs = [(s, seed) for s in a.subsets for seed in a.seeds]
        if a.jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=a.jobs)(delayed(run_one)(s, seed, a.output_dir) for s, seed in jobs)
        else:
            for s, seed in jobs:
                run_one(s, seed, a.output_dir)
    res = calibration_curve(a.output_dir)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

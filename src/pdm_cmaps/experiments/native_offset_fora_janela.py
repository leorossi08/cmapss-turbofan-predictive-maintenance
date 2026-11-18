"""Deslocamento do motor estimado FORA da janela de calibração: o falso alarme nativo deixa de ser otimista.

WHAT: nos motores nativos (H = 50, mesma divisão por motor e mesmos regimes da camada de monitoramento) compara, para o T50 EWMA e para a
MEWMA de 13 resíduos, quatro tratamentos do deslocamento do motor:
    sem               nenhum deslocamento removido
    janela H          deslocamento = média do z nos primeiros 50 ciclos (o do texto); o falso alarme da janela saudável é medido nos
                      mesmos ciclos que estimam o deslocamento, onde a média é nula por construção (otimista)
    janela 25, dentro deslocamento nos primeiros 25 ciclos, mas calibração (escala, μ, Σ, limite) e falso alarme em todos os 50
    janela 25, fora   deslocamento nos primeiros 25 ciclos; calibração e falso alarme só nos ciclos 26 a 50 (a calibração vê o mesmo erro de
                      estimativa do deslocamento que o monitoramento depois de H verá)
WHY: calibrar o deslocamento dentro da própria janela de calibração é otimista (ressalva metodológica). As medidas honestas, iguais para os quatro tratamentos, são:
    far_26_50_pct   falso alarme por ciclo nos ciclos 26 a 50 (fora da janela de 25 ciclos, dentro da janela saudável)
    far_ref_pct     falso alarme por ciclo depois de H nos ciclos de motor saudável (RUL verdadeira >= 125)
    early_alarm_pct motores cujo 1º alarme vem com RUL verdadeira >= 125 (alarme prematuro depois da janela)
    far_ctrl_s1     falsos alarmes por 100 ciclos monitorados em controles independentes (reamostragem dos ciclos saudáveis do próprio motor, sem
                    deriva), com o limite NOMINAL de cada detector: mede a calibração sem a confusão da degradação inicial real
Também conta quantas linhas monitoradas de motor multirregime ficam sem correção (menos de 3 linhas do regime na janela do deslocamento).

Run:  .venv/bin/python -m pdm_cmaps.experiments.native_offset_fora_janela [--subsets FD001 ...] [--seeds 42 7 123] [--jobs 3]
Saída: results/native_offset_fora_janela/{runs/*.csv,por_semente.csv,resumo.csv}
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
from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.domain.schema import DEFAULT_RUL_CAP
from pdm_cmaps.experiments.fault_detection_study import HORIZON_CTRL, MIN_POOL, alarm_cycles, bootstrap_series, control_far
from pdm_cmaps.experiments.fault_injection import blind_columns

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "native_offset_fora_janela"
H_NATIVE = 50
W_SPLIT = 25
K = 2
N_CTRL = 6              # controles independentes por motor de teste
CTRL_ID0 = 1_000_000    # unit_id dos controles (não colide com os motores reais)
# (rótulo, engine_offset, offset_window, calibrate_out_of_window)
TREATMENTS = (
    ("sem deslocamento", False, None, False),
    ("deslocamento, janela H", True, None, False),
    ("deslocamento, janela 25, calibração dentro", True, W_SPLIT, False),
    ("deslocamento, janela 25, calibração fora", True, W_SPLIT, True),
)


def _late_metrics(per_cycle: pd.DataFrame, first_alarm: dict, H: int, eol: pd.Series) -> dict:
    """Medidas de falso alarme iguais para todos os tratamentos. ``per_cycle``: unit_id, cycle, breach, rul_true."""
    win = (per_cycle["cycle"] > W_SPLIT) & (per_cycle["cycle"] <= H)
    ref = (per_cycle["cycle"] > H) & (per_cycle["rul_true"] >= DEFAULT_RUL_CAP)
    early = [a is not None and (eol[u] - a) >= DEFAULT_RUL_CAP for u, a in first_alarm.items()]
    return {"far_26_50_pct": 100 * float(per_cycle.loc[win, "breach"].mean()),
            "far_ref_pct": 100 * float(per_cycle.loc[ref, "breach"].mean()) if ref.any() else float("nan"),
            "early_alarm_pct": 100 * float(np.mean(early))}


def _control_far(ratio: np.ndarray, gate: np.ndarray, ids: np.ndarray, cycles: np.ndarray, H: int) -> float:
    """Falsos alarmes por 100 ciclos monitorados (> H) no limite nominal (s = 1) das séries de controle."""
    alarms, last = [], []
    for u in np.unique(ids):
        m = ids == u
        alarms.append(alarm_cycles(ratio[m], gate[m], cycles[m], H + 1, np.array([1.0]), K))
        last.append(cycles[m].max())
    return float(control_far(np.vstack(alarms), np.array(last), H)[0][0])


def _controls(test_df: pd.DataFrame, regimes, seed: int) -> pd.DataFrame:
    """Séries saudáveis independentes (bootstrap iid dos ciclos de cada motor de teste com RUL >= 125), no esquema bruto e já com regime."""
    rng = np.random.default_rng(seed)
    frames, bid = [], CTRL_ID0
    for _, g in test_df.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        if (g["rul_true"] >= DEFAULT_RUL_CAP).sum() < MIN_POOL:
            continue
        for _ in range(N_CTRL):
            bid += 1
            frames.append(blind_columns(bootstrap_series(g, HORIZON_CTRL, rng), bid))
    return regimes.transform(pd.concat(frames, ignore_index=True))


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
    te_all = pd.concat([te, ctrl], ignore_index=True)  # motores reais + controles, pontuados pelo mesmo detector ajustado
    rows = []
    for label, offset, window, out_of_window in TREATMENTS:
        base = {"Subset": subset, "Seed": seed, "H": H, "Tratamento": label}
        vs = VirtualSensor(target_sensor="T50", mode="exogenous", ewma_alpha=0.10, min_consecutive_breaches=K, healthy_cycle_limit=H,
                           random_state=seed, engine_offset=offset, offset_window=window, calibrate_out_of_window=out_of_window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_all, per_all, _ = vs.evaluate(tr, te_all)
        real = trace_all["unit_id"] < CTRL_ID0
        trace, per_engine = trace_all[real], per_all[per_all["unit_id"] < CTRL_ID0]
        summary = _summary(per_engine)
        c = trace_all[~real].sort_values(["unit_id", "cycle"])
        far_ctrl_t50 = _control_far((c["ewma_abs_z"] / c["ewma_ucl"]).to_numpy(), (c["z_score"].abs() >= vs.threshold_sigma).to_numpy(),
                                    c["unit_id"].to_numpy(), c["cycle"].to_numpy(), H)
        alarms = {int(r.unit_id): (None if pd.isna(r.first_alarm_cycle) else int(r.first_alarm_cycle)) for r in per_engine.itertuples()}
        breach = (trace["ewma_abs_z"] >= trace["ewma_ucl"]) & (trace["z_score"].abs() >= vs.threshold_sigma)
        pc = trace[["unit_id", "cycle"]].assign(breach=breach.to_numpy())
        pc["rul_true"] = rul.reindex(pd.MultiIndex.from_frame(pc[["unit_id", "cycle"]])).to_numpy()
        rows.append({**base, "Detector": "T50 EWMA", **summary, **_late_metrics(pc, alarms, H, eol), "far_ctrl_s1": far_ctrl_t50})

        mv = MultivariateResidualMonitor(healthy_cycle_limit=H, random_state=seed, engine_offset=offset, offset_window=window,
                                         calibrate_out_of_window=out_of_window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.fit(tr)
            scored_all = mv.score(te_all)
        real = scored_all["unit_id"] < CTRL_ID0
        scored = scored_all[real]
        c = scored_all[~real].sort_values(["unit_id", "cycle"])
        far_ctrl_mv = _control_far((c["t2"] / c["ucl"]).to_numpy(), np.ones(len(c), bool), c["unit_id"].to_numpy(), c["cycle"].to_numpy(), H)
        mv_alarms, _, mv_summary = mv.alarms(scored)
        pc = scored[["unit_id", "cycle", "breach"]].copy()
        pc["rul_true"] = rul.reindex(pd.MultiIndex.from_frame(pc[["unit_id", "cycle"]])).to_numpy()
        mv_summary.pop("n_sensors", None)
        rows.append({**base, "Detector": "MEWMA (13 resíduos)", **mv_summary, **_late_metrics(pc, mv_alarms, H, eol),
                     "far_ctrl_s1": far_ctrl_mv, "linhas_sem_correcao_pct": mv.offset_uncovered_pct(te) if offset else 0.0})
        print(f"[{subset} seed {seed}] {label}", flush=True)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _summary(per_engine: pd.DataFrame) -> dict:
    """Mesmo resumo que ``VirtualSensor.evaluate`` devolve, só nos motores reais."""
    det = per_engine.loc[per_engine["detected"]]
    return {"total_engines": len(per_engine), "detection_rate_pct": float(per_engine["detected"].mean() * 100.0),
            "mean_lead_time_cycles": float(per_engine["lead_time_cycles"].mean()),
            "mean_lead_time_detected_cycles": float(det["lead_time_cycles"].mean()) if len(det) else 0.0,
            "mean_healthy_false_alarm_pct": float(per_engine["healthy_far"].mean() * 100.0),
            "premature_alarm_pct": float(per_engine["premature_alarm"].mean() * 100.0)}


METRICS = ["detection_rate_pct", "mean_lead_time_detected_cycles", "mean_healthy_false_alarm_pct", "premature_alarm_pct",
           "far_26_50_pct", "far_ref_pct", "early_alarm_pct", "far_ctrl_s1", "linhas_sem_correcao_pct"]


def aggregate(out: Path) -> pd.DataFrame:
    per = pd.concat([pd.read_csv(p) for p in sorted((out / "runs").glob("*.csv"))], ignore_index=True)
    per.to_csv(out / "por_semente.csv", index=False)
    order = {t[0]: i for i, t in enumerate(TREATMENTS)}
    g = per.groupby(["Subset", "Detector", "Tratamento"], sort=False)[METRICS]
    res = g.mean().round(3).add_suffix("_mean").join(g.std().round(3).add_suffix("_std")).reset_index()
    res["n_seeds"] = g.size().to_numpy()
    res = res.sort_values(["Subset", "Detector", "Tratamento"], key=lambda c: c.map(order) if c.name == "Tratamento" else c).reset_index(drop=True)
    res.to_csv(out / "resumo.csv", index=False)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--jobs", type=int, default=3)
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
    res = aggregate(a.output_dir)
    cols = ["Subset", "Detector", "Tratamento", "detection_rate_pct_mean", "mean_lead_time_detected_cycles_mean", "mean_healthy_false_alarm_pct_mean",
            "far_26_50_pct_mean", "far_ref_pct_mean", "early_alarm_pct_mean", "far_ctrl_s1_mean"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

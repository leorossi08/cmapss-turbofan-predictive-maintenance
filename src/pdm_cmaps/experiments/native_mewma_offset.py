"""Detecção da MEWMA nos motores nativos, com e sem o deslocamento do motor.

WHAT: para cada subconjunto e semente (mesma divisão por motor e mesmos regimes da camada de monitoramento, H = 50), ajusta a MEWMA de
13 resíduos com ``engine_offset`` desligado e ligado e mede detecção, antecedência, falso alarme na janela saudável e alarmes prematuros,
como em ``monitoring_repeated_spc_mv_summary.csv``. Sem a RUL, o conformal nem o custo: só a parte da MEWMA.
WHY: o deslocamento do motor já vale no T50; esta é a medida da mesma correção na MEWMA (extensão natural do estudo do T50).

Run:  .venv/bin/python -m pdm_cmaps.experiments.native_mewma_offset [--subsets FD001 ...] [--seeds 42 7 123]
Saída: results/native_mewma_offset/{por_semente.csv,resumo.csv}
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from pdm_cmaps.config import DEFAULT_RESULTS_DIR
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.splits import split_engine_units
from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.domain.rul import compute_piecewise_rul

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "native_mewma_offset"
H_NATIVE = 50


def run_one(subset: str, seed: int, H: int = H_NATIVE) -> list[dict]:
    loader = CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    train_df, test_df = frame.loc[frame["unit_id"].isin(train_units)], frame.loc[frame["unit_id"].isin(test_units)]
    regimes = RegimeDecoupler(n_clusters=6 if subset.upper() in {"FD002", "FD004"} else 1, random_state=seed).fit(train_df)
    tr, te = regimes.transform(train_df), regimes.transform(test_df)
    rows = []
    for offset in (False, True):
        mv = MultivariateResidualMonitor(healthy_cycle_limit=H, random_state=seed, engine_offset=offset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.fit(tr)
            _, _, summary = mv.alarms(mv.score(te))
        rows.append({"Subset": subset, "Seed": seed, "H": H, "engine_offset": offset, **summary})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in a.subsets:
        for seed in a.seeds:
            rows += run_one(s, seed)
            print(f"[{s} seed {seed}] ok", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(a.output_dir / "por_semente.csv", index=False)
    cols = ["detection_rate_pct", "mean_lead_time_detected_cycles", "mean_healthy_false_alarm_pct", "premature_alarm_pct"]
    res = per.groupby(["Subset", "engine_offset"])[cols].mean().round(2).reset_index()
    res.to_csv(a.output_dir / "resumo.csv", index=False)
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

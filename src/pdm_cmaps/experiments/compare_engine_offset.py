"""Compara o sensor virtual com e sem o deslocamento do motor (VirtualSensor(engine_offset=True)).

WHAT: lê duas pastas de runs do estudo de injeção a falso alarme igualado (fault_detection_study: uma com --no-engine-offset e outra
com o padrão) e as duas pastas do estudo do sensor virtual (virtual_sensor_models, também sem e com o deslocamento), e grava duas
tabelas: poder e atraso pareados (mesma semente, forma e magnitude) e R² da janela tardia por sensor e modelo.
WHY: em regime único o modelo vira a média da frota e o deslocamento entre motores domina; este é o teste de melhorar o
sensor virtual do FD003. O detector padrão não muda: a opção fica desligada por padrão.

Run:  .venv/bin/python -m pdm_cmaps.experiments.compare_engine_offset --base-runs DIR [DIR ...] --offset-runs DIR [DIR ...] \
          --vs-base results/virtual_sensor_models --vs-offset results/virtual_sensor_models_offset --out results/sensor_virtual_offset
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

KEYS = ["Detector", "kind", "magnitude", "Subset", "Seed"]
DETECTORS = ["T50 EWMA (com regime)", "T50 EWMA (com regime, sem portão)", "MEWMA (13 resíduos, com regime)"]


def _runs(dirs: list[Path], kind: str = "pareado") -> pd.DataFrame:
    files = [f for d in dirs for f in glob.glob(str(Path(d) / "runs" / f"*_{kind}.csv"))]
    if not files:
        raise SystemExit(f"nenhum *_{kind}.csv em {dirs}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def power_table(base_dirs: list[Path], offset_dirs: list[Path]) -> pd.DataFrame:
    """Poder, falso alarme e atraso pareados por subconjunto e detector; 'celulas_*' conta as células (semente, forma, magnitude)."""
    m = _runs(base_dirs).merge(_runs(offset_dirs), on=KEYS, suffixes=("_base", "_offset"))
    m = m[m["Detector"].isin(DETECTORS)].assign(ganho=lambda d: d["power_pct_offset"] - d["power_pct_base"])
    return m.groupby(["Subset", "Detector"]).agg(
        poder_base=("power_pct_base", "mean"), poder_offset=("power_pct_offset", "mean"), ganho_medio=("ganho", "mean"),
        celulas=("ganho", "size"), celulas_melhores=("ganho", lambda x: int((x > 0).sum())), celulas_piores=("ganho", lambda x: int((x < 0).sum())),
        far_base=("far_ctrl_per100_base", "mean"), far_offset=("far_ctrl_per100_offset", "mean"),
        atraso_base=("delay_med_base", "mean"), atraso_offset=("delay_med_offset", "mean")).round(2).reset_index()


def r2_table(vs_base: Path, vs_offset: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(vs_offset).glob("FD*")):
        o, b = pd.read_csv(d / "model_scores.csv"), pd.read_csv(Path(vs_base) / d.name / "model_scores.csv")
        m = b.merge(o, on=["sensor", "model"], suffixes=("_base", "_offset"))
        for col in ("r2_late_model", "r2_holdout", "detection_rate_pct", "mean_lead_time_detected_cycles", "mean_healthy_false_alarm_pct"):
            m[col + "_delta"] = m[col + "_offset"] - m[col + "_base"]
        rows.append(m.assign(Subset=d.name))
    keep = ["Subset", "sensor", "model", "r2_late_model_base", "r2_late_motor_base_base", "r2_late_model_offset", "r2_holdout_base", "r2_holdout_offset",
            "detection_rate_pct_base", "detection_rate_pct_offset", "mean_lead_time_detected_cycles_base", "mean_lead_time_detected_cycles_offset",
            "mean_healthy_false_alarm_pct_base", "mean_healthy_false_alarm_pct_offset"]
    return pd.concat(rows, ignore_index=True)[keep].round(3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-runs", nargs="+", type=Path, required=True)
    ap.add_argument("--offset-runs", nargs="+", type=Path, required=True)
    ap.add_argument("--vs-base", type=Path, default=Path("results/virtual_sensor_models"))
    ap.add_argument("--vs-offset", type=Path, default=Path("results/virtual_sensor_models_offset"))
    ap.add_argument("--out", type=Path, default=Path("results/sensor_virtual_offset"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    p = power_table(a.base_runs, a.offset_runs)
    p.to_csv(a.out / "comparacao_poder.csv", index=False)
    print(p.to_string(index=False))
    if a.vs_offset.exists():
        r = r2_table(a.vs_base, a.vs_offset)
        r.to_csv(a.out / "comparacao_r2.csv", index=False)
        print(r.groupby(["Subset", "model"])[["r2_late_model_base", "r2_late_model_offset", "r2_late_motor_base_base"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

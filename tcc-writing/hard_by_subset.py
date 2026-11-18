"""Ganho do ML sobre a média do regime por subconjunto (falhas sorteadas; MEWMA e sensor afetado; IC por bootstrap de motores).
Uso (da raiz): python tcc-writing/hard_by_subset.py --results results/falhas_dificeis --out tcc-writing/tables/falhas_dificeis_por_subconjunto.csv"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from pdm_cmaps.experiments import fault_hard_study as fh

ap = argparse.ArgumentParser(); ap.add_argument("--results", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
series, manifest = fh._load(a.results, "series"), fh._load(a.results, "manifesto")
rows = []
for sub in sorted(series["Subset"].unique()):
    d = fh.paired_delta(series[series["Subset"] == sub], manifest[manifest["Subset"] == sub], np.random.default_rng(0))
    rows.append(d.assign(Subset=sub))
out = pd.concat(rows)[["Subset", "Mode", "Estimator", "Chart", "family", "n_faults", "n_clusters", "delta_pp", "ci_lo", "ci_hi"]]
a.out.parent.mkdir(parents=True, exist_ok=True); out.to_csv(a.out, index=False)
pd.set_option("display.width", 220)
print(out[(out.Mode == "exogenous") & (out.Chart == "MEWMA")].round(1).to_string(index=False))

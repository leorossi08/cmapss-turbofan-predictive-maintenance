"""Taxa de recuo do deslocamento do motor (o_{e,r} = 0): fração de pares (motor, regime) com menos de três ciclos nos primeiros H ciclos.

Uso (da raiz): python scripts/taxa_fallback_deslocamento.py [--out tcc-writing/tables/taxa_fallback_deslocamento.csv]
Responde: com H = 25 e seis regimes, quantos motores caem no recuo (o_{e,r} = 0) em algum regime, e o quanto isso muda com H = 50.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdm_cmaps.data.loader import CMapssLoader  # noqa: E402
from pdm_cmaps.detection.regime import RegimeDecoupler  # noqa: E402


def taxa(subset: str, H: int, min_rows: int = 3) -> dict:
    frame = CMapssLoader().load(subset)
    n = 6 if subset in ("FD002", "FD004") else 1
    f = RegimeDecoupler(n_clusters=n).fit(frame).transform(frame) if n > 1 else frame.assign(regime_id=0)
    first = f[f["cycle"] <= H]
    cnt = first.groupby(["unit_id", "regime_id"]).size().unstack(fill_value=0)
    pairs = cnt.size
    low = int((cnt < min_rows).sum().sum())
    return {"Subset": subset, "H": H, "motores": int(cnt.shape[0]), "regimes": int(cnt.shape[1]), "pares_motor_regime": int(pairs),
            "pares_com_menos_de_3_ciclos_pct": 100 * low / pairs, "motores_com_algum_recuo_pct": 100 * float((cnt < min_rows).any(axis=1).mean()),
            "ciclos_por_regime_media": float(cnt.to_numpy().mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "tcc-writing" / "tables" / "taxa_fallback_deslocamento.csv")
    a = ap.parse_args()
    d = pd.DataFrame([taxa(s, H) for s in ("FD002", "FD004") for H in (25, 50)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.round(2).to_csv(a.out, index=False)
    print(d.round(2).to_string(index=False))

"""Teste de robustez da série saudável da injeção cega (revisão da banca, item 3).

O protocolo principal (fault_detection_study.bootstrap_series) gera a série saudável por bootstrap i.i.d.
dos ciclos saudáveis do próprio motor (RUL >= 125). Este script compara-o com um bootstrap em blocos
contíguos de 25 ciclos, que preserva a ordem local (autocorrelação e deriva lenta). Todo o resto fica igual:
detectores, limiar s* a falso alarme igualado (0,2 por 100 ciclos), falhas injetadas, H = 25.

Modos:
  diag  SUBSET SEED            autocorrelação e desvio do T50 padronizado: ciclos saudáveis reais,
                               reamostragem i.i.d. e reamostragem em blocos
  run   METODO SUBSET SEED     injeção cega completa; METODO = iid (protocolo atual), bloco
                               (série inteira em blocos) ou bloco_cal (calibração i.i.d., resto em blocos)
  resumir                      junta as duas execuções em tabelas (poder, falso alarme, s*), pareadas por semente

Uso, a partir da raiz do projeto (no Colab, com o mesmo layout):
  python scripts/teste_bloco_injecao.py diag FD004 42
  python scripts/teste_bloco_injecao.py run bloco FD004 42
  python scripts/teste_bloco_injecao.py resumir
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdm_cmaps.config import DEFAULT_RESULTS_DIR  # noqa: E402
from pdm_cmaps.data.loader import CMapssLoader  # noqa: E402
from pdm_cmaps.domain.schema import DEFAULT_RUL_CAP  # noqa: E402
from pdm_cmaps.experiments import fault_detection_study as fds  # noqa: E402
from pdm_cmaps.experiments.fault_injection import INJECT_SENSORS, load_world, standardize  # noqa: E402

H = 25
BLOCK = 25
LAGS = (1, 5, 25)
SERIES_LEN = 300  # mesmo HORIZON_CTRL do protocolo
RESULTS = DEFAULT_RESULTS_DIR / "teste_bloco"
TABLES = ROOT / "tcc-writing" / "tables"
_ORIG_BOOTSTRAP = fds.bootstrap_series


def bloco_bootstrap(engine: pd.DataFrame, horizon: int, rng: np.random.Generator) -> pd.DataFrame | None:
    """Bootstrap em blocos contíguos de BLOCK ciclos sobre os ciclos saudáveis (RUL >= cap), em ordem."""
    pool = engine.loc[engine["rul_true"] >= DEFAULT_RUL_CAP].sort_values("cycle")
    n = len(pool)
    if n < fds.MIN_POOL:
        return None
    if n < BLOCK:
        return _ORIG_BOOTSTRAP(engine, horizon, rng)
    n_blocks = int(np.ceil(horizon / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, n_blocks)
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:horizon]
    base = pool.iloc[idx].reset_index(drop=True).copy()
    base["cycle"] = np.arange(1, horizon + 1)
    return base


def bloco_cal_bootstrap(engine: pd.DataFrame, horizon: int, rng: np.random.Generator) -> pd.DataFrame | None:
    """Variante de controle: a janela de calibração (primeiros H ciclos) continua i.i.d. sobre todo o histórico
    saudável, como no protocolo; só o restante da série usa blocos contíguos de BLOCK ciclos."""
    pool = engine.loc[engine["rul_true"] >= DEFAULT_RUL_CAP].sort_values("cycle")
    n = len(pool)
    if n < fds.MIN_POOL:
        return None
    if n < BLOCK or horizon <= H:
        return _ORIG_BOOTSTRAP(engine, horizon, rng)
    cal = pool.iloc[rng.integers(0, n, H)]
    rest_len = horizon - H
    n_blocks = int(np.ceil(rest_len / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, n_blocks)
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:rest_len]
    base = pd.concat([cal, pool.iloc[idx]], ignore_index=True)
    base["cycle"] = np.arange(1, horizon + 1)
    return base


def _acf(x: np.ndarray, k: int) -> float:
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0.0 or len(x) <= k:
        return np.nan
    return float(np.sum(x[:-k] * x[k:]) / denom)


def diag(subset: str, seed: int) -> pd.DataFrame:
    world = load_world(subset, seed, H, loader=CMapssLoader())
    rng = np.random.default_rng(seed)
    t50 = INJECT_SENSORS.index("T50")
    engines = {int(u): g.sort_values("cycle") for u, g in world.test.groupby("unit_id")}
    rows = []
    for fonte, sampler in (("real", None), ("iid", _ORIG_BOOTSTRAP), ("bloco", bloco_bootstrap)):
        acfs = {k: [] for k in LAGS}
        sds, n_motores = [], 0
        for _, eng in engines.items():
            if sampler is None:
                frame = eng.loc[eng["rul_true"] >= DEFAULT_RUL_CAP].sort_values("cycle")
                if len(frame) < 60:
                    continue
            else:
                frame = sampler(eng, SERIES_LEN, rng)
                if frame is None:
                    continue
            frame = world.regimes.transform(frame)
            z = standardize(world, frame)[:, t50]
            n_motores += 1
            sds.append(float(np.std(z)))
            for k in LAGS:
                acfs[k].append(_acf(z, k))
        for k in LAGS:
            rows.append({"subset": subset, "seed": seed, "fonte": fonte, "lag": k,
                         "acf_medio": float(np.nanmean(acfs[k])), "n_motores": n_motores})
        rows.append({"subset": subset, "seed": seed, "fonte": fonte, "lag": 0,
                     "acf_medio": np.nan, "n_motores": n_motores, "desvio_padrao_z": float(np.mean(sds))})
    return pd.DataFrame(rows)


def run(metodo: str, subset: str, seed: int) -> None:
    if metodo == "bloco":
        fds.bootstrap_series = bloco_bootstrap
    elif metodo == "bloco_cal":
        fds.bootstrap_series = bloco_cal_bootstrap
    elif metodo != "iid":
        raise SystemExit(f"metodo desconhecido: {metodo}")
    out = RESULTS / metodo
    fds.run_one(subset, seed, H, out)
    print(f"FIM {metodo} {subset} seed {seed}", flush=True)


def resumir() -> None:
    frames = []
    for metodo in ("iid", "bloco", "bloco_cal"):
        out = RESULTS / metodo
        if not (out / "runs").exists():
            continue
        par = fds._load(out, "pareado")
        par = par.assign(metodo=metodo)
        frames.append(par)
    par = pd.concat(frames, ignore_index=True)
    keys = ["metodo", "Subset", "Detector", "kind", "magnitude"]
    g = par.groupby(keys)[["s_star", "far_ctrl_per100", "power_pct"]]
    resumo = g.mean().add_suffix("_media").join(g.std().add_suffix("_dp")).reset_index()
    resumo["n_sementes"] = g.size().to_numpy()
    TABLES.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(TABLES / "teste_bloco_injecao_resumo.csv", index=False)
    par.to_csv(TABLES / "teste_bloco_injecao_por_semente.csv", index=False)
    print(resumo.to_string(), flush=True)


if __name__ == "__main__":
    modo = sys.argv[1]
    if modo == "diag":
        df = diag(sys.argv[2], int(sys.argv[3]))
        RESULTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS / f"diag_{sys.argv[2]}_seed{sys.argv[3]}.csv", index=False)
        print(df.to_string(), flush=True)
    elif modo == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif modo == "resumir":
        resumir()
    else:
        raise SystemExit(__doc__)

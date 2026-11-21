"""Intervalos de confiança do poder de detecção por bootstrap sobre os motores (e das diferenças pareadas entre detectores).

WHAT: refaz as séries cegas do estudo de injeção (mesmas sementes, mesmo ``build_series``, mesmos detectores) e GUARDA, para cada detector, a
matriz de alarmes [séries x limiares] e o manifesto (motor receptor, papel, forma, magnitude, t0). Com isso:
  * ``collect`` grava ``alarmes/<subconj>_seed<s>_H<H>.npz`` + ``_manifesto.csv`` (e confere o poder pareado com o versionado);
  * ``analyse`` faz o bootstrap em cluster por motor receptor (controles e falhas reamostrados JUNTOS, dentro de cada semente), RECALCULA o limiar
    de falso alarme igualado (s*) em cada réplica a partir dos controles reamostrados e tira a média das sementes, como a tabela de poder;
  * as diferenças entre detectores usam as MESMAS réplicas (pareadas), com valor-p do bootstrap e correção de Holm por família de afirmações.
WHY: o texto comparava diferenças de poder com o desvio entre três sementes, que não é a régua certa para uma diferença pareada, e o limiar
estimado em 6 controles por motor tinha incerteza que não era propagada. Aqui a incerteza vem dos motores (e do limiar).
LIMITES: o IC condiciona às três divisões treino/teste (não reamostra a divisão) e trata as sementes como independentes, embora reaproveitem motores;
cobre o erro de amostragem de motores e de limiar, não o modelo da falha injetada.

Run:  .venv/bin/python -m pdm_cmaps.experiments.poder_bootstrap collect [--config principal|so_t50] [--jobs 3]
      .venv/bin/python -m pdm_cmaps.experiments.poder_bootstrap analyse [--config ...] [--boot 4000]
Saída: results/injecao_deteccao_ic/<config>/{alarmes/,poder_ic.csv,contrastes_ic.csv}
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pdm_cmaps.config import DEFAULT_RESULTS_DIR
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.experiments.fault_detection_study import (ADOPTED_H, BUDGETS, DET_IFOREST, DET_MV, DET_RAW_MV, DET_RAW_T50, DET_T50_NOGATE, DET_T50_SIGNED,
                                                         DETECTORS, EXTRA_DETECTORS, alarm_matrix, build_series, extra_scores, score_all)
from pdm_cmaps.experiments.fault_injection import load_world

OUT_DEFAULT = DEFAULT_RESULTS_DIR / "injecao_deteccao_ic"
BUDGET = BUDGETS[0]
CONFIGS = {"principal": dict(engine_offset=True, mewma_offset=True, versioned="injecao_deteccao"),
           "so_t50": dict(engine_offset=True, mewma_offset=False, versioned="injecao_deteccao_offset_so_t50")}
N_BOOT = 4000
SEED_BOOT = 20261118

# Contrastes (A - B) e a família de afirmações de cada um; Holm corrige dentro da família e da configuração.
CONTRASTS = {
    "regime na MEWMA (com - sem)": (DET_MV, DET_RAW_MV),
    "T50 sem portão - MEWMA com regime": (DET_T50_NOGATE, DET_MV),
    "T50 sem portão - T50 bruto": (DET_T50_NOGATE, DET_RAW_T50),
}
# config "extras": a MEWMA contra a EWMA clássica com sinal e o Isolation Forest (degraus e rampas pequenos) e a EWMA clássica contra a do módulo, onde a
# do módulo deve ganhar (variância) e onde deve perder (degrau, rampa, viés de um sensor)
CONTRASTS_EXTRAS = {
    "MEWMA com regime - EWMA clássica com sinal": (DET_MV, DET_T50_SIGNED),
    "MEWMA com regime - Isolation Forest": (DET_MV, DET_IFOREST),
    "EWMA clássica com sinal - EWMA do módulo (T50 sem portão)": (DET_T50_SIGNED, DET_T50_NOGATE),
}
FAMILIES_EXTRAS = {
    "MEWMA com regime - EWMA clássica com sinal": (("degrau", 0.10), ("degrau", 0.05), ("rampa", 0.10)),
    "MEWMA com regime - Isolation Forest": (("degrau", 0.10), ("degrau", 0.05), ("rampa", 0.10)),
    "EWMA clássica com sinal - EWMA do módulo (T50 sem portão)": (("degrau", 0.10), ("rampa", 0.10), ("variancia", 0.5), ("vies_sensor", 1.0)),
}
FAMILIES = {  # contraste -> células (forma, magnitude) da família
    "regime na MEWMA (com - sem)": (("degrau", 0.10), ("degrau", 0.05), ("rampa", 0.10)),
    "T50 sem portão - MEWMA com regime": (("degrau", 0.25), ("rampa", 0.25), ("vies_sensor", 1.0), ("variancia", 1.0)),
    "T50 sem portão - T50 bruto": (("degrau", 0.25), ("vies_sensor", 1.0)),
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ============================================================================
# collect
# ============================================================================

def collect_one(subset: str, seed: int, H: int, out: Path, config: str, n_ctrl: int = 6, n_rep: int = 2) -> Path:
    path = out / config / "alarmes" / f"{subset}_seed{seed}_H{H}.npz"
    if path.exists():
        return path
    cfg = CONFIGS[config]
    world = load_world(subset, seed, H, loader=CMapssLoader())
    blind, meta = build_series(world, seed, n_ctrl, n_rep, None, False)
    scores = score_all(world, blind, seed, cfg["engine_offset"], cfg["mewma_offset"])
    if cfg.get("extras"):
        scores.update(extra_scores(world, blind, seed, cfg["engine_offset"], cfg["mewma_offset"]))
    midx = meta.set_index("blind_id")
    arrays: dict[str, np.ndarray] = {}
    for name, sc in scores.items():
        ids, mat, last = alarm_matrix(sc, np.maximum(midx["t0"], H + 1))
        arrays[f"{_slug(name)}__ids"], arrays[f"{_slug(name)}__mat"], arrays[f"{_slug(name)}__last"] = ids, mat.astype(np.float32), last
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.assign(Subset=subset, Seed=seed, H=H).to_csv(path.with_name(path.stem + "_manifesto.csv"), index=False)
    np.savez_compressed(path, **arrays)
    print(f"[{subset} seed {seed}] {len(meta)} séries, {len(scores)} detectores guardados", flush=True)
    return path


# ============================================================================
# bootstrap
# ============================================================================

class Run:
    """Uma execução (subconjunto, semente) com tudo que o bootstrap precisa, por detector."""

    def __init__(self, npz: Path):
        self.subset, self.seed, self.H = re.match(r"(FD\d{3})_seed(\d+)_H(\d+)", npz.stem).groups()
        self.seed, self.H = int(self.seed), int(self.H)
        meta = pd.read_csv(npz.with_name(npz.stem + "_manifesto.csv")).set_index("blind_id")
        data = np.load(npz)
        self.det: dict[str, dict] = {}
        self.names = [n for n in (*DETECTORS, *EXTRA_DETECTORS) if f"{_slug(n)}__mat" in data.files]
        for name in self.names:
            slug = _slug(name)
            ids, mat, last = data[f"{slug}__ids"], data[f"{slug}__mat"].astype(float), data[f"{slug}__last"]
            m = meta.loc[ids]
            ctrl = (m["role"] == "controle").to_numpy()
            has = ~np.isnan(mat)
            expo = np.where(has, mat - self.H, (last - self.H)[:, None])
            self.det[name] = {"ctrl_has": has[ctrl], "ctrl_expo": expo[ctrl], "fault_has": has[~ctrl], "ctrl_rec": m["recipient"].to_numpy()[ctrl],
                              "fault_rec": m["recipient"].to_numpy()[~ctrl], "groups": m.loc[~ctrl, ["kind", "magnitude", "t0"]].reset_index(drop=True)}
        ref = self.det[DETECTORS[0]]
        self.recipients = np.unique(np.concatenate([ref["ctrl_rec"], ref["fault_rec"]]))
        code = {r: i for i, r in enumerate(self.recipients)}
        keys = ref["groups"].astype(str).agg("|".join, axis=1)
        self.group_keys = pd.unique(keys)
        gcode = {k: i for i, k in enumerate(self.group_keys)}
        self.group_table = ref["groups"].assign(_k=keys).drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
        for d in self.det.values():
            d["ctrl_code"] = np.array([code[r] for r in d["ctrl_rec"]])
            d["fault_code"] = np.array([code[r] for r in d["fault_rec"]])
            d["gcode"] = np.array([gcode[k] for k in d["groups"].astype(str).agg("|".join, axis=1)])

    def power(self, name: str, counts: np.ndarray, budget: float = BUDGET) -> np.ndarray:
        """Poder (%) por grupo de falha no s* mais baixo cujo falso alarme dos controles (ponderados por ``counts``) cabe no orçamento; NaN se nenhum."""
        d = self.det[name]
        wc = counts[d["ctrl_code"]].astype(float)
        events, exposure = wc @ d["ctrl_has"], wc @ d["ctrl_expo"]
        with np.errstate(divide="ignore", invalid="ignore"):
            far = 100 * events / exposure
        ok = np.flatnonzero(far <= budget)
        n_groups = len(self.group_keys)
        if not ok.size:
            return np.full(n_groups, np.nan)
        wf = counts[d["fault_code"]].astype(float)
        det = d["fault_has"][:, ok[0]].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 100 * np.bincount(d["gcode"], wf * det, n_groups) / np.bincount(d["gcode"], wf, n_groups)


def point_power(runs: list[Run], name: str) -> pd.DataFrame:
    """Poder pontual (todos os motores, peso 1) por (subconjunto, semente, forma, magnitude, t0)."""
    rows = []
    for r in runs:
        p = r.power(name, np.ones(len(r.recipients)))
        rows.append(r.group_table.assign(Subset=r.subset, Seed=r.seed, Detector=name, power_pct=p[np.arange(len(r.group_keys))]))
    return pd.concat(rows, ignore_index=True)


def check_against_versioned(runs: list[Run], versioned_dir: Path) -> float:
    """Maior diferença absoluta de poder (pontos) entre as matrizes guardadas e os ``*_pareado.csv`` versionados (esperado: 0)."""
    worst = 0.0
    for name in runs[0].names:
        if name in EXTRA_DETECTORS:
            continue  # não existem no estudo versionado
        mine = point_power(runs, name)
        for (subset, seed), g in mine.groupby(["Subset", "Seed"]):
            old = pd.read_csv(versioned_dir / "runs" / f"{subset}_seed{seed}_H{ADOPTED_H}_pareado.csv")
            old = old[(old["Detector"] == name) & (old["budget"] == BUDGET)][["kind", "magnitude", "t0", "power_pct"]]
            merged = g.merge(old, on=["kind", "magnitude", "t0"], suffixes=("", "_old"))
            assert len(merged) == len(g), f"grupos diferentes em {subset} seed {seed} {name}"
            worst = max(worst, float(np.nanmax(np.abs(merged["power_pct"] - merged["power_pct_old"]))))
    return worst


def bootstrap(runs: list[Run], n_boot: int = N_BOOT, seed: int = SEED_BOOT) -> dict[str, np.ndarray]:
    """Réplicas do poder médio das sementes, por subconjunto: {subset: array [n_boot, n_detectores, n_grupos]} (grupos na ordem de ``group_table``)."""
    rng = np.random.default_rng(seed)
    by_subset: dict[str, list[Run]] = {}
    for r in runs:
        by_subset.setdefault(r.subset, []).append(r)
    out = {}
    for subset, rs in by_subset.items():
        ref_keys = list(rs[0].group_keys)
        assert all(list(r.group_keys) == ref_keys for r in rs), "os grupos de falha diferem entre as sementes"
        names = rs[0].names
        boot = np.full((n_boot, len(names), len(ref_keys)), np.nan)
        for b in range(n_boot):
            per_seed = []
            for r in rs:
                counts = np.bincount(rng.integers(0, len(r.recipients), len(r.recipients)), minlength=len(r.recipients))
                per_seed.append(np.vstack([r.power(name, counts) for name in names]))
            with np.errstate(all="ignore"):
                boot[b] = np.nanmean(np.stack(per_seed), axis=0)
        out[subset] = boot
    return out


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(runs: list[Run], boot: dict[str, np.ndarray], contrasts: dict | None = None, families: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(poder com IC por detector e célula, contrastes pareados com IC, p do bootstrap e p de Holm por família)."""
    power_rows, contrast_rows = [], []
    contrasts = CONTRASTS if contrasts is None else contrasts
    families = FAMILIES if families is None else families
    names = runs[0].names
    didx = {n: i for i, n in enumerate(names)}
    for subset, b in boot.items():
        rs = [r for r in runs if r.subset == subset]
        table = rs[0].group_table
        pts = {n: np.nanmean(np.vstack([r.power(n, np.ones(len(r.recipients))) for r in rs]), axis=0) for n in names}
        for g, cell in table.iterrows():
            for n in names:
                lo, hi = np.nanpercentile(b[:, didx[n], g], [2.5, 97.5])
                power_rows.append({"Subset": subset, "Detector": n, "kind": cell["kind"], "magnitude": cell["magnitude"], "t0": cell["t0"],
                                   "poder_pct": pts[n][g], "ic_lo": lo, "ic_hi": hi, "n_seeds": len(rs)})
            for label, (a, c) in contrasts.items():
                d = b[:, didx[a], g] - b[:, didx[c], g]
                d = d[np.isfinite(d)]
                if not d.size:
                    continue
                p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
                p = max(p, 1.0 / (len(d) + 1))
                contrast_rows.append({"Subset": subset, "Contraste": label, "kind": cell["kind"], "magnitude": cell["magnitude"], "t0": cell["t0"],
                                      "diferenca_pp": pts[a][g] - pts[c][g], "ic_lo": np.percentile(d, 2.5), "ic_hi": np.percentile(d, 97.5), "p_boot": p})
    power, contr = pd.DataFrame(power_rows), pd.DataFrame(contrast_rows)
    # Holm dentro da família (contraste x células listadas x subconjuntos)
    contr["familia"] = False
    contr["p_holm"] = np.nan
    for label, cells in families.items():
        sel = contr["Contraste"].eq(label) & contr[["kind", "magnitude"]].apply(tuple, axis=1).isin([(k, m) for k, m in cells]) & (contr["t0"] == ADOPTED_H + 10)
        contr.loc[sel, "familia"] = True
        if sel.any():
            contr.loc[sel, "p_holm"] = holm(contr.loc[sel, "p_boot"].to_numpy())
    return power, contr


# ============================================================================
# CLI
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["collect", "analyse"])
    ap.add_argument("--config", choices=list(CONFIGS), default="principal")
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(argv)
    if a.action == "collect":
        jobs = [(s, seed) for s in a.subsets for seed in a.seeds]
        if a.jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=a.jobs)(delayed(collect_one)(s, seed, ADOPTED_H, a.output_dir, a.config) for s, seed in jobs)
        else:
            for s, seed in jobs:
                collect_one(s, seed, ADOPTED_H, a.output_dir, a.config)
        return 0
    runs = [Run(p) for p in sorted((a.output_dir / a.config / "alarmes").glob("*.npz"))]
    versioned = DEFAULT_RESULTS_DIR / CONFIGS[a.config]["versioned"]
    if (versioned / "runs").exists():
        print(f"conferência com {versioned.name}: maior diferença de poder = {check_against_versioned(runs, versioned):.6f} ponto")
    ex = a.config == "extras"
    power, contr = summarize(runs, bootstrap(runs, a.boot), CONTRASTS_EXTRAS if ex else None, FAMILIES_EXTRAS if ex else None)
    power.round(3).to_csv(a.output_dir / a.config / "poder_ic.csv", index=False)
    contr.round(4).to_csv(a.output_dir / a.config / "contrastes_ic.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(contr[contr["familia"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

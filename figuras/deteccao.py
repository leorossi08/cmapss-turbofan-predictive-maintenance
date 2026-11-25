"""Figuras 5.12 e 5.13 refeitas com destaques e rótulos diretos (curvas operacionais e detecção acumulada), fundo branco.

Lê as saídas do estudo `fault_detection_study` (results/injecao_deteccao/runs), que precisam ser regeradas para o FD004:
    .venv/bin/python -m pdm_cmaps.experiments.fault_detection_study --subsets FD004 --windows 25 --jobs 3
Uso: python figuras/gerar.py N   (N = número da figura no PDF)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdm_cmaps.experiments import fault_detection_study as fds  # noqa: E402
from pdm_cmaps.reporting import viz_style as vs  # noqa: E402

FIG = Path(__file__).resolve().parent / "saida"
H, SUBSET, BUDGET = 25, "FD004", 0.2
SHORT = {fds.DET_MV: "MEWMA com regime", fds.DET_RAW_MV: "MEWMA sem regime", fds.DET_T50_NOGATE: "T50 com regime, sem portão",
         fds.DET_RAW_T50: "T50 sem regime", fds.DET_T50: "T50 com regime, com portão"}
STY = {fds.DET_MV: (vs.SERIES[1], "-", "o"), fds.DET_RAW_MV: (vs.SERIES[1], "--", "s"), fds.DET_T50_NOGATE: (vs.SERIES[2], "-", "^"),
       fds.DET_RAW_T50: (vs.SERIES[0], "--", "D"), fds.DET_T50: (vs.SERIES[0], "-", "v")}
CASES = [("degrau", 0.10), ("rampa", 0.10), ("degrau", 0.25)]  # com o deslocamento em todos os detectores o degrau de 0,25 satura na MEWMA: os casos informativos são os de 0,10


def operating(cur: pd.DataFrame, par: pd.DataFrame) -> None:
    vs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.9), sharey=True)
    for ax, (kind, mag) in zip(axes, CASES):
        for det in fds.DETECTORS:
            d = cur[(cur["Subset"] == SUBSET) & (cur["H"] == H) & (cur["kind"] == kind) & np.isclose(cur["magnitude"], mag) & (cur["t0"] == H + 10) & (cur["Detector"] == det)]
            m = d.groupby("s")[["far_ctrl_per100", "power_pct"]].mean().sort_index()
            col, ls, mk = STY[det]
            ax.plot(m["far_ctrl_per100"].clip(lower=1e-3), m["power_pct"], color=col, ls=ls, lw=2.2, label=SHORT[det])
            row = par[(par["Subset"] == SUBSET) & (par["H"] == H) & (par["kind"] == kind) & np.isclose(par["magnitude"], mag) & (par["Detector"] == det) & (par["budget"] == BUDGET) & (par["t0"] == H + 10)]
            if len(row):
                pw = float(row["power_pct"].mean())
                ax.scatter([BUDGET], [pw], s=70, color=col, marker=mk, zorder=6, edgecolor=vs.SURFACE, linewidth=1.5)
        ax.axvline(BUDGET, color=vs.INK_2, lw=1.4, ls=":")
        ax.set_xscale("log"), ax.set_xlim(2e-3, 10), ax.set_ylim(0, 102)
        ax.set_title(f"{kind} de {f"{mag:.2f}".replace('.', ',')} da assinatura", fontsize=12.5, loc="left")
        ax.set_xlabel("falsos alarmes por 100 ciclos (controles)", fontsize=11)
        ax.tick_params(labelsize=10.5)
        vs.style_axes(ax, grid="both")
        if ax is axes[0]:  # a seta de "melhor" só no primeiro painel: nos outros cruzaria as curvas
            ax.annotate("melhor", xy=(0.0028, 97), xytext=(0.0034, 66), fontsize=11.5, color="#12805a", fontweight="bold", ha="left", arrowprops=dict(arrowstyle="-|>", color=vs.STATUS[0], lw=2))
    axes[0].set_ylabel("poder de detecção (%)", fontsize=11)
    # a legenda mostra também o marcador de cada detector (o ponto grande no orçamento de falso alarme)
    h = [plt.Line2D([], [], color=STY[d][0], ls=STY[d][1], lw=2.2, marker=STY[d][2], ms=8, markeredgecolor=vs.SURFACE) for d in fds.DETECTORS]
    l = [SHORT[d] for d in fds.DETECTORS]
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=10.5, frameon=False)
    fig.suptitle("Curvas operacionais: cada ponto grande é o detector operado no orçamento de falso alarme", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(FIG / "F11_curvas_operacionais.png", dpi=170, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def cumulative(dl: pd.DataFrame) -> None:
    vs.apply_style()
    cases = CASES  # mesmos casos e mesma ordem das curvas operacionais
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.9), sharey=True)
    grid = np.arange(0, fds.POST_DET + 1)
    for ax, (kind, mag) in zip(axes, cases):
        for det in fds.DETECTORS:
            d = dl[(dl["Subset"] == SUBSET) & (dl["H"] == H) & (dl["kind"] == kind) & np.isclose(dl["magnitude"], mag) & (dl["t0"] == H + 10) & (dl["budget"] == BUDGET) & (dl["Detector"] == det)]
            if d.empty:
                continue
            x = d["delay"].to_numpy()
            col, ls, _ = STY[det]
            curve = np.array([100 * np.mean(x <= g) for g in grid])
            ax.step(grid, curve, where="post", color=col, ls=ls, lw=2.2, label=SHORT[det])
            hit = np.flatnonzero(curve >= 50)
            if len(hit):
                ax.scatter([grid[hit[0]]], [50], s=50, color=col, zorder=6, edgecolor=vs.SURFACE, linewidth=1.5)
        ax.axhline(50, color=vs.INK_2, lw=1, ls=":")
        # o rótulo fica à direita, na faixa livre acima da linha de 50%; só na rampa de 0,10 a curva da MEWMA com regime sobe por ali, então vai à esquerda
        if not (kind.startswith('ramp') and mag < 0.2):
            ax.text(fds.POST_DET - 1, 51.5, "50% detectadas", ha="right", va="bottom", fontsize=10.5, color=vs.INK_2)
        else:
            ax.text(1, 52.5, "50% detectadas", ha="left", va="bottom", fontsize=10.5, color=vs.INK_2)
        ax.set_title(f"{kind} de {f"{mag:.2f}".replace('.', ',')} da assinatura", fontsize=12.5, loc="left")
        ax.set_xlabel("ciclos após o início da falha (t0)", fontsize=11), ax.set_ylim(0, 102)
        ax.tick_params(labelsize=10.5)
        vs.style_axes(ax, grid="both")
    axes[0].set_ylabel("séries detectadas até esse atraso (%)", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=10.5, frameon=False)
    fig.suptitle("Detecção acumulada: quanto mais rápida a curva sobe, menor o atraso (0,2 falsos alarmes por 100 ciclos)", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(FIG / "F12_deteccao_acumulada.png", dpi=170, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "injecao_deteccao")
    a = ap.parse_args()
    cur, par, dl = fds._load(a.runs, "curvas"), fds._load(a.runs, "pareado"), fds._load(a.runs, "atrasos")
    operating(cur, par)
    cumulative(dl)
    print("ok")

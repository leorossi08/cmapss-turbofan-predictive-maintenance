"""Exemplos visuais da injeção cega (Seção 4.9): série saudável reamostrada e formas de falha.

  F21a  de um motor real a uma série saudável (bootstrap dos ciclos com RUL >= 125) e ao seu gêmeo com falha
  F21b  as cinco formas de falha, como o deslocamento somado ao T50 (em σ saudável)
Usa as funções de `fault_injection` e `fault_detection_study` só para leitura (sem alterá-las).
Uso: python figuras/gerar.py N   (N = número da figura no PDF)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdm_cmaps.experiments import fault_detection_study as fds  # noqa: E402
from pdm_cmaps.experiments import fault_injection as fi  # noqa: E402
from pdm_cmaps.reporting import viz_style as vs  # noqa: E402

FIG = Path(__file__).resolve().parent / "saida"
H, T0, SUBSET, SEED = 25, 35, "FD004", 42


def main() -> None:
    vs.apply_style()
    world = fi.load_world(SUBSET, SEED, H)
    j = fi.INJECT_SENSORS.index("T50")
    test = world.test
    unit = int(sorted(test["unit_id"].unique())[3])
    eng = test[test.unit_id == unit].sort_values("cycle")
    rng = np.random.default_rng(SEED)
    base = fds.bootstrap_series(eng, 120, rng)
    ctrl = base.copy()
    dz = fi.delta_z(world, base, "degrau", 0.25, T0, rng)
    faulty = fi.inject(world, base, dz)

    # ---- F21a ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.6))
    ax = axes[0]
    ax.scatter(eng["cycle"], eng["T50"], s=8, color=vs.MUTED)
    healthy = eng[eng["rul_true"] >= 125]
    ax.scatter(healthy["cycle"], healthy["T50"], s=10, color=vs.SERIES[0], label="ciclos saudáveis (RUL real ≥ 125)")
    ax.set_title(f"1. Motor {unit} do {SUBSET} (holdout): T50 bruto; em azul, a vida saudável", fontsize=10.5, loc="left")
    ax.set_xlabel("ciclo original"), ax.legend(loc="lower left", fontsize=9)
    ax = axes[1]
    ax.scatter(ctrl["cycle"], ctrl["T50"], s=8, color=vs.SERIES[0])
    ax.set_title(f"2. Série saudável (motor {unit} do {SUBSET}): ciclos saudáveis sorteados com reposição e renumerados (controle)", fontsize=10.5, loc="left")
    ax.set_xlabel("ciclo da série")
    ax = axes[2]
    diff = faulty["T50"].to_numpy() - ctrl["T50"].to_numpy()
    ax.plot(ctrl["cycle"], diff, color=vs.STATUS[3], lw=2)
    ax.axvline(T0, color=vs.INK_2, lw=1.4, ls="--")
    ax.axvspan(0, H, color=vs.GRID, alpha=0.6)
    ax.text(T0 + 1, diff.max() * 0.55, f"início da falha t0 = {T0}\n(oculto dos detectores)", va="center", fontsize=9.5, color=vs.INK_2, fontweight="bold")
    ax.text(H / 2, diff.max() * 0.9, "calibração", ha="center", fontsize=9, color=vs.INK_2)
    ax.set_title(f"3. Gêmeo com falha menos controle (motor {unit} do {SUBSET}): zero antes de t0; depois, deslocamento pequeno", fontsize=10.5, loc="left")
    ax.set_xlabel("ciclo da série")
    axes[2].set_ylabel("diferença no T50 (°R)")
    for a in axes[:2]:
        a.set_ylabel("T50 (°R)")
    for a in axes:
        vs.style_axes(a)
    axes[0].sharey = None
    fig.tight_layout()
    fig.savefig(FIG / "F21a_injecao_serie_saudavel.png", dpi=170)
    plt.close(fig)

    # ---- F21b ----
    cyc = base["cycle"].to_numpy()
    shapes = [("degrau", 0.25, "Degrau\n(25% da assinatura)"), ("rampa", 0.25, "Rampa de 40 ciclos\n(25% da assinatura)"), ("vies_sensor", 2.0, "Viés em um sensor (T50)\n(2σ)"),
              ("variancia", 1.0, "Variância aumentada\n(1σ, todos os sensores)"), ("um_regime", 0.50, "Rampa restrita a um regime\n(50% da assinatura)")]
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.6), sharey=True)
    for ax, (kind, mag, title) in zip(axes, shapes):
        d = fi.delta_z(world, base, kind, mag, T0, np.random.default_rng(1))[:, j]
        ax.plot(cyc, d, color=vs.STATUS[3] if kind != "variancia" else vs.SERIES[1], lw=1.6)
        ax.axvline(T0, color=vs.INK_2, lw=1, ls="--")
        ax.set_title(title, fontsize=10.5, loc="left"), ax.set_xlabel("ciclo", fontsize=10.5)
        ax.tick_params(labelsize=9.5)
        vs.style_axes(ax, grid="y")
    axes[0].set_ylabel("deslocamento somado\nao T50 (σ saudável)", fontsize=10.5)
    fig.suptitle("As cinco formas de falha injetadas (tracejado = início t0)", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "F21b_injecao_formas_de_falha.png", dpi=170)
    plt.close(fig)
    print("ok", unit)


if __name__ == "__main__":
    main()

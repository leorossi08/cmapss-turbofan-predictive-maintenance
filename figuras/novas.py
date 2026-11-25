"""Figuras 1.1 e 5.20, feitas só com dados já versionados (nenhum estudo é rodado).

F24  Figura 1.1   T50 bruto e resíduo do sensor virtual do mesmo motor (results/maintenance_guide_fd004/tables)
F26  Figura 5.20  custo relativo ao oráculo por limiar crítico (tcc-writing/tables/margin_check_point_thresholds_raw.csv)
Uso: python figuras/gerar.py N   (N = número da figura no PDF)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdm_cmaps.reporting import viz_style as vs  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG, TAB, RES = HERE / "saida", ROOT / "tcc-writing" / "tables", ROOT / "results"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
MOTOR, JANELA = 88, 50  # o mesmo motor da figura dos desvios do envelope; janela saudável do pipeline nativo


def _virgula(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


def bruto_vs_residuo() -> None:
    d = pd.read_csv(RES / "maintenance_guide_fd004" / "tables" / "virtual_sensor_residuals_t50_h50.csv")
    d = d[d["unit_id"] == MOTOR].sort_values("cycle")
    vs.apply_style()
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True)
    a.scatter(d["cycle"], d["T50"], s=9, color=vs.SERIES[0])
    a.set_ylabel("T50 medido (°R)", fontsize=10)
    a.set_title("Sinal bruto: o que se vê é o regime de voo (cada faixa é um regime)", fontsize=10.5, loc="left", fontweight="bold")
    b.axvspan(d["cycle"].min(), JANELA, color="#ecebe5", lw=0)
    b.text(JANELA / 2, 0.93, "calibração", transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=9, color=vs.INK_2)
    b.axhline(0, color=vs.INK_2, lw=1)
    b.scatter(d["cycle"], d["residual"], s=9, color="#b9b7ae", label="resíduo de cada voo")
    b.plot(d["cycle"], d["residual"].rolling(15, min_periods=5).mean(), color=vs.SERIES[1], lw=2.4, label="média dos 15 voos anteriores")
    b.set_ylabel("T50 medido − esperado (°R)", fontsize=10), b.set_xlabel("ciclo (voo)", fontsize=10)
    b.set_title("Depois de remover o regime: aparece o desgaste (escala vertical ampliada)", fontsize=10.5, loc="left", fontweight="bold")
    b.legend(loc="upper left", bbox_to_anchor=(0.16, 0.99), fontsize=9, frameon=False)
    for ax in (a, b):
        ax.tick_params(labelsize=9.5)
        vs.style_axes(ax)
    fig.suptitle(f"Motor {MOTOR} do FD004 (holdout, sem descontinuidade)", fontsize=11.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "F24_bruto_vs_residuo.png", dpi=200)
    plt.close(fig)




def custo_por_limiar() -> None:
    d = pd.read_csv(TAB / "margin_check_point_thresholds_raw.csv")
    m = d.groupby(["S", "L", "strat", "c"])["rel"].mean().reset_index()
    est = {"point": ("Previsão pontual", vs.SERIES[0], "o"), "interval": ("Limite inferior do intervalo", vs.SERIES[1], "s")}
    prazos = (0, 10)
    vs.apply_style()
    fig, axes = plt.subplots(len(prazos), len(SUBSETS), figsize=(8.4, 5.6), sharex=True, sharey=True)
    for i, L in enumerate(prazos):
        for j, sub in enumerate(SUBSETS):
            ax = axes[i, j]
            for k, (lab, col, mk) in est.items():
                g = m[(m["S"] == sub) & (m["L"] == L) & (m["strat"] == k)].sort_values("c")
                ax.plot(g["c"], g["rel"], color=col, marker=mk, ms=4.5, lw=1.8, label=lab)
                best = g.loc[g["rel"].idxmin()]
                ax.scatter([best["c"]], [best["rel"]], s=95, facecolor="none", edgecolor=col, linewidth=1.8, zorder=5)
                acima = k == "interval"
                ax.annotate(_virgula(best["rel"]), (best["c"], best["rel"]), xytext=(0, 8 if acima else -8), textcoords="offset points",
                            ha="center", va="bottom" if acima else "top", fontsize=8.5, color=col, fontweight="bold", zorder=7,
                            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9))  # fundo branco: a curva não cobre o rótulo
            ax.set_yscale("log"), ax.set_ylim(0.8, 10), ax.set_xlim(3, 54)
            ax.set_yticks([1, 1.5, 2, 3, 5, 8]), ax.set_yticklabels(["1", "1,5", "2", "3", "5", "8"])
            ax.minorticks_off()
            ax.set_xticks([10, 20, 30, 40, 50]), ax.tick_params(labelsize=9)
            if i == 0:
                ax.set_title(sub, fontsize=10.5, loc="left", fontweight="bold")
            vs.style_axes(ax)
        axes[i, 0].set_ylabel(("sem prazo de logística" if L == 0 else f"prazo de {L} ciclos") + "\ncusto relativo ao oráculo", fontsize=9.5)
    axes[-1, 0].set_xlabel("limiar crítico c (ciclos de RUL)", fontsize=9, loc="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], ls="", marker="o", markersize=10, markerfacecolor="none", markeredgecolor=vs.INK_2, markeredgewidth=1.6))
    l.append("menor custo de cada estratégia")
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle("Custo por limiar crítico (razão de custo 10; escala logarítmica; 1 = oráculo)\no limite inferior só foi avaliado com limiares de 10 a 20",
                 fontsize=10.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(FIG / "F26_custo_por_limiar.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    bruto_vs_residuo()
    custo_por_limiar()
    print("ok")

"""Figuras da injeção cega e da criticidade, feitas das tabelas de `tcc-writing/tables/` e de `results/`, sem rodar estudos.

F06/F07  Figuras 5.2 e 5.18  comparação de modelos com IC e trajetórias de RUL com intervalo (rótulos no vocabulário do texto)
F18      Figura B.1          ganho de poder de cada modelo sobre a média do regime
F23      Figura 5.13         decomposição do ganho no degrau de 0,10, com o deslocamento do motor em todos os detectores
F08      Figura B.2          compromisso de criticidade com as quatro estratégias do texto
F27      Figura 5.19         conformal do modelo ajustado (CQR)
Uso: python figuras/gerar.py N   (N = número da figura no PDF; ver MANIFESTO.md)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdm_cmaps.experiments import fault_detection_study as fds  # noqa: E402
from pdm_cmaps.reporting import viz_style as vs  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG, TAB = HERE / "saida", ROOT / "tcc-writing" / "tables"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
POLICY = "critical<=15"
# (rótulo como na tabela de custo, marcador, cor, vazado)
STRATEGIES = {
    "point": ("Previsão pontual", "o", vs.SERIES[0], False),
    "interval": ("Limite inferior", "s", vs.SERIES[1], False),
    "interval+anomaly": ("Limite inferior + T50", "s", vs.SERIES[2], True),
    "interval+anomaly_mv": ("Limite inferior + MEWMA", "D", "#52514e", False),
}


H, BUDGET = 25, 0.2
DESC = {"FD001": "1 regime\n1 modo de falha", "FD002": "6 regimes\n1 modo de falha", "FD003": "1 regime\n2 modos de falha", "FD004": "6 regimes\n2 modos de falha"}








def decomposicao() -> None:
    """Poder da MEWMA no degrau de 0,10 (o de 0,25 satura em 100 %) com três versões do sinal, com o deslocamento do motor em todos os detectores."""
    par = pd.read_csv(TAB / "pareado_far_resumo.csv")
    dec = pd.read_csv(TAB / "decomposicao_regime_modelo.csv")
    sel = lambda d: d[(d["H"] == H) & (d["budget"] == BUDGET) & (d["kind"] == "degrau") & (d["magnitude"] == 0.10)]
    par, dec = sel(par[par["t0"] == H + 10]), sel(dec)
    series = [("Sensores brutos (sem regime)", par[par["Detector"] == fds.DET_RAW_MV], "#b9b7ae"),
              ("Média do regime", dec[(dec["Estimator"] == "Média do regime (sem ML)") & dec["Detector"].str.startswith("MEWMA")], vs.SERIES[0]),
              ("Extra Trees (sensor virtual)", dec[(dec["Estimator"] == "Extra Trees (padrão)") & dec["Detector"].str.startswith("MEWMA")], vs.SERIES[1])]
    vs.apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    w = 0.26
    for k, (lab, d, col) in enumerate(series):
        d = d.set_index("Subset").loc[list(SUBSETS)]
        x = [i + (k - 1) * w for i in range(len(SUBSETS))]
        ax.bar(x, d["power_pct_mean"], width=w * 0.92, color=col, label=lab, yerr=d["power_pct_std"], error_kw=dict(ecolor=vs.INK_2, elinewidth=1, capsize=2.5))
        for xi, v, e in zip(x, d["power_pct_mean"], d["power_pct_std"]):
            ax.text(xi, v + e + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=vs.INK)
    ax.set_xticks(range(len(SUBSETS))), ax.set_xticklabels([f"{s}\n{DESC[s]}" for s in SUBSETS], fontsize=9.5)
    ax.set_ylim(0, 118), ax.set_yticks([0, 20, 40, 60, 80, 100]), ax.set_ylabel("poder de detecção (%)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9.5)
    vs.style_axes(ax)
    ax.legend(loc="upper center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("De onde vem o ganho: poder da MEWMA de 13 sinais em um degrau de 0,10\nnível de cada motor corrigido; mesmo falso alarme (0,2/100 ciclos); barras = desvio entre sementes",
                 fontsize=10.2, fontweight="bold", x=0.015, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "F23_decomposicao_ganho.png", dpi=200)
    plt.close(fig)


def criticidade() -> None:
    d = pd.read_csv(TAB / "monitoring_repeated_summary.csv")
    d = d[d["Policy"] == POLICY]
    vs.apply_style()
    # grade 2 x 2: na largura da página, quatro painéis lado a lado ficavam pequenos demais para ler
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.4), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, sub in zip(axes, SUBSETS):
        vs.style_axes(ax, grid="both")
        for key in ("interval+anomaly", "interval+anomaly_mv", "point", "interval"):  # o vazado primeiro, para não esconder o cheio
            _, mk, col, hollow = STRATEGIES[key]
            r = d[(d["Subset"] == sub) & (d["Strategy"] == key)].iloc[0]
            ax.errorbar(100 * r["False_alarm_rate_normal_mean"], 100 * r["Missed_critical_rate_mean"],
                        xerr=100 * r["False_alarm_rate_normal_std"], yerr=100 * r["Missed_critical_rate_std"],
                        fmt=mk, color=col, ecolor=col, elinewidth=1.4, capsize=3, markersize=14 if hollow else 9, clip_on=False,
                        markerfacecolor=vs.SURFACE if hollow else col, markeredgecolor=col if hollow else vs.SURFACE, markeredgewidth=2.0)
        ax.set_title(sub, fontsize=12, loc="left")
        ax.tick_params(labelsize=10.5)
    for ax in axes[2:]:
        ax.set_xlabel("falsos alertas em ciclos normais (%)", fontsize=11)
    for ax in axes[::2]:
        ax.set_ylabel("críticos não alertados (%)", fontsize=11)
    xtop = 100 * (d["False_alarm_rate_normal_mean"] + d["False_alarm_rate_normal_std"]).max()
    ytop = 100 * (d["Missed_critical_rate_mean"] + d["Missed_critical_rate_std"]).max()
    axes[0].set_xlim(-0.06 * xtop, 1.08 * xtop), axes[0].set_ylim(-0.08 * ytop, 1.08 * ytop)
    handles = [Line2D([], [], ls="", marker=mk, markersize=11 if hollow else 9, markerfacecolor=vs.SURFACE if hollow else col,
                      markeredgecolor=col if hollow else vs.SURFACE, markeredgewidth=2.0, label=lab) for lab, mk, col, hollow in STRATEGIES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11, frameon=False)
    fig.suptitle("Compromisso das estratégias: mais perto do canto inferior esquerdo é melhor\n(média ± desvio entre sementes; crítico ≤ 15 ciclos)",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(FIG / "F08_criticidade_tradeoff.png", dpi=170)
    plt.close(fig)


RES = ROOT / "results"
NOMES_MODELOS = {"PyTorch CNN": "CNN", "PyTorch RNN (LSTM)": "LSTM", "PyTorch Transformer": "Transformer", "XGBoost (NASA Loss)": "XGBoost (perda NASA)",
                 "Support Vector Machine (SVM)": "SVM linear", "MiniRocket + RidgeCV": "MiniRocket + Ridge", "Random Forest": "Random forest"}


def vocabulario() -> None:
    """F06 e F07 pelas funções do pacote, só com os rótulos no vocabulário do texto (nada muda em `src/`)."""
    from pdm_cmaps.reporting import monitoring_figures as mf

    vs.apply_style()
    tabs = RES / "maintenance_guide_fd004" / "tables"
    boot = pd.read_csv(tabs / "thesis_bootstrap_ci.csv")
    boot["Model"] = boot["Model"].replace(NOMES_MODELOS)
    boot = boot.nlargest(9, "R2_Observed")  # só os nove de maior R² (0,82 a 0,85); a legenda da monografia cita a faixa dos demais
    mf.plot_model_ci(boot, FIG / "F06_comparacao_modelos_ic_fd004.png")

    timeline = pd.read_csv(tabs / "alert_timeline.csv")
    eol = timeline.groupby("unit_id")["cycle"].max().sort_values()
    picks = list(dict.fromkeys(int(eol.index[int(q * (len(eol) - 1))]) for q in (0.1, 0.4, 0.7, 0.95)))  # mesma escolha de `monitoring.layer`
    salvar = mf.save_figure

    def _com_motor(fig, path):
        for ax in fig.axes:
            for loc in ("left", "center", "right"):
                if ax.get_title(loc):
                    t = ax.get_title(loc).replace("Ativo", "Motor")
                    if "Motor " in t and "FD004" not in t:
                        t = f"{t} (FD004)"
                    ax.set_title(t, loc=loc, pad=11)  # logo acima da faixa de criticidade
        if fig._suptitle is not None:
            fig._suptitle.set_text("RUL com incerteza e nível de criticidade ao longo da vida (FD004, motores 178, 204, 224 e 174 do holdout)")
        return salvar(fig, path)

    mf.save_figure = _com_motor
    try:
        mf.plot_rul_trajectories(timeline, picks, FIG / "F07_trajetorias_intervalo_conformal.png")
    finally:
        mf.save_figure = salvar


MODOS = {"exogenous": ("modo exógeno", vs.SERIES[0], "-"), "cross_sensor": ("modo entre sensores", vs.SERIES[1], "--")}
CASOS = [("MEWMA", "processo", "MEWMA\nfalhas de processo"), ("MEWMA", "sensor", "MEWMA\nfalhas de sensor"), ("sensor afetado", "sensor", "Carta do sensor afetado\nfalhas de sensor")]
ESTIMADORES = ("Extra Trees (padrão)", "LightGBM", "MLP", "Ridge")


def ganho_ml() -> None:
    """Ganho de poder de cada modelo sobre a média do regime, com IC 95 % (tabela `diferenca_vs_media_regime.csv`)."""
    d = pd.read_csv(TAB / "diferenca_vs_media_regime.csv")
    vs.apply_style()
    fig, axes = plt.subplots(1, len(CASOS), figsize=(8.4, 4.0), sharex=True, sharey=True)
    for ax, (chart, fam, titulo) in zip(axes, CASOS):
        for k, (modo, (lab, col, _)) in enumerate(MODOS.items()):
            g = d[(d["Chart"] == chart) & (d["family"] == fam) & (d["Mode"] == modo)].set_index("Estimator").loc[list(ESTIMADORES)]
            y = [i + (k - 0.5) * 0.38 for i in range(len(ESTIMADORES))]
            ax.barh(y, g["delta_pp"], height=0.34, color=col, label=lab, xerr=[g["delta_pp"] - g["ci_lo"], g["ci_hi"] - g["delta_pp"]],
                    error_kw=dict(ecolor=vs.INK_2, elinewidth=1, capsize=2.5))
        ax.axvline(0, color=vs.INK, lw=1)
        ax.set_yticks(range(len(ESTIMADORES))), ax.set_yticklabels([e.replace(" (padrão)", "") for e in ESTIMADORES], fontsize=10)
        ax.invert_yaxis(), ax.tick_params(axis="x", labelsize=9.5)
        ax.set_title(titulo, fontsize=10, loc="left", fontweight="bold")
        vs.style_axes(ax, grid="x")
    fig.supxlabel("ganho de poder sobre a média do regime (pontos percentuais)", fontsize=9.5, y=0.075)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle("Ganho de detecção de cada modelo sobre a média do regime nos quatro subconjuntos juntos\nbarras finas = IC 95 % por bootstrap sobre motores", fontsize=10.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.11, 1, 0.91))
    fig.savefig(FIG / "F18_ganho_ml_vs_media_regime.png", dpi=200)
    plt.close(fig)




def conformal_ajustado() -> None:
    """F27: compara Mondrian sobre LightGBM (dente de serra) vs XGBoost ajustado + CQR + histerese."""
    from matplotlib.patches import Patch, Rectangle

    traj_path = RES / "estudos_complementares" / "exp_a_trajetorias_seed42.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"F27 precisa de {traj_path} (estudos_complementares, Exp. A)")
    d = pd.read_csv(traj_path)
    # Escolhe 2 motores representativos com forte efeito dente-de-serra no baseline
    ref = d[(d["Model"] == "LightGBM (ref.)") & (d["Conformal_method"] == "Mondrian 3 tercis")].copy()
    stats = []
    for u, g in ref.groupby("unit_id"):
        g = g.sort_values("cycle")
        jumps = int(((np.abs(np.diff(g["RUL_lower"])) >= 15) & (np.abs(np.diff(g["RUL_pred"])) <= 5)).sum())
        revs = int((np.diff(g["level_interval"]) < 0).sum())
        stats.append((u, jumps, revs, int(g["cycle"].max())))
    st_df = pd.DataFrame(stats, columns=["unit_id", "jumps", "revs", "eol"]).sort_values(["jumps", "revs"], ascending=False)
    picks = [int(u) for u in st_df["unit_id"].head(2)]
    if 204 in set(ref["unit_id"]) and 204 not in picks:
        picks[1] = 204

    panels = [
        ("LightGBM (ref.)", "Mondrian 3 tercis", "LightGBM (ref.) + Mondrian (3 tercis)"),
        ("XGBoost (ajustado)", "CQR + histerese", "XGBoost (ajustado) + CQR + histerese"),
    ]
    vs.apply_style()
    fig, axes = plt.subplots(len(picks), 2, figsize=(11.2, 6.2), sharex=False, sharey=True)
    ymax, strip_lo, strip_hi = 146, 131, 143
    for row_i, uid in enumerate(picks):
        for col_j, (mod, conf, label) in enumerate(panels):
            ax = axes[row_i, col_j]
            vs.style_axes(ax, "y")
            eng = d[(d["unit_id"] == uid) & (d["Model"] == mod) & (d["Conformal_method"] == conf)].sort_values("cycle")
            c = eng["cycle"].to_numpy()
            span = max(int(c.max() - c.min()), 1)
            ax.fill_between(c, eng["RUL_lower"], eng["RUL_upper"], color=vs.SERIES[1], alpha=0.18, lw=0)
            ax.plot(c, eng["RUL_true"], color=vs.INK, lw=vs.LINE_W)
            ax.plot(c, eng["RUL_pred"], color=vs.SERIES[1], lw=vs.LINE_W)
            lvls = eng["level_interval"].to_numpy(dtype=int)
            # Runs
            cuts = np.flatnonzero(lvls[1:] != lvls[:-1]) + 1
            for s_i, e_i in zip(np.r_[0, cuts], np.r_[cuts, len(lvls)] - 1):
                lvl = int(lvls[s_i])
                ax.add_patch(Rectangle((c[s_i] - 0.5, strip_lo), c[e_i] - c[s_i] + 1, strip_hi - strip_lo,
                                       facecolor=vs.STATUS[lvl], edgecolor=vs.SURFACE, linewidth=1.3, clip_on=False))
                if (c[e_i] - c[s_i] + 1) / span >= 0.16:
                    ax.text((c[s_i] + c[e_i]) / 2, (strip_lo + strip_hi) / 2, vs.LEVEL_NAMES_PT[lvl], ha="center", va="center",
                            color="#ffffff" if lvl in (0, 3) else vs.INK, fontsize=7.5)
            ax.set_ylim(0, ymax)
            ax.set_yticks([0, 25, 50, 75, 100, 125])
            ax.set_title(f"Motor {uid} · {label}", fontsize=10, loc="left", pad=9)
            if row_i == len(picks) - 1:
                ax.set_xlabel("Ciclo")
        axes[row_i, 0].set_ylabel("RUL (ciclos)")
    handles = [
        Line2D([], [], color=vs.INK, lw=vs.LINE_W), Line2D([], [], color=vs.SERIES[1], lw=vs.LINE_W),
        Patch(color=vs.SERIES[1], alpha=0.30), *[Patch(color=vs.STATUS[i]) for i in range(4)],
    ]
    labels_leg = ["RUL real", "RUL prevista", "Intervalo de 90%", *[f"Nível {n}" for n in vs.LEVEL_NAMES_PT]]
    fig.legend(handles, labels_leg, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Eliminação do dente de serra e estreitamento do intervalo: Mondrian vs. XGBoost ajustado + CQR + histerese (FD004, holdout)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(FIG / "F27_conformal_ajustado_cqr.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    vocabulario()
    ganho_ml()
    decomposicao()
    criticidade()
    conformal_ajustado()
    print("ok")

"""Thesis figures of the monitoring layer (dataviz method; tokens and mark specs in viz_style.py).

Each function draws ONE message and has a CSV as its table view:
  plot_regime_variance          regime_variance_by_subset.csv           why regime decoupling is needed
  plot_residual_control_chart   alert_timeline / spc residual trace     what an alarm looks like on one engine
  plot_rul_trajectories         alert_timeline.csv                      RUL, uncertainty and criticality over time
  plot_strategy_tradeoff        criticality_policy_sensitivity.csv      missed-critical vs false-alarm trade-off
  plot_strategy_tradeoff_repeated  monitoring_repeated_strategies_raw.csv   same, mean +- std over seeds
  plot_model_ci                 thesis_bootstrap_ci.csv                 model comparison with uncertainty
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

from pdm_cmaps.reporting.plots import save_figure
from pdm_cmaps.reporting.viz_style import INK, INK_2, LEVEL_NAMES_PT, LINE_W, MARKER_PT, MUTED, SERIES, STATUS, SURFACE, apply_style, ring_marker_kwargs, style_axes

apply_style()

STRATEGY_ORDER = ("point", "interval", "interval+anomaly")
STRATEGY_LABEL = {
    "point": "Ponto (RUL prevista)",
    "interval": "Limite inferior do intervalo",
    "interval+anomaly": "Intervalo + anomalia SPC",
}
STRATEGY_MARKER = {"point": "o", "interval": "s", "interval+anomaly": "s"}
STRATEGY_COLOR = dict(zip(STRATEGY_ORDER, SERIES))


def _policy_label(policy: str) -> str:
    return policy.replace("critical<=", "crítico ≤ ")


def _level_runs(levels: np.ndarray):
    start = 0
    for i in range(1, len(levels) + 1):
        if i == len(levels) or levels[i] != levels[start]:
            yield start, i - 1, int(levels[start])
            start = i


# ---------------------------------------------------------------------------
# regime variance (the thesis' strongest evidence)
# ---------------------------------------------------------------------------

def plot_regime_variance(table: pd.DataFrame, path: Path):
    """One series: share of T50 variance explained by flight regime, per C-MAPSS subset."""
    d = table.groupby(["Subset", "Regimes"], as_index=False)["Between_regime_pct"].mean().sort_values("Subset", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(d) + 1.7))
    y = np.arange(len(d))
    ax.barh(y, d["Between_regime_pct"], height=0.26, color=SERIES[0])
    ax.set_yticks(y, [f"{s}  ·  {int(k)} regime{'s' if k > 1 else ''}" for s, k in zip(d["Subset"], d["Regimes"])], color=INK_2)
    for yi, v in zip(y, d["Between_regime_pct"]):
        ax.text(v + 1.2, yi, f"{v:.1f}%".replace(".", ","), va="center", ha="left", color=INK, fontsize=10)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
    style_axes(ax, "x")
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Variância do T50 explicada pelo regime de voo (motores de teste)")
    ax.set_title("Em voo com vários regimes, o regime domina o sinal bruto")
    fig.tight_layout()
    return save_figure(fig, path)


# ---------------------------------------------------------------------------
# detection on one engine
# ---------------------------------------------------------------------------

def plot_residual_control_chart(trace: pd.DataFrame, unit_id: int, first_alarm_cycle, path: Path, healthy_cycle_limit: int):
    """Residual z-score and EWMA control chart for ONE engine; the alarm is marked on the EWMA."""
    eng = trace.loc[trace["unit_id"] == unit_id].sort_values("cycle")
    cycles = eng["cycle"].to_numpy()
    eol = int(cycles.max())
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw={"hspace": 0.28})
    for ax in axes:
        style_axes(ax, "y")
        ax.axvspan(0.5, healthy_cycle_limit + 0.5, color=MUTED, alpha=0.10, lw=0)
    top, bottom = axes
    top.plot(cycles, eng["z_score"], color=SERIES[0], lw=LINE_W)
    for level in (3, -3):
        top.axhline(level, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    top.text(healthy_cycle_limit + 6, 3.0, "limite ±3σ", ha="left", va="bottom", color=INK_2, fontsize=8.5)
    top.text(healthy_cycle_limit / 2, top.get_ylim()[1] * 0.95, f"calibração saudável\n(ciclos 1–{healthy_cycle_limit})", ha="center", va="top", color=INK_2, fontsize=8.5)
    top.set_ylabel("z do resíduo")
    top.set_title("Resíduo do sensor virtual (real − esperado em condição saudável), padronizado")

    bottom.plot(cycles, eng["ewma_abs_z"], color=SERIES[0], lw=LINE_W, label="EWMA de |z| (λ = 0,10)")
    bottom.plot(cycles, eng["ewma_ucl"], color=INK_2, lw=1.2, ls=(0, (4, 3)), label="Limite de controle (3σ)")
    handles = [Line2D([], [], color=SERIES[0], lw=LINE_W), Line2D([], [], color=INK_2, lw=1.2, ls=(0, (4, 3)))]
    bottom.legend(handles, ["EWMA de |z| (λ = 0,10)", "Limite de controle (3σ)"], loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02))
    if first_alarm_cycle is not None and not pd.isna(first_alarm_cycle):
        a = int(first_alarm_cycle)
        y_alarm = float(eng.loc[eng["cycle"] == a, "ewma_abs_z"].iloc[0])
        bottom.vlines(a, 0, y_alarm, color=INK_2, lw=1.0)
        bottom.plot([a], [y_alarm], marker="o", **ring_marker_kwargs(STATUS[3], 10))
        bottom.annotate(f"Primeiro alarme persistente: ciclo {a}\nantecedência de {eol - a} ciclos até o fim de vida",
                        xy=(a, y_alarm), xytext=(-12, 6), textcoords="offset points", ha="right", va="bottom", color=INK, fontsize=9)
    bottom.set_ylabel("EWMA de |z|")
    bottom.set_xlabel("Ciclo operacional (o último ciclo é o fim de vida)")
    bottom.set_title("Carta de controle")
    fig.suptitle(f"Ativo {unit_id} (motor do holdout)", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.subplots_adjust(top=0.88)
    return save_figure(fig, path)


# ---------------------------------------------------------------------------
# RUL with uncertainty and criticality
# ---------------------------------------------------------------------------

def plot_rul_trajectories(timeline: pd.DataFrame, unit_ids: list[int], path: Path, strategy: str = "interval+anomaly", alpha: float = 0.10):
    """RUL real vs prevista com intervalo conformal; o fundo mostra o nível de criticidade ao longo do tempo."""
    n = len(unit_ids)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.3 * rows + 0.9), squeeze=False, sharey=True)
    ymax, strip_lo, strip_hi = 146, 131, 143
    for ax, unit in zip(axes.ravel(), unit_ids):
        style_axes(ax, "y")
        eng = timeline.loc[timeline["unit_id"] == unit].sort_values("cycle")
        c = eng["cycle"].to_numpy()
        span = max(c.max() - c.min(), 1)
        ax.fill_between(c, eng["RUL_lower"], eng["RUL_upper"], color=SERIES[1], alpha=0.18, lw=0)
        ax.plot(c, eng["RUL_true"], color=INK, lw=LINE_W)
        ax.plot(c, eng["RUL_pred"], color=SERIES[1], lw=LINE_W)
        # criticality strip: status color + printed level name (never color alone); 2 px surface gap between runs
        for s_i, e_i, lvl in _level_runs(eng[f"level_{strategy}"].to_numpy()):
            ax.add_patch(Rectangle((c[s_i] - 0.5, strip_lo), c[e_i] - c[s_i] + 1, strip_hi - strip_lo,
                                   facecolor=STATUS[lvl], edgecolor=SURFACE, linewidth=1.5, clip_on=False))
            if (c[e_i] - c[s_i] + 1) / span >= 0.14:  # name only where it fits; the legend carries the rest
                ax.text((c[s_i] + c[e_i]) / 2, (strip_lo + strip_hi) / 2, LEVEL_NAMES_PT[lvl], ha="center", va="center",
                        color="#ffffff" if lvl in (0, 3) else INK, fontsize=8)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, 25, 50, 75, 100, 125])
        ax.set_title(f"Ativo {unit}")
        ax.set_xlabel("Ciclo")
    for ax in axes[:, 0]:
        ax.set_ylabel("RUL (ciclos)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles = [
        Line2D([], [], color=INK, lw=LINE_W), Line2D([], [], color=SERIES[1], lw=LINE_W),
        Patch(color=SERIES[1], alpha=0.30),
        *[Patch(color=STATUS[i]) for i in range(4)],
    ]
    labels = ["RUL real", "RUL prevista", f"Intervalo de {int(round((1 - alpha) * 100))}%", *[f"Nível {name}" for name in LEVEL_NAMES_PT]]
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("RUL com incerteza e nível de criticidade ao longo da vida (motores do holdout)", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    return save_figure(fig, path)


# ---------------------------------------------------------------------------
# criticality trade-off
# ---------------------------------------------------------------------------

def _strategy_handles() -> list[Line2D]:
    out = []
    for s in STRATEGY_ORDER:
        hollow = s == "interval+anomaly"
        out.append(Line2D([], [], marker=STRATEGY_MARKER[s], **ring_marker_kwargs(STRATEGY_COLOR[s], 14 if hollow else MARKER_PT + 1, hollow)))
    return out


def plot_strategy_tradeoff(sensitivity: pd.DataFrame, path: Path):
    """Críticos não alertados × falsos alertas; cada ponto é uma política de limiares."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    style_axes(ax, "both")
    d = sensitivity.copy()
    d["x"], d["y"] = 100 * d["False_alarm_rate_normal"], 100 * d["Missed_critical_rate"]
    for s in ("interval+anomaly", "point", "interval"):  # hollow ring first so the filled marker on top does not hide it
        g = d.loc[d["Strategy"] == s]
        hollow = s == "interval+anomaly"
        ax.plot(g["x"], g["y"], marker=STRATEGY_MARKER[s], clip_on=False, **ring_marker_kwargs(STRATEGY_COLOR[s], 14 if hollow else MARKER_PT + 1, hollow))
    for row in d.loc[d["Strategy"] == "interval"].itertuples():
        ax.annotate(_policy_label(row.Policy), (row.x, row.y), xytext=(7, 7), textcoords="offset points", color=INK_2, fontsize=8.5)
    a = d.loc[d["Strategy"] == "interval", ["x", "y"]].to_numpy()
    b = d.loc[d["Strategy"] == "interval+anomaly", ["x", "y"]].to_numpy()
    if a.shape == b.shape and np.allclose(a, b):
        ax.text(0.98, 0.04, "“Limite inferior” e “intervalo + anomalia” coincidem:\no alarme SPC não mudou nenhum nível.",
                transform=ax.transAxes, ha="right", va="bottom", color=INK_2, fontsize=8.5)
    ax.set_xlim(left=-0.06 * max(d["x"].max(), 1e-9))
    ax.set_ylim(bottom=-0.08 * max(d["y"].max(), 1e-9))
    ax.set_xlabel("Falsos alertas em estado normal (% dos ciclos)")
    ax.set_ylabel("Estados críticos não alertados (% dos ciclos)")
    ax.set_title("Compromisso entre estratégias de criticidade\nmais perto da origem é melhor; cada ponto é uma política de limiares")
    ax.legend(_strategy_handles(), [STRATEGY_LABEL[s] for s in STRATEGY_ORDER], loc="upper right")
    fig.tight_layout()
    return save_figure(fig, path)


def plot_strategy_tradeoff_repeated(strat: pd.DataFrame, path: Path, policy: str = "critical<=15"):
    """Um painel por subconjunto: média ± desvio entre sementes, política fixa."""
    default = strat.loc[strat["Policy"] == policy]
    subsets = sorted(default["Subset"].unique())
    fig, axes = plt.subplots(1, len(subsets), figsize=(3.4 * len(subsets) + 0.6, 4.3), sharey=True, squeeze=False)
    for ax, subset in zip(axes.ravel(), subsets):
        style_axes(ax, "both")
        d = default.loc[default["Subset"] == subset]
        for s in ("interval+anomaly", "point", "interval"):
            g = d.loc[d["Strategy"] == s]
            hollow = s == "interval+anomaly"
            ax.errorbar(100 * g["False_alarm_rate_normal"].mean(), 100 * g["Missed_critical_rate"].mean(),
                        xerr=100 * g["False_alarm_rate_normal"].std(), yerr=100 * g["Missed_critical_rate"].std(),
                        fmt=STRATEGY_MARKER[s], color=STRATEGY_COLOR[s], ecolor=STRATEGY_COLOR[s], elinewidth=1.4, capsize=3,
                        markersize=14 if hollow else MARKER_PT + 1, clip_on=False,
                        markerfacecolor=SURFACE if hollow else STRATEGY_COLOR[s],
                        markeredgecolor=STRATEGY_COLOR[s] if hollow else SURFACE, markeredgewidth=2.0)
        ax.set_title(subset)
        ax.set_xlabel("Falsos alertas (%)")
    # limits from mean + std of every subset/strategy, set once after plotting: with sharey, a
    # one-sided set_ylim inside the loop froze the top at the first panel and clipped later ones
    g = default.groupby(["Subset", "Strategy"])
    ytop = 100 * (g["Missed_critical_rate"].mean() + g["Missed_critical_rate"].std().fillna(0)).max()
    ytop = max(ytop, 1.0)
    axes.ravel()[0].set_ylim(-0.06 * ytop, 1.08 * ytop)
    for ax, subset in zip(axes.ravel(), subsets):
        gx = default.loc[default["Subset"] == subset].groupby("Strategy")["False_alarm_rate_normal"]
        xtop = max(100 * (gx.mean() + gx.std().fillna(0)).max(), 1.0)
        ax.set_xlim(-0.08 * xtop, 1.08 * xtop)
    axes.ravel()[0].set_ylabel("Críticos não alertados (%)")
    fig.legend(_strategy_handles(), [STRATEGY_LABEL[s] for s in STRATEGY_ORDER], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Compromisso das estratégias (média ± desvio entre sementes; {_policy_label(policy)} ciclos)", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    return save_figure(fig, path)


# ---------------------------------------------------------------------------
# model comparison
# ---------------------------------------------------------------------------

def plot_model_ci(boot: pd.DataFrame, path: Path, metric: str = "R2"):
    """Modelos ordenados pelo valor observado, com IC 95% por bootstrap de motores (uma série, sem legenda)."""
    d = boot.sort_values(f"{metric}_Observed").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(d) + 1.9))
    style_axes(ax, "x")
    y = np.arange(len(d))
    obs = d[f"{metric}_Observed"].to_numpy()
    lo, hi = d[f"{metric}_CI_Lower"].to_numpy(), d[f"{metric}_CI_Upper"].to_numpy()
    ax.hlines(y, lo, hi, color=SERIES[0], lw=LINE_W, alpha=0.5)
    ax.plot(obs, y, marker="o", **ring_marker_kwargs(SERIES[0]))
    ax.set_yticks(y, d["Model"], color=INK_2)
    best = len(d) - 1
    ax.annotate(f"{obs[best]:.3f}".replace(".", ","), (hi[best], best), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))
    ax.set_xlabel(f"{metric.replace('R2', 'R²')} nos motores do holdout 75/25 (IC 95%, bootstrap por motor)")
    ax.set_title("Modelos de RUL: intervalos marginais sobrepostos exigem teste pareado")
    fig.tight_layout()
    return save_figure(fig, path)

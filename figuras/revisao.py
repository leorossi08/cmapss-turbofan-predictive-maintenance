"""Figuras 5.3, 5.4, 5.6 a 5.9 e 5.10 a 5.12, sem alterar o pacote `pdm_cmaps`.

  F15       Figura 5.3       RUL prevista x real sem a coluna em RUL real = 125 (teto da verdade truncada)
  F05       Figura 5.4       carta de controle do motor 179 com a manutenção (ciclo 242) rotulada
  F14_linha1..4  Figuras 5.6 a 5.9  motor 179 do FD004, uma faixa por figura
  F17a/b/c  Figuras 5.10 a 5.12  desvios do envelope: motor completo, desvios isolados, persistência na frota

Usa results/maintenance_guide_fd00x/tables e as funções de `pdm_cmaps.experiments.envelope_ignored_cases` e `rul_prediction_plots` só para leitura.
Uso: python figuras/gerar.py N   ou   python figuras/revisao.py [--so f05|f14|f15|f17]
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
from pdm_cmaps.experiments import envelope_ignored_cases as env  # noqa: E402
from pdm_cmaps.experiments.rul_prediction_plots import EVENT_COLOR, PRED_COLOR, VERIFIED_MAINTENANCE  # noqa: E402
from pdm_cmaps.reporting import viz_style as vs  # noqa: E402

FIG = Path(__file__).resolve().parent / "saida"
CAP = 125


def f15(results: Path, model: str = "XGBoost (+temporal)", subset: str = "FD004") -> None:
    from sklearn.metrics import r2_score

    vs.apply_style()
    df = pd.read_csv(results / f"maintenance_guide_{subset.lower()}" / "tables" / "rul_holdout_predictions.csv")
    col = next(c for c in df.columns if c.startswith("RUL_predicted_") and "xgboost" in c.lower() and "temporal" in c.lower() and "nasa" not in c.lower())
    truth, pred = df["RUL_actual"].to_numpy(), df[col].to_numpy()
    r2_all = r2_score(truth, pred)
    keep = truth < CAP
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(truth[keep], pred[keep], s=7, alpha=0.22, color=PRED_COLOR, edgecolors="none", label="ciclos do holdout")
    for (sub, unit), cyc in VERIFIED_MAINTENANCE.items():
        if sub != subset:
            continue
        m = (df["unit_id"] == unit).to_numpy() & keep
        after = m & (df["cycle"] >= cyc).to_numpy()
        if (m & ~after).any():
            ax.scatter(truth[m & ~after], pred[m & ~after], s=15, color=vs.SERIES[1], label=f"motor {unit}, antes do ciclo {cyc}")
        if after.any():
            ax.scatter(truth[after], pred[after], s=15, color=EVENT_COLOR, label=f"motor {unit}, após a descontinuidade (ciclo ≥ {cyc})")
    lo, hi = 0, CAP + 3
    ax.plot([lo, hi], [lo, hi], color=vs.INK_2, ls="--", lw=1.4, label="previsão perfeita")
    ax.set_xlim(lo, hi), ax.set_ylim(lo - 2, 150)
    vs.style_axes(ax, grid="both")
    ax.set_xlabel("RUL real (ciclos)"), ax.set_ylabel("RUL prevista (ciclos)")
    n_fmt = f"{len(truth):,}".replace(",", " ")
    r2_fmt = f"{r2_all:.3f}".replace(".", ",")
    ax.set_title(f"{subset} · XGBoost com atributos temporais\nR² no holdout = {r2_fmt} (todos os ciclos, n = {n_fmt})", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8)
    n_omit = f"{int((~keep).sum()):,}".replace(",", " ")
    ax.text(0.99, 0.02, f"{n_omit} ciclos com RUL real truncada em {CAP} não são mostrados", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=vs.INK_2)
    fig.tight_layout()
    fig.savefig(FIG / "F15_dispersao_predito_real_xgboost_temporal.png", dpi=150)
    plt.close(fig)
    print("F15 ok:", model, "omitidos", int((~keep).sum()))


def f05(results: Path, unit: int = 179, maint_cycle: int = 242) -> None:
    """Carta de controle do motor 179 (Figura 5.5) com a manutenção realizada no ciclo 242 rotulada nos dois painéis."""
    vs.apply_style()
    trace = pd.read_csv(results / "maintenance_guide_fd004" / "tables" / "envelope_scored_trace_h50.csv")
    eng = trace[trace.unit_id == unit].sort_values("cycle")
    cyc = eng["cycle"].to_numpy()
    eol = int(cyc.max())
    flag = ((eng["ewma_abs_z"] > eng["ewma_ucl"]) & (eng["z_score"].abs() >= 3.0) & (eng["cycle"] > 50)).to_numpy()
    first, run = None, 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= 2:
            first = int(cyc[i - 1])
            break
    gold = "#c98500"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True, gridspec_kw={"hspace": 0.3})
    for ax in axes:
        vs.style_axes(ax, "y")
        ax.axvspan(0.5, 50.5, color=vs.MUTED, alpha=0.10, lw=0)
        ax.axvline(maint_cycle, color=gold, lw=2.2, ls=(0, (5, 3)))
    top, bot = axes
    top.plot(cyc, eng["z_score"], color=vs.SERIES[0], lw=vs.LINE_W)
    for lv in (3, -3):
        top.axhline(lv, color=vs.INK_2, lw=1.2, ls=(0, (4, 3)))
    top.text(56, 3.05, "limite ±3σ", color=vs.INK_2, fontsize=8.5)
    top.text(25, top.get_ylim()[1] * 0.95, "calibração saudável\n(ciclos 1–50)", ha="center", va="top", color=vs.INK_2, fontsize=8.5)
    top.set_ylabel("z do resíduo")
    top.set_title("Resíduo do sensor virtual (real − esperado em condição saudável), padronizado")
    top.text(maint_cycle + 5, -2.3, "descontinuidade\n(ciclo 242)", color=gold, fontsize=9.5, fontweight="bold", va="center")
    bot.plot(cyc, eng["ewma_abs_z"], color=vs.SERIES[0], lw=vs.LINE_W, label="EWMA de |z| (λ = 0,10)")
    bot.plot(cyc, eng["ewma_ucl"], color=vs.INK_2, lw=1.2, ls=(0, (4, 3)), label="Limite de controle (3σ)")
    bot.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02))
    if first is not None:
        y = float(eng.loc[eng["cycle"] == first, "ewma_abs_z"].iloc[0])
        bot.vlines(first, 0, y, color=vs.INK_2, lw=1.0)
        bot.plot([first], [y], marker="o", **vs.ring_marker_kwargs(vs.STATUS[3], 10))
        bot.annotate(f"Primeiro alarme persistente: ciclo {first}\nantecedência de {eol - first} ciclos até o fim de vida", xy=(first, y), xytext=(-12, 6), textcoords="offset points", ha="right", va="bottom", color=vs.INK, fontsize=9)
    bot.annotate("descontinuidade\n(ciclo 242)", xy=(maint_cycle, 4.0), xytext=(maint_cycle + 18, 5.4), ha="left", color=gold, fontsize=9.5, fontweight="bold", arrowprops=dict(arrowstyle="->", color=gold, lw=1.8))
    bot.set_ylabel("EWMA de |z|"), bot.set_xlabel("Ciclo operacional (o último ciclo é o fim de vida)"), bot.set_title("Carta de controle")
    fig.suptitle(f"Motor {unit} do FD004 (holdout, com descontinuidade no ciclo {maint_cycle})", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.88)
    fig.savefig(FIG / "F05_carta_controle_residuo_fd004.png", dpi=164)
    plt.close(fig)
    print("F05 ok: primeiro alarme", first, "fim de vida", eol)


def _cases(r, runs, subset="FD004"):
    rs, ru = r[r["Subset"] == subset], runs[runs["Subset"] == subset]
    mon = ru[ru["monitored"]]
    ign_by_unit = mon[~mon["alert"]].groupby("unit_id").size()
    firsts = ru[ru["first_alert"]].set_index("unit_id")["start"]
    cand = [(int(u), int(n)) for u, n in ign_by_unit.items() if u in firsts.index and (subset, int(u)) not in env.RESET_ENGINES]
    top = max(cand, key=lambda t: (t[1], t[0]))[0] if cand else int(rs["unit_id"].iloc[0])
    runs_top = ru[ru["unit_id"] == top]
    healthy_ign = mon[(~mon["alert"]) & mon["phase"].str.startswith("vida saudável") & ~mon["unit_id"].map(lambda u: (subset, int(u)) in env.RESET_ENGINES)]
    deg_ign = mon[(~mon["alert"]) & mon["phase"].str.startswith("degradação")]
    case_b = healthy_ign.iloc[(healthy_ign["max_excess_sigma"] - healthy_ign["max_excess_sigma"].median()).abs().argsort()].iloc[0]
    case_c = deg_ign.iloc[(deg_ign["max_excess_sigma"] - deg_ign["max_excess_sigma"].median()).abs().argsort()].iloc[0] if len(deg_ign) else case_b
    case_d = runs_top[runs_top["first_alert"]].iloc[0]
    return rs, ru, mon, top, runs_top, case_b, case_c, case_d


def f17(results: Path, subset: str = "FD004") -> None:
    vs.apply_style()
    frames = [env.fleet_envelope(results, s)[0] for s in env.SUBSETS]
    r = pd.concat(frames, ignore_index=True)
    runs = env.breach_runs(r)
    summ = env.summarize(r, runs)
    rs, ru, mon, top, runs_top, case_b, case_c, case_d = _cases(r, runs, subset)
    g_top = rs[rs["unit_id"] == top]

    # ---- F17a: um motor inteiro, com todos os desvios ----
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    env._band(ax)
    ax.plot(g_top["cycle"], g_top["res_sigma"], color=vs.INK, lw=1.0, label="resíduo do T50 (em σ)")
    env._mark(ax, g_top, runs_top[runs_top["monitored"]], 70)
    fa = int(runs_top[runs_top["first_alert"]]["alert_cycle"].iloc[0])
    ax.axvline(fa, color=vs.STATUS[3], lw=1.8, ls="--")
    ax.axvspan(g_top["cycle"].min(), env.H, color=vs.STATUS[0], alpha=0.10, label=f"janela de calibração (até o ciclo {env.H}, não monitorada)")
    ymax = g_top["res_sigma"].max() + 1.5
    ax.annotate(f"primeiro alerta: ciclo {fa}\n(2 ciclos seguidos fora do envelope)", (fa, ymax - 0.4), xytext=(-12, 0), textcoords="offset points", ha="right", va="top", fontsize=9.5, color=vs.STATUS[3], fontweight="bold")
    n_ign = int(((~runs_top["alert"]) & runs_top["monitored"]).sum())
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8.5, ncol=2, loc="lower left")
    ax.set_title(f"Motor {top} do {subset} (holdout, sem descontinuidade): {n_ign} desvios isolados ignorados; alerta só quando persiste", fontsize=10.8)
    ax.set_xlabel("ciclo (voo)"), ax.set_ylabel("resíduo do T50 (σ saudável)")
    ax.set_ylim(min(-4.8, g_top["res_sigma"].min() - 0.5), ymax)
    vs.style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "F17a_envelope_motor_completo.png", dpi=150)
    plt.close(fig)

    # ---- F17b: tres exemplos de zoom ----
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    specs = [(case_b, f"Desvio isolado, vida saudável\nmotor {int(case_b['unit_id'])} ({subset}): 1 ciclo, {case_b['max_excess_sigma']:.2f}σ além", vs.INK),
             (case_c, f"Desvio isolado, em degradação\nmotor {int(case_c['unit_id'])} ({subset}): 1 ciclo, {case_c['max_excess_sigma']:.2f}σ além", vs.INK),
             (case_d, f"Desvio persistente: vira alerta\nmotor {top} ({subset}): ≥ 2 ciclos seguidos", vs.STATUS[3])]
    for ax, (run, ttl, color) in zip(axes, specs):
        gz = rs[rs["unit_id"] == int(run["unit_id"])]
        env._zoom(ax, gz, run, ttl, color, ru[(ru["unit_id"] == int(run["unit_id"])) & ru["monitored"] & (ru["end"] >= run["start"] - 14) & (ru["start"] <= run["end"] + 14)])
        ax.set_title(ttl, fontsize=10.5, color=color, loc="left", fontweight="semibold")
        ax.set_ylabel("resíduo do T50 (σ)")
        leg = ax.get_legend()
        if leg:
            leg.remove()
    h, l = axes[2].get_legend_handles_labels()
    allh, alll = [], []
    for ax in axes:
        hh, ll = ax.get_legend_handles_labels()
        allh += hh
        alll += ll
    uniq = dict(zip(alll, allh))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "F17b_envelope_desvios_isolados.png", dpi=150)
    plt.close(fig)

    # ---- F17c: o que a regra ensina na frota ----
    fig, (axe, axf) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    pool_ign = runs[runs["monitored"] & ~runs["alert"]]["max_excess_sigma"]
    pool_per = runs[runs["monitored"] & runs["alert"]]["max_excess_sigma"]
    bins = np.linspace(0, max(8, pool_per.quantile(0.9)), 22)
    axe.hist(pool_ign, bins=bins, alpha=0.9, color=vs.STATUS[1], label=f"ignorados (n = {len(pool_ign)})", edgecolor=vs.SURFACE)
    axe.hist(pool_per, bins=bins, alpha=0.85, color=vs.STATUS[3], label=f"viraram alerta (n = {len(pool_per)})", edgecolor=vs.SURFACE)
    axe.set_xlabel("excesso máximo além do envelope (σ do resíduo saudável)"), axe.set_ylabel("sequências de desvio")
    axe.set_title("Os desvios ignorados são pequenos;\nos persistentes são maiores", fontsize=10.5)
    axe.legend(), vs.style_axes(axe)
    x = np.arange(len(summ))
    a_, b_ = summ["false_alert_per100_without_persistence"], summ["false_alert_per100_with_persistence"]
    axf.bar(x - 0.2, a_, 0.4, color=vs.INK_2, label="alertar em todo desvio")
    axf.bar(x + 0.2, b_, 0.4, color=vs.SERIES[0], label="exigir 2 ciclos seguidos")
    axf.scatter(x + 0.2, b_, marker="_", s=420, color=vs.SERIES[0], linewidths=3.5, zorder=4, clip_on=False)  # barras de valor zero continuam visíveis
    for xi, (a, b) in enumerate(zip(a_, b_)):
        axf.text(xi - 0.2, a + 0.004, f"{a:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=9)
        axf.text(xi + 0.2, b + 0.004, f"{b:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=9)
    axf.set_xticks(x), axf.set_xticklabels(summ["Subset"])
    axf.set_ylabel("alertas falsos por 100 ciclos saudáveis")
    axf.set_title("O que a persistência evita: alertas falsos\nna vida saudável (RUL ≥ 125, após a calibração)", fontsize=10.5)
    axf.legend(), vs.style_axes(axf)
    fig.tight_layout()
    fig.savefig(FIG / "F17c_envelope_persistencia_frota.png", dpi=150)
    plt.close(fig)
    print("F17a/b/c ok: motor", top, "casos", int(case_b["unit_id"]), int(case_c["unit_id"]))


def f14_linhas(results: Path, unit: int = 179, subset: str = "FD004", maint_cycle: int = 242) -> None:
    """Gera F14_linha1..4.png com cabeçalho explícito do motor 179 do FD004 em cada faixa."""
    from pdm_cmaps.experiments.rul_prediction_plots import NOMES_PT, REAL_COLOR

    vs.apply_style()
    df = pd.read_csv(results / f"maintenance_guide_{subset.lower()}" / "tables" / "rul_holdout_predictions.csv")
    eng = df[df["unit_id"] == unit].sort_values("cycle")
    models = [c.replace("RUL_predicted_", "") for c in df.columns if c.startswith("RUL_predicted_") and "MiniRocket" not in c]  # MiniRocket fora da monografia
    ymax = float(eng["RUL_actual"].max())
    ncol = 5
    nrow = int(np.ceil(len(models) / ncol))
    for r_idx in range(nrow):
        row_models = models[r_idx * ncol : (r_idx + 1) * ncol]
        fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 2.50), sharex=True, sharey=True)
        for ax in axes[len(row_models):]:
            ax.set_visible(False)
        for j, (ax, model) in enumerate(zip(axes, row_models)):
            pred = eng[f"RUL_predicted_{model}"]
            rmse = float(np.sqrt(np.mean((pred - eng["RUL_actual"]) ** 2)))
            ax.plot(eng["cycle"], eng["RUL_actual"], color=REAL_COLOR, lw=1.5)
            ax.plot(eng["cycle"], pred, color=PRED_COLOR, lw=1.5)
            ax.axvline(maint_cycle, color=EVENT_COLOR, lw=1.6, ls="--", zorder=1)
            ax.text(maint_cycle + 8, ymax * 1.13, f"descontinuidade (ciclo {maint_cycle})", ha="left", va="center", fontsize=8.5, color=EVENT_COLOR, fontweight="bold")
            vs.style_axes(ax)
            label = NOMES_PT.get(model, model)
            ax.set_title(f"{label}\n" + f"RMSE {rmse:.1f}".replace(".", ","), fontsize=9.5)
            ax.set_xlabel("Ciclo", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("RUL (ciclos)", fontsize=9)
        fig.suptitle(f"Motor {unit} do {subset} (holdout, linha {r_idx + 1} de {nrow}): RUL real (preto) × prevista (azul) — descontinuidade no ciclo {maint_cycle}",
                     fontsize=11, fontweight="bold", x=0.01, ha="left", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(FIG / f"F14_linha{r_idx + 1}.png", dpi=300)
        plt.close(fig)
    print("F14_linha1..4 ok: motor", unit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--so", choices=("f05", "f14", "f15", "f17"))
    a = ap.parse_args()
    if a.so in (None, "f05"):
        f05(a.results)
    if a.so in (None, "f14"):
        f14_linhas(a.results)
    if a.so in (None, "f15"):
        f15(a.results)
    if a.so in (None, "f17"):
        f17(a.results)

"""Small deviations that leave the envelope but do NOT become alerts: cases, numbers and figure for the TCC.

WHAT: the prediction envelope of the T50 virtual sensor is  y_pred ± k·sigma_e  (k = 3, sigma_e from the OUT-OF-FOLD
healthy residuals of the training engines). A cycle outside the envelope is a "breach"; an ALERT needs `k_persist`
consecutive breaches after the calibration window (default 2: the Western Electric persistence rule of the pipeline).
An isolated breach (one cycle) is therefore ignored. This module shows those ignored cases on ALL test engines and
measures what the rule buys (false alerts avoided) and what it costs (alert issued one cycle later).
WHY: with a 3-sigma limit even healthy noise leaves the envelope sometimes (the healthy residual is not exactly Gaussian,
and 62 engines x ~200 cycles give thousands of chances); alerting on every breach would flood the engineer with false
alarms. The persistence rule trades one cycle of delay for far fewer false alerts.

[LEAKAGE] nothing is refit: the table `envelope_scored_trace_t50.csv` (sample engine) carries the calibration of the
pipeline run, from which mu_e and sigma_e are recovered EXACTLY (upper - prediction = mu_e + k·sigma_e), then applied to
the residuals of every test engine in `virtual_sensor_residuals_t50.csv` (same seed 42, H = 50 run).
[ASSUMPTION] "healthy zone" = cycles after H whose piecewise RUL label is 125; engine 179 (FD004) is excluded from the
false-alert statistics because its core resets at cycle 242 (its early life is really degrading, label notwithstanding).

Run:  python -m pdm_cmaps.experiments.envelope_ignored_cases --results /path/to/results
Output: results/envelope_ignored_cases/F17_envelope_desvios_ignorados.png (--figure), tcc-writing/tables/envelope_desvios_{resumo,casos}.csv
"""

from __future__ import annotations

from pdm_cmaps.config import DEFAULT_RESULTS_DIR

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from pdm_cmaps.reporting import viz_style

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
K_SIGMA = 3.0
PERSIST = 2
H = 50
RUL_CAP = 125
RESET_ENGINES = {("FD004", 179)}


def fleet_envelope(results: Path, subset: str) -> tuple[pd.DataFrame, dict]:
    """T50, prediction and envelope for every test engine, with the calibration recovered from the pipeline run."""
    base = results / f"maintenance_guide_{subset.lower()}" / "tables"
    trace = pd.read_csv(base / "envelope_scored_trace_t50.csv")
    margin = float(pd.read_csv(base / "envelope_summary_t50.csv")["envelope_margin"].iloc[0])
    mu = float((trace["envelope_upper"] - trace["prediction"]).iloc[0] - margin)
    r = pd.read_csv(base / "virtual_sensor_residuals_t50.csv").sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    r["upper"] = r["prediction"] + mu + margin
    r["lower"] = r["prediction"] + mu - margin
    r["breach"] = (r["T50"] > r["upper"]) | (r["T50"] < r["lower"])
    sigma = margin / K_SIGMA
    r["excess_sigma"] = (np.maximum(0, r["T50"] - r["upper"]) + np.maximum(0, r["lower"] - r["T50"])) / sigma
    r["res_sigma"] = (r["T50"] - r["prediction"] - mu) / sigma  # envelope = flat band of +-K_SIGMA in this scale
    r["Subset"] = subset
    return r, {"margin": margin, "sigma": sigma, "mu": mu}


def breach_runs(r: pd.DataFrame, persist: int = PERSIST, h: int = H) -> pd.DataFrame:
    """Every maximal run of consecutive breaches, with its length, size, phase and whether it raised an alert."""
    rows = []
    for (sub, unit), g in r.groupby(["Subset", "unit_id"], sort=True):
        b, c = g["breach"].to_numpy(), g["cycle"].to_numpy()
        ex, rul = g["excess_sigma"].to_numpy(), g["RUL"].to_numpy()
        i = 0
        while i < len(b):
            if not b[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(b) and b[j + 1]:
                j += 1
            monitored = c[i] > h
            phase = "janela de calibração (não monitorada)" if not monitored else ("vida saudável (RUL = 125)" if rul[i] >= RUL_CAP else "degradação (RUL < 125)")
            rows.append({"Subset": sub, "unit_id": int(unit), "start": int(c[i]), "end": int(c[j]), "length": j - i + 1, "max_excess_sigma": float(ex[i:j + 1].max()),
                         "RUL_label": float(rul[i]), "monitored": monitored, "phase": phase, "alert": bool(monitored and (j - i + 1) >= persist)})
            i = j + 1
    out = pd.DataFrame(rows)
    out["alert_cycle"] = np.where(out["alert"], out["start"] + persist - 1, np.nan)  # the alert is issued when the k-th breach arrives
    first = out[out["alert"]].groupby(["Subset", "unit_id"])["start"].transform("min")
    out["first_alert"] = out["alert"] & (out["start"] == first.reindex(out.index))
    return out


def summarize(r: pd.DataFrame, runs: pd.DataFrame, persist: int = PERSIST, h: int = H) -> pd.DataFrame:
    rows = []
    for sub in SUBSETS:
        rs, ru = r[r["Subset"] == sub], runs[runs["Subset"] == sub]
        if rs.empty:
            continue
        keep = ~rs["unit_id"].map(lambda u: (sub, int(u)) in RESET_ENGINES)
        healthy_cycles = int(((rs["cycle"] > h) & (rs["RUL"] >= RUL_CAP) & keep).sum())
        rh = ru[ru["phase"].str.startswith("vida saudável") & ~ru["unit_id"].map(lambda u: (sub, int(u)) in RESET_ENGINES)]
        mon = ru[ru["monitored"]]
        ign, per = mon[~mon["alert"]], mon[mon["alert"]]
        firsts = ru[ru["first_alert"]]
        pre = []
        for u, f in firsts.groupby("unit_id"):
            pre.append(int(((ru["unit_id"] == u) & ru["monitored"] & ~ru["alert"] & (ru["start"] < f["start"].iloc[0])).sum()))
        rows.append({
            "Subset": sub, "engines": rs["unit_id"].nunique(), "healthy_cycles_monitored": healthy_cycles,
            "breach_runs_monitored": len(mon), "runs_ignored": len(ign), "runs_alert": len(per), "pct_runs_ignored": 100 * len(ign) / max(len(mon), 1),
            "healthy_runs_total": len(rh), "healthy_runs_ignored": int((~rh["alert"]).sum()), "healthy_runs_alert": int(rh["alert"].sum()),
            "false_alert_per100_without_persistence": 100 * len(rh) / max(healthy_cycles, 1),
            "false_alert_per100_with_persistence": 100 * int(rh["alert"].sum()) / max(healthy_cycles, 1),
            "excess_sigma_median_ignored": float(ign["max_excess_sigma"].median()) if len(ign) else np.nan,
            "excess_sigma_p90_ignored": float(ign["max_excess_sigma"].quantile(0.9)) if len(ign) else np.nan,
            "excess_sigma_median_alert": float(per["max_excess_sigma"].median()) if len(per) else np.nan,
            "engines_with_alert": int(firsts["unit_id"].nunique()), "engines_with_ignored_before_first_alert": int(sum(p > 0 for p in pre)),
            "mean_ignored_before_first_alert": float(np.mean(pre)) if pre else np.nan, "alert_delay_cycles_by_rule": persist - 1,
        })
    return pd.DataFrame(rows)


def _band(ax) -> None:
    ax.axhspan(-K_SIGMA, K_SIGMA, color=viz_style.GRID, alpha=0.9, label="envelope (±3σ)")
    ax.axhline(0, color=viz_style.MUTED, lw=1, ls="--", label="previsto pelo sensor virtual")


def _mark(ax, g: pd.DataFrame, runs_g: pd.DataFrame, size: int) -> None:
    for _, rr in runs_g.iterrows():
        seg = g[(g["cycle"] >= rr["start"]) & (g["cycle"] <= rr["end"])]
        if seg.empty:
            continue
        ax.scatter(seg["cycle"], seg["res_sigma"], s=size, color=viz_style.STATUS[3] if rr["alert"] else viz_style.STATUS[1], edgecolor=viz_style.INK, zorder=5,
                   label="alerta (≥ 2 ciclos seguidos fora)" if rr["alert"] else "desvio isolado, ignorado")


def _zoom(ax, g: pd.DataFrame, run: pd.Series, title: str, color: str, runs_g: pd.DataFrame) -> None:
    lo, hi = run["start"] - 14, run["end"] + 14
    w = g[(g["cycle"] >= lo) & (g["cycle"] <= hi)]
    _band(ax)
    ax.plot(w["cycle"], w["res_sigma"], color=viz_style.INK, lw=1.4, marker="o", ms=3.5, label="resíduo de T50 (σ)")
    _mark(ax, w, runs_g, 90)
    ax.set_title(title, fontsize=9, color=color, loc="left")
    ax.set_xlabel("ciclo", fontsize=8)
    ax.set_ylabel("resíduo de T50 (σ saudável)", fontsize=8)
    ax.set_ylim(min(-4.2, w["res_sigma"].min() - 0.5), max(4.2, w["res_sigma"].max() + 0.5))
    viz_style.style_axes(ax)


def make_figure(r: pd.DataFrame, runs: pd.DataFrame, summ: pd.DataFrame, path: Path, subset: str = "FD004") -> dict:
    viz_style.apply_style()
    rs, ru = r[r["Subset"] == subset], runs[runs["Subset"] == subset]
    mon = ru[ru["monitored"]]
    ign_by_unit = mon[~mon["alert"]].groupby("unit_id").size()
    firsts = ru[ru["first_alert"]].set_index("unit_id")["start"]
    cand = [(int(u), int(n)) for u, n in ign_by_unit.items() if u in firsts.index and (subset, int(u)) not in RESET_ENGINES and (mon[(mon["unit_id"] == u) & ~mon["alert"] & (mon["start"] < firsts[u])].shape[0] >= 2)]
    top = max(cand, key=lambda t: (t[1], t[0]))[0] if cand else int(rs["unit_id"].iloc[0])
    g_top = rs[rs["unit_id"] == top]
    runs_top = ru[ru["unit_id"] == top]

    healthy_ign = mon[(~mon["alert"]) & mon["phase"].str.startswith("vida saudável") & ~mon["unit_id"].map(lambda u: (subset, int(u)) in RESET_ENGINES)]
    deg_ign = mon[(~mon["alert"]) & mon["phase"].str.startswith("degradação")]
    case_b = healthy_ign.iloc[(healthy_ign["max_excess_sigma"] - healthy_ign["max_excess_sigma"].median()).abs().argsort()].iloc[0]
    case_c = deg_ign.iloc[(deg_ign["max_excess_sigma"] - deg_ign["max_excess_sigma"].median()).abs().argsort()].iloc[0] if len(deg_ign) else case_b
    case_d = runs_top[runs_top["first_alert"]].iloc[0]

    fig = plt.figure(figsize=(15.5, 13.5))
    gs = GridSpec(3, 6, figure=fig, hspace=0.55, wspace=0.7, height_ratios=[1.1, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    _band(ax)
    ax.plot(g_top["cycle"], g_top["res_sigma"], color=viz_style.INK, lw=1.0, label="resíduo de T50 (σ)")
    _mark(ax, g_top, runs_top[runs_top["monitored"]], 70)
    fa = runs_top[runs_top["first_alert"]]["alert_cycle"].iloc[0]
    ax.axvline(fa, color=viz_style.STATUS[3], lw=1.5, ls="--")
    ax.axvspan(g_top["cycle"].min(), H, color=viz_style.STATUS[0], alpha=0.10, label=f"janela de calibração (≤ {H}, não monitorada)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8, frameon=False, ncol=3, loc="upper left")
    n_ign = int(((~runs_top["alert"]) & runs_top["monitored"]).sum())
    ax.set_title(f"(A) {subset}, motor {top}: {n_ign} desvios isolados ignorados; o alerta sai no ciclo {int(fa)}, quando 2 ciclos seguidos saem do envelope", fontsize=10.5, loc="left")
    ax.set_xlabel("ciclo")
    ax.set_ylabel("resíduo de T50 (σ saudável)")
    ax.set_ylim(min(-4.5, g_top["res_sigma"].min() - 0.5), g_top["res_sigma"].max() + 1.5)
    viz_style.style_axes(ax)

    for k, (run, ttl, col, label) in enumerate([
        (case_b, f"(B) motor {int(case_b['unit_id'])}, vida saudável:\n1 ciclo, {case_b['max_excess_sigma']:.2f}σ além do envelope", viz_style.INK_2, "b"),
        (case_c, f"(C) motor {int(case_c['unit_id'])}, em degradação:\n1 ciclo isolado, {case_c['max_excess_sigma']:.2f}σ além", viz_style.INK_2, "c"),
        (case_d, f"(D) motor {top}: 2 ou mais ciclos\nseguidos fora viram alerta", viz_style.STATUS[3], "d"),
    ]):
        axz = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        gz = rs[rs["unit_id"] == int(run["unit_id"])]
        _zoom(axz, gz, run, ttl, col, ru[(ru["unit_id"] == int(run["unit_id"])) & ru["monitored"] & (ru["end"] >= run["start"] - 14) & (ru["start"] <= run["end"] + 14)])

    axe = fig.add_subplot(gs[2, :3])
    pool_ign = runs[runs["monitored"] & ~runs["alert"]]["max_excess_sigma"]
    pool_per = runs[runs["monitored"] & runs["alert"]]["max_excess_sigma"]
    bins = np.linspace(0, max(8, pool_per.quantile(0.9)), 22)
    axe.hist(pool_ign, bins=bins, alpha=0.8, color=viz_style.STATUS[1], label=f"ignorados (n = {len(pool_ign)})", edgecolor=viz_style.SURFACE)
    axe.hist(pool_per, bins=bins, alpha=0.75, color=viz_style.STATUS[3], label=f"viraram alerta (n = {len(pool_per)})", edgecolor=viz_style.SURFACE)
    axe.set_xlabel("excesso máximo além do envelope (em σ do resíduo saudável)")
    axe.set_ylabel("sequências de desvio")
    axe.set_title("(E) Desvios ignorados são pequenos; os persistentes\nsão maiores (4 subconjuntos, motores de teste)", fontsize=10, loc="left")
    axe.legend(frameon=False, fontsize=8)
    viz_style.style_axes(axe)

    axf = fig.add_subplot(gs[2, 3:])
    x = np.arange(len(summ))
    axf.bar(x - 0.2, summ["false_alert_per100_without_persistence"], 0.4, color=viz_style.INK_2, label="alertar em todo desvio")
    axf.bar(x + 0.2, summ["false_alert_per100_with_persistence"], 0.4, color=viz_style.SERIES[0], label="exigir 2 ciclos seguidos")
    for xi, (a, b) in enumerate(zip(summ["false_alert_per100_without_persistence"], summ["false_alert_per100_with_persistence"])):
        axf.text(xi - 0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        axf.text(xi + 0.2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    axf.set_xticks(x)
    axf.set_xticklabels(summ["Subset"])
    axf.set_ylabel("alertas falsos por 100 ciclos de vida saudável")
    axf.set_title("(F) O que a persistência evita: alertas falsos\nna vida saudável (RUL = 125, após a calibração)", fontsize=10, loc="left")
    axf.legend(frameon=False, fontsize=8)
    viz_style.style_axes(axf)
    fig.suptitle("Pequenos desvios do envelope que não viram alerta (T50, sensor virtual, limite 3σ, persistência de 2 ciclos)", fontsize=12.5, y=0.93)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"top_engine": top, "case_b": (int(case_b["unit_id"]), int(case_b["start"])), "case_c": (int(case_c["unit_id"]), int(case_c["start"])), "case_d": (top, int(case_d["start"]))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--figure", type=Path, default=DEFAULT_RESULTS_DIR / "envelope_ignored_cases" / "F17_envelope_desvios_ignorados.png")
    p.add_argument("--tables", type=Path, default=Path("tcc-writing/tables"))
    a = p.parse_args(argv)
    frames = []
    for s in SUBSETS:
        try:
            frames.append(fleet_envelope(a.results, s)[0])
        except FileNotFoundError:
            print(f"[skip] {s}: tabelas do pipeline ausentes")
    r = pd.concat(frames, ignore_index=True)
    runs = breach_runs(r)
    summ = summarize(r, runs)
    a.tables.mkdir(parents=True, exist_ok=True)
    summ.to_csv(a.tables / "envelope_desvios_resumo.csv", index=False)
    runs[runs["monitored"]].sort_values(["Subset", "unit_id", "start"]).to_csv(a.tables / "envelope_desvios_casos.csv", index=False)
    cases = make_figure(r, runs, summ, a.figure)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.round(2).T.to_string())
    print(cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

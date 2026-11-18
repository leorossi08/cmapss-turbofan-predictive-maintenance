"""Interactive (Plotly) figures for the TCC, behind one clickable index page.

WHAT: builds every interactive figure from the CSVs already written by the pipeline and groups them in
    results/plotly/index.html        <- the only file you need to open
    results/plotly/tcc/*.html        main figures (one per planned TCC figure; see TCC_ESQUELETO.md)
    results/plotly/exploracao/*.html extra figures to dig into the data
    results/plotly/assets/plotly.min.js   shared offline copy of Plotly (works from file://, no internet)
WHY: the PNGs are fixed pictures; here you can hover for exact values, toggle detectors/models in the
legend, pick the subset / fault / engine in dropdowns, and export each figure as a PNG (camera icon in the
modebar, 3x scale) for the text.

Run (from the repo root):
    .venv/bin/python -m pdm_cmaps.reporting.plotly_hub --open        # build everything and open the index in the browser
    .venv/bin/python -m pdm_cmaps.reporting.plotly_hub --only t05 e01 # rebuild a few figures
A figure whose data is missing is skipped and listed in the index with the command that creates the data.

Every figure is a pure function  data (CSV) -> plotly Figure  or an "explorer" page (a generic
dropdown/checkbox time-series viewer driven by embedded JSON). Colors come from viz_style (categorical
slots blue/orange/aqua, status palette only for criticality), text in ink tokens.
"""

from __future__ import annotations

from pdm_cmaps.config import DEFAULT_RESULTS_DIR

import argparse
import datetime as dt
import html
import json
import re
import subprocess
import unicodedata
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from pdm_cmaps.reporting import viz_style

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SUBSET_COLORS = dict(zip(SUBSETS, [*viz_style.SERIES, viz_style.INK_2]))
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
ADOPTED_H = 25  # healthy window adopted in the TCC (detection sweep 25/35/50)
PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "figura_tcc"}}
VERIFIED_RESET = {("FD004", 179): 242}  # core reset found by rul_prediction_plots.scan_discontinuities (see its docstring)
DET_COLORS = {"T50 bruto (sem regime)": (viz_style.SERIES[0], "dash"), "T50 EWMA (com regime)": (viz_style.SERIES[0], "solid"),
              "MEWMA bruto (sem regime)": (viz_style.SERIES[1], "dash"), "MEWMA (13 resíduos, com regime)": (viz_style.SERIES[1], "solid")}
DET_COLORS["T50 EWMA (com regime, sem portão)"] = (viz_style.SERIES[2], "solid")
KINDS_PT = {"degrau": "degrau", "rampa": "rampa", "um_regime": "um regime", "variancia": "variância", "vies_sensor": "viés de sensor"}

TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=viz_style.INK, size=13), paper_bgcolor=viz_style.SURFACE, plot_bgcolor=viz_style.SURFACE,
    colorway=list(viz_style.SERIES) + [viz_style.INK_2, viz_style.MUTED],
    xaxis=dict(gridcolor=viz_style.GRID, linecolor=viz_style.AXIS, zeroline=False, ticks="", automargin=True),
    yaxis=dict(gridcolor=viz_style.GRID, linecolor=viz_style.AXIS, zeroline=False, ticks="", automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(font=dict(family=FONT)), margin=dict(l=70, r=30, t=90, b=70)))


class MissingData(Exception):
    """Raised by a builder when an input table does not exist yet; the message says how to create it."""


@dataclass
class Ctx:
    results: Path
    out: Path
    repo: Path = Path(".")

    def csv(self, rel: str, hint: str) -> pd.DataFrame:
        path = self.results / rel
        if not path.exists():
            raise MissingData(f"falta {path}: {hint}")
        return pd.read_csv(path)

    def glob_csv(self, pattern: str, hint: str) -> pd.DataFrame:
        files = sorted(self.results.glob(pattern))
        if not files:
            raise MissingData(f"falta {self.results / pattern}: {hint}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


@dataclass
class Chart:
    id: str
    section: str  # "tcc" | "exploracao"
    title: str
    desc: str
    ref: str      # where it is used in the TCC
    builder: Callable[[Ctx], "go.Figure | str"]
    tags: list[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        ascii_title = unicodedata.normalize("NFKD", self.title).encode("ascii", "ignore").decode()
        return f"{self.id}_{re.sub(r'[^a-z0-9]+', '-', ascii_title.lower()).strip('-')[:48]}.html"


# ============================================================================
# page helpers
# ============================================================================

def _layout(fig: go.Figure, title: str, subtitle: str = "", height: int = 560) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, title=dict(text=f"<b>{title}</b>" + (f"<br><sup>{subtitle}</sup>" if subtitle else ""), x=0.01, xanchor="left", y=0.97, yanchor="top"))
    return fig


def _nav(title: str, back: str = "../index.html") -> str:
    return (f'<div style="font-family:{FONT};padding:12px 20px;border-bottom:1px solid {viz_style.GRID};background:{viz_style.SURFACE}">'
            f'<a href="{back}" style="color:{viz_style.SERIES[0]};text-decoration:none;font-weight:600">← Índice dos gráficos</a>'
            f'<span style="color:{viz_style.MUTED};margin-left:14px">{html.escape(title)}</span></div>')


def write_figure(fig: go.Figure, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page = fig.to_html(full_html=True, include_plotlyjs="../assets/plotly.min.js", config=PLOT_CONFIG)
    page = page.replace("<head>", f"<head><title>{html.escape(title)}</title>", 1)
    page = page.replace("<body>", f'<body style="margin:0;background:{viz_style.SURFACE}">{_nav(title)}', 1)
    path.write_text(page, encoding="utf-8")


def _target(ctx: "Ctx", cid: str) -> Path:
    chart = next(c for c in CHARTS if c.id == cid)
    return ctx.out / chart.section / chart.filename


def dropdown_figure(fig: go.Figure, options: list[tuple[str, list[tuple[go.BaseTraceType, int, int]]]], label: str = "", x: float = 0.0) -> go.Figure:
    """options = [(label, [(trace, row, col), ...])]; one option visible at a time (buttons toggle trace visibility)."""
    owner, k = [], 0
    for oi, (_, traces) in enumerate(options):
        for tr, row, col in traces:
            tr.visible = oi == 0
            if getattr(fig, "_grid_ref", None) is None:
                fig.add_trace(tr)
            else:
                fig.add_trace(tr, row=row, col=col)
            owner.append(oi)
            k += 1
    buttons = [dict(label=(f"{label}: {lab}" if label else lab), method="update", args=[{"visible": [o == oi for o in owner]}]) for oi, (lab, _) in enumerate(options)]
    has_grid = getattr(fig, "_grid_ref", None) is not None
    fig.update_layout(margin=dict(l=70, r=30, t=200, b=70),
                      updatemenus=[dict(buttons=buttons, direction="down", x=x, xanchor="left", y=1.0, yanchor="bottom", pad=dict(b=40 if has_grid else 12),
                                        showactive=True, bgcolor=viz_style.SURFACE, bordercolor=viz_style.AXIS, font=dict(size=13))])
    return fig


# ============================================================================
# generic explorer (dropdowns + checkboxes, embedded JSON)
# ============================================================================

EXPLORER_TEMPLATE = r"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><script src="../assets/plotly.min.js"></script>
<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--blue:#2a78d6}
body{margin:0;background:var(--surface);color:var(--ink);font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
nav{padding:12px 20px;border-bottom:1px solid var(--grid)} nav a{color:var(--blue);text-decoration:none;font-weight:600}
header{padding:14px 20px 0} h1{margin:0 0 4px;font-size:20px} p.sub{margin:0;color:var(--ink2);font-size:14px;max-width:1100px}
#controls{display:flex;flex-wrap:wrap;gap:14px 22px;padding:14px 20px;align-items:flex-end}
label{display:flex;flex-direction:column;font-size:12px;color:var(--ink2);gap:4px}
select,button{font:inherit;font-size:14px;padding:6px 9px;border:1px solid var(--axis);background:var(--surface);color:var(--ink);border-radius:6px}
button{cursor:pointer} #models{display:flex;flex-wrap:wrap;gap:6px 14px;padding:0 20px 8px}
#models label{flex-direction:row;align-items:center;gap:6px;font-size:13px;color:var(--ink)} .sw{width:12px;height:3px;display:inline-block}
#plot{height:620px;padding:0 10px} #note{padding:4px 20px 24px;color:var(--ink2);font-size:13px;max-width:1100px}
</style></head><body>
<nav><a href="../index.html">← Índice dos gráficos</a></nav>
<header><h1>__TITLE__</h1><p class="sub">__SUBTITLE__</p></header>
<div id="controls">
 <label>__SETLABEL__<select id="set"></select></label>
 <label>__UNITLABEL__<select id="unit"></select></label>
 <label id="modeWrap">Modo<select id="mode"><option value="series">Série ao longo dos ciclos</option><option value="scatter">Dispersão predito × real (todos os motores)</option></select></label>
 <button id="all">marcar todos</button><button id="none">desmarcar</button><button id="reset">padrão</button>
</div>
<div id="models"></div><div id="plot"></div><div id="note">__NOTE__</div>
<script>
const DATA=__DATA__, CFG=__CONFIG__;
const PAL=["#2a78d6","#eb6834","#1baf7a","#7c3aed","#d03b3b","#b8860b","#0f766e","#be185d","#4b5563","#0891b2","#65a30d","#9333ea"];
const $=id=>document.getElementById(id);
const base=n=>n.split(" · ")[0], isCtrl=n=>n.endsWith(" · controle");
let checked=new Set(CFG.defaultOn);
const setSel=$("set"), unitSel=$("unit"), modeSel=$("mode");
if(!CFG.scatter) $("modeWrap").style.display="none";
Object.keys(DATA).forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent=k;setSel.appendChild(o)});
function names(){return DATA[setSel.value].names}
function colorOf(n){const b=names().filter(x=>!isCtrl(x)||true).map(base);const u=[...new Set(b)];return PAL[u.indexOf(base(n))%PAL.length]}
function fillUnits(){const U=DATA[setSel.value].units;unitSel.innerHTML="";
 Object.keys(U).sort((a,b)=>a-b).forEach(u=>{const o=document.createElement("option");o.value=u;
  const mk=U[u].markers||[];o.textContent=(mk.some(m=>m.kind==="verified")?"★ ":mk.some(m=>m.kind==="candidate")?"◆ ":"")+CFG.unitLabel+" "+u;unitSel.appendChild(o)});
 if(CFG.defaultUnit&&DATA[setSel.value].units[CFG.defaultUnit]&&setSel.value===CFG.defaultSet)unitSel.value=CFG.defaultUnit}
function fillModels(){const box=$("models");box.innerHTML="";
 names().forEach(n=>{const l=document.createElement("label"),c=document.createElement("input");c.type="checkbox";c.checked=checked.has(n);
  c.onchange=()=>{c.checked?checked.add(n):checked.delete(n);draw()};
  const sw=document.createElement("span");sw.className="sw";sw.style.background=colorOf(n);if(isCtrl(n))sw.style.opacity=.45;
  l.appendChild(c);l.appendChild(sw);l.appendChild(document.createTextNode(n));box.appendChild(l)})}
function draw(){const S=DATA[setSel.value],U=S.units[unitSel.value];const mode=CFG.scatter?modeSel.value:"series";let tr=[],shapes=[],ann=[];
 if(mode==="series"){
  if(U.truth)tr.push({x:U.x,y:U.truth,name:CFG.truthName,mode:"lines",line:{color:"#0b0b0b",width:3},hovertemplate:"%{y:.1f}"});
  names().filter(n=>checked.has(n)).forEach(n=>tr.push({x:U.x,y:U.series[n],name:n,mode:"lines",line:{color:colorOf(n),width:2,dash:isCtrl(n)?"dot":"solid"},hovertemplate:"%{y:.1f}"}));
  (U.markers||[]).forEach(m=>{const col=(m.kind==="verified"||m.kind==="fault")?"#d03b3b":"#898781";
   shapes.push({type:"line",x0:m.x,x1:m.x,y0:0,y1:1,yref:"paper",line:{color:col,width:2,dash:(m.kind==="candidate"||m.kind==="eol")?"dot":"dash"}});
   ann.push({x:m.x,y:1,yref:"paper",text:m.label,showarrow:false,xanchor:"left",yanchor:"top",font:{size:11,color:col},bgcolor:"rgba(252,252,251,.85)"})});
  Plotly.react("plot",tr,{height:600,margin:{l:70,r:30,t:30,b:60},paper_bgcolor:"#fcfcfb",plot_bgcolor:"#fcfcfb",hovermode:"x unified",
   xaxis:{title:CFG.xlabel,gridcolor:"#e1e0d9",linecolor:"#c3c2b7"},yaxis:{title:CFG.ylabel,gridcolor:"#e1e0d9",linecolor:"#c3c2b7"},
   legend:{orientation:"h",y:-0.18},shapes:shapes,annotations:ann},CFG.plotConfig);
 } else {
  const all=Object.values(S.units),truth=[].concat(...all.map(u=>u.truth)),sel=U;let lo=0,hi=130;
  names().filter(n=>checked.has(n)).forEach(n=>{const y=[].concat(...all.map(u=>u.series[n]));
   tr.push({x:truth,y:y,name:n,type:"scattergl",mode:"markers",marker:{size:4,opacity:.25,color:colorOf(n)},hoverinfo:"skip"});
   tr.push({x:sel.truth,y:sel.series[n],name:n+" (motor "+unitSel.value+")",type:"scattergl",mode:"markers",marker:{size:7,color:colorOf(n),line:{color:"#0b0b0b",width:1}}})});
  tr.push({x:[lo,hi],y:[lo,hi],name:"predição perfeita",mode:"lines",line:{color:"#52514e",dash:"dash",width:1.5}});
  Plotly.react("plot",tr,{height:600,margin:{l:70,r:30,t:30,b:60},paper_bgcolor:"#fcfcfb",plot_bgcolor:"#fcfcfb",
   xaxis:{title:"RUL real (ciclos)",gridcolor:"#e1e0d9",linecolor:"#c3c2b7"},yaxis:{title:"RUL predito (ciclos)",gridcolor:"#e1e0d9",linecolor:"#c3c2b7"},legend:{orientation:"h",y:-0.18}},CFG.plotConfig)}}
setSel.onchange=()=>{fillUnits();fillModels();draw()};unitSel.onchange=draw;modeSel.onchange=draw;
$("all").onclick=()=>{names().forEach(n=>checked.add(n));fillModels();draw()};
$("none").onclick=()=>{checked.clear();fillModels();draw()};
$("reset").onclick=()=>{checked=new Set(CFG.defaultOn);fillModels();draw()};
if(CFG.defaultSet)setSel.value=CFG.defaultSet;fillUnits();fillModels();draw();
</script></body></html>"""


def _round(a, nd: int = 1) -> list:
    return [None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else round(float(v), nd) for v in a]


def explorer_page(path: Path, title: str, subtitle: str, sets: dict, cfg: dict, set_label: str = "Subconjunto", note: str = "") -> None:
    """sets = {set: {"names": [...], "units": {unit: {"x": [...], "truth": [...]|None, "series": {name: [...]}, "markers": [...]}}}}"""
    base_cfg = {"truthName": "RUL real", "xlabel": "Ciclo", "ylabel": "RUL (ciclos)", "unitLabel": "Motor", "scatter": False,
                "defaultOn": [], "plotConfig": PLOT_CONFIG}
    page = (EXPLORER_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__SUBTITLE__", html.escape(subtitle))
            .replace("__SETLABEL__", set_label).replace("__UNITLABEL__", cfg.get("unitLabel", "Motor")).replace("__NOTE__", html.escape(note))
            .replace("__DATA__", json.dumps(sets, separators=(",", ":"))).replace("__CONFIG__", json.dumps({**base_cfg, **cfg}, separators=(",", ":"))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


# ============================================================================
# TCC figures
# ============================================================================

def t01_regime_variance(ctx: Ctx) -> go.Figure:
    d = ctx.csv("monitoring_repeated/tables/regime_variance_by_subset.csv", "python -m pdm_cmaps.monitoring.layer --regime-variance-only")
    fig = go.Figure(go.Bar(x=d["Subset"], y=d["Between_regime_pct_mean"], error_y=dict(type="data", array=d["Between_regime_pct_std"].fillna(0)),
                           marker_color=[SUBSET_COLORS[s] for s in d["Subset"]], text=[f"{v:.2f} %" for v in d["Between_regime_pct_mean"]], textposition="outside",
                           customdata=np.c_[d["Regimes"], d["OOS_R2_mean"]], hovertemplate="%{x}<br>variância entre regimes: %{y:.3f} %<br>regimes: %{customdata[0]}<br>R² fora da amostra: %{customdata[1]:.3f}<extra></extra>"))
    fig.update_yaxes(title="Variância de T50 explicada pelo regime (%)", range=[0, 112])
    return _layout(fig, "O regime de operação domina a variância do T50", "Média de 3 seeds nos motores de teste; FD001/FD003 têm 1 regime e servem de controle", 520)


def t02_benchmark(ctx: Ctx) -> go.Figure:
    d = ctx.csv("benchmark_repeated/benchmark_repeated_summary.csv", "python -m pdm_cmaps.experiments.benchmark_repeated")
    order = d.groupby("Model")["RMSE_mean"].mean().sort_values().index.tolist()
    fig = make_subplots(rows=1, cols=1)
    opts = []
    for proto, lab in (("official", "Protocolo oficial (último ciclo do test_FDxxx)"), ("holdout", "Holdout 75/25 por motor (todos os ciclos)")):
        tr = []
        for s in SUBSETS:
            x = d[(d["Protocol"] == proto) & (d["Subset"] == s)].set_index("Model").reindex(order)
            tr.append((go.Bar(x=order, y=x["RMSE_mean"], error_y=dict(type="data", array=x["RMSE_std"].fillna(0)), name=s, marker_color=SUBSET_COLORS[s],
                              hovertemplate="%{x}<br>RMSE %{y:.2f} ciclos<extra>" + s + "</extra>"), 1, 1))
        opts.append((lab, tr))
    dropdown_figure(fig, opts, "Protocolo")
    fig.update_layout(barmode="group", legend_title="Subconjunto")
    fig.update_yaxes(title="RMSE (ciclos), menor é melhor")
    fig.update_xaxes(tickangle=-35)
    return _layout(fig, "Modelos de RUL: RMSE por subconjunto", "Média ± desvio de 3 seeds. No protocolo oficial o desvio é só de inicialização (modelos determinísticos têm 0)", 620)


def t03_detection(ctx: Ctx) -> go.Figure:
    a = ctx.csv("monitoring_repeated/tables/monitoring_repeated_spc_summary.csv", "python -m pdm_cmaps.monitoring.layer")
    b = ctx.csv("monitoring_repeated/tables/monitoring_repeated_spc_mv_summary.csv", "python -m pdm_cmaps.monitoring.layer")
    metrics = [("detection_rate_pct", "Detecção (% dos motores)"), ("mean_lead_time_cycles", "Antecedência média (ciclos)"),
               ("mean_healthy_false_alarm_pct", "Falso alarme na janela saudável (%)"), ("premature_alarm_pct", "Motores com alarme prematuro (%)")]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[m[1] for m in metrics], horizontal_spacing=0.07)
    for i, (col, _) in enumerate(metrics, start=1):
        for df, name, color in ((a, "T50 EWMA (1 resíduo)", viz_style.SERIES[0]), (b, "MEWMA (13 resíduos)", viz_style.SERIES[1])):
            fig.add_trace(go.Bar(x=df["Subset"], y=df[f"{col}_mean"], error_y=dict(type="data", array=df[f"{col}_std"].fillna(0)), name=name, marker_color=color,
                                 showlegend=i == 1, legendgroup=name, hovertemplate="%{x}: %{y:.1f}<extra>" + name + "</extra>"), row=1, col=i)
    fig.update_layout(barmode="group")
    return _layout(fig, "Detecção antecipada: T50 sozinho × MEWMA de 13 resíduos", "Sensor virtual calibrado na janela saudável (H = 50, pipeline principal); média ± desvio de 3 seeds", 480)


def _power_traces(d: pd.DataFrame, subset: str, kinds: list[str], metric: str, ysd: str | None, show_legend: bool):
    out = []
    for j, kind in enumerate(kinds, start=1):
        for det, (col, dash) in DET_COLORS.items():
            x = d[(d["Subset"] == subset) & (d["kind"] == kind) & (d["Detector"] == det)].sort_values("magnitude")
            if x.empty:
                continue
            out.append((go.Scatter(x=x["magnitude"], y=x[metric], error_y=dict(type="data", array=x[ysd].fillna(0), thickness=1) if ysd else None, mode="lines+markers", name=det,
                                   line=dict(color=col, dash=dash, width=2.5), marker=dict(size=7), legendgroup=det, showlegend=show_legend and j == 1,
                                   hovertemplate="magnitude %{x}<br>poder %{y:.0f} %<extra>" + det + "</extra>"), 1, j))
    return out


def t05_ablation(ctx: Ctx) -> go.Figure:
    d = ctx.csv("injecao_deteccao/tabelas/pareado_far_resumo.csv", "python -m pdm_cmaps.experiments.fault_detection_study")
    d = d[(d["H"] == ADOPTED_H) & (d["budget"] == 0.2) & (d["t0"] == ADOPTED_H + 10)]
    kinds = [k for k in KINDS_PT if k in set(d["kind"])]
    fig = make_subplots(rows=1, cols=len(kinds), subplot_titles=[KINDS_PT[k] for k in kinds], horizontal_spacing=0.05, shared_yaxes=True)
    opts = [(s, _power_traces(d, s, kinds, "power_pct_mean", "power_pct_std", True)) for s in SUBSETS if s in set(d["Subset"])]
    dropdown_figure(fig, opts, "Subconjunto")
    fig.update_yaxes(range=[-3, 103], title_text="Poder de detecção (%)", col=1)
    fig.update_xaxes(title_text="magnitude da falha", matches=None)
    return _layout(fig, "Ablação do desacoplamento de regime: poder de detecção a falso alarme igualado", "Tracejado = sem remover o regime, cheio = com regime. Falso alarme ≤ 0,2 por 100 ciclos, H = 25, média ± dp de 3 seeds. Clique na legenda para ocultar um detector", 520)


def _curve_options(ctx: Ctx, kind: str) -> list:
    raw = ctx.glob_csv(f"injecao_deteccao/runs/*_H{ADOPTED_H}_{kind}.csv", "python -m pdm_cmaps.experiments.fault_detection_study")
    return raw


def t06_operating(ctx: Ctx) -> go.Figure:
    c = _curve_options(ctx, "curvas")
    c = c[(c["H"] == ADOPTED_H) & (c["t0"] == ADOPTED_H + 10)]
    m = c.groupby(["Subset", "kind", "magnitude", "Detector", "s"])[["far_ctrl_per100", "power_pct"]].mean().reset_index()
    fig = go.Figure()
    opts = []
    for (s, kind, mag), g in m.groupby(["Subset", "kind", "magnitude"], sort=True):
        tr = []
        for det, (col, dash) in DET_COLORS.items():
            x = g[g["Detector"] == det].sort_values("s")
            tr.append((go.Scatter(x=x["far_ctrl_per100"].clip(lower=1e-3), y=x["power_pct"], mode="lines", name=det, line=dict(color=col, dash=dash, width=2.5),
                                  customdata=x["s"], hovertemplate="falso alarme %{x:.3f}/100 ciclos<br>poder %{y:.0f} %<br>limite s = %{customdata:.2f}<extra>" + det + "</extra>"), 1, 1))
        opts.append((f"{s} · {KINDS_PT[kind]} {mag:g}", tr))
    dropdown_figure(fig, opts, "Subconjunto · falha · magnitude")
    fig.add_vline(x=0.2, line=dict(color=viz_style.MUTED, dash="dot"), annotation_text="orçamento 0,2", annotation_position="top")
    fig.update_xaxes(type="log", title="Falsos alarmes por 100 ciclos nos controles (escala log; 0 aparece em 0,001)")
    fig.update_yaxes(title="Poder de detecção (%)", range=[-3, 103])
    return _layout(fig, "Curvas operacionais: poder × falso alarme (limite varrido)", "Cada ponto da curva é um limite de controle diferente; escolha a falha no menu", 560)


def t07_cumulative(ctx: Ctx) -> go.Figure:
    d = ctx.glob_csv(f"injecao_deteccao/runs/*_H{ADOPTED_H}_atrasos.csv", "python -m pdm_cmaps.experiments.fault_detection_study")
    d = d[(d["H"] == ADOPTED_H) & (d["t0"] == ADOPTED_H + 10) & (d["budget"] == 0.2)]
    grid = np.arange(0, 81)
    fig = go.Figure()
    opts = []
    for (s, kind, mag), g in d.groupby(["Subset", "kind", "magnitude"], sort=True):
        tr = []
        for det, (col, dash) in DET_COLORS.items():
            x = g.loc[g["Detector"] == det, "delay"].to_numpy(dtype=float)
            y = [100 * np.mean(x <= v) if len(x) else 0 for v in grid]
            tr.append((go.Scatter(x=grid, y=y, mode="lines", line=dict(color=col, dash=dash, width=2.5, shape="hv"), name=det,
                                  hovertemplate="até %{x} ciclos: %{y:.0f} % detectadas<extra>" + det + "</extra>"), 1, 1))
        opts.append((f"{s} · {KINDS_PT[kind]} {mag:g}", tr))
    dropdown_figure(fig, opts, "Subconjunto · falha · magnitude")
    fig.update_xaxes(title="Ciclos após o início da falha (t0)")
    fig.update_yaxes(title="Séries detectadas até o atraso (%)", range=[-3, 103])
    return _layout(fig, "Detecção acumulada por atraso", "Falso alarme igualado a 0,2 por 100 ciclos, H = 25; séries nunca detectadas ficam censuradas no fim", 540)


def t08_min_magnitude(ctx: Ctx) -> go.Figure:
    d = ctx.csv("injecao_deteccao/tabelas/magnitude_minima_detectavel.csv", "python -m pdm_cmaps.experiments.fault_detection_study")
    d = d[(d["H"] == ADOPTED_H) & (d["budget"] == 0.2) & (d["t0"] == ADOPTED_H + 10)]
    dets = list(DET_COLORS)
    rows = d.pivot_table(index=["Subset", "kind"], columns="Detector", values="mdm_mediana").reindex(columns=dets).reset_index()
    cells = [rows["Subset"], rows["kind"].map(KINDS_PT)] + [[("não atinge" if pd.isna(v) else f"{v:g}") for v in rows[c]] for c in dets]
    fig = go.Figure(go.Table(header=dict(values=["Subconjunto", "Falha", *dets], fill_color=viz_style.GRID, align="left", font=dict(color=viz_style.INK, size=13)),
                             cells=dict(values=cells, align="left", fill_color=viz_style.SURFACE, height=26)))
    return _layout(fig, "Magnitude mínima detectável (poder ≥ 80 %)", "Mediana entre seeds; magnitude em fração da assinatura de fim de vida (degrau, rampa, um regime) ou em σ (variância, viés). Falso alarme ≤ 0,2 por 100 ciclos", 190 + 27 * len(rows))


def t04_window(ctx: Ctx) -> go.Figure:
    d = ctx.csv("injecao_deteccao/tabelas/pareado_far_resumo.csv", "python -m pdm_cmaps.experiments.fault_detection_study --windows 25 35 50")
    d = d[(d["budget"] == 0.2) & (d["t0"] == d["H"] + 10)]
    if d["H"].nunique() < 2:
        raise MissingData("a varredura de H não foi rodada: python -m pdm_cmaps.experiments.fault_detection_study --windows 25 35 50")
    options = [(("degrau", 0.25), "degrau 0,25"), (("rampa", 0.25), "rampa 0,25"), (("degrau", 0.5), "degrau 0,5")]
    fig = make_subplots(rows=2, cols=4, subplot_titles=list(SUBSETS), shared_xaxes=True, vertical_spacing=0.1, horizontal_spacing=0.05)
    opts = []
    for (kind, mag), lab in options:
        tr = []
        for j, s in enumerate(SUBSETS, start=1):
            for r, (m, sd) in enumerate((("power_pct_mean", "power_pct_std"), ("delay_med_mean", "delay_med_std")), start=1):
                for det, (col, dash) in DET_COLORS.items():
                    x = d[(d["Subset"] == s) & (d["kind"] == kind) & np.isclose(d["magnitude"], mag) & (d["Detector"] == det)].sort_values("H")
                    tr.append((go.Scatter(x=x["H"], y=x[m], error_y=dict(type="data", array=x[sd].fillna(0), thickness=1), mode="lines+markers", name=det,
                                          line=dict(color=col, dash=dash, width=2.5), legendgroup=det, showlegend=r == 1 and j == 1,
                                          hovertemplate="H=%{x}: %{y:.1f}<extra>" + det + "</extra>"), r, j))
        opts.append((lab, tr))
    dropdown_figure(fig, opts, "Falha")
    fig.update_yaxes(title_text="Poder (%)", range=[-3, 103], row=1, col=1)
    fig.update_yaxes(title_text="Atraso mediano (ciclos)", row=2, col=1)
    fig.update_xaxes(title_text="Janela saudável H (ciclos)", tickvals=sorted(d["H"].unique()), row=2)
    return _layout(fig, "Aumentar a janela saudável reduz a detecção", "A falso alarme igualado (0,2 por 100 ciclos): o poder cai e o atraso sobe com H. Adotado: H = 25 (a diferença para 50 é pequena). Média ± dp de 3 seeds", 720)


def t09_conformal(ctx: Ctx) -> go.Figure:
    d = ctx.csv("monitoring_repeated/tables/monitoring_repeated_uncertainty_summary.csv", "python -m pdm_cmaps.monitoring.layer")
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[r["mean_width_mean"]], y=[r["coverage_mean"]], mode="markers+text", text=[r["Subset"]], textposition="top center", name=r["Subset"],
                                 marker=dict(size=14, color=SUBSET_COLORS[r["Subset"]]), error_x=dict(type="data", array=[r["mean_width_std"]]), error_y=dict(type="data", array=[r["coverage_std"]]),
                                 hovertemplate=f"{r['Subset']}<br>cobertura {r['coverage_mean']:.3f}<br>largura {r['mean_width_mean']:.1f} ciclos<br>"
                                               f"cobertura p10 por motor {r['Per_engine_coverage_p10_mean']:.2f}<extra></extra>", showlegend=False))
    fig.add_hline(y=0.9, line=dict(color=viz_style.STATUS[3], dash="dash"), annotation_text="nominal 90 %", annotation_position="bottom right")
    fig.update_xaxes(title="Largura média do intervalo (ciclos)")
    fig.update_yaxes(title="Cobertura empírica", range=[0.8, 0.98])
    return _layout(fig, "Incerteza da RUL: cobertura do intervalo conformal", "Média ± dp de 3 seeds; FD003 fica abaixo do nominal (reportado, não corrigido)", 520)


def t10_cost(ctx: Ctx) -> go.Figure:
    d = ctx.csv("monitoring_repeated/tables/monitoring_repeated_costs_summary.csv", "python -m pdm_cmaps.monitoring.layer")
    policies = sorted(d["Policy"].unique(), key=lambda p: int(p.split("<=")[1]))
    strategies = ["point", "interval", "interval+anomaly", "interval+anomaly_mv"]
    colors = dict(zip(strategies, [viz_style.INK_2, viz_style.SERIES[0], viz_style.SERIES[1], viz_style.SERIES[2]]))
    fig = make_subplots(rows=1, cols=len(policies), subplot_titles=policies, shared_yaxes=True, horizontal_spacing=0.04)
    opts = []
    for s in SUBSETS:
        for lead in sorted(d["Lead"].unique()):
            tr = []
            for j, p in enumerate(policies, start=1):
                for st in strategies:
                    x = d[(d["Subset"] == s) & (d["Lead"] == lead) & (d["Policy"] == p) & (d["Strategy"] == st)].sort_values("Cost_ratio")
                    tr.append((go.Scatter(x=x["Cost_ratio"], y=x["Relative_to_oracle_mean"], mode="lines+markers", name=st, line=dict(color=colors[st], width=2.5), legendgroup=st,
                                          showlegend=j == 1, hovertemplate="razão de custo %{x}:1<br>%{y:.2f}× o oráculo<extra>" + st + "</extra>"), 1, j))
            opts.append((f"{s} · antecedência logística {lead}", tr))
    dropdown_figure(fig, opts, "Subconjunto · prazo de logística")
    fig.update_xaxes(type="log", title_text="Razão custo falha : custo preventivo")
    fig.update_yaxes(type="log", title_text="Custo por ciclo relativo ao oráculo (1 = ideal)", col=1)
    return _layout(fig, "Custo da decisão por estratégia de criticidade", "Três políticas de limiar (crítico ≤ 10/15/20). A estratégia 'point' só recebe o limiar ≤ 20, o que confunde a comparação com a margem de segurança", 560)


def t11_tradeoff(ctx: Ctx) -> go.Figure:
    d = ctx.csv("monitoring_repeated/tables/monitoring_repeated_summary.csv", "python -m pdm_cmaps.monitoring.layer")
    sym = {"critical<=10": "circle", "critical<=15": "square", "critical<=20": "diamond"}
    strategies = ["point", "interval", "interval+anomaly", "interval+anomaly_mv"]
    colors = dict(zip(strategies, [viz_style.INK_2, viz_style.SERIES[0], viz_style.SERIES[1], viz_style.SERIES[2]]))
    fig = go.Figure()
    opts = []
    for s in (*SUBSETS,):
        tr = []
        for st in strategies:
            x = d[(d["Subset"] == s) & (d["Strategy"] == st)]
            tr.append((go.Scatter(x=100 * x["False_alarm_rate_normal_mean"], y=100 * x["Missed_critical_rate_mean"], mode="markers", name=st,
                                  marker=dict(size=13, color=colors[st], symbol=[sym.get(p, "circle") for p in x["Policy"]], line=dict(color=viz_style.SURFACE, width=2)),
                                  error_x=dict(type="data", array=100 * x["False_alarm_rate_normal_std"].fillna(0), thickness=1), error_y=dict(type="data", array=100 * x["Missed_critical_rate_std"].fillna(0), thickness=1),
                                  text=x["Policy"], hovertemplate="%{text}<br>falsos alertas %{x:.1f} %<br>críticos perdidos %{y:.1f} %<extra>" + st + "</extra>"), 1, 1))
        opts.append((s, tr))
    dropdown_figure(fig, opts, "Subconjunto")
    fig.update_xaxes(title="Falsos alertas em estado normal (%)")
    fig.update_yaxes(title="Críticos não alertados (%)", rangemode="tozero")
    return _layout(fig, "Criticidade: críticos perdidos × falsos alertas", "Formas = política de limiar (círculo ≤ 10, quadrado ≤ 15, losango ≤ 20). Mais perto da origem é melhor", 560)


def _holdout_sets(ctx: Ctx, with_markers: bool = True) -> dict:
    sets = {}
    for s in SUBSETS:
        p = ctx.results / f"maintenance_guide_{s.lower()}" / "tables" / "rul_holdout_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        models = [c for c in df.columns if c.startswith("RUL_predicted_")]
        names = [c.replace("RUL_predicted_", "") for c in models]
        cand = ctx.results / "rul_predicoes" / s / "candidatos_descontinuidade.csv"
        cand_df = pd.read_csv(cand) if (cand.exists() and with_markers) else pd.DataFrame()
        units = {}
        for u, g in df.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
            mk = []
            if with_markers and (s, int(u)) in VERIFIED_RESET:
                mk.append({"x": VERIFIED_RESET[(s, int(u))], "label": "manutenção / reset do núcleo (ciclo %d)" % VERIFIED_RESET[(s, int(u))], "kind": "verified"})
            if len(cand_df):
                for r in cand_df[(cand_df["unit_id"] == u) & cand_df["candidate"] & ~cand_df["verified_maintenance"]].itertuples():
                    mk.append({"x": int(r.cycle), "label": "queda brusca de Nc (candidato, não verificado)", "kind": "candidate"})
            units[str(int(u))] = {"x": g["cycle"].astype(int).tolist(), "truth": _round(g["RUL_actual"]), "markers": mk,
                                  "series": {n: _round(g[c]) for n, c in zip(names, models)}}
        sets[s] = {"names": names, "units": units}
    if not sets:
        raise MissingData("falta results/maintenance_guide_*/tables/rul_holdout_predictions.csv: python -m pdm_cmaps --all")
    return sets


def t12_pred_vs_real(ctx: Ctx) -> str:
    sets = _holdout_sets(ctx)
    path = _target(ctx, "t12")
    explorer_page(path, "RUL predito × real, por modelo e por motor", "Holdout (seed 42, modelos treinados com H = 50). Escolha subconjunto, motor e modelos; ★ = manutenção verificada, ◆ = queda brusca candidata.", sets,
                  {"scatter": True, "defaultOn": ["Ridge", "XGBoost (+temporal)", "PyTorch RNN (LSTM)"], "defaultSet": "FD004", "defaultUnit": "179", "truthName": "RUL real (rótulo, teto 125)"},
                  note="O rótulo de RUL é linear por partes a partir da falha final (teto 125) e não reinicia na manutenção; após o reset do motor 179 (ciclo 242) os modelos saltam de volta, o rótulo não.")
    return str(path)


def t13_motor179(ctx: Ctx) -> go.Figure:
    from pdm_cmaps.data.loader import CMapssLoader
    from pdm_cmaps.detection.regime import RegimeDecoupler

    raw = ctx.repo / "data" / "CMaps"
    if not (raw / "train_FD004.txt").exists():
        raise MissingData(f"falta {raw}/train_FD004.txt (dados do C-MAPSS)")
    d = RegimeDecoupler(n_clusters=6, random_state=42).fit_transform(CMapssLoader().load("FD004"))
    m = d[d["unit_id"] == 179].sort_values("cycle").reset_index(drop=True)
    sensors = {"Nc": "Nc (rotação do núcleo)", "T30": "T30 (saída HPC)", "T50": "T50 (escape)", "P30": "P30 (saída HPC)", "phi": "phi (combustível/Ps30)",
               "Nf": "Nf (rotação do fan)", "T24": "T24 (saída LPC)", "BPR": "BPR (razão de bypass)"}
    core = ["Nc", "T30", "T50", "P30", "phi"]
    pre, post, healthy = (m["cycle"].between(210, 241)), (m["cycle"].between(242, 270)), (m["cycle"] <= 50)
    z, delta = {}, []
    for s in sensors:
        base = m.groupby("regime_id")[s].transform("mean")
        res = m[s] - base
        z[s] = (res / (res[healthy].std() or 1.0)).rolling(5, min_periods=1).mean()
        delta.append((sensors[s], s in core, float((100 * res / base)[post].mean() - (100 * res / base)[pre].mean()), float((res[post].mean() - res[pre].mean()) / (res[healthy].std() or 1.0))))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.64, 0.36], horizontal_spacing=0.09,
                        subplot_titles=("Resíduo por regime, em desvios-padrão da janela saudável (média móvel de 5)", "Mudança do resíduo em 242 (média 242–270 menos 210–241), em % do valor"))
    pal = [viz_style.SERIES[0], viz_style.SERIES[1], viz_style.SERIES[2], viz_style.STATUS[3], viz_style.STATUS[1], "#7c3aed", viz_style.INK_2, "#0f766e"]
    for i, s in enumerate(sensors):
        fig.add_trace(go.Scatter(x=m["cycle"], y=z[s], name=sensors[s], mode="lines", line=dict(width=2.2, color=pal[i], dash="solid" if s in core else "dash"), legendgroup=s), row=1, col=1)
    fig.add_vrect(x0=1, x1=50, fillcolor=viz_style.STATUS[0], opacity=0.10, line_width=0, row=1, col=1)
    fig.add_vline(x=242, line=dict(color=viz_style.STATUS[3], dash="dash", width=2), row=1, col=1)
    fig.add_annotation(x=242, y=1, xref="x", yref="paper", text=" ciclo 242", showarrow=False, xanchor="left", yanchor="top", font=dict(color=viz_style.STATUS[3]))
    delta.sort(key=lambda t: abs(t[2]))
    fig.add_trace(go.Bar(y=[t[0] for t in delta], x=[t[2] for t in delta], orientation="h", marker_color=[viz_style.STATUS[3] if t[1] else viz_style.SERIES[0] for t in delta],
                         text=[f"{t[2]:+.2f} %" for t in delta], textposition="outside", customdata=[t[3] for t in delta], showlegend=False,
                         hovertemplate="%{y}<br>%{x:+.3f} % do valor<br>%{customdata:+.1f} σ saudável<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Ciclo do motor 179", row=1, col=1)
    fig.update_xaxes(title_text="Mudança (% do valor do sensor)", range=[-3.4, 2.6], row=1, col=2)
    return _layout(fig, "Motor 179 (FD004): queda abrupta em quase todos os sensores no ciclo 242",
                   "Em σ da janela saudável todos mudam (Nf, com variância saudável minúscula, até mais que o núcleo); em % do valor o núcleo (vermelho) muda 0,6 a 2,4 %, Nf e T24 menos de 0,13 %, e o BPR 1,5 %", 600)


# ============================================================================
# exploration figures
# ============================================================================

def e01_control_chart(ctx: Ctx) -> str:
    sets = {}
    for s in SUBSETS:
        base = ctx.results / f"maintenance_guide_{s.lower()}" / "tables"
        if not (base / "virtual_sensor_residuals_t50.csv").exists():
            continue
        r = pd.read_csv(base / "virtual_sensor_residuals_t50.csv")
        al = pd.read_csv(base / "fleet_anomalies_t50.csv").set_index("unit_id") if (base / "fleet_anomalies_t50.csv").exists() else pd.DataFrame()
        units = {}
        for u, g in r.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
            mk = []
            if len(al) and u in al.index and pd.notna(al.loc[u, "first_alarm_cycle"]):
                mk.append({"x": int(al.loc[u, "first_alarm_cycle"]), "label": f"1º alarme (ciclo {int(al.loc[u, 'first_alarm_cycle'])})", "kind": "fault"})
            mk.append({"x": int(g["cycle"].max()), "label": "fim de vida (EOL)", "kind": "eol"})
            units[str(int(u))] = {"x": g["cycle"].astype(int).tolist(), "truth": None, "markers": mk,
                                  "series": {"EWMA de |z| do resíduo de T50": _round(g["ewma_abs_z"], 3), "Limite de controle (UCL)": _round(g["ewma_ucl"], 3)}}
        sets[s] = {"names": ["EWMA de |z| do resíduo de T50", "Limite de controle (UCL)"], "units": units}
    if not sets:
        raise MissingData("falta results/maintenance_guide_*/tables/virtual_sensor_residuals_t50.csv: python -m pdm_cmaps --all")
    path = _target(ctx, "e01")
    explorer_page(path, "Carta de controle do sensor virtual (T50), por motor", "EWMA do resíduo padronizado contra o limite de controle, nos motores de teste (seed 42, H = 50).", sets,
                  {"defaultOn": ["EWMA de |z| do resíduo de T50", "Limite de controle (UCL)"], "ylabel": "EWMA de |z|", "defaultSet": "FD004"})
    return str(path)


def e02_benchmark_heatmap(ctx: Ctx) -> go.Figure:
    d = ctx.csv("benchmark_repeated/benchmark_repeated_summary.csv", "python -m pdm_cmaps.experiments.benchmark_repeated")
    order = d.groupby("Model")["RMSE_mean"].mean().sort_values().index.tolist()
    fig = go.Figure()
    opts = []
    for proto in ("official", "holdout"):
        for metric, lab, scale in (("RMSE_mean", "RMSE (ciclos)", "Blues"), ("R2_mean", "R²", "Blues_r")):
            p = d[d["Protocol"] == proto].pivot(index="Model", columns="Subset", values=metric).reindex(order)
            opts.append((f"{proto} · {lab}", [(go.Heatmap(z=p.values, x=list(p.columns), y=list(p.index), colorscale=scale, text=np.round(p.values, 2), texttemplate="%{text}",
                                                          hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>", colorbar=dict(title=lab)), 1, 1)]))
    dropdown_figure(fig, opts, "Protocolo · métrica")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Todos os modelos × subconjuntos", "Média de 3 seeds; modelos ordenados pelo RMSE médio", 640)


def e03_error_by_engine(ctx: Ctx) -> go.Figure:
    d = ctx.glob_csv("rul_predicoes/*/metricas_por_motor.csv", "python -m pdm_cmaps.experiments.rul_prediction_plots")
    fig = go.Figure()
    order = d.groupby("Model")["RMSE"].median().sort_values().index.tolist()
    opts = []
    for s in SUBSETS:
        x = d[d["Subset"] == s]
        if x.empty:
            continue
        opts.append((s, [(go.Box(x=g["Model"], y=g["RMSE"], name=m, marker_color=viz_style.SERIES[0], boxpoints="outliers", customdata=g["unit_id"], showlegend=False,
                                 hovertemplate="motor %{customdata}<br>RMSE %{y:.1f}<extra>" + m + "</extra>"), 1, 1) for m, g in x.groupby("Model") if m in order]))
    dropdown_figure(fig, opts, "Subconjunto")
    fig.update_xaxes(categoryorder="array", categoryarray=order, tickangle=-35)
    fig.update_yaxes(title="RMSE por motor (ciclos)")
    return _layout(fig, "Erro de RUL por motor e por modelo (holdout)", "Cada caixa resume os motores do holdout; passe o mouse nos pontos para ver o número do motor", 620)


def e04_window_calibration(ctx: Ctx) -> go.Figure:
    d = ctx.csv("healthy_window_study/tables/healthy_window_summary.csv", "python -m pdm_cmaps.experiments.healthy_window_study")
    metrics = {"contamination_pct_mean": "Contaminação da calibração (% linhas com RUL < 125)", "calib_cv_pct_mean": "Ruído de calibração (CV bootstrap, %)",
               "far_common_pct_mean": "Falso alarme por ciclo (ciclos > 100 com RUL ≥ 125, %)", "early_alarm_pct_mean": "Motores com alarme precoce (%)",
               "mean_lead_time_detected_cycles_mean": "Antecedência média (ciclos)", "detection_rate_pct_mean": "Taxa de detecção (%)"}
    fig = go.Figure()
    opts = []
    for col, lab in metrics.items():
        tr = []
        for det in d["Detector"].unique():
            for s in SUBSETS:
                x = d[(d["Detector"] == det) & (d["Subset"] == s)].sort_values("H_mean")
                tr.append((go.Scatter(x=x["H_mean"], y=x[col], mode="lines+markers", name=f"{s} · {det}", line=dict(color=SUBSET_COLORS[s], dash="solid" if det.startswith("MEWMA") else "dash", width=2.2),
                                      text=x["Window_kind"], hovertemplate="%{text}: %{y:.2f}<extra>" + f"{s} · {det}" + "</extra>"), 1, 1))
        opts.append((lab, tr))
    dropdown_figure(fig, opts, "Métrica")
    fig.update_xaxes(title="Janela saudável H (ciclos; 25 % da vida ≈ 50 a 58)")
    return _layout(fig, "Estudo da janela saudável: calibração e métricas independentes de H", "Cheio = MEWMA, tracejado = T50 EWMA; média de 3 seeds", 620)


def e05_power_grid(ctx: Ctx) -> go.Figure:
    d = ctx.csv("injecao_deteccao/tabelas/pareado_far_resumo.csv", "python -m pdm_cmaps.experiments.fault_detection_study --windows 50 75 100 --late-t0 --budgets 0.2 1.0")
    fig = make_subplots(rows=1, cols=4, subplot_titles=list(SUBSETS), shared_yaxes=True, horizontal_spacing=0.04)
    opts = []
    for (H, t0, budget, kind), g in d.groupby(["H", "t0", "budget", "kind"], sort=True):
        tr = []
        for j, s in enumerate(SUBSETS, start=1):
            for det, (col, dash) in DET_COLORS.items():
                x = g[(g["Subset"] == s) & (g["Detector"] == det)].sort_values("magnitude")
                if x.empty:
                    continue
                tr.append((go.Scatter(x=x["magnitude"], y=x["power_pct_mean"], error_y=dict(type="data", array=x["power_pct_std"].fillna(0), thickness=1), mode="lines+markers", name=det,
                                      line=dict(color=col, dash=dash, width=2.5), legendgroup=det, showlegend=j == 1, hovertemplate="magnitude %{x}: %{y:.0f} %<extra>" + det + "</extra>"), 1, j))
        opts.append((f"H={H} · t0={t0} · FAR≤{budget:g} · {KINDS_PT[kind]}", tr))
    dropdown_figure(fig, opts, "H · t0 · falso alarme · falha")
    fig.update_yaxes(range=[-3, 103], title_text="Poder (%)", col=1)
    fig.update_xaxes(title_text="magnitude")
    return _layout(fig, "Grade completa de poder de detecção", "Todas as combinações de janela H, início t0, orçamento de falso alarme e tipo de falha que foram rodadas", 560)


def e06_default_rule(ctx: Ctx) -> go.Figure:
    d = ctx.csv("injecao_deteccao/tabelas/regra_padrao_s1.csv", "python -m pdm_cmaps.experiments.fault_detection_study")
    d = d[(d["H"] == ADOPTED_H)].groupby(["Subset", "Detector"])[["far_ctrl_per100", "arl0"]].mean().reset_index()
    fig = go.Figure()
    for det, (col, dash) in DET_COLORS.items():
        x = d[d["Detector"] == det]
        fig.add_trace(go.Bar(x=x["Subset"], y=x["far_ctrl_per100"], name=det, marker_color=col, marker_pattern_shape="/" if dash == "dash" else "",
                             customdata=x["arl0"], hovertemplate="%{x}<br>%{y:.3f} por 100 ciclos<br>ARL0 ≈ %{customdata:.0f} ciclos<extra>" + det + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Falsos alarmes por 100 ciclos (controles)")
    return _layout(fig, "Regra padrão (limite s = 1): falso alarme nos controles saudáveis", "ARL0 infinito (nenhum alarme observado) aparece como barra zero; é um limite inferior, não uma medida", 500)


def e07_transplant(ctx: Ctx) -> str:
    sets = {}
    for s in SUBSETS:
        base = ctx.results / "injecao_de_falhas" / f"{s}_seed42"
        if not (base / "saidas" / "predicoes_rul.csv").exists():
            continue
        pred, man = pd.read_csv(base / "saidas" / "predicoes_rul.csv"), pd.read_csv(base / "selado" / "manifesto_rul.csv")
        models = [c for c in pred.columns if c not in ("blind_id", "cycle")]
        names = [f"{m} · {k}" for m in models for k in ("falha", "controle")]
        units = {}
        for rec, g in man.groupby("recipient"):
            f, c = g[g["role"] == "transplante"].iloc[0], g[g["role"] == "controle"].iloc[0]
            pf, pc = (pred[pred["blind_id"] == int(i)].sort_values("cycle") for i in (f["blind_id"], c["blind_id"]))
            cyc = pf["cycle"].to_numpy()
            series = {}
            for m in models:
                series[f"{m} · falha"], series[f"{m} · controle"] = _round(pf[m]), _round(pc[m])
            units[str(int(rec))] = {"x": cyc.tolist(), "truth": _round(np.minimum(int(f["eol_cycle"]) - cyc, 125)), "series": series,
                                    "markers": [{"x": int(f["t0"]), "label": f"início da falha (t0 = {int(f['t0'])}), só nós sabíamos", "kind": "fault"}]}
        sets[s] = {"names": names, "units": units}
    if not sets:
        raise MissingData("falta results/injecao_de_falhas: python -m pdm_cmaps.experiments.fault_injection --subset FD004 --include-deep")
    path = _target(ctx, "e07")
    explorer_page(path, "Transplante de degradação: como cada modelo de RUL reage", "Motor receptor (saudável) com a degradação de outro motor a partir de t0. Cheio = com falha; pontilhado = o mesmo motor sem falha (controle).", sets,
                  {"defaultOn": ["Ridge · falha", "Ridge · controle", "XGBoost (+temporal) · falha", "PyTorch RNN (LSTM) · falha"], "truthName": "RUL verdadeira (por construção)", "unitLabel": "Receptor"},
                  note="Análise secundária: o foco do TCC é a detecção antecipada. A assinatura transplantada vem de outro motor, plausível mas não física.")
    return str(path)


def e08_candidates(ctx: Ctx) -> go.Figure:
    d = ctx.glob_csv("rul_predicoes/*/candidatos_descontinuidade.csv", "python -m pdm_cmaps.experiments.rul_prediction_plots")
    fig = go.Figure()
    opts = []
    for s in SUBSETS:
        x = d[d["Subset"] == s].sort_values("drop_sigma").head(25)
        if x.empty:
            continue
        colors = [viz_style.STATUS[3] if v else (viz_style.INK_2 if c else viz_style.MUTED) for v, c in zip(x["verified_maintenance"], x["candidate"])]
        opts.append((s, [(go.Bar(x=[f"motor {u}" for u in x["unit_id"]], y=x["drop_sigma"], marker_color=colors, customdata=np.c_[x["cycle"], x["eol_cycle"]],
                                 hovertemplate="%{x}<br>queda de %{y:.1f}σ no ciclo %{customdata[0]} (EOL %{customdata[1]})<extra></extra>"), 1, 1)]))
    dropdown_figure(fig, opts, "Subconjunto")
    fig.add_hline(y=-3, line=dict(color=viz_style.MUTED, dash="dot"), annotation_text="limiar de candidato (−3σ)")
    fig.update_yaxes(title="Maior queda sustentada de Nc (σ da janela saudável)")
    return _layout(fig, "Busca de manutenção/reset: maior queda de Nc por motor", "Vermelho = manutenção verificada (motor 179, ciclo 242); cinza escuro = candidato não verificado. Apenas o FD004 tem candidatos", 520)


def t14_ml_gain(ctx: Ctx) -> go.Figure:
    d = ctx.csv("falhas_dificeis/tabelas/diferenca_vs_media_regime.csv", "python -m pdm_cmaps.experiments.fault_hard_study")
    cases = [(c, f) for c in ("MEWMA", "sensor afetado") for f in ("processo", "sensor") if not d[(d["Chart"] == c) & (d["family"] == f)].empty]
    fig = make_subplots(rows=1, cols=len(cases), subplot_titles=[f"{c} · falhas de {f}" for c, f in cases], shared_xaxes=True, horizontal_spacing=0.18)
    colors = {"exogenous": viz_style.SERIES[0], "cross_sensor": viz_style.SERIES[1]}
    for j, (chart, fam) in enumerate(cases, start=1):
        x = d[(d["Chart"] == chart) & (d["family"] == fam)].sort_values(["Mode", "Estimator"])
        for mode in ("exogenous", "cross_sensor"):
            g = x[x["Mode"] == mode]
            fig.add_trace(go.Bar(y=g["Estimator"].str.replace(" (padrão)", "", regex=False), x=g["delta_pp"], orientation="h", name="exógeno (settings + ambiente)" if mode == "exogenous" else "cross-sensor (vê os outros sensores)",
                                 marker_color=colors[mode], legendgroup=mode, showlegend=j == 1,
                                 error_x=dict(type="data", symmetric=False, array=g["ci_hi"] - g["delta_pp"], arrayminus=g["delta_pp"] - g["ci_lo"]),
                                 customdata=np.c_[g["n_faults"], g["n_clusters"]], hovertemplate="%{y}<br>%{x:+.1f} pontos<br>%{customdata[0]} falhas, %{customdata[1]} motores<extra></extra>"), row=1, col=j)
        fig.add_vline(x=0, line=dict(color=viz_style.INK, width=1), row=1, col=j)
    fig.update_xaxes(title_text="poder − poder da média do regime (pontos)")
    fig.update_layout(barmode="group")
    return _layout(fig, "Quanto o modelo de ML ganha sobre a média do regime (sem ML)?", "Falhas sorteadas, falso alarme ≤ 0,2 por 100 ciclos, H = 25; barras de erro = IC 95 % por bootstrap sobre motores. Ganho só vale se o IC exclui 0", 520)


def e09_hard_power(ctx: Ctx) -> go.Figure:
    d = ctx.csv("falhas_dificeis/tabelas/poder_por_magnitude.csv", "python -m pdm_cmaps.experiments.fault_hard_study")
    fig = go.Figure()
    opts = []
    ests = list(dict.fromkeys(d["Estimator"]))
    pal = dict(zip(ests, [*viz_style.SERIES, viz_style.STATUS[3], viz_style.INK_2, viz_style.MUTED]))
    for chart in ("MEWMA", "sensor afetado"):
        for fam in ("processo", "sensor"):
            x = d[(d["Chart"] == chart) & (d["family"] == fam)]
            if x.empty:
                continue
            tr = []
            for (est, mode), g in x.groupby(["Estimator", "Mode"]):
                g = g.sort_values("mag_mid")
                tr.append((go.Scatter(x=g["mag_mid"], y=g["power_pct"], mode="lines+markers", name=f"{est} · {mode}", line=dict(color=pal[est], dash="solid" if mode == "exogenous" else "dash", width=2.5),
                                      customdata=g["n"], hovertemplate="magnitude %{x:.2f}: %{y:.0f} % (n=%{customdata})<extra>" + f"{est} · {mode}" + "</extra>"), 1, 1))
            opts.append((f"{chart} · falhas de {fam}", tr))
    dropdown_figure(fig, opts, "Detector · família")
    fig.update_xaxes(type="log", title="Magnitude (fração da assinatura para processo; σ para sensor)")
    fig.update_yaxes(title="Poder (%)", range=[-3, 103])
    return _layout(fig, "Falhas sorteadas: poder por magnitude, modelo e modo", "Cheio = exógeno, tracejado = cross-sensor; falso alarme ≤ 0,2 por 100 ciclos", 560)


def e10_envelope_cases(ctx: Ctx) -> str:
    from pdm_cmaps.experiments import envelope_ignored_cases as eic

    frames = []
    for sub in SUBSETS:
        try:
            frames.append(eic.fleet_envelope(ctx.results, sub)[0])
        except FileNotFoundError:
            continue
    if not frames:
        raise MissingData("falta results/maintenance_guide_*/tables/envelope_scored_trace_t50.csv: python -m pdm_cmaps --all")
    r = pd.concat(frames, ignore_index=True)
    runs = eic.breach_runs(r)
    sets = {}
    for sub, g in r.groupby("Subset"):
        units = {}
        for u, gu in g.groupby("unit_id"):
            mk = []
            for rr in runs[(runs["Subset"] == sub) & (runs["unit_id"] == u) & runs["monitored"]].itertuples():
                if rr.alert and not rr.first_alert:
                    continue  # only the first alert of an engine is drawn, to keep the plot readable
                mk.append({"x": int(rr.start), "kind": "fault" if rr.alert else "candidate",
                           "label": f"1º alerta (ciclo {int(rr.alert_cycle)})" if rr.alert else f"ignorado ({rr.max_excess_sigma:.1f}σ)"})
            units[str(int(u))] = {"x": gu["cycle"].astype(int).tolist(), "truth": None, "markers": mk,
                                  "series": {"resíduo de T50 (σ)": _round(gu["res_sigma"], 2), "limite superior (+3σ)": [3.0] * len(gu), "limite inferior (−3σ)": [-3.0] * len(gu)}}
        sets[sub] = {"names": ["resíduo de T50 (σ)", "limite superior (+3σ)", "limite inferior (−3σ)"], "units": units}
    path = _target(ctx, "e10")
    explorer_page(path, "Envelope do sensor virtual: desvios isolados ignorados e alertas", "Cada motor de teste (seed 42, H = 50). Linhas cinza pontilhadas = desvios de 1 ciclo ignorados pela regra de persistência; vermelha = primeiro alerta (2 ciclos seguidos fora do ±3σ).", sets,
                  {"defaultOn": ["resíduo de T50 (σ)", "limite superior (+3σ)", "limite inferior (−3σ)"], "ylabel": "resíduo de T50 (σ saudável)", "defaultSet": "FD004", "truthName": ""},
                  note="O envelope é previsão ± 3σ, com σ do resíduo saudável fora da amostra. Um ciclo fora do envelope não gera alerta; são necessários 2 ciclos seguidos (regra de persistência de Western Electric).")
    return str(path)


CHARTS = [
    Chart("t01", "tcc", "O regime domina a variância do T50", "Fração da variância explicada pelo regime, por subconjunto.", "5.1", t01_regime_variance, ["regime", "tese"]),
    Chart("t02", "tcc", "RMSE dos modelos de RUL", "Benchmark em 2 protocolos, 4 subconjuntos, 3 seeds.", "5.2", t02_benchmark, ["RUL", "modelos"]),
    Chart("t03", "tcc", "Detecção: T50 × MEWMA", "Detecção, antecedência, falso alarme e alarme prematuro.", "5.3", t03_detection, ["detecção"]),
    Chart("t04", "tcc", "Janela saudável: H maior reduz a detecção", "Poder e atraso × H (25, 35, 50), a falso alarme igualado. Adotado H = 25.", "5.4", t04_window, ["janela", "injeção"]),
    Chart("t05", "tcc", "Ablação do desacoplamento de regime", "Poder de detecção com e sem regime, 5 tipos de falha injetada.", "5.7", t05_ablation, ["injeção", "regime"]),
    Chart("t06", "tcc", "Curvas operacionais (poder × falso alarme)", "Limite varrido; escolha o subconjunto e a falha.", "5.7", t06_operating, ["injeção"]),
    Chart("t07", "tcc", "Detecção acumulada por atraso", "Fração de falhas detectadas até cada atraso.", "5.7", t07_cumulative, ["injeção"]),
    Chart("t08", "tcc", "Magnitude mínima detectável", "Tabela por subconjunto, falha e detector (poder ≥ 80 %).", "5.7", t08_min_magnitude, ["injeção"]),
    Chart("t09", "tcc", "Cobertura do intervalo conformal", "Cobertura empírica × largura do intervalo.", "5.5", t09_conformal, ["incerteza"]),
    Chart("t10", "tcc", "Custo da decisão por estratégia", "Custo relativo ao oráculo; subconjunto e prazo no menu.", "5.6", t10_cost, ["criticidade", "custo"]),
    Chart("t11", "tcc", "Críticos perdidos × falsos alertas", "Trade-off das estratégias e políticas de limiar.", "5.6", t11_tradeoff, ["criticidade"]),
    Chart("t12", "tcc", "RUL predito × real (explorador)", "Por motor e modelo, com a manutenção verificada do motor 179; modo dispersão.", "5.8", t12_pred_vs_real, ["RUL", "explorador"]),
    Chart("t13", "tcc", "Motor 179: queda abrupta no ciclo 242", "Todos os sensores em σ e a mudança em % do valor (núcleo × fan).", "5.1", t13_motor179, ["regime", "manutenção"]),
    Chart("t14", "tcc", "Ganho do ML sobre a média do regime", "Diferença pareada de poder com IC 95 %, exógeno × cross-sensor.", "5.8", t14_ml_gain, ["modelos", "injeção"]),
    Chart("e01", "exploracao", "Carta de controle por motor", "EWMA do resíduo de T50 e limite, com o 1º alarme.", "—", e01_control_chart, ["detecção", "explorador"]),
    Chart("e02", "exploracao", "Mapa de calor: modelos × subconjuntos", "RMSE e R² em ambos os protocolos.", "—", e02_benchmark_heatmap, ["RUL"]),
    Chart("e03", "exploracao", "Erro por motor e modelo", "Distribuição do RMSE por motor do holdout.", "—", e03_error_by_engine, ["RUL"]),
    Chart("e04", "exploracao", "Janela saudável: calibração", "Contaminação, ruído de calibração e métricas independentes de H.", "—", e04_window_calibration, ["janela"]),
    Chart("e05", "exploracao", "Grade completa de poder", "Todas as combinações de H, t0, falso alarme e falha.", "—", e05_power_grid, ["injeção"]),
    Chart("e06", "exploracao", "Regra padrão: falso alarme nos controles", "Falso alarme e ARL0 com o limite s = 1.", "—", e06_default_rule, ["injeção"]),
    Chart("e07", "exploracao", "Transplante de degradação (RUL)", "Reação dos modelos de RUL a uma falha com início conhecido (análise secundária).", "apêndice", e07_transplant, ["RUL", "injeção", "explorador"]),
    Chart("e09", "exploracao", "Falhas sorteadas: poder por magnitude", "Poder por modelo e modo ao longo das magnitudes sorteadas.", "—", e09_hard_power, ["injeção", "modelos"]),
    Chart("e10", "exploracao", "Envelope: desvios ignorados × alertas", "Motor a motor: quais desvios do envelope viram alerta e quais são ignorados.", "5.3", e10_envelope_cases, ["detecção", "envelope", "explorador"]),
    Chart("e08", "exploracao", "Busca de reset: quedas de Nc", "Candidatos a manutenção em todos os motores.", "—", e08_candidates, ["manutenção"]),
]


# ============================================================================
# index + build
# ============================================================================

INDEX_CSS = f"""
:root{{--surface:{viz_style.SURFACE};--ink:{viz_style.INK};--ink2:{viz_style.INK_2};--muted:{viz_style.MUTED};--grid:{viz_style.GRID};--axis:{viz_style.AXIS};--blue:{viz_style.SERIES[0]}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--surface);color:var(--ink);font-family:{FONT}}}
main{{max-width:1180px;margin:0 auto;padding:28px 20px 60px}} h1{{margin:0 0 6px;font-size:26px}} p.lead{{color:var(--ink2);margin:0 0 18px;max-width:820px;line-height:1.5}}
h2{{margin:34px 0 4px;font-size:19px}} h2 small{{font-weight:400;color:var(--muted);font-size:13px;margin-left:8px}}
input{{font:inherit;padding:9px 12px;border:1px solid var(--axis);border-radius:8px;width:100%;max-width:420px;background:var(--surface);color:var(--ink)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:12px}}
a.card{{display:block;border:1px solid var(--grid);border-radius:10px;padding:14px 16px;text-decoration:none;color:inherit;background:var(--surface);transition:border-color .12s,transform .12s}}
a.card:hover{{border-color:var(--blue);transform:translateY(-1px)}} a.card b{{display:block;font-size:15px;margin-bottom:4px}} a.card span{{display:block;color:var(--ink2);font-size:13px;line-height:1.4}} a.card span.tag{{display:inline-block}}
.meta{{margin-top:8px;font-size:12px;color:var(--muted)}} .tag{{display:inline-block;margin:6px 6px 0 0;padding:1px 8px;border:1px solid var(--grid);border-radius:99px;font-size:11px;color:var(--ink2)}}
.miss{{border:1px dashed var(--axis);border-radius:10px;padding:12px 14px;color:var(--ink2);font-size:13px;margin-top:10px}} code{{background:var(--grid);padding:1px 6px;border-radius:4px}}
footer{{margin-top:40px;color:var(--muted);font-size:12px}}
"""


def build_index(out: Path, built: list[Chart], skipped: list[tuple[Chart, str]]) -> None:
    def cards(section: str) -> str:
        items = []
        for c in built:
            if c.section != section:
                continue
            tags = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in c.tags)
            items.append(f'<a class="card" href="{section}/{c.filename}" data-q="{html.escape((c.title + " " + c.desc + " " + " ".join(c.tags)).lower())}"><b>{html.escape(c.title)}</b>'
                         f'<span>{html.escape(c.desc)}</span><div class="meta">Seção do TCC: {html.escape(c.ref)}</div>{tags}</a>')
        return f'<div class="grid">{"".join(items)}</div>'

    missing = "".join(f'<div class="miss"><b>{html.escape(c.title)}</b> indisponível. {html.escape(msg)}</div>' for c, msg in skipped)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        commit = ""
    page = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gráficos interativos do TCC</title><style>{INDEX_CSS}</style></head>
<body><main><h1>Gráficos interativos do TCC</h1>
<p class="lead">Manutenção preditiva em turbofans (NASA C-MAPSS). Clique em um gráfico para abrir; passe o mouse para ver os valores, clique na legenda para ocultar séries e use o ícone da câmera para exportar a figura em PNG.</p>
<input id="q" placeholder="Filtrar: regime, injeção, RUL, custo, janela…" autofocus>
<h2>Figuras principais do TCC <small>pasta <code>tcc/</code>: uma por figura planejada</small></h2>{cards("tcc")}
<h2>Exploração <small>pasta <code>exploracao/</code>: o resto, para investigar os dados</small></h2>{cards("exploracao")}
{('<h2>Indisponíveis</h2>' + missing) if missing else ''}
<footer>Gerado em {dt.datetime.now():%Y-%m-%d %H:%M}{(' · commit ' + commit) if commit else ''} · <code>python -m pdm_cmaps.reporting.plotly_hub --open</code> reconstrói tudo a partir de <code>results/</code>.</footer></main>
<script>const q=document.getElementById('q');q.oninput=()=>{{const t=q.value.toLowerCase().trim();document.querySelectorAll('a.card').forEach(a=>a.style.display=!t||a.dataset.q.includes(t)?'':'none')}}</script></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def build(ctx: Ctx, only: list[str] | None = None) -> tuple[list[Chart], list[tuple[Chart, str]]]:
    (ctx.out / "assets").mkdir(parents=True, exist_ok=True)
    (ctx.out / "assets" / "plotly.min.js").write_text(pyo.get_plotlyjs(), encoding="utf-8")
    built, skipped = [], []
    for c in CHARTS:
        if only and c.id not in only:
            if (ctx.out / c.section / c.filename).exists():
                built.append(c)
            continue
        for old in (ctx.out / c.section).glob(f"{c.id}_*.html"):  # drop stale files from a previous title
            if old.name != c.filename:
                old.unlink()
        try:
            res = c.builder(ctx)
            if not isinstance(res, str):  # explorer pages write their own file
                write_figure(res, ctx.out / c.section / c.filename, c.title)
            built.append(c)
            print(f"[ok]   {c.id} {c.title}", flush=True)
        except MissingData as e:
            skipped.append((c, str(e)))
            print(f"[skip] {c.id} {c.title}: {e}", flush=True)
    build_index(ctx.out, built, skipped)
    return built, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", type=Path, default=None, help="default: <results>/plotly")
    p.add_argument("--repo", type=Path, default=Path("."), help="folder with data/CMaps (for the motor 179 figure)")
    p.add_argument("--only", nargs="+", help="figure ids to rebuild (t01 … e09)")
    p.add_argument("--open", action="store_true", help="open the index in the default browser")
    a = p.parse_args(argv)
    out = a.output_dir or a.results / "plotly"
    ctx = Ctx(results=a.results, out=out, repo=a.repo)
    built, skipped = build(ctx, a.only)
    index = (out / "index.html").resolve()
    print(f"\n{len(built)} figuras, {len(skipped)} indisponíveis.\nAbra: {index.as_uri()}")
    if a.open:
        webbrowser.open(index.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

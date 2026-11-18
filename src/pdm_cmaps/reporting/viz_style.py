"""Shared figure style for the thesis figures (tokens + helpers), following the dataviz method.

Only colors, line/marker specs and axis styling live here.

Rules encoded here
  * Categorical series use fixed slots in fixed order (blue, orange, aqua). The first three validate
    all-pairs (scatter / small multiples). Aqua is below 3:1 contrast on the surface, so every figure
    that uses it also prints direct labels and has its CSV as the table equivalent.
  * Criticality levels use the STATUS palette (good / warning / serious / critical). Status colors
    never carry meaning alone: the level NAME is always printed next to the color.
  * Marks: 2 px lines, markers >= 8 px with a 2 px surface ring, hairline SOLID gridlines, no
    top/right spines. Text uses ink tokens, never a series color.
  * Print-only figures: no hover or dark mode. The CSV next to each figure is its table view.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (blue, orange, aqua)
STATUS = ("#0ca30c", "#fab219", "#ec835a", "#d03b3b")  # good, warning, serious, critical
LEVEL_NAMES_PT = ("Normal", "Atenção", "Alerta", "Crítico")

LINE_W = 2.0
MARKER_PT = 8.0  # points; >= 8 px at the 100 dpi reference
RING_W = 2.0


def apply_style() -> None:
    """Global rcParams: system sans, surface background, ink text tokens."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.axisbelow": True,
    })


def style_axes(ax, grid: str = "y") -> None:
    """Hairline solid recessive grid and axes; no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=0, pad=4)
    ax.grid(False)
    if grid in ("y", "both"):
        ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    if grid in ("x", "both"):
        ax.grid(axis="x", color=GRID, linewidth=0.8, linestyle="-")


def ring_marker_kwargs(color: str, size: float = MARKER_PT, hollow: bool = False) -> dict:
    """Marker with a 2 px surface ring so it stays legible over lines and other markers."""
    return {
        "markersize": size,
        "markerfacecolor": SURFACE if hollow else color,
        "markeredgecolor": color if hollow else SURFACE,
        "markeredgewidth": RING_W,
        "linestyle": "none",
    }

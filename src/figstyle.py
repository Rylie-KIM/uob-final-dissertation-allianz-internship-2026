"""Canonical figure style for the dissertation — one look across every notebook.

Import at the top of any notebook (notebooks add ``src`` to ``sys.path`` already)::

    import figstyle
    figstyle.apply()                       # sets fonts, sizes, grid, colour cycle
    ...
    fig, ax = plt.subplots(figsize=figstyle.FIG_1)
    ax.plot(...)                           # colours come from SERIES, in order
    figstyle.save(fig, "03_evaluation_gap")   # -> figures/03_evaluation_gap.png (+ .pdf)

The single most important thing this module does: it fixes the **order** in which
colours are assigned. ``apply()`` installs ``SERIES`` as matplotlib's colour cycle,
so the first plotted series is ``SERIES[0]``, the second ``SERIES[1]``, and so on —
identically in every figure, every notebook. Do not pass ``color=`` for ordinary
categorical series; let the cycle assign them. Reach for the named colours
(``PRIMARY``, ``CAUTION`` …) only when a mark carries a *fixed meaning*.

The values mirror the data-viz skill's CVD-validated light-surface palette; the
slot ordering is the colour-blind-safety mechanism, not cosmetic — keep it.

The prose spec that accompanies this module is ``notebook/FIGURE_TEMPLATE.md``.
"""
from __future__ import annotations

from pathlib import Path

# ── Ordered categorical palette ─────────────────────────────────────────────
# Assigned in THIS order to series 1, 2, 3, …  Never reorder; never cycle past 8
# (an 8th+ category folds into "Other" or becomes small multiples).
BLUE    = "#2a78d6"   # slot 1
RED     = "#e34948"   # slot 6
GREEN   = "#008300"   # slot 4

VIOLET  = "#4a3aa7"   # slot 5
ORANGE  = "#eb6834"   # slot 8

AQUA    = "#1baf7a"   # slot 2
YELLOW  = "#eda100"   # slot 3
MAGENTA = "#e87ba4"   # slot 7
SERIES = [BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE]

# ── Named roles (use only when the colour means something fixed) ─────────────
PRIMARY = BLUE       # the headline / recommended quantity
NEUTRAL = "#898781"  # baseline / "for reference" series
CAUTION = RED        # a biased / not-to-be-trusted quantity
GOOD    = "#0ca30c"  # meets target / desirable
WARNING = "#fab219"  # attention, sub-target

# ── Ink & chrome (text and structure never wear a series colour) ─────────────
INK        = "#0b0b0b"   # primary text: titles, annotations
INK_SOFT   = "#52514e"   # secondary text: axis labels
MUTED      = "#898781"   # ticks, minor labels
GRID       = "#e1e0d9"   # hairline gridlines
AXIS       = "#c3c2b7"   # spines / baseline

# ── Sequential single hue (continuous magnitude: heatmaps, density) ──────────
SEQUENTIAL = "Blues"     # matplotlib cmap name, light -> dark

# ── Figure sizes (inches) — sized to a ~6.3in dissertation text column ───────
FIG_1    = (6.3, 3.9)    # single panel
FIG_2    = (11.0, 4.0)   # two panels side by side
FIG_WIDE = (11.0, 3.6)   # one wide/timeline panel
FIG_SQ   = (5.0, 5.0)    # square (scatter, confusion matrix)

# figures/ lives next to src/ so saving works regardless of notebook cwd
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def apply() -> None:
    """Install the house style into matplotlib rcParams for the session."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        # typography — one sans face everywhere, no serif/display
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.0,
        "axes.titlesize": 13.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "figure.titlesize": 15.0,
        "figure.titleweight": "bold",
        # colour — the fixed categorical order
        "axes.prop_cycle": cycler(color=SERIES),
        # ink & chrome
        "text.color": INK,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SOFT,
        "ytick.labelcolor": INK_SOFT,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        # recessive grid, only where it helps reading
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        # de-boxed axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        # marks
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "patch.linewidth": 0.0,
        # surfaces & output
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, name: str) -> Path:
    """Save ``fig`` to ``figures/<name>.png`` (and ``.pdf`` for LaTeX) consistently.

    Returns the PNG path. PDF is vector — prefer it when inserting into the thesis.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png)
    return png
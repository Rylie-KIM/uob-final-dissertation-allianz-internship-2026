"""Canonical figure style for the dissertation — one look across every notebook.

Import at the top of any notebook (notebooks add ``src`` to ``sys.path`` already)::

    import figstyle
    figstyle.apply()                       # sets fonts, sizes, grid, colour cycle
    ...
    fig, ax = plt.subplots(figsize=figstyle.FIG_1)
    ax.plot(...)                           # colours come from SERIES, in order
    figstyle.save_fig(fig, "03_evaluation_gap")   # -> figures/03_evaluation_gap.png (+ .pdf)

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

# ── Alias / real-name routing (opt-in; SHAP / SHAP-DiD notebooks only) ───────────────
# A notebook that produces a real-Allianz-feature-name figure/table alongside an anonymised
# "alias_"-prefixed twin (00_SHAP.ipynb, shap_did/04_02, shap_did/04_03) sets ALIAS_SPLIT = True
# next to its own FIG_DIR assignment. save()/save_table() then route each output into
# FIG_DIR/alias/ or FIG_DIR/real_named/ by name alone, so a reader can treat "alias/" as the only
# thesis-safe half and never open "real_named/" — no other call site in those notebooks needs to
# change. A notebook that never sets it (03_01-03_05 etc.) keeps saving flat into FIG_DIR, exactly
# as before.
#
# An output that carries no feature name at all (a DiD/dose/concentration summary table, a score
# distribution) stays flat at FIG_DIR instead of either bucket — sharing it is safe, but it has
# nothing to do with the alias/real-name distinction. Missing a name from _NEUTRAL_SUFFIXES below
# is a cosmetic bug only (the file nests under real_named/ needlessly): the ONLY way anything
# reaches alias/ is its own name starting with "alias_", which every alias-twin call already does.
ALIAS_SPLIT = False

_NEUTRAL_SUFFIXES = (
    # 00_SHAP.ipynb — aggregate, no per-feature content
    "_training_config",
    "_feature_summary",
    "_03a_score_distribution",
    # shap_did/04_01_shap_did_inputs.ipynb — coverage table, no per-feature content
    "05_shap_did_input_coverage",
    # shap_did/04_02_shap_did_concentration.ipynb — DiD/dose/drift summary tables
    "04_02_shap_did_assumptions",
    "04_02_dose_context",
    "04_02_cross_version_estimate",
    "04_02_local_band_h_selection",
    "04_02_local_cross_version_estimate",
    "04_02_within_version_confound_check",
    "_04_02_score_drift_scatter",
    "_04_02_shared_claims_scores",
    "04_02_v2_v3_score_drift_and_dose",
    # shap_did/04_03_shap_did_mitigation_delta.ipynb — concentration/DiD summary tables
    "_04_03_step1_within_version",
    "04_03_step2_corruption_footprint",
    "04_03_step2_shap_did_delta",
    "04_03_step3_local_h_selection",
    "04_03_step3_shap_did_delta_local",
    "04_03_whole_vs_local",
)


def _out_dir(name: str) -> Path:
    """FIG_DIR itself, or its alias/real_named child, depending on ALIAS_SPLIT and ``name``."""
    if not ALIAS_SPLIT:
        return FIG_DIR
    if name.startswith("alias_"):
        return FIG_DIR / "alias"
    if any(name.endswith(suffix) for suffix in _NEUTRAL_SUFFIXES):
        return FIG_DIR
    return FIG_DIR / "real_named"


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


def save_fig(fig, name: str, subdir: str | None = None) -> Path:
    """Save ``fig`` to ``figures/<name>.png`` (and ``.pdf`` for LaTeX) consistently.

    ``subdir``, if given, nests one more level under the resolved directory (e.g. ``"internal"``
    for a claim-level table that should not be casually shared even though it lives in
    ``figures/`` — see ``save_table``).

    Returns the PNG path. PDF is vector — prefer it when inserting into the thesis.
    """
    out_dir = _out_dir(name)
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png)
    return png


def save_table(df, name: str, index: bool = True, subdir: str | None = None) -> Path:
    """Save ``df`` to ``figures/<name>.csv``, index kept by default (normally the feature name).

    The CSV twin of ``save()`` — same directory, same naming convention — for a table that
    backs a figure (SHAP importance, association) and is worth keeping in FULL, not just the
    ``head()`` a notebook prints. Pass ``index=False`` for a table whose index is a meaningless
    RangeIndex (e.g. one row per metrics run) rather than a real key.

    ``subdir``, if given, nests one more level under the resolved directory — e.g.
    ``subdir="internal"`` for a claim-level diagnostic (a table that carries raw `claim_id`
    values, one row per claim) that still belongs in ``figures/`` next to that notebook's other
    output (every OTHER informational table already lives there — only actual data artefacts
    stay under ``src/data/real/``), but is worth keeping visually apart from the rest.
    """
    out_dir = _out_dir(name)
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{name}.csv"
    df.to_csv(csv, index=index)
    return csv
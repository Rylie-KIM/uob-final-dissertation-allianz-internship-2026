"""Plotting primitives for SFP-detection reports (matplotlib).

Deliberately kept OUT of the Strategy interface (``algorithm/base.py``): a ``DetectionAlgorithm`` is
an oracle-free function of (scores, labels) that depends only on pandas + abc, so detection stays
importable and testable without a plotting stack. Figures are a *reporting* concern and live here.

These are small primitives — a residual-PDF panel and a generic across-generations trajectory line.
Notebooks compose multi-panel figures (annotations, the 0.985 target line, the ~15% anchor band)
from them; the panel layout stays in the notebook, the reusable drawing lives here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde

from detector.algorithm.residual_peak import peak0, residual

if TYPE_CHECKING:  # keep this module importable without a plotting stack (see module docstring)
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def plot_residual_pdf(
    scores_by_version: dict[str, ArrayLike],
    y_true: ArrayLike,
    *,
    ax: Axes | None = None,
    xlim: tuple[float, float] = (-0.25, 0.25),
    n: int = 400,
    styles: dict[str, dict] | None = None,
) -> Axes:
    """KDE of the SFP residual (genuine − score) per version, each labelled with ψ = peak0.

    scores_by_version : {version label: scores on the common garage-verified rows}
    y_true            : genuine 0/1 outcome on those same rows
    styles            : optional {version label: matplotlib line kwargs} (color, ls, lw, …)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    x = np.linspace(xlim[0], xlim[1], n)
    y = np.asarray(y_true).astype(int)
    for name, scores in scores_by_version.items():
        r = residual(y, scores)
        kw = {"lw": 2, "label": f"{name}  ψ={peak0(r):.2f}"}
        if styles and name in styles:
            kw.update(styles[name])
            kw.setdefault("label", f"{name}  ψ={peak0(r):.2f}")
        ax.plot(x, gaussian_kde(r)(x), **kw)
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.set_xlabel("residual = genuine label − score")
    ax.set_ylabel(r"density  $f_t$(residual)")
    return ax


def plot_metric_trajectory(
    series: dict[str, Sequence[float]],
    *,
    ax: Axes | None = None,
    xticklabels: Sequence[str] | None = None,
    ylabel: str = "",
    title: str = "",
    styles: dict[str, dict] | None = None,
    hline: tuple[float, dict] | None = None,
    band: tuple[float, float, dict] | None = None,
) -> Axes:
    """One line per chain across generations. ``series`` = {label: [y_gen1, y_gen2, …]}.

    styles : optional {label: matplotlib line kwargs}
    hline  : optional (y, kwargs) — e.g. the 0.985 precision target
    band   : optional (lo, hi, kwargs) — e.g. the ~15% scrap-rate anchor
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    xs: list[int] = []
    for name, ys in series.items():
        xs = list(range(len(ys)))
        kw = {"marker": "o", "lw": 2, "ms": 8, "label": name}
        if styles and name in styles:
            kw.update(styles[name])
        ax.plot(xs, ys, **kw)
    if band is not None:
        lo, hi, bkw = band
        ax.axhspan(lo, hi, **({"color": "#999999", "alpha": 0.18} | bkw))
    if hline is not None:
        hy, hkw = hline
        ax.axhline(hy, **({"color": "k", "ls": "--", "lw": 1} | hkw))
    if xs and xticklabels is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax

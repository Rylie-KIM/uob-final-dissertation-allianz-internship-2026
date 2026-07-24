# Figure template — dissertation house style

**Load this file into any prompt that produces a figure.** It is the single
source of truth for how every chart in the notebooks and the dissertation looks:
fonts, sizes, and — above all — **the fixed order in which colours are used**.

The rules here are enforced in code by `src/figstyle.py`. The prose and the code
must agree; if you change one, change the other.

---

## How to use it in a notebook

```python
import sys; sys.path.insert(0, str(ROOT / "src"))   # notebooks already do this
import figstyle
figstyle.apply()                                     # ← call once, first thing

fig, ax = plt.subplots(figsize=figstyle.FIG_1)
ax.plot(x, a)        # first series  → colour slot 1 (blue)   automatically
ax.plot(x, b)        # second series → colour slot 2 (aqua)   automatically
ax.legend([...])     # ≥2 series ALWAYS get a legend
figstyle.save(fig, "03_evaluation_gap")              # → figures/*.png + *.pdf
```

Do **not** pass `color=` for ordinary categorical series — let the cycle assign
them so the order is identical in every figure. Pass an explicit named colour
only when a mark carries a *fixed meaning* (see § Named roles).

---

## 1. Colour — the ordered categorical palette

Multiple series in one figure take these colours **in this exact order**. The
first thing you plot is slot 1, the next is slot 2, and so on. Never reorder,
never skip, never cycle past slot 8.

| # | name | hex | typical use |
|---|------|-----|-------------|
| 1 | blue | `#2a78d6` | the first / headline series |
| 2 | aqua | `#1baf7a` | second series |
| 3 | yellow | `#eda100` | third |
| 4 | green | `#008300` | fourth |
| 5 | violet | `#4a3aa7` | fifth |
| 6 | red | `#e34948` | sixth |
| 7 | magenta | `#e87ba4` | seventh |
| 8 | orange | `#eb6834` | eighth |

Why this order and not another: it is the colour-blind-safe ordering (maximises
the minimum separation between *adjacent* slots under CVD simulation). It comes
from the data-viz skill's validated light-surface palette — treat it as fixed.

**More than 8 categories?** Do not invent a 9th colour. Fold the tail into
"Other", switch to small multiples, or add a second encoding (line style / marker).

## 2. Named roles — colour that *means* something

Use these only when a colour must carry fixed semantics (not mere identity):

| role | hex | meaning |
|------|-----|---------|
| `PRIMARY` | `#2a78d6` (blue) | the headline / recommended quantity |
| `NEUTRAL` | `#898781` (grey) | a "for reference" baseline |
| `CAUTION` | `#e34948` (red) | a biased / not-to-be-trusted quantity |
| `GOOD` | `#0ca30c` | meets target / desirable |
| `WARNING` | `#fab219` | attention / sub-target |

Example: the naive-vs-verified-vs-IPS bar chart — naive is `CAUTION`
(optimistically biased), verified is `NEUTRAL` (subpopulation reference), IPS is
`PRIMARY` (the metric to report). That mapping is *semantic*, so it overrides the
default cycle.

## 3. Sequential (continuous magnitude)

One hue, light→dark. Default cmap **`Blues`** (`figstyle.SEQUENTIAL`). Heatmaps,
density, choropleths. Never a rainbow. For a second sequential context at once,
use a `Greens` ramp (the next categorical hue as its own ramp).

## 4. Typography

One sans face everywhere — no serif, no display face.

| element | size (pt) | weight | colour |
|---------|-----------|--------|--------|
| figure suptitle | 15 | bold | `#0b0b0b` ink |
| axes title | 13 | bold | `#0b0b0b` ink |
| axis label | 11 | normal | `#52514e` soft ink |
| tick labels | 10 | normal | `#52514e` |
| legend / annotation | 10 | normal | `#0b0b0b` |

Text never wears a series colour — values and labels stay in ink; the coloured
mark beside them carries identity.

## 5. Ink & chrome

| role | hex |
|------|-----|
| primary ink (title, annotation) | `#0b0b0b` |
| secondary ink (axis label) | `#52514e` |
| muted (ticks) | `#898781` |
| gridline (hairline) | `#e1e0d9` |
| spine / baseline | `#c3c2b7` |

Top and right spines off. Grid recessive and *below* the data. Bars/marks have no
outline stroke.

## 6. Figure sizes (inches) — sized to a ~6.3 in text column

| constant | size | use |
|----------|------|-----|
| `FIG_1` | 6.3 × 3.9 | single panel |
| `FIG_2` | 11 × 4.0 | two panels side by side |
| `FIG_WIDE` | 11 × 3.6 | one wide / timeline panel |
| `FIG_SQ` | 5 × 5 | square (scatter, confusion matrix) |

Screen dpi 110; **saved** dpi 150. `figstyle.save(fig, name)` writes a PNG to
`figures/`. For the thesis, also export a vector PDF of the final figure
(`fig.savefig("figures/<name>.pdf")`) — vector scales cleanly in LaTeX.

## 7. Non-negotiables (from the data-viz method)

- Categorical hues assigned in fixed order, never cycled to a 9th.
- **One y-axis** — never a dual-axis chart. Two scales → two panels or index to a
  common base.
- Colour follows the entity, not its rank — a filter that drops a series must not
  repaint the survivors.
- Sequential = one hue light→dark; diverging = two hues + grey midpoint; never a
  rainbow.
- A legend is always present for ≥ 2 series (none needed for one — the title names
  it); at most ~4 series may also be direct-labelled.
- Status colours (`GOOD`/`WARNING`/…) are reserved for state and ship with a label,
  never reused as "series N".

## 8. Standing project rule — AUC never travels alone

Wherever a figure or table reports **AUC**, the *same* artefact also shows
**precision, recall, and the confusion matrix (TN / FP / FN / TP)**. This is a
project rule, not a style choice, but it belongs in the figure checklist.

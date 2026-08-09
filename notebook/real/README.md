# notebook/real

Three notebooks. The two `00_` notebooks answer different questions about SHAP; `01_` is the
Layer-1 primary detector.

## 0 · `01_error_inheritance.ipynb` — does the next model repeat its predecessor's verified mistakes?

The primary detection test (paper §3.3.1). Uses v1's mobility-segmented cutoffs: in the band
`(0.75, 0.85]` mobile vehicles were garaged and verified while immobile ones were scrapped, so
verified-repairable high-score rows exist — if v2 trained on v1's forced labels, v2 should
over-score exactly those rows. Needs only each model's *scores* joined by claim id (no common
feature space), runs in the analysis `.venv`, reads everything through `loaders`/`config`, and
gates every estimate on band row counts. Also runs the weaker v2→v3 near-boundary analogue.

| | `00_SHAP.ipynb` | `00_shap_attribution.ipynb` |
|---|---|---|
| question | **what does THIS version look at?** | **how does that change v1 → v2 → v3?** |
| scope | one version at a time | all versions at once |
| kernel | that version's env (`src/envs/v<k>/.venv`) | the analysis `.venv` (`sfp-detection`) |
| opens the model pickle | **yes** | no — reads parquet only |
| run it | **three times, once per version** | once, after the three runs |

Everything real (paths, column names, cutoffs, hyperparameters) comes from `src/config.py`. Neither
notebook contains a repo name, a filename or a threshold.

---

## 1 · `00_SHAP.ipynb` — one version, inside its own environment

The three FTTL models were built seven years apart — **xgboost 0.72.1 (v1) · 1.4.2 (v2) · 3.2.0
(v3)** — and a pickle only loads under the release it was serialised with. So this notebook runs on
that version's own kernel. It names no version: it detects which one from the interpreter path and
checks the running xgboost against `config`, raising if they disagree (a mismatched load produces
figures, not errors — that is the whole danger).

**Kernel setup, once per env:**

```bash
src/envs/v2/.venv/bin/python -m pip install ipykernel matplotlib pandas pyarrow
src/envs/v2/.venv/bin/python -m ipykernel install --user --name fttl-v2 --display-name "FTTL v2 (env-v2)"
```

`shap` is optional. If it is installed the values come from `shap.TreeExplainer`; if not, from the
booster's own `pred_contribs`. Exact TreeSHAP either way — see §4 of the notebook for what differs.

**What it produces, per version:**

| § | output |
|---|---|
| 1 | training configuration + that version's decision rule, printed before any figure |
| 2 | feature-space profile — count, types, missingness, cardinality, one-hot families (L0, never collapsed) |
| 3 | score distribution and the fast-track cutoff |
| 4 | SHAP values + the additivity check (φ must sum to the model's own margin) |
| 5 | input distributions of the top drivers, below vs above τ |
| 6 | global: mean\|SHAP\| bar · signed beeswarm · magnitude beeswarm coloured by mean\|SHAP\| |
| 7 | **interactions**: TreeSHAP interaction matrix, strength heatmap, top pairs — and the same pairs' plain association in X for contrast |
| 8 | dependence plots, coloured by the feature that actually interacts most |
| 9 | local: waterfall + force for the strongest fast-track, and for the pair straddling τ |
| 10 | mean\|SHAP\| by score band inside the fast-track region |
| 11 | `src/data/real/detection/<v>_attributions.parquet` + `_meta.json` |

§7 is the expensive one: interaction values cost O(rows × p²), so they run on a few hundred rows —
enough to *rank* pairs, which is all that is needed to choose what to plot. (Rows are the only
lever: the model needs every feature column to predict, so columns cannot be dropped from the input.)
It also draws the distinction the section exists for: **a SHAP interaction is what the model does
jointly with a pair; a correlation is whether the two columns are related in the data.** They
disagree routinely, and only the first explains vertical dispersion in a dependence plot.

Figures land in `figures/` prefixed with the version (`v2_03_shap_beeswarm.png`, …).

All the plotting lives in **`src/shap_kit.py`** — numpy/pandas/matplotlib only, every figure drawn
from the raw φ matrix, because `shap.plots.beeswarm` and the `Explanation` object do not exist in
the stack that pairs with xgboost 0.72.1.

## 2 · `00_shap_attribution.ipynb` — the cross-version comparison

Only after all three have been attributed. Reads the parquets, never a model, and runs in the
analysis `.venv`. It reports the L0 feature overlap, the concentration measures
(Hill / Shannon / Simpson / Gini / top-k, `src/estimator/concentration.py`), and refuses to compare
versions whose attributions were produced under different SHAP backends.

Its input can equally be produced headlessly, without opening a notebook per env:

```bash
python src/scoring/attribute_all.py --dry-run     # resolve paths, write nothing
python src/scoring/attribute_all.py --rows 5000 --background 500
```

That driver additionally fixes **one shared claim set** across versions, which `00_SHAP.ipynb` does
not (it samples its own rows — correct for a single-version analysis, confounded with case-mix for a
comparison). Before reporting any cross-version difference, use the driver's ids or state the caveat.

## File map

| file | runs in | role |
|---|---|---|
| `src/shap_kit.py` | any version env | detection + guards, SHAP values, and every plot |
| `src/scoring/attribute.py` | a version env | the headless equivalent of §4/§10 — φ to parquet, no figures |
| `src/scoring/attribute_all.py` | analysis `.venv` | runs the above for all versions on one shared claim set |
| `src/estimator/concentration.py` | analysis `.venv` | concentration measures + the comparability guard |
| `src/config.py` | both | the only file holding a real path, column, cutoff or hyperparameter |

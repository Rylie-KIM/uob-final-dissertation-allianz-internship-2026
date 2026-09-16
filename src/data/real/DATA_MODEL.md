# `src/data/real/` — data model

## 0 · Split row counts (user-supplied 2026-09-12)

| version | split | n_claims |
|---|---|---:|
| v1 | train | 183,758 |
| v1 | test | 43,471 |
| v1 | val1 | 35,254 |
| v1 | val2 (oot) | 51,189 |
| v2 | train | 200,716 |
| v2 | val | 62,662 |
| v2 | test (oot) | 50,179 |
| v3 | train | 231,295 |
| v3 | test | 57,823 |
| v3 | oot | 46,229 |

## 0.1a · v3 decision threshold — precision-tuned, never deployed

v3 has no regime history (`config.DECISION_RULES["v3"]["shape"] == "global"`, confirmed
2026-07-29): a single scalar $\tau$, but which scalar depends on which precision target it is
tuned to hit — the two documented points are:

| target precision | $\tau$ |
|---|---:|
| $\ge 0.985$ (the business floor, Eq. `tau-floor`) | **0.984** |
| relaxed to $0.97$ | **0.970** |

`config.DECISION_RULES["v3"]["threshold"] = 0.984` encodes only the precision-floor value — the
one every `config.threshold_on("v3", ...)` / `threshold.apply("v3", ...)` call in this codebase
actually uses (SHAP-DiD region/region\_local tagging included). The $0.970$ relaxed-target value
is documented in the thesis (`subsec:decision-threshold`, the recall-collapse paragraph) as a
sensitivity comparison only — it is not wired into `config.py`, so no code path returns it. If an
analysis ever needs to run under the relaxed target, it must pass `0.970` explicitly; nothing
here does that automatically.

## 0.1b · v1 decision reconstruction — single-threshold fallback (2026-09-14)

v1's real rule is SEGMENTED on mobility (`config.DECISION_RULES["v1"]`: `0.75` immobile / `0.85`
mobile, confirmed off the serving code), but `loaders.VersionData.decisions` cannot apply that
rule for v1: mobility is unrecoverable — absent from the raw dataset (confirmed 2026-08-27) AND
from the production log, which does not exist for v1 at all (destroyed; there is no `log` kind to
merge it from — merely accessing `self.log` for v1 raises `FileNotFoundError`, which is what
`04_02_shap_did_concentration.ipynb`'s "v1 log missing" error traced back to, before the fix below).

**Fallback**: `VersionData.decisions` treats **`0.85`, the HIGHER of the two segment thresholds,
as a single global threshold for every v1 row** — `score > 0.85 → 1`, `score ≤ 0.85 → 0` — instead
of raising. This is a deliberate approximation, not a bug:

| score range | real rule (mobility known) | fallback (`score > 0.85`) | error |
|---|---|---|---|
| `≤ 0.75` | 0 for both mobile and immobile | 0 | none |
| `(0.75, 0.85]` — the overlap band | immobile → 1, mobile → 0 (mixed) | 0 | immobile rows only: false negative (real 1, fallback 0) |
| `> 0.85` | 1 for both mobile and immobile | 1 | none |

No false positives anywhere; the only error is undercounting immobile scraps inside the
`(0.75, 0.85]` overlap band — the same permanently-undecided band `project_v1_mobility_not_a_feature.md`
already documents as unresolvable without mobility, and the same convention `00_SHAP.ipynb`'s own
`TAU = max(rule["thresholds"].values())` already uses for v1's score-distribution plot. The score
being thresholded is v1's own **train-time / in-sample recomputed score** (`self.scores`, "as
recomputed by scoring/predict_v1.py") — v1's real production serving score does not survive
(§0.1's row for `log`) — so this reconstructs "what the frozen model would have decided", not a
literal replay of history. `VersionData.decisions` prints `[v1] decisions: FALLBACK single
threshold 0.85 in use ...` whenever this branch runs, so any notebook cell that touches `.decisions`
for v1 shows the fallback is active rather than silently reconstructing decisions from a rule the
row-level data cannot actually support.

## 0.1 · Which notebook/script writes which kind, in what format

| kind | producer | format written | one-line purpose |
|---|---|---|---|
| `processed_inputs`, `targets`, `raw_dataset` | `01_export_v1.ipynb` | **CSV** (env-v1 has no parquet engine) | v1's model-dev matrix / label / raw claims |
| `processed_inputs`, `targets`, `raw_dataset` | `01_export_v2.ipynb`, `01_export_v3.ipynb` | parquet | same, v2/v3 |
| `log`, `log_raw`, `log_features`, `log_scores`, `log_targets` | `01_export_v2_logs.ipynb` | parquet | v2's production serving log, split by column (v2 only — see §6) |
| `scores` | `scoring/predict.py` (v2/v3), `scoring/predict_v1.py` (v1) | parquet (v2/v3) · **CSV then converted** (v1) | recomputed baseline scores |
| `attributions` | `00_SHAP.ipynb` (v2/v3) | parquet | per-row SHAP, baseline model |
| `attributions` | `00_SHAP_v1.ipynb` | **CSV** (`shap_kit_v1` has no parquet engine either) | same, v1 |
| `corrector_targets` | `mitigation/03_01_corrector_inputs.ipynb` | parquet (+ diagnostic CSV siblings) | targets joined to the recorded treatment (decision [+ score]) — v3 from the v2 log, v2 from the vehicle-status file |
| `corrected` | `mitigation/03_02_reweight_mitigation.ipynb` | parquet, **one file per correction axis** | de-contaminated (`label`, `weight`) pairs — naive/rarity/transport/pnu, see §5b |
| `mitigated` | `mitigation/03_03_retrain.ipynb` §2 (`training/retrain.py`) | pickle, **one file per (split, axis)** | model retrained on one `corrected` axis — filename convention diverges from the declared `mitigated` kind path, see §5b caveat |
| `reeval_scores` | `mitigation/03_03_retrain.ipynb` §4 (`scoring/predict.py`) | parquet, **one file per (scored split, axis)** | scores from each mitigated model — NOT produced by 03_04 (that notebook only reads them) |
| *(ad hoc, no declared kind)* | `mitigation/03_05_reevaluation.ipynb` §3/§3b | parquet, **one file per (model, τ)** | `reeval/<v>_decisions_<split>_<model>_tau<value>.parquet` — claim_id/score/decision at one tuned or grid cutoff, see §5b |
| *(no new kind)* | `mitigation/03_04_prediction.ipynb`, `mitigation/03_05_reevaluation.ipynb` (its figures/tables only) | — | file-level diagnostics of the above: integrity, distributions, PR curves, operating-point metrics — figures/CSVs under `figures/`, not under this directory |
| *(retired)* | `mitigation/03_06_shap.ipynb` | — | stub only — moved to `shap_did/04_03_shap_did_mitigation_delta.ipynb`, see §5a |
| `shap_did_input` | `shap_did/04_01_shap_did_inputs.ipynb` | parquet | region(A/B) + era tags, per (version, split) — v2/v3 `train`+OOT only, see §5a |
| `mitigated_attributions` | `00_SHAP.ipynb` (2026-09-14, `MODEL_PATH` pointed at a mitigated pkl — auto-detected, no toggle) | parquet, **one file per (split, axis)** | per-row SHAP, MITIGATED model — the "after" arm of `ShapDiDDelta`, see §5 |

**CSV → parquet conversion history.** `scripts/v1_csv_to_parquet.py` runs in the analysis `.venv` (env-v1
has no parquet engine at all — pinned to Python 3.5.6, no pyarrow/fastparquet). It originally
covered only `processed_inputs` / `targets` / `scores` / `raw_dataset` (v1's `01_export_v1.ipynb`
outputs). **Extended 2026-09-12** to also convert `00_SHAP_v1.ipynb`'s `attributions` CSVs
(`detection/shap/v1/v1_attributions_<split>[_native].csv`) — that notebook was already writing CSV
and printing "run this by hand" one-liners as a stopgap; the script now globs and converts them
the same way, respecting `--splits`. Sidecar `_meta.json` files are already JSON and need no
conversion.

Column reference for every artefact under this directory. Paths resolve only through
`config.path(kind, version, source, split=...)`.

Feature columns are written here as `<v>_feat1 … <v>_featN`. The real names live in each version's
pickle (`features/registry/<v>.json`) and cross-version correspondence only in
`features/feature_overlap.json`.

---

## 1 · Index

| kind | path | grain | key | per split |
|---|---|---|---|---|
| `processed_inputs` | `inputs/features_<v>_<split>.parquet` | claim | `claim_id` | yes |
| `targets` | `inputs/targets_<v>_<split>.parquet` | claim | `claim_id` | yes |
| `raw_dataset` | `inputs/raw_<v>.parquet` | claim | `claim_id` | no |
| `scores` | `detection/<v>_scores_<split>.parquet` | claim | `claim_id` | yes |
| `attributions` | `detection/shap/<v>/<v>_attributions_<split>[_<backend>].parquet` | claim | `claim_id` | yes |
| *(sidecar)* | `detection/shap/<v>/<v>_attributions_<split>[_<backend>]_meta.json` | file | — | yes |
| `log` | `logs/v2.parquet` | claim | `claim_id` | no |
| `log_raw` | `logs/v2_raw.parquet` | scoring event | `claim_id`, `correlation_id` | no |
| `log_features` | `logs/v2_features.parquet` | scoring event | `claim_id`, `correlation_id` | no |
| `log_scores` | `logs/v2_scores.parquet` | scoring event | `claim_id`, `correlation_id` | no |
| `log_targets` | `logs/v2_targets.parquet` | scoring event | `claim_id`, `correlation_id` | no |
| `corrector_targets` | `mitigation/inputs/corrector_targets_<v>_<split>.parquet` | claim | `claim_id` | yes |
| *(sidecar)* | `mitigation/inputs/corrector_targets_<v>_<split>_meta.json` | file | — | yes |
| `corrected` | `mitigation/<v>_corrected_<split>_<axis>.parquet` (declared kind has no `<axis>`, see §5b) | claim (not unique for `transport`/`pnu`, §5b) | `claim_id` | yes |
| *(sidecar)* | `mitigation/<v>_corrected_<split>_<axis>_meta.json` | file | — | yes |
| `mitigated` | actually written as `src/models/real/mitigated/<v>_<split>_<axis>.pkl` — the declared kind's fallback (`src/models/{source}/mitigated/{v}.pkl`) names a file 03_03 never produces, see §5b caveat | — | — | no (model, not per-split data) |
| *(sidecar)* | `src/models/real/mitigated/<v>_<split>_<axis>_meta.json` | file | — | yes |
| `reeval_scores` | actually written as `src/data/real/reeval/<v>_mitigated_scores_<split>_<axis>.parquet` (declared kind fallback has no `<axis>` either) | claim | `claim_id` | yes |
| *(ad hoc, no declared kind)* | `reeval/<v>_decisions_<split>_<axis>_tau<value>.parquet` | claim | `claim_id` | yes (eval split only) |
| `shap_did_input` | `estimation/shap_did/<v>/<v>_shap_did_input_<split>.parquet` | claim | `claim_id` | yes · v2/v3 `train`+OOT only |
| *(sidecar)* | `estimation/shap_did/<v>/<v>_shap_did_input_<split>_meta.json` | file | — | yes |
| `mitigated_attributions` | `mitigation/shap/<v>/<v>_mitigated_attributions_<split>_<axis>.parquet` (declared kind has no `<axis>`, see §5) | claim | `claim_id` | yes |

`logs/*` exist for **v2 only**.

---

## 2 · Split names

| version | splits | out-of-time holdout |
|---|---|---|
| v1 | `train`, `test`, `val1`, `val2` | `val2` |
| v2 | `train`, `val`, `test` | `test` |
| v3 | `train`, `test`, `oot` | `oot` |

No unsplit base file exists for any per-split kind.

---

## 3 · Column naming

| group | columns | naming |
|---|---|---|
| canonical | `claim_id`, `date`, `score`, `decision`, `observed`, `mobility` | `src/schema.py`; translated once at ingest from `config.VERSIONS[v]["columns"]` |
| version's own | all feature columns, `raw_dataset`, `log_raw` | never canonicalised |
| unkeyed | `correlation_id` | no canonical name; keeps its source name |

Per-version real names of the canonical columns:

| canonical | v1 | v2 | v3 |
|---|---|---|---|
| `claim_id` | `claimnumber` | `Claimnumber_CLAIM` | `Claimnumber_CLAIM` — the business claim number, **not** the repo's `ID_CLAIM` row key (the first export used `ID_CLAIM`; re-keyed in place 2026-09-03 — join onto raw_v3's Claimnumber_CLAIM, old id dropped) |
| `date` | `lossdate` | `ReportedDate` | `ReportedDate_CLAIM` |
| `observed` | `veh_total_loss` | `veh_total_loss` | `Fttl` |
| `score` | — | `FastTrackerProbablity` | — |
| `decision` | — | `FastTrackerDecision` | — |
| `mobility` | `vehicle_mobility_status` | — | — |

---

## 4 · `inputs/`

### `processed_inputs` — `inputs/features_<v>_<split>.parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | key |
| `<v>_feat1 … <v>_featN` | int64 / float64 | encoded feature names, fit order |
| *version's target column* | int64 | v1/v2 `veh_total_loss`, v3 `Fttl`; present in all three versions |

### `targets` — `inputs/targets_<v>_<split>.parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | key |
| `date` | datetime64 | |
| `observed` | int64 | contaminated label; no `score`, no `decision` |

### `raw_dataset` — `inputs/raw_<v>.parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | key |
| *version's date column* | datetime64 | see §3 |
| *version's target column* | int64 | see §3 |
| *remaining raw columns* | mixed | version's own names; `mobility` in v1 only |

`raw_v3.parquet` additionally keeps **`ID_CLAIM`** (the v3 repo's `project_params.KEY`, a database row id) under its own name beside `claim_id`. It is the only place both keys sit together — the bridge the 2026-09-03 re-key used, and the way back to the Z: / DB row if one is ever needed. `features_v3_*` may carry it too (the transformed file's own column, never a model feature).

---

## 5 · `detection/`

### `scores` — `detection/<v>_scores_<split>.parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | key |
| `model_<v>_score` | float64 | recomputed by `scoring/predict.py`, not the logged score |

### `attributions` — `detection/shap/<v>/<v>_attributions_<split>[_<backend>].parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | key |
| `<v>_feat1 … <v>_featN` | float64 | per-row φ; same names and order as `processed_inputs` |
| `_base_value` | float64 | not a feature |

Filename is assembled in three steps: `config.path()` appends `_<split>`; the caller appends the
backend suffix (`_native` for tree-path-dependent, none for interventional); the sidecar is
`<stem>_meta.json`. The backend is not part of the path key.

### `*_meta.json`

| field | type | written by `attribute.py` | written by `00_SHAP.ipynb` |
|---|---|---|---|
| `version` | str | yes | yes |
| `split` | str | yes | yes |
| `model_path` | str | yes | yes |
| `features_path` | str | yes | — |
| `features_provenance` | str | — | yes |
| `estimator` | str | yes | yes |
| `estimator_params` | dict | yes | yes |
| `feature_order` | str | yes | yes |
| `backend` | str | yes | yes |
| `perturbation` | str | yes | yes |
| `model_output` | str | yes | yes |
| `note` | str | — | yes |
| `n_rows` | int | yes | yes |
| `n_features` | int | yes | yes |
| `feature_names` | list[str] | yes | yes |
| `background_n` | int | yes | yes |
| `explain_ids_file` | str \| null | yes | yes |
| `background_ids_file` | str \| null | yes | yes |
| `seed` | int | yes | yes |
| `tau_used` | float | — | yes |
| `tau_note` | str | — | yes |
| `base_value` | float | yes | yes |
| `python` / `numpy` / `pandas` | str | yes | — |
| `env` | dict | — | yes |
| `dummy` | bool | — | — (dummy generator only) |

`backend`, `perturbation` and `model_output` must match across versions or
`concentration.require_comparable()` raises.

`feature_order` takes one of four values, from `shap_kit.feature_order()` /
`shap_kit_v1.feature_order()`:

| value | meaning |
|---|---|
| `exact` | columns are the trained names in the trained order |
| `reordered` | same set, different order — φ are attributed to the wrong features |
| `set_mismatch` | different columns altogether |
| `exact (via registry: <source>)` | the pickle named nothing; the order matched `features/registry/<v>.json`, whose own order came from `<source>` (`booster.feature_names` = the fit order; `get_feature_names_out` = the preprocessing head's output order, an inference; `unrecorded` = registry built before the source was written) |
| `unverified (the estimator exposes no trained feature names)` | neither the pickle nor the registry could answer |

`attribute.py` exits on `reordered` / `set_mismatch` and writes no file; the notebooks reorder
via `align()` and record the result. Files written before 2026-08-31 have no field at all —
backfill with `src/scoring/backfill_feature_order.py`.

---

## 5a · `estimation/shap_did/` — SHAP-DiD inputs (`notebook/real/shap_did/`)

Built by `shap_did/04_01_shap_did_inputs.ipynb` only. **v1 is excluded** — its era jump alone
breaks parallel trends, so only v2 and v3 are built. **Not every declared split is built either**:
`BUILD = {v: [train, OOT_SPLIT[v]] for v in (v2, v3)}`, i.e. v2 gets `train` + `test` (its OOT) but
never `val`, and v3 gets `train` + `oot` but never `test`. A missing file for e.g. `(v2, val)` is
expected, not a bug.

The other three notebooks in `shap_did/` (`04_02_shap_did_concentration.ipynb`,
`04_03_shap_did_mitigation_delta.ipynb`, `04_04_regime_break_robustness.ipynb`) are **read-only**
against this directory plus `attributions` / `mitigated_attributions` / `corrector_targets` /
`mitigated` / `corrected` — they write no new kind under `src/data/real/`. Their output is
figures and CSV tables under the repo-root `figures/` directory (`figstyle.save()` /
`figstyle.save_table()`), which is a separate concern from this data model — see
`reference_figure_template.md`-style conventions, not this file.

`04_03` was rebuilt 2026-09-15 into three steps: **Step 1** (within-version, no era — `baseline`
vs every corrected axis and `naive` vs every OTHER axis, on each version's OWN population, train
AND OOT) absorbs what used to be a separate `04_05_shap_did_within_version.ipynb` (retired,
deleted); **Step 2** (cross-version, matching tag only, **OOT ONLY** — never train, see the
notebook's own §2 markdown for why) mirrors `04_02`'s `cross_version_estimate` structure repeated
per tag; **Step 3** is Step 2's local-RDD robustness check. Train-split comparisons in Step 1
restrict to the `corrector_targets`-covered population (the claims every corrected model for that
version actually trained on) — otherwise baseline (fit on the full population) and a corrected
axis (fit on a filtered subset) would be compared on partly different in-sample/out-of-sample
footing. OOT needs no such restriction: no model, baseline or corrected, ever trained on OOT.

### At a glance — every file 04_01–04_04 write, in run order

Confirmed by grepping all four notebooks for `to_parquet`/`to_csv`/`write_text`/`joblib.dump`/
`open(`: **only 04_01 writes anything under `src/data/real/`**. 04_02/04_03/04_04 call nothing
but `figstyle.save()`/`figstyle.save_table()`, which land in the repo-root `figures/` directory —
outside this file's scope (see §5's producer table, §0.1).

| # | file (path relative to `src/data/real/`) | type | producer | kind |
|---|---|---|---|---|
| 1 | `estimation/shap_did/<v>/<v>_shap_did_input_<split>.parquet` | parquet, per (v ∈ {v2, v3}) × (split ∈ `train`, that version's OOT) — 4 files on a full run | 04_01 | `shap_did_input` |
| 2 | `estimation/shap_did/<v>/<v>_shap_did_input_<split>_meta.json` | json sidecar, one per row 1 | 04_01 | — |

04_02's `local_cross_version_estimate` / `within_version_confound_check`, 04_03's mitigation-delta
and local-RDD checks, and 04_04's regime-break robustness all read rows 1–2 above (plus
`attributions`, `corrector_targets`, `mitigated`, `corrected` from §5/§5b) and write only figures
and CSV tables — none of it under `src/data/real/`, so it isn't enumerated here.

### `shap_did_input` — `estimation/shap_did/<v>/<v>_shap_did_input_<split>.parquet`

A small claim-level tag table — deliberately does not carry the SHAP φ matrix itself. 04-02/04-03
join it onto `attributions` (or `mitigated_attributions`) by `claim_id` at read time.

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | key |
| `date` | datetime64 | canonical |
| `score` | float64 | that version's own score column, renamed to `score` |
| `decision` | int64 | that version's own decision rule, 1 iff `score` > τ |
| `region` | str | `"B"` if `decision==1` (score > τ, strict), else `"A"` |
| `era` | str | `"early"` / `"late"`, split at the **median date** within this (version, split) population — a PROVISIONAL placeholder, not a regime-aware cutoff |
| `observed` | canonical dtype | contaminated label, unchanged from `load()` |

### `*_meta.json` (shap_did_input sidecar)

| field | type | notes |
|---|---|---|
| `version` | str | |
| `split` | str | |
| `n_claims` | int | |
| `era_cutoff_date` | str | the median date used to split `era` |
| `era_rule` | str | fixed text noting the median-split is provisional |
| `region_rule` | str | fixed text: `decision==1 -> B (score>tau, strict); else A` |
| `cell_counts` | dict | `{"early/A": n, "early/B": n, "late/A": n, "late/B": n}` |
| `columns` | list[str] | the parquet's column names |

---

### `mitigated_attributions` — `mitigation/shap/<v>/<v>_mitigated_attributions_<split>_<axis>.parquet`

Same column shape as `attributions` (claim_id + per-row φ + `_base_value`), but for a MITIGATED
model instead of the baseline. Lives under `mitigation/shap/<v>/` — its own artefact family
alongside `corrected` / `mitigated` / `reeval_scores` — not inside `detection/shap/<v>/`, so it
can never be mistaken for a baseline attributions file. The per-version directory was added
2026-09-16 (mirroring `attributions` above) once version × split × axis started multiplying flat
filenames past the point of being readable in one listing.

**The declared kind path (`config.path("mitigated_attributions", ...)`) carries no axis** — unlike
`corrected`/`mitigated`/`reeval_scores`, which do. `00_SHAP.ipynb` is the producer and appends the
axis suffix itself: it detects a mitigated run **off `MODEL_PATH` alone** (no separate toggle — a
`<v>_<trained_split>_<axis>.pkl` under the `mitigated` kind's own directory), and its default
`OUT_PATH` is `config.path("mitigated_attributions", v, source, split=split)` with `_<axis>`
appended, matching the axis actually loaded. `<trained_split>` in the pickle's own filename need
not equal the `split` this notebook is currently explaining — 03_03 only ever retrains on
`"train"`, but 00_SHAP scores/explains that same pickle on other splits too (e.g. the OOT split)
by changing `SPLIT` while leaving `MODEL_PATH` pointed at the train-tagged pickle;
`mitigated_axis_tag()` matches on `version` then peels off whichever of that version's own split
names prefixes the remainder, so this never has to be a literal match (2026-09-16 fix — it used
to require an exact match, so a mitigated model scored on `"test"`/`"oot"` was silently treated as
a baseline run and its output landed under the baseline paths below instead of the mitigated
ones). Every 00_SHAP run's figures/tables land under a `<version>/<split>[/<axis>]` subfolder —
baseline: `figures/00_shap/baseline/<v>/<split>/`, mitigated:
`figures/00_shap/mitigated/<v>/<split>/<axis>/` (see §8) — so a mitigated run never overwrites the
baseline figures for the same (version, split), different axes at the same (version, split) never
overwrite each other, and different (version, split) runs never collide either.
`shap_did/04_03_shap_did_mitigation_delta.ipynb` reads this kind through a single
`MITIGATION_AXIS` constant (§1) — the ONE axis it compares for both v2 and v3 — appending the
identical suffix on read; that constant must name the same axis `00_SHAP.ipynb` was run with, or
it silently reads whichever axis happens to be sitting at that (version, split).

---

## 5b · `mitigation/` and `reeval/` — the corrector pipeline (`notebook/real/mitigation/`)

Five live notebooks, run in order (`03_06_shap.ipynb` is a retired stub — see §0.1/§5a): **03_01**
joins each version's own recorded treatment onto `targets` (v3 from the v2 serving log's
`log_scores`, v2 from a company-laptop-only vehicle-status file — v1 is out of scope, no log
exists for it); **03_02** turns that into a `(label, weight)` correction, one file per axis;
**03_03** retrains and scores each axis inside the version's own env; **03_04** and **03_05** are
read-only diagnostics against 03_01–03_03's output, except that **03_05 also writes new per-cutoff
decision files** (below). Like `shap_did_input` (§5a), `corrector_targets` is built only for v2/v3
and only for `BUILD = {train, that version's OOT split}` — never every declared split, never v1.

### At a glance — every file 03_01–03_05 write, in run order

`<axis>` ∈ `{naive, rarity_regime, rarity_fixed, transport, pnu_regime, pnu_fixed}` (rarity/pnu
skipped when there's no `score`, i.e. always for v2). `03_04_prediction.ipynb` and
`03_06_shap.ipynb` write **nothing** under `src/data/real/` or `src/models/real/` — 03_04's output
is figures/CSVs under the repo-root `figures/` directory, 03_06 is a retired stub (see §5a).

| # | file (path relative to `src/data/real/` unless noted) | type | producer | kind |
|---|---|---|---|---|
| 1 | `mitigation/inputs/corrector_targets_<v>_<split>.parquet` | parquet | 03_01 | `corrector_targets` |
| 2 | `mitigation/inputs/corrector_targets_<v>_<split>_meta.json` | json sidecar | 03_01 | — |
| 3 | `mitigation/inputs/corrector_targets_<v>_<split>_0301_<section>_missing_claim_ids.csv` | csv, conditional (any unmatched) | 03_01 §2 | ad hoc |
| 4 | `mitigation/inputs/corrector_targets_<v>_<split>_0301_<section>_observed_mismatches.csv` | csv, conditional, v2 only | 03_01 §2 | ad hoc |
| 5 | `mitigation/inputs/corrector_targets_v3_<split>_0301_sec3b_missing_by_month.csv` | csv, v3 only, conditional | 03_01 §3b | ad hoc |
| 6 | `mitigation/inputs/corrector_targets_v3_<split>_0301_sec3b_missing_observed_mix.csv` | csv, v3 only, conditional | 03_01 §3b | ad hoc |
| 7 | `mitigation/inputs/corrector_targets_0301_sec5_summary.csv` | csv, one file total | 03_01 §5 | ad hoc |
| 8 | `mitigation/<v>_corrected_<split>_<axis>.parquet` | parquet, one per axis run | 03_02 §3 | `corrected` |
| 9 | `mitigation/<v>_corrected_<split>_<axis>_meta.json` | json sidecar | 03_02 §3 | — |
| 10 | `mitigation/<v>_<split>_0302_sec2c_band_table.csv` | csv, conditional (`SELECT_PARAMS` and has `score`) | 03_02 §2c | ad hoc |
| 11 | `mitigation/<v>_<split>_0302_sec2c_clip_table.csv` | csv, same condition | 03_02 §2c | ad hoc |
| 12 | `mitigation/<v>_<split>_0302_sec3_diag_summary.csv` | csv, one row per axis | 03_02 §3 | ad hoc |
| 13 | `src/models/real/mitigated/<v>_<split>_<axis>.pkl` | pickle, one per axis | 03_03 §2 | `mitigated` |
| 14 | `src/models/real/mitigated/<v>_<split>_<axis>_meta.json` | json sidecar | 03_03 §2 | — |
| 15 | `src/models/real/mitigated/<v>_<split>_0303_sec3_fitted_summary.csv` | csv, one row per axis | 03_03 §3 | ad hoc |
| 16 | `reeval/<v>_mitigated_scores_<split>_<axis>.parquet` | parquet, one per (axis, scored split) | 03_03 §4 | `reeval_scores` |
| 17 | `reeval/<v>_decisions_<split>_<axis>_tau<value>.parquet` | parquet, one per (axis, τ actually tuned) | 03_05 §3/§3b | ad hoc |
| 18 | `reeval/<v>_0305_sec3_reweight_metrics_<split>.csv` | csv | 03_05 §3 | ad hoc |
| 19 | `reeval/<v>_0305_sec3_tau_by_source_<split>.csv` | csv | 03_05 §3 | ad hoc |
| 20 | `reeval/<v>_0305_sec3b_reweight_metrics_<split>_all.csv` | csv | 03_05 §3b | ad hoc |
| 21 | `reeval/<v>_0305_sec3b_precision_gap_<split>.csv` | csv | 03_05 §3b | ad hoc |
| 22 | `reeval/<v>_0305_sec3b_tau_by_source_<split>_all.csv` | csv | 03_05 §3b | ad hoc |

Rows 3–7, 10–12, 15, 17–22 have **no declared `config.py` kind** — they're written with plain
`Path` string-building next to a real kind's file, named after the notebook+section that produced
them (`0301`/`0302`/`0303`/`0305`), never read back by another notebook's `config.path()` call.
Full column/field reference for each numbered row follows below.

### `corrector_targets` — `mitigation/inputs/corrector_targets_<v>_<split>.parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | targets' own dtype | key |
| `date` | datetime64 | from `targets` |
| `observed` | int64 | from `targets`, contaminated label |
| `decision` | int64 | the recorded treatment — v3: fast-track flag off the v2 log; v2: derived from the vehicle-status vocabulary (`V2_STATUS_MAP`, filled in on the company laptop) |
| `score` | float64 | **v3 only** (`has_score` in the meta below) — v2 has no v1-era deciding score to join |

Rows the treatment source never covers are dropped (left-join `notna` gate, floored at
`MIN_COVERAGE=0.5`) — `n_unmatched_dropped` in the meta counts them.

### `*_meta.json` (corrector_targets sidecar)

| field | type | notes |
|---|---|---|
| `version`, `split` | str | |
| `treatment_source` | str | path/description of the treatment file joined |
| `n_targets` | int | rows in `targets` before the join |
| `n_unmatched_dropped` | int | rows the treatment source didn't cover |
| `n_out` | int | rows actually written |
| `has_score` | bool | true for v3, false for v2 |
| `n_scrapped` | int | `decision.sum()` |
| `n_observed_mismatch` | int \| null | v2 only — rows where `observed` disagrees with the status-implied outcome |
| `missing_claim_ids_file` / `observed_mismatch_file` | str \| null | paths to the diagnostic CSVs below, if written |
| `columns` | list[str] | |
| `id_dtype` | str | |
| `status_dropped` / `status_kept` | dict | **v2 only** — per-status-value row counts (`extra_meta`) |

### Diagnostic CSV siblings (same folder, no declared kind)

| file | when | purpose |
|---|---|---|
| `corrector_targets_<v>_<split>_0301_<section>_missing_claim_ids.csv` | any unmatched rows | which `claim_id`s the treatment source didn't cover |
| `corrector_targets_<v>_<split>_0301_<section>_observed_mismatches.csv` | v2 only, on disagreement | `observed` vs status-implied outcome, with the raw status string |
| `corrector_targets_v3_<split>_0301_sec3b_missing_by_month.csv`, `..._missing_observed_mix.csv` | v3 only, exploratory §3b | whether missing claim_ids cluster by month / skew the `observed` mix |
| `corrector_targets_0301_sec5_summary.csv` | always, one file | coverage/mismatch summary across the whole `BUILD` |

`section` ∈ `{sec3 (v3 build), sec4 (v2 build)}` names which cell of 03_01 wrote the file — it
plays no role in the join and is not a kind key.

### `corrected` — `mitigation/<v>_corrected_<split>_<axis>.parquet`

One file per axis actually run: `naive`, `rarity_regime`, `rarity_fixed`, `transport`,
`pnu_regime`, `pnu_fixed` — `rarity_*`/`pnu_*` are skipped whenever `corrector_targets` has no
`score` (always true for v2). `BAND_H`/`CLIP_HI` are auto-selected per split by §2c's grid search
(power + variance gates only, never a metric on contaminated labels) before §3 runs the axes.

| column | dtype | notes |
|---|---|---|
| `claim_id` | | **not unique** for `transport`/`pnu` — each scrapped (U) row is split into a label=1 row (weight `g·w`) and a label=0 row (weight `(1-g)·w`), both keeping the same `claim_id` |
| `label` | int64 | corrected target |
| `weight` | float64 | 1.0 for naive and for transport's garage rows; a rarity multiplier or transport probability elsewhere |

### `*_meta.json` (corrected sidecar)

| field | notes |
|---|---|
| `version`, `split`, `features`, `targets`, `scheme` | run identity |
| `n_total`, `n_garage`, `n_scrapped`, `n_scrapped_observed_0` | population counts before correction |
| `n_out_rows`, `weight_mean`, `weight_sum`, `weighted_pos_share` | the written file's own shape |
| `tau_mode`, `decider`, `tau_min`, `tau_max`, `band_h`, `clip`, `cells` (per-cell n/freq/multiplier), `n_garage_above_tau`, `n_scrapped_at_or_below_tau` | **rarity/pnu only** |
| `transport` (`estimator`, `n_fit_garage`, `n_scored_U`, `n_feature_cols`, `g_mean_U`, `g_min_U`, `g_max_U`) | **transport/pnu only** |

Diagnostic CSVs from 03_02, same folder as `corrected`: `<v>_<split>_0302_sec2c_band_table.csv` /
`..._sec2c_clip_table.csv` (the BAND_H/CLIP_HI grid search) and `<v>_<split>_0302_sec3_diag_summary.csv`
(one row per axis, the non-dict meta fields).

### `mitigated` — one file per (split, axis), routed through config since 2026-09-14

`config.py` declares `mitigated` as a `SPLIT_KIND` with fallback
`src/models/{source}/mitigated/{v}.pkl`; that template names the version+split base only, the same
way `attributions`' backend suffix and `corrected`'s axis suffix already work — the axis tag is
appended by the caller. **Before 2026-09-14, 03_03 §2 didn't call `config.path()`/`split_path()`
at all** — it hardcoded `MODEL_DIR = ROOT/"src/models/real/mitigated"` directly, and
`shap_did/04_03`'s existence check called `config.path("mitigated", "v2")` with no `split=`, which
raises (`ValueError`) since `mitigated` is a `SPLIT_KIND`. **Fixed 2026-09-14**: 03_03 now builds
`mitigated_path(tag)` from `config.split_path("mitigated", VERSION, SPLIT)` + `"_" + tag` (same
pattern as 03_02's `corrected_path()`), and `shap_did/04_03` §3 globs
`config.split_path("mitigated", "v2", split).parent` for `v2_<split>_*.pkl` instead of checking one
bare path. Output filenames are unchanged from before the fix (no per-version override was ever
declared for this kind), so **no earlier 03_01–03_03 run needs redoing** for this alone.

Sidecar: `<same stem>_meta.json` — `version`, `baseline`, `features`, `labels`, `random_state`
(null ⇒ the baseline estimator had no fixed seed, so the retrain is not bit-for-bit reproducible),
`n_rows`, `n_features`, `feature_selection`, `weighted`, `weight_mean`, `weight_max`,
`label_pos_rate`, `estimator`. Diagnostic CSV:
`src/models/real/mitigated/<v>_<split>_0303_sec3_fitted_summary.csv` — one row per axis.

### `reeval_scores` — `reeval/<v>_mitigated_scores_<split>_<axis>.parquet`

Written by 03_03 §4 (`scoring/predict.py`), for every axis retrained, on every split in
`{the trained split, that version's OOT split}`. Same column shape as `scores`: `claim_id`,
`model_<v>_score`.

### `decisions` — `reeval/<v>_decisions_<split>_<axis>_tau<value>.parquet` (no declared kind)

Written by 03_05 §3/§3b — one file per (axis, τ) that 03_05 itself tunes (`tau_source` ∈
`{tuned, tuned_oot, tuned_all, tuned_oot_all}`; the plain `grid` reference cutoffs are never
materialised as a decisions file, only scored in memory). `split` here is always the version's
**eval** (OOT) split.

| column | dtype | notes |
|---|---|---|
| `claim_id` | | key |
| `score` | float64 | that axis's score on the eval split |
| `decision` | int64 | `score > tau`, strict |

Other 03_05 diagnostics (CSV, same `reeval/` folder, not a declared kind): `<v>_0305_sec3_reweight_metrics_<split>.csv`,
`..._sec3_tau_by_source_<split>.csv`, `..._sec3b_reweight_metrics_<split>_all.csv`,
`..._sec3b_precision_gap_<split>.csv`, `..._sec3b_tau_by_source_<split>_all.csv` — AUC/precision/recall/CM
per (model, τ), both on the verified-garage slice and on the whole split, plus the precision gap
between them that quantifies forced-label inflation.

---

## 6 · `logs/` — v2 only

### `log` — `logs/v2.parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | key, unique |
| `date` | datetime64 | |
| `score` | float64 | logged score |
| `decision` | int64 | logged fast-track flag, not derived from `score` |
| `observed` | float64 | v3 extract's `Fttl`; **null = unobserved, not 0** |

Column order is as listed. Read by `loaders.load("v2").log` and `threshold.read_off()`.

### `log_raw` — `logs/v2_raw.parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | key 1 |
| `correlation_id` | str | key 2 |
| *remaining raw columns* | mixed | v2's own names; neither model feature nor prediction |

### `log_features` — `logs/v2_features.parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | key 1 |
| `correlation_id` | str | key 2 |
| `v2_feat1 … v2_featN` | int64 / float64 | `param.py` `MODEL_FEATURES`, fit order, `_transformed` suffix stripped |

Same column names as `processed_inputs`, different row set (serving history, not training splits).

### `log_scores` — `logs/v2_scores.parquet`

| column | dtype |
|---|---|
| `claim_id` | int64 |
| `correlation_id` | str |
| `score` | float64 |
| `decision` | int64 |

### `log_targets` — `logs/v2_targets.parquet`

| column | dtype | notes |
|---|---|---|
| `claim_id` | int64 | |
| `correlation_id` | str | |
| `date` | datetime64 | |
| `observed` | float64 | nullable |

---

## 7 · Local dummy

Written by `python src/data/make_dummy_real.py --force` (`_DUMMY_DATA` marker present) and
`src/data/make_dummy_shap.py`. Seeded RNG; shapes only.

| | v1 | v2 | v3 |
|---|---|---|---|
| rows per split | 1738 / 435 / 353 / 512 | 1600 / 400 / 500 | 1500 / 375 / 480 |
| features | 20 | 18 | 15 |
| date window | 2016-01 → 2021-12 | 2018-01 → 2020-09 | 2023-06 → 2026-05 |
| log | — | 2575 events / 2500 claims | — |

| dummy property | value |
|---|---|
| repeat scoring events | 75 |
| null `observed` in log | 199 |
| feature names | plausible placeholders (`make_FORD`, …); `log_raw` / `log_features` use `v2_raw*` / `v2_feat*` |
| `feature_order` in meta | hardcoded `"exact"`, with `"dummy": true` beside it |
| v2 threshold regimes | one only (dummy window sits inside regime 1) |

---

## 8 · Per-notebook output paths — figures / tables / dataset (target structure)

Placeholders: `<v>` version · `<split>` that version's split · `<axis>` mitigation axis ·
`<b>` = `ACTIVE_BACKEND` (`interventional` / `tree_path_dependent`) · `<feat>`, `<fa>`/`<fb>`
resolved at runtime · `<label>` ∈ `{strongest_fast-track, just_above_τ, just_below_τ}`.

### `00_SHAP.ipynb`

Figures (`figstyle.save`, `.png`) and tables (`figstyle.save_table`, `.csv`) share one root per run:

```
figures/00_shap/baseline/<v>/<split>/
figures/00_shap/mitigated/<v>/<split>/<axis>/
```

Filenames under that root:

| real name | alias twin | type | guard |
|---|---|---|---|
| `<v>_training_config` | — | table | |
| `<v>_feature_summary` | — | table | |
| `<v>_feature_profile` | `alias_<v>_feature_profile` | table | |
| `<v>_onehot_family_counts` | — | table | |
| `<v>_03a_score_distribution` | — | figure | |
| `<v>_04a_shap_bar_backend_comparison` | `alias_<v>_04a_shap_bar_backend_comparison` | figure | |
| `<v>_shap_mean_abs_by_backend` | `alias_<v>_shap_mean_abs_by_backend` | table | |
| `<v>_<b>_05a_input_distributions` | `alias_<v>_<b>_05a_input_distributions` | figure | |
| `<v>_<b>_06a_shap_bar` | `alias_<v>_<b>_06a_shap_bar` | figure | |
| `<v>_<b>_06b_shap_beeswarm` | `alias_<v>_<b>_06b_shap_beeswarm` | figure | |
| `<v>_<b>_06c_shap_beeswarm_abs` | `alias_<v>_<b>_06c_shap_beeswarm_abs` | figure | |
| `<v>_<b>_06d_shap_native_beeswarm` | `alias_<v>_<b>_06d_shap_native_beeswarm` | figure | `shap.plots` available |
| `<v>_<b>_06e_shap_native_bar` | `alias_<v>_<b>_06e_shap_native_bar` | figure | `shap.plots` available |
| `<v>_<b>_06f_shap_native_waterfall` | `alias_<v>_<b>_06f_shap_native_waterfall` | figure | `shap.plots` available |
| `<v>_07a_interaction_heatmap` | `alias_<v>_07a_interaction_heatmap` | figure | interaction values computed |
| `<v>_shap_interaction_strength` | `alias_<v>_shap_interaction_strength` | table | interaction values computed |
| `<v>_shap_interaction_pairs` | `alias_<v>_shap_interaction_pairs` | table | interaction values computed |
| `<v>_feature_association_spearman` | `alias_<v>_feature_association_spearman` | table | |
| `<v>_feature_association_pairs` | `alias_<v>_feature_association_pairs` | table | |
| `<v>_shap_interaction_vs_association` | `alias_<v>_shap_interaction_vs_association` | table | interaction values computed |
| `<v>_<b>_08a_dependence_<feat>` | `alias_<v>_<b>_08a_dependence_<feat>` | figure | one per `<feat>` (≤4) |
| `<v>_<b>_08b_pair_<fa>__<fb>` | `alias_<v>_<b>_08b_pair_<fa>__<fb>` | figure | interaction values computed, top 3 pairs |
| `<v>_<b>_09a_waterfall_<label>` | `alias_<v>_<b>_09a_waterfall_<label>` | figure | one per `<label>` |
| `<v>_<b>_09b_force_<label>` | `alias_<v>_<b>_09b_force_<label>` | figure | one per `<label>` |
| `<v>_<b>_09c_near_tau_band_comparison` | `alias_<v>_<b>_09c_near_tau_band_comparison` | figure + table | |
| `<v>_<b>_09d_near_tau_individual_heatmap` | `alias_<v>_<b>_09d_near_tau_individual_heatmap` | figure | both bands non-empty |
| `<v>_<b>_09d_near_tau_above_phi` | `alias_<v>_<b>_09d_near_tau_above_phi` | table | both bands non-empty |
| `<v>_<b>_09d_near_tau_below_phi` | `alias_<v>_<b>_09d_near_tau_below_phi` | table | both bands non-empty |
| `<v>_<b>_10a_band_bars` | `alias_<v>_<b>_10a_band_bars` | figure + table | |

Dataset (unchanged from §5/§5a, repeated here for completeness):

```
src/data/real/detection/shap/<v>/<v>_attributions_<split>[_native].parquet                  baseline
src/data/real/detection/shap/<v>/<v>_attributions_<split>[_native]_meta.json
src/data/real/mitigation/shap/<v>/<v>_mitigated_attributions_<split>_<axis>[_native].parquet     mitigated
src/data/real/mitigation/shap/<v>/<v>_mitigated_attributions_<split>_<axis>[_native]_meta.json
```

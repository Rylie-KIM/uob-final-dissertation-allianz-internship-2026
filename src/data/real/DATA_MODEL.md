# `src/data/real/` — data model

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
| `log_scores` | `logs/v2_score.parquet` | scoring event | `claim_id`, `correlation_id` | no |
| `log_targets` | `logs/v2_targets.parquet` | scoring event | `claim_id`, `correlation_id` | no |
| `corrected` | `mitigation/<v>_corrected.parquet` | claim | `claim_id` | yes · not produced yet |
| `reeval_scores` | `reeval/<v>_mitigated_scores.parquet` | claim | `claim_id` | yes · not produced yet |

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

### `log_scores` — `logs/v2_score.parquet`

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

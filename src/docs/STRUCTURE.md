# App Structure



```
repo/
├── pyproject.toml · uv.lock · .python-version
├── v1_csv_to_parquet.py · v1_parquet_to_csv.py
│                          # the v1 format bridge, BOTH directions. env-v1 is py3.5.6 with no
│                          #   parquet engine, so 01_export_v1 writes CSV -> parquet (in), and
│                          #   parquet -> CSV (out) for anything env-v1 must read back. Both
│                          #   formats stay on disk; neither is a copy to delete. Run in the
│                          #   analysis .venv, never in env-v1. v2/v3 need neither.
│
├── model_repos/
│   ├── practice/
│   │   ├── fttl-v1/
│   │   ├── fttl-v2/
│   │   └── fttl-v3/
│   └── v1/ · v2/ · v3/
│
└── src/
    ├── config.py             # every real name/path, per version. The ONLY file that may hold one.
    ├── schema.py             # [now] canonical column names + to_canonical()/require(). The
    │                         #   translation applied once at ingest — see § "Paths are resolved by KIND"
    ├── figstyle.py           # canonical matplotlib style for all dissertation figures (spec: notebook/FIGURE_TEMPLATE.md)
    ├── trained_order.py      # [now 2026-09-01] the trained column ORDER, defined once: model_feature_names ·
    │                         #   feature_order (exact / reordered / set_mismatch / unverified) · align ·
    │                         #   read_registry · select_features (the booster-then-registry ladder that
    │                         #   picks the model's columns out of an export matrix carrying the target).
    │                         #   shap_kit.py, shap_kit_v1.py and training/retrain.py all import these
    │                         #   four names from here and re-export them, so a `feature_order` verdict
    │                         #   means the same thing in a v1 meta as in a v3 one. It was three
    │                         #   copy-pasted copies until 2026-09-01 and they had DRIFTED: only the v1
    │                         #   one rejected xgboost's fabricated f0/f1/... names, only the retrain one
    │                         #   returned None rather than [] — so one pickle could be "exact" to one
    │                         #   caller and "unverified" to another. Stdlib only, Python 3.5 syntax,
    │                         #   ASCII-only strings, because env-v1 imports it too. NOT the producer
    │                         #   side: features/extract_features.py keeps its own (deliberately
    │                         #   different, source-labelled) ladder — see § "One notebook, three
    │                         #   environments".
    ├── model_io.py           # [now 2026-09-01] unpickling a fitted model, defined once: load_any() tries
    │                         #   sklearn.externals.joblib -> joblib -> load_compatibility -> pickle and
    │                         #   REPORTS which one answered; load_estimator() unwraps a Pipeline to its
    │                         #   final step. v1's 2018 artefact does not open under plain joblib.load
    │                         #   (sklearn < 0.23 vendored a different joblib; the standalone unpickler
    │                         #   desyncs with KeyError). The ladder was copy-pasted in shap_kit_v1.py and
    │                         #   features/extract_features_v1.py, each with a comment saying it had to
    │                         #   stay identical to the other -- a rule no code enforced. Same 3.5 +
    │                         #   ASCII + stdlib rules as trained_order.py, and the same reason.
    ├── shap_kit.py           # [now] the ONE module notebook/real/00_SHAP.ipynb imports. Runs INSIDE
    │                         #   env-v1/v2/v3, so it may use only numpy/pandas/matplotlib: every
    │                         #   figure (beeswarm · waterfall · force · dependence · band bars ·
    │                         #   interaction heatmap) is drawn from raw φ rather than through shap's
    │                         #   plotting layer, which does not exist in the xgboost-0.72-era
    │                         #   stack. Also holds the interaction/association split: TreeSHAP
    │                         #   interaction values (what the MODEL does jointly) vs
    │                         #   pearson/spearman/mutual-info on X (what the DATA does) — kept as
    │                         #   separate functions because they routinely disagree. See § "One
    │                         #   notebook, three environments".
    ├── shap_kit_v1.py        # [now 2026-08-24] Python 3.5 twin of shap_kit.py for env-v1 ONLY —
    │                         #   shap_kit/config/figstyle are 3.7+ syntax, so env-v1 cannot import
    │                         #   them. Carries the v1 constants (splits, decision rule, xgboost pin,
    │                         #   CSV path builders) + the native-TreeSHAP subset (pred_contribs;
    │                         #   tree_path_dependent only, no shap package). Imported only by
    │                         #   notebook/real/00_SHAP_v1.ipynb. Same rule as extract_features_v1.py:
    │                         #   never retrofit v1 into the shared file.
    ├── threshold.py          # [now] τ, in ONE place: tune() (fallback when a repo exposes none) ·
    │                         #   apply() dispatches on the rule SHAPE (v1 segmented / v2 piecewise /
    │                         #   v3 global) · read_off() — see § "τ has two sources"
    ├── docs/STRUCTURE.md · DESIGN.md · ENV_MANAGEMENT.md
    │
    ├── pipeline/
    │   └── pipeline.py
    ├── detector/             # "is there a loop?"        — (scores, labels); never opens a pkl
    │   ├── sfp_detector.py
    │   └── algorithm/
    ├── estimator/            # "how bad / by what mechanism?" — see § "Five layers"
    │   ├── concentration.py       # [now] Hill / Shannon / Simpson / Gini / top-k on mean|φ|, plus
    │   │                          #   require_comparable() — refuses versions attributed under
    │   │                          #   different SHAP backends. Encoded feature names only.
    │   ├── effect_estimator.py    # [plan] EffectEstimator ABC: assumptions() + falsify() + estimate()
    │   ├── rdd.py                 # notebook 04_01 — τ as a sharp discontinuity (needs payout)
    │   ├── shap_did.py            # notebook 04_02 — corruption footprint in the feature-dependence
    │   └── logit_adjust.py        # notebook 04_03 — conditional odds ratio; falsify() FAILS by design
    ├── mitigator/            # "how do we fix it?"
    │   ├── sfp_mitigator.py
    │   ├── corrector/             # TrainingDataCorrector ABC → ReweightCorrector
    │   │                          #   (03_02's naive/rarity/transport/pnu schemes, 2026-09-01;
    │   │                          #   transport/pnu emit DUPLICATED claim_ids — retrain.py's join
    │   │                          #   handles it)
    │   └── policy/
    ├── reeval/               # [plan] "what CHANGED?" — the only layer that reads TWO artefact sets
    │   ├── reevaluator.py         # ReEvaluator: composes detector/estimator over (before, after)
    │   └── metrics/               # ReEvalMetric ABC → DecisionFlipCount · DetectionDelta ·
    │                              #   ShapDiDDelta (footprint erased?) · OracleValidation (SYNTHETIC-ONLY)
    ├── loaders/              # [now] the ONE way a notebook reaches a version's artefacts
    │   └── version_data.py   #   load(v) -> VersionData: .frame · .tau · .decisions · .features
    │                         #   Reads parquet; a declared `.pkl` source goes through joblib.load,
    │                         #   NOT pd.read_pickle — the Z: artefacts wear a .pkl extension but are
    │                         #   joblib dumps, and pandas dies on them (fixed 2026-09-01).
    │                         #   No synthetic/real subclasses — config's `source` handles that.
    │
    ├── scoring/              # VERSION LAYER — runs inside env-vX; the only code that opens a pkl
    │   │                      # (ingest.py deleted 2026-08-09: the log_source → log translation is
    │   │                      #   absorbed into notebook/real/01_export_v2.ipynb — only v2 has a
    │   │                      #   production log; column renames still go through schema.py)
    │   │                      # (build_inputs.py DELETED 2026-08-19 — it wrote the LOG's rows into
    │   │                      #  targets_<v>_<split>, the export notebooks' own file, so its only
    │   │                      #  remaining effect was to overwrite a split's targets with a
    │   │                      #  different population. The export notebooks are the sole producer.)
    │   ├── predict.py         # model + features → detection/<v>_scores. [2026-09-02] the MODERN
    │   │                      #   (>=3.10) shared v2/v3 worker: importable predict() function
    │   │                      #   (03_03 calls it in-kernel); the CLI main wraps the same call.
    │   ├── predict_v1.py      # [2026-09-02] its FROZEN py3.5 twin — the env-v1 scorer (CSV I/O,
    │   │                      #   no config import, ASCII console). score_all.py dispatches on
    │   │                      #   config.NO_PARQUET_ENVS. Behaviour changes land in predict.py
    │   │                      #   FIRST, then are hand-ported here in 3.5 syntax.
    │   ├── attribute.py       # [now] model + features → detection/shap/<v>/<v>_attributions + _meta.json
    │   │                      #   (per-row φ; the sibling of predict.py — SHAP must run where the
    │   │                      #   pkl unpickles). Backends: shap TreeExplainer (interventional,
    │   │                      #   shared background) or the booster's own pred_contribs (needs no
    │   │                      #   `shap` in a frozen env, but is tree-path-dependent). The meta
    │   │                      #   records which — and the estimator's hyperparameters (§1.4c).
    │   ├── score_all.py       # [now] driver: loops the versions, calls each env's interpreter.
    │   │                      #   Replaced score_all.sh (deleted 2026-07-31).
    │   ├── attribute_all.py   # [now] the attribution driver. Also picks ONE claim set (id-column
    │   │                      #   intersection + seeded sample) so every version explains the same
    │   │                      #   rows — otherwise a concentration shift can be case-mix.
    │   ├── backfill_feature_order.py
    │   │                      # [now] adds `feature_order` to metas written before the check
    │   │                      #   existed (2026-08-31). Reads meta["feature_names"] against the
    │   │                      #   pickle's own trained names — no phi are recomputed, no parquet
    │   │                      #   is opened. v2/v3 only; v1 gets its own file (py3.5).
    │   ├── backfill_feature_order_v1.py
    │   │                      # [now] the py3.5 twin. Separate because env-v1 cannot import
    │   │                      #   config.py at all (PEP 563/585 syntax) and because v1 resolves
    │   │                      #   the column order from features/registry/v1.json when xgboost
    │   │                      #   0.72 exposes only f0/f1/… — the status then says "(via registry)".
    │   └── load_scores.py     # (inspect_pickle.py DELETED 2026-08-19 — the one-off onboarding
    │                          #  util that reported a prod pickle's schema + date range. Its job
    │                          #  is done: the export notebooks 01_export_v{1,2,3} now read every
    │                          #  Z: source inside its own env and write canonical parquet.)
    ├── preprocessing/
    │   ├── base.py
    │   └── v1.py · v2.py · v3.py
    ├── training/             # VERSION LAYER — the worker runs inside env-vX, the driver does not
    │   ├── retrain.py         # MITIGATED retrainer — clones baseline hyperparams, weighted.
    │   │                      #  The ONLY trainer left. [2026-09-02] MODERN (>=3.10): v1 is never
    │   │                      #  retrained, so the py3.5 discipline it used to carry protected an
    │   │                      #  interpreter that can never reach it. Importable retrain()
    │   │                      #  function (03_03 calls it in-kernel); the CLI main wraps the same
    │   │                      #  call. Paths still arrive as ARGUMENTS, never from config — the
    │   │                      #  caller resolves them (driver / notebook). Feature columns come
    │   │                      #  from the BASELINE BOOSTER's own feature_names, never "everything
    │   │                      #  except claim_id" — the export notebooks write the TARGET into
    │   │                      #  the matrix.
    │   ├── retrain_all.py     # [now] its config-aware driver, in the ANALYSIS env: resolves
    │   │                      #  model/processed_inputs/corrected/mitigated by KIND and launches
    │   │                      #  each version's interpreter. Same split as score_all -> predict.
    │   └── spec.py            # (train.py + train_all.py DELETED 2026-08-19 — BASELINE training
    │                          #  is dead code on real data: all three versions declare
    │                          #  paths.model, so the baseline is each version's own production
    │                          #  pickle, LOADED not refitted. The driver skipped all three, and
    │                          #  its --force path wrote over the REAL pickle. Hyperparameters
    │                          #  survive in config.TRAINING_CONFIG if a refit is ever needed.)
    ├── envs/                 # specs + build notes; the scripts here inspect envs, never build them
    │   ├── v1/ · v2/ · v3/    #  v1/requirements.txt records how env-v1 was REALLY built;
    │   │                      #  v1/check_shap.py probes whether a `shap` wheel actually works
    │   │                      #  there (py3.5.6 + xgboost 0.72) — interventional vs the native
    │   │                      #  pred_contribs fallback, run BEFORE switching v1 over.
    │   │                      #  v1/compare_margin_reference.py explains the additivity fix of
    │   │                      #  2026-08-24: one phi, scored against the OLD reference
    │   │                      #  (est.predict(output_margin=True), which xgboost < 0.81's
    │   │                      #  sklearn wrapper does not honour) and the NEW one
    │   │                      #  (shap_kit_v1.model_margin → the booster) side by side
    │   └── LOCAL_STANDIN.md   # what the LOCAL .venvs are and are not — v3 matches its pin
    │                          #   (xgboost 3.2.0), v1/v2 cannot and deliberately carry none
    │
    ├── models/
    │   ├── synthetic/
    │   │   ├── baseline/  v1.pkl · v2.pkl · v3.pkl
    │   │   └── mitigated/ v1.pkl · v2.pkl · v3.pkl
    │   └── real/
    │       ├── baseline/  …
    │       └── mitigated/ …
    │
    └── data/
        ├── make_dummy_real.py # writes a FAKE data/real/ tree with the REAL shapes, so the chain
        │                      #   runs on the local laptop; seeded RNG, no Allianz value. Guarded
        │                      #   by a _DUMMY_DATA marker and --clean. Never run it where the
        │                      #   real exports live. See § "The local dummy tree".
        ├── make_dummy_shap.py # the one kind make_dummy_real cannot reach: attributions come out
        │                      #   of scoring/attribute.py, which must OPEN A PICKLE inside that
        │                      #   version's env. Writes detection/shap/<v>/ for every version ×
        │                      #   split × backend so 00_shap_attribution.ipynb has all three of
        │                      #   its runs to read. Same marker, same guard.
        ├── synthetic/         # ⚠ THE WHOLE SUBTREE IS A STAND-IN — deleted when real data lands.
        │   │                  #   App code must NEVER import from it. Notebooks may use ALL of it.
        │   ├── run.py · evaluate.py · export_thresholds.py
        │   ├── generate/      # ← WORLD ①: the DGP. 70,000 rows × 54 cols; v1/v2a/v2b/v3a/v3b.
        │   │   └── csv/ · parquet/   (+ parquet/features/)     What every NOTEBOOK reads today.
        │   ├── script/
        │   ├── logs/          #  ┐ WORLD ②: the app dry-run. 4,000 rows × 2 feats (make,
        │   ├── inputs/        #  │ repair_ratio), v1/v2/v3, from model_repos/practice/.
        │   ├── detection/     #  │ Unrelated to ① — a scaffold that proves the app WIRING,
        │   ├── mitigation/    #  │ not a dataset anyone analyses. Mirrors data/real/ exactly.
        │   └── reeval/        #  ┘ (see § "Three worlds", and the two @TODOs there)
        └── real/          # gitignored (src/data/real/) — real exports, or the dummy stand-in
            ├── _DUMMY_DATA    #   present ⇒ this tree is fake and disposable; absent ⇒ treat as real
            ├── DATA_MODEL.md  #   EVERY artefact's columns, grain and keys, one page. The single
            │                  #   .gitignore exception in this tree, because it must reach the
            │                  #   company laptop. Layout lives here; COLUMNS live there.
            ├── logs/          # only v2 has a production log (v1's destroyed, v3 never deployed)
            │   ├── v2.parquet         # kind `log`       — ONE ROW PER CLAIM, canonical names
            │   ├── v2_raw.parquet      # `log_raw`      ┐ one row per SCORING EVENT, keyed
            │   ├── v2_features.parquet # `log_features` │ (ClaimNumber, correlation_id).
            │   ├── v2_scores.parquet   # `log_scores`   │ NOT pre-joined — raw/features keep
            │   └── v2_targets.parquet  # `log_targets`  ┘ v2's names, score/targets ours.
            ├── inputs/
            ├── detection/     # `scores`: <v>_scores_<split>.parquet, flat
            │   └── shap/
            │       └── <v>/   # [2026-08-24] `attributions` gets a per-version DIRECTORY — it is the
            │                  #   one kind that multiplies (splits x backends x parquet + _meta.json),
            │                  #   so a flat shap/ mixed all three models' runs together. Holds
            │                  #   <v>_attributions_<split>[_<backend>].parquet / _meta.json; the <v>_
            │                  #   prefix is kept so a file copied out still names its model.
            ├── mitigation/
            └── reeval/
```




> **Two layers, not three apps (decided 2026-07-01).** The project is split **by concern**, not by model version. **Version Layer — per-version model worker** (`envs/v1│v2│v3`): preprocessing → (re)train → score for ONE version's artefact, each in its own isolated env / `pyproject.toml`. **Analysis Layer** (`pipeline/` + `detector/` + `mitigator/` + re-evaluation): a **single, version-agnostic** app, never triplicated — the SFP signal lives *between* versions and the comparison needs identical code. The layers never share a process (Version Layer writes `inputs/features_<v>.parquet` + `detection/<v>_scores.parquet`; Analysis Layer reads + merges on `claim_id`). Note Version Layer owns **(re)train + score**, not only score: after-mitigation retraining runs in the version's own env on the Analysis-Layer-corrected training set, holding that version's preprocessing fixed across the pre→post comparison. Full rationale + the re-evaluation invariant: `README.md` § "Application Architecture — Split by Concern (Two Layers), Not by Model Version" and `problem.md` §2.5 #12.

> **Directory reorganisation (decided 2026-07-03).** The old `src/data/scores/{features,labels}/` bag-of-artefacts and `src/model/envs/` were reorganised along two clean axes:
> - **`src/model/` → `src/envs/`.** That directory only ever held per-version `.venv`s (no models live there — the pkls are elsewhere), so the name `model/` was misleading. Renamed to `envs/`, which is what it is. Source-agnostic (an env is a library stack), so it is **not** split by synthetic/real.
> - **Data is split by *source* then *pipeline stage*:** `data/{synthetic,real}/{inputs,detection,mitigation,reeval}/`. `inputs/` holds the fixed *materials* (per-version `features_<v>` + `targets_<v>`); the three stage dirs hold pure *outputs* (`detection/` = baseline scores → detector; `mitigation/` = corrected targets ← mitigator; `reeval/` = post-retrain scores). ⚠️ `inputs/features_<v>` is the one file used by **both** scoring and retrain — it is a shared material, not a stage output, hence its own `inputs/` bucket. It is also the likely edit point if feature engineering is revisited.
> - **Model pkls leave `data/` entirely → `src/models/{synthetic,real}/{baseline,mitigated}/`.** A fitted `.pkl` is a model artefact, not data. `baseline/` = the version repo's code retrained on its **original (contaminated)** target = production reproduction; `mitigated/` = the same code retrained on the **corrector's de-contaminated** target. Folder carries the meaning → filename is just `v<k>.pkl`.
>
> **Why baseline pkls are ours to regenerate, not vendored.** Industry convention: code lives in git, **fitted model binaries do not** (they bloat history, don't diff, and are reproducible from code + data + pinned env — they belong in a model registry / object store). The version repos under `model_repos/` are therefore treated as **code only**; every pkl this project scores — baseline *and* mitigated — is **regenerated here** by retraining from that repo's code in its pinned env. Hence both land in `src/models/`, and `model_repos/` is never written to.

> **Legend:** `[now]` = exists on disk today · `[plan]` = target design, not yet created · `[move]` = exists on disk but under the *old* path, pending the reorg above. **The `src/` application code was scaffolded on 2026-07-01 and then deleted the same day** (it was premature); everything under the layer headings is therefore `[plan]` again. The design it describes still stands; only the code is gone.

> **Note — `data/` is data-only; code lives in the layer packages.** Source-first means the synthetic *generator* (`run.py`, `generate/` — which holds the DGP output `csv/` + `parquet/` — and `script/`) and the *staged artefacts* (`logs/`, `inputs/…reeval/`) sit under the **same** `data/synthetic/` node — the generator is what makes this source. The stage dirs (`logs/` + the four `inputs/…reeval/`) mirror exactly under `data/real/` (which has no generator — real data arrives from Allianz). Analysis/pipeline **code never lives in `data/`**: score-ingestion (`load_scores.py`) and the per-version scoring/attribution workers live under `scoring/` (log-ingestion and the log→inputs split are gone — `ingest.py` and `build_inputs.py`, absorbed into the export notebooks), and the planned `DataLoader` classes under `loaders/` — keeping `data/` purely for stored artefacts.

> **Note — synthetic tests read ONLY `data/synthetic/generate/` (added 2026-07-06; ~~superseded 2026-07-14~~).** ⚠️ **The restrictive half of this note is retired — see § "Three worlds" and § "The one invariant" below.** Notebooks may now use **anything** under `src/data/synthetic/`, including `generate.*`. What survives from the original note, and is still true: the `logs/` · `inputs/` · `detection/` · `mitigation/` · `reeval/` subtrees are **not a dataset anyone analyses** — they are a synthetic **dry-run of the real-data pipeline**, materialised only to pin down and validate the application wiring (source→stage layout; ingest/build/score/retrain) **before** any real Allianz resource is connected. Once real data arrives it flows through the identical `data/real/{logs,inputs,detection,mitigation,reeval}/` tree. Treat them as scaffolding for the real-data integration.

> **Note — log ingestion is the one name-translation point (added 2026-07-06; ~~manifest.json~~ → config, and the translation extended to COLUMNS, 2026-07-31).** Each model application emits its production log under whatever name IT chose, with whatever column names IT chose. `scoring/ingest.py` is the SINGLE place that translates both into ours: it reads `config.path("log_source", v)`, renames the columns via `schema.to_canonical`, checks the required ones survived (`schema.require`, which names the offending config entry when one did not), and writes `config.path("log", v)`. Everything downstream reads only the canonical log. `logs/manifest.json` is retired — the source path is now `paths.log_source` in config, so the file name and the column names are edited in the same one place.

> **Reality vs target (OO layers built 2026-07-04).** The reorg **and** the Analysis-Layer class hierarchy are **done and verified** — `src/pipeline/pipeline.py` (`SFPPipeline`) runs the full synthetic chain end-to-end (detect → mitigate → re-eval for v1/v2/v3), identical results to the retired procedural `run_cycle.py`. On disk and working: `src/config.py`; the design docs under `src/docs/`; the per-version envs at **`src/envs/v{1,2,3}/`**; the source→stage data tree **`src/data/synthetic/{inputs,detection,mitigation,reeval}/`**; all pkls at **`src/models/synthetic/{baseline,mitigated}/v{1,2,3}.pkl`**; the **Analysis-Layer OO impl** — `pipeline/pipeline.py` (`SFPPipeline`), `detector/sfp_detector.py` (`SFPDetector`) + `detector/algorithm/` (`DetectionAlgorithm` ABC → `ResidualPeakAlgorithm`), `mitigator/sfp_mitigator.py` (`SFPMitigator`) + `mitigator/corrector/` (`TrainingDataCorrector` ABC → `IPSCorrector`, deleted 2026-09-03 — `ReweightCorrector` is the thesis corrector) + `mitigator/policy/` (`InvestigationPolicy` ABC); the scoring I/O (`scoring/predict.py`, `score_all.py`, `load_scores.py`; `ingest.py` and `build_inputs.py` were here then, deleted 2026-08-09 / 2026-08-19); the **training I/O** (`training/retrain.py` = mitigated retrainer; the baseline trainer `train.py` and its `train_all.py` driver were here then, both deleted 2026-08-19); the canonical **log-ingestion landing zone** `data/synthetic/logs/` (`manifest.json` + `<v>.parquet`); the `src/data/synthetic/` generator tree; and the **working practice repos** `model_repos/practice/fttl-v{1,2,3}/` (code-only — pkls moved out). Still **design-only**: `preprocessing/`, `training/spec.py`, `loaders/`, concrete `InvestigationPolicy` impls, and the whole `data/real/` + `models/real/` side (arrive with the real version repos). The repo-root `pyproject.toml` + `uv.lock` + `.python-version` are the **uv-managed analysis env** (`.venv`, py3.11 — `uv sync`). `xgboost` needs system OpenMP (`brew install libomp`). The entry point is `src/pipeline/pipeline.py`; there is no `src/main.py`.

> **Per-version features, built through each version's own repo (both sources).** Each version is scored on its own `inputs/features_<v>.parquet`, produced by **that version's repo preprocessing** — the `preprocessing/v{1,2,3}.py` adapters run each repo's feature builder. This is now **identical for synthetic and real** (decided 2026-07-03): synthetic is only a temporary stand-in, so it goes through the *same* external-repo path, not a special shared recipe. The single difference is where the raw claims come from — synthetic **generates** them (`data/synthetic/` DGP), real **receives** them from Allianz. Because v1/v2/v3 preprocessing genuinely diverges (`V2/V3FeatureBuilder`; `problem.md` §2.5 #10/#11), `features_<v>` files differ across versions on **both** sources. See `DESIGN.md` § "Per-version feature matrices". **This invariant is load-bearing and must be kept** — see § "What synthetic cannot rehearse" below.

---

## Three worlds (clarified 2026-07-14)

Three *unrelated* datasets live in this repo, and confusing them has already cost time. They are:

| | what it is | size | versions | who reads it |
|---|---|---|---|---|
| **① DGP** `data/synthetic/generate/` | synthetic claims + simulated production models, made by `run.py` | **70,000 × 54** | v1, v2a, v2b, v3a, v3b | **every notebook** |
| **② app dry-run** `data/synthetic/{logs,inputs,detection,mitigation,reeval}/` | a toy pipeline pass, fed by `model_repos/practice/fttl-v{1,2,3}/` | **4,000 × 2** (`make`, `repair_ratio`) | v1, v2, v3 | **the app** (`SFPPipeline`) |
| **③ real** `data/real/` | the actual Allianz logs + version repos | — | v1, v2, v3 | nothing yet |

**① and ② are not connected.** Different rows, different features, different version names. The
consequence, stated plainly so nobody rediscovers it: **the dissertation's METHOD grows on ①, while
the app's STRUCTURE grows on ②.** That is deliberate for now — ① is where the research happens, ② is
where the wiring is proven — but it means the app has only ever been exercised on a 2-feature toy.

**What ① actually is.** The DGP is not "a test fixture". It plays the role of **Allianz's whole
Version Layer**: it manufactures claims, trains each model version, applies each version's τ, and
emits scores/decisions/observed outcomes. It is the stand-in for *real claims + real production
logs*. `model_repos/practice/` is a separate stand-in for *real model code*.

**The target state**, once real resources land: the DGP's output enters through the **same canonical
doorway** real data uses (`logs/<v>.parquet` → `inputs/`), so the Analysis Layer never learns whether
it is looking at synthetic or real. That is the design; it is **not built yet**:

- **@TODO (real-data arrival)** — connect ① to the canonical doorway, *or* delete it. Blocked today
  because we clone the v1/v2/v3 model repos, and a pre-baked 70k dataset does not respect that
  contract.
- **@TODO (real-data arrival)** — retire ② and `model_repos/practice/`.
- **@TODO (real-data arrival)** — delete `data/synthetic/` entirely; **nothing else should change.**
  If deleting it breaks app code, the invariant below has been violated.

## The one invariant

> **Application code — `detector/` `estimator/` `mitigator/` `reeval/` `pipeline/` `scoring/`
> `training/` — must NEVER import `generate.*`.**
> **Notebooks may import anything under `src/data/synthetic/`, `generate.*` included.**

That is the whole rule, and it is mechanically checkable. Notebooks are a workbench for picking the
logic apart on synthetic data; coupling them to the DGP costs nothing, because when real data lands
only their *data paths* change. Application code is different: it must survive the deletion of
`data/synthetic/`, so it may only ever touch canonical artefacts.

## Paths are resolved by KIND, never by filename (added 2026-07-31)

The three real version repos have **different internal structures** — different folder names,
different pickle filenames, different column names, different places for the production log. So
`src/` code may not name a file; it names a **kind** and a **version**, and `config.py` answers with
a path.

```python
config.path("model", "v2")                                # -> that version's pickle, wherever it lives
config.path("processed_inputs", "v2", split="test")       # -> that split's feature matrix
config.path("scores", "v2", split="test")                 # -> where WE write its recomputed scores
config.column("v2", "date")            # -> "ReportedDate"   (v1: "lossdate", v3: "ReportedDate_CLAIM")
```

`KINDS` splits in two, by **who made the artefact** — not by who it belongs to:

| | kinds | resolution |
|---|---|---|
| `READ_KINDS` | `model` · `preprocessor` · `log_source` · `processed_inputs` · `raw_dataset` | already exist in the real repo → **declared** per version in `VERSIONS[v]["paths"]` |
| `WRITE_KINDS` | `log` · `log_raw` · `log_features` · `log_scores` · `log_targets` · `targets` · `scores` · `attributions` · `corrected` · `mitigated` · `reeval_scores` | this project produces them → **`FALLBACK`** template under our own tree |

**A declared path always beats the template, for every kind.** That is the whole point: a kind moves
from "we generate it" to "the real repo already has it" by filling in one line of config, with no
code change anywhere. `paths.model` is the live example — leave it blank and we score a baseline pkl
we retrained ourselves; fill it in and we score that version's real production pickle. All three
versions declare it today, so the baseline is always the measured pickle, **loaded and never
refitted** — which is why the baseline trainer was deleted (2026-08-19). ⚠️ Corrected 2026-08-19:
v1's *training* data is NOT lost — `inputs_transformed.pkl` survives on the `Z:` drive
(user-confirmed) and 01_export_v1 exports it, so v1 supports the mitigation dataset and
`retrain.py` like the others. What was destroyed is v1's production **log**, which is a different
artefact (`paths.log_source = None`).

**Which columns of the matrix are model inputs comes from the REGISTRY, never from config**
*(unified 2026-08-20)*. `processed_inputs` is not feature-only — every version's export carries the
target beside the inputs, and v3's also carries its own saved predictions — so
`df.drop(columns=["claim_id"])` is a bug wherever it appears. The answer is
`features/registry/<v>.json`, written by `features/extract_features.py` **inside each version env**
off that version's own pickles: `raw_features` (what the preprocessor expects) and `model_features`
(what the booster consumes). `config.registry_path(v)` resolves the path; config never stores the
names, because a copy would be a second source of truth free to drift from the pickles — the
failure mode `01_export_v2.ipynb` documents for its pasted `MODEL_FEATURES`. Consumers try the open
booster first and the registry second, and refuse rather than guess when neither route answers.

*(revised 2026-09-01.)* That ladder is now **one function** — `trained_order.select_features`,
imported by `training/retrain.py` and `scoring/predict.py` alike, so fitting and scoring cannot
pick different columns; `notebook/real/00_SHAP.ipynb` §2 reaches it through `shap_kit`. A process
that cannot open the pickle at all asks `config.model_features(v)`, which reads the same registry
— that is the analysis-env route, and `mitigator/corrector/reweight.py` takes it (its transport
model g(x) would otherwise be fitted on every column but `claim_id`, i.e. on the outcome it
exists to recover).
`scoring/predict.py` had the same `drop(columns=["claim_id"])` bug and now selects by trained
name, which fixes the column ORDER at the same time: xgboost matches positionally when names are
absent and raises `feature_names mismatch` when they are present, so a reordered frame is either
silently wrong or a hard stop.

**Columns are translated once, at ingest.** `VERSIONS[v]["columns"]` maps our canonical name →
that version's real name (`config.rename_map(v)` hands back the `df.rename` dict). Downstream code
only ever says `claim_id` / `date` / `score` / `decision` / `observed` / `mobility`. A version that
genuinely lacks a column declares `None` rather than a placeholder — v2 and v3 have no `mobility`,
because only v1's decision rule is segmented on it.

**The kind is `targets`, not `labels`.** What a version was trained against is *not* a ground
truth — for v2 and v3 it is the previous model's `observed_outcome`, forced to 1 for every scrapped
car. Calling it `labels` invites reading it as truth, which is the exact error the whole project is
about. (The fallback filename follows: `inputs/targets_<v>.parquet`. The legacy synthetic dry-run
still has `labels_<v>.parquet` on disk — rename those, or declare the path, if ② is ever re-run.)

**Synthetic names are gone from application code (2026-07-31).** `config.VERSIONS[v]["columns"]
["observed"]` briefly carried `pre_ml_label` / `model_v1_observed_outcome` / `model_v2_observed_outcome`
marked "confirmed". They are **synthetic DGP column names** — they sit in
`src/data/synthetic/csv/claims_v1_log.csv`, not in any Allianz log — and were reset to placeholders.
`trained_on` survives as a concept (the contamination chain `pre_ml -> v1 -> v2`), no longer as a
column name. `config.FEATURE_COLS` / `CATEGORICAL_COLS` were deleted for the same reason: the DGP's
schema had no business in a real-data config, and nothing imported them. Notebooks still import
`SCRAP_THRESHOLD` and `TARGET_PRECISION`, so those stay. The oracle column `true_garage_outcome` is
now **optional everywhere** (`pipeline.py --oracle-col`, off by default) rather than required, since
real data cannot have it: a scrapped car is never sent to a garage.

### One env cannot read parquet, so paths have two flavours (added 2026-09-01)

`config.path()` is the **analysis env's** truth: every artefact is parquet, identically for all
three versions, and that is what `loaders/` and the notebooks resolve. env-v1 cannot honour it —
Python 3.5.6 has no pyarrow and no fastparquet, and neither can be built for it.

```python
config.NO_PARQUET_ENVS                                   # ("v1",) — a legacy carve-out, not a list that grows
config.path("scores", "v1", split="test")                # detection/v1_scores_test.parquet   <- analysis reads
config.worker_path("scores", "v1", split="test")         # detection/v1_scores_test.csv       <- env-v1 writes
config.worker_path("scores", "v2", split="test")         # …v2_scores_test.parquet — unchanged
```

Same directory, same stem, different extension — exactly the convention `01_export_v1.ipynb`
writes under and `v1_csv_to_parquet.py` converts from. **Hand `worker_path` to anything running
inside a version env** (`score_all.py` does, for `--features` and `--out`) and keep `path` for
everything the analysis env reads. A v1 run therefore leaves CSV behind and is not finished until
the converter has run; `score_all.py` prints that reminder rather than assuming it.

CSV is the *transport* out of env-v1, not a second storage format. Unifying v1 on CSV would push
a per-version branch into every reader and re-infer dtypes on every read instead of fixing them
once at conversion — including `claim_id`, whose dtype decides whether
`attribute_all.common_sample()` finds a cross-version intersection at all.

### Some kinds exist ONLY per split (added 2026-08-18)

The export notebooks (`notebook/real/01_export_v{1,2,3}.ipynb`) write **one file per split and no
unsplit base file** — `features_v2_train.parquet`, `features_v2_val.parquet`,
`features_v2_test.parquet`, and nothing called `features_v2.parquet`. `config.SPLIT_KINDS` names
the kinds this applies to: `processed_inputs` · `targets` · `scores`, plus everything derived from
one of them (`attributions` · `corrected` · `mitigated` · `reeval_scores`).

So `config.path()` **requires** `split=` for those kinds and **rejects** it for the rest, rather
than resolving to a filename nothing produces:

```python
config.path("processed_inputs", "v2", split="test")   # inputs/features_v2_test.parquet
config.path("processed_inputs", "v2")                 # ValueError — which split?
config.path("log", "v2", split="test")                # ValueError — the log has no splits
config.path("scores", "v2", split="oot")              # KeyError  — v2's splits are train/val/test
```

**Split names are each version's own and are never unified** (`config.SPLITS`): v1 and v2 invert
what `test` means, only v3 has `oot`, only v1 has `val1`/`val2`. `config.OOT_SPLIT` records which
one is each version's **out-of-time holdout** — `{"v1": "val2", "v2": "test", "v3": "oot"}`
(v1 confirmed 2026-08-18; note v1's OOT is *not* its `test`). A cross-version comparison should
put every version on the same *kind* of split, or part of the difference is
in-sample-vs-out-of-sample rather than the fitted functions. That is why the drivers take a
mapping — `--split v2=test v3=oot` — and a bare `--split test` is only shorthand for "this name,
for every version named". `config.resolve_splits()` is the one parser; it validates before any
subprocess is launched.

**Why this is not just plumbing.** The split decides what a number means. SHAP concentration
measured on `train` describes the fitted function on data it saw; on a holdout it describes
generalisation; one version on train against another on a holdout is a confound, not a finding. So
the split is stamped into the output filename **and** into the attribution sidecar meta, and
`concentration.require_comparable()` prints it back when the versions disagree.

**Pooling is available but must be typed — at the call site.** One `loaders.load()` is one split;
the loader has no pooled mode. Where the analysis is a *population* statement across versions —
`02_error_inheritance` is the case — the notebook concatenates `config.SPLITS[version]` itself, in
a helper the reader can see, and carries no split marker, since v1's `test` and v2's `test` are
different sets.

**What this does NOT unify:** the decision rule. `DECISION_RULES` keeps v1 segmented, v2
piecewise, v3 global, because that is a difference in *meaning*, not in naming — see § "τ has two
sources". Config unifies names; it must never flatten semantics.

`python src/config.py` reports remaining placeholders; `--paths` and `--columns` print the full
resolved mapping.

### The v2 production log is FOUR artefacts (2026-08-19, revised 2026-08-20)

`notebook/real/01_export_v2_logs.ipynb` reads the live log through the v2 repo's own
`lvanalytics/evaluation_helpers.py` (inside env-v2 — the frame is bound to that stack).

**`v2_logs` is not one kind of data.** What `eh` hands back is the raw claim frame joined to the
transformed matrix — on `ClaimNumber` + `correlation_id`, `suffix="_transformed"` — with the
prediction columns alongside. Exported as a single file, nothing in it separates a raw value from
what the model actually consumed, so the export **splits it by column**. Two traps it avoids:

* **The suffix is not the test.** `_transformed` lands only on *colliding* names, so a transformed
  column with no raw namesake arrives bare. The judge is `param.py`'s `MODEL_FEATURES`: for each
  fitted feature `f`, `f_transformed` is the transformed column if it exists (and bare `f` is then
  the raw one), otherwise bare `f` is. `log_features` writes them back under the model's own
  names — SHAP, the registry and `feature_overlap.json` all key on those.
* **`ClaimNumber` alone is not a key.** One claim can be scored more than once, so
  `correlation_id` is part of the row identity and every piece carries both keys.

| kind | file | grain | columns |
|---|---|---|---|
| `log_raw` | `logs/v2_raw.parquet` | scoring event | keys + every column that is neither a model feature nor a prediction, v2's own names |
| `log_features` | `logs/v2_features.parquet` | scoring event | both keys + `param.py` `MODEL_FEATURES` in fit order (that order is only knowable inside env-v2, so the file carries it) |
| `log_scores` | `logs/v2_scores.parquet` | scoring event | both keys + `score` · `decision` (canonical) |
| `log_targets` | `logs/v2_targets.parquet` | scoring event | both keys + `date` · `observed` (canonical; `observed` is the v3 extract's `Fttl`) |
| `log` | `logs/v2.parquet` | **claim** | `claim_id · date · score · decision · observed` |

All four carry the same key pair — `claim_id` (canonical) + `correlation_id` (its own name; there
is no canonical name for a scoring event) — so they always rejoin the same way. Only the VALUE
columns differ: `log_scores` / `log_targets` use canonical names (`schema.SCORE`, `DECISION`,
`DATE`, `OBSERVED`), while `log_raw` / `log_features` keep v2's own, because a feature name IS
the measurement.

Only `log` is collapsed to one row per claim — that is the shape
`loaders.load("v2").log` and `threshold.read_off` require. Collapsing needs a rule for which
scoring event *is* the claim's decision; the notebook's `EVENT_PICK` makes that an explicit,
recorded choice and **raises rather than defaulting** when a claim has more than one event.
The four event-grain pieces are the same row set cut by column, so they rejoin on both keys exactly.

**Why the log's score column is `score`, not `model_v2_score`.** The `scores` kind carries a
version-tagged name because a cross-version test merges two versions' scores into one frame on
`claim_id` — `02_error_inheritance` joins v1's and v2's, and untagged they would collide into
`score_x`/`score_y`. A log is one version's own
record, is never merged with another version's log (there are no others), and is read by
`threshold.apply(..., score_col=schema.SCORE)`, which expects the canonical name. `make_dummy_real.py` renames in the same direction on purpose. If a log ever
needs a version tag, rename on read: `d.log.rename(columns={schema.SCORE: d.score_col})`.

**Neither kind is a SPLIT kind.** train/val/test is a property of model *training*, not of
production serving: the log spans the serving history and most of it postdates training. So
`config.path("log_features", "v2")` takes no `split=`, and `"log"` must never be added to
`config.SPLITS` — that would make production rows resolvable as if they were a training split.
The log's natural partition is the **regime** (`DECISION_RULES["v2"]["regimes"]`, five of them),
computed from `date` on read, not baked into filenames.

**`observed` does not come from the log.** The live log carries no outcome column; the label is
joined in from the v3 extract's `Fttl` on claim number (left join — the log's row count is
preserved). Rows outside the extract's window (2023-06 → 2026-05) therefore have `observed = null`,
which means *not observed*, never 0.

**`decision` must be the log's own fast-track flag.** Reconstructing it as `score >= τ` is wrong:
v2's τ moved four times, so a single cutoff mislabels four of the five regimes. If the flag turns
out to be absent, a regime-aware reconstruction via `threshold.apply` is the fallback and must be
recorded as reconstructed — the divergence between the rule and what was actually actioned is part
of what the analysis is looking for.

### The local dummy tree (added 2026-08-18)

`src/data/real/` is gitignored and holds one of two things, depending on the machine:

| machine | what is there | marker |
|---|---|---|
| company laptop | the real exports from `notebook/real/01_export_v{1,2,3}.ipynb` | none |
| local laptop | a fake set from `python src/data/make_dummy_real.py` | `_DUMMY_DATA` |

The generator writes every kind at its real path with its real per-split filenames, canonical
columns, each version's own encoded feature names, each version's date window, and score distributions
with deliberate mass **above** each cutoff and in the **boundary band** just below it — so the
detector, the RDD window, `03_02`'s corrector and `02_error_inheritance`'s band all have rows to
work on. v1 and v2 share a claim-id pool (v2's window sits inside v1's production era) while v3
does not, which is the real join structure.

For v2 it also writes the `log_features` matrix beside the log, pooled off the split matrices —
on the real tree those rows are production rows, not the training splits, and the dummy owes the
chain only the shape. It fakes none of `log_raw` / `log_scores` / `log_targets`: they carry v2's own encoded column
names and the `correlation_id` event key, which the generator does not know, and nothing
downstream reads them yet.

Two things it does not fake, on purpose: **v1's and v3's production logs** (destroyed / never
deployed — the single hardest constraint the project works under), and the **model pickles**
(`paths.model` is declared per version and resolves through `repo_dir`, which is a placeholder
off-site). So locally the data half of the chain runs and the model half stops with a message
naming the config entry to fill — which is the correct outcome, not a gap in the dummy set.

**Attributions are generated separately** (`make_dummy_shap.py`, added 2026-08-25), because they
are the one kind that cannot be drawn from thin air the way a matrix can: on a real tree they come
out of `scoring/attribute.py`, which opens a pickle inside that version's env. It writes every
version × split × **both backends** — the canonical no-suffix file labelled `shap`/`interventional`
and its `_native` sibling labelled `native`/`tree_path_dependent`, the naming `attribute_all.py
--out-suffix` uses and `00_shap_attribution.ipynb` reads. Three properties are faithful because the
notebook actually tests them: feature columns come from `features/registry/<v>.json` (so §2's
cross-check passes rather than being skipped), the backend labels are distinct (so
`concentration.require_comparable` is exercised instead of trivially satisfied), and φ is additive —
`φ.sum(axis=1) + base_value == logit(score)` to ~1e-15. What is **not** faithful, on purpose: no
version is drawn more concentrated than another. Baking the SFP direction into the stand-in would
let a figure "confirm" the hypothesis before any real data exists.

`--clean` removes exactly the files listed in the marker, plus the `.csv` sidecar of any listed
parquet (`v1_parquet_to_csv.py` writes those beside every v1 parquet and cannot record them —
it also runs on the real tree, where there is no marker). The `_guard` refuses to write into a
tree holding files it did not write, so the first run on the company laptop stops before touching
a real export.

## Five layers, split by which artefacts each one opens

The layer boundaries are not conceptual — they follow mechanically from **what each layer must read**:

| layer | reads | answers | opens a pkl? |
|---|---|---|---|
| `detector/` | `detection/<v>_scores` + `inputs/targets_<v>` | **Is there a loop?** | no |
| **`estimator/`** ★ | the above + `detection/shap/<v>/<v>_attributions` + τ | **How harmful? By what mechanism?** | **no** — see below |
| `mitigator/` | `inputs/{features,labels}_<v>` | **How do we fix it?** | no |
| **`reeval/`** ★ | **TWO artefact sets** (before + after) | **What changed?** | no |
| `scoring/` · `training/` | the model itself | (produces everything above) | **yes — only these** |

**Why `estimator/` is its own layer.** SHAP needs the *model function*, not just its scores — which
naively makes it the first analysis layer that must open a pkl. It must not. **The pkls only unpickle
inside their version env** (they carry that repo's `FeatureBuilder`; loading `models/*/baseline/v1.pkl`
from the analysis `.venv` raises `ModuleNotFoundError: fttl_v1`, and this will be equally true of the
real pkls). So attribution runs in the Version Layer — **`scoring/attribute.py`, the sibling of
`predict.py`** — and emits `detection/shap/<v>/<v>_attributions.parquet`. `estimator/` then reads that parquet
and never touches a model. The two-layer split is preserved.

**Built 2026-08-01** (`attribute.py` · `attribute_all.py` · `estimator/concentration.py` ·
`notebook/real/00_shap_attribution.ipynb`, guide in `notebook/real/README.md`). Three things the
implementation had to decide, all of them recorded in the per-run `<v>_attributions_meta.json` rather
than assumed:
> - **Which TreeSHAP.** Interventional against a *shared* background makes a cross-version difference
>   a difference in the model **function**; the booster's own `pred_contribs` needs no `shap` package
>   in a frozen version env but is **tree-path-dependent**, so part of any difference is a training
>   distribution. `concentration.require_comparable()` **raises** on a mixed set — silently mixed
>   backends produce a plausible-looking number.
> - **Which rows.** `attribute_all.py` samples **per version**, because the windows leave no claim
>   common to all three (v2 2018-01→2020-09, v3 2023-06→2026-05). Every cross-version difference is
>   therefore case-mix confounded, as a **standing** caveat. `--shared-claims` takes the
>   intersection instead, for a pair that genuinely overlaps in time (v1/v2), and refuses an empty
>   one rather than sampling silently. Even a shared claim set would not make the φ vectors
>   directly comparable — the versions consume different feature matrices under different column
>   names, so the reading still goes through `features/feature_overlap.json`.
> - **Which configuration produced it.** The estimator's hyperparameters are captured in the meta, so
>   the notebook can print them beside the concentration figures — the standing requirement of
>   `problem.md` §1.4c (no adjacent version pair is configuration-matched, and L1 concentrates
>   importance by construction).

### One notebook, three environments (added 2026-08-01)

Everything above concerns *headless* Version-Layer work. There is a second, equally legitimate way to
use a version env: **`notebook/real/00_SHAP.ipynb` runs interactively inside it**, opens that
version's pickle and produces the whole single-version SHAP analysis — feature-space profile, input
distributions either side of the cutoff, global bar / beeswarm / magnitude beeswarm, dependence,
per-claim waterfall + force, and mean|φ| by score band.

It is **one file, run three times on three kernels**, not three notebooks. It names no version: it
detects which one it is from the interpreter path and reads the rest from `config`. Two constraints
follow from where it runs, and `src/shap_kit.py` exists to absorb them:

> - **The library floor is numpy + pandas + matplotlib.** The envs span **xgboost 0.72 / 1.4.2 /
>   3.2.0** — seven years — so `shap.plots.beeswarm` and the `Explanation` object cannot be assumed
>   to exist. Every figure is drawn from the raw φ matrix. `shap` is used for the *values* when
>   present, with the booster's own `pred_contribs` as the fallback.
> - **A wrong kernel must fail, not plot.** `config.VERSIONS[v]["xgboost"]` pins each version's
>   release and `shap_kit.env_report(strict=True)` raises on a mismatch — loading a pickle under the
>   wrong stack otherwise yields a complete set of plausible figures for the wrong model.
>
> It writes the *same* `detection/shap/<v>/<v>_attributions.parquet` + `_meta.json` as `attribute.py`, so the
> interactive and headless routes are interchangeable and `00_shap_attribution.ipynb` (the
> cross-version comparison) reads whichever produced them.

**Why `estimator/` mandates `assumptions()` and `falsify()`.** RDD and DiD are identified only under
assumptions that are, in general, **untestable** (parallel trends; continuity at the cutoff). P7's
discipline is: state them, test their observable implications, bound the damage when they fail. The
ABC encodes that discipline in the type system — **an estimator that has not run its falsification
gates cannot report a number.** (Notebook 04-01 gates the RDD on density / covariate continuity /
placebo cutoffs; 04-02 *measures* the parallel-trends bias with a version pair that carries no
corruption at either end.)

**Why `reeval/` is its own layer.** It is not a new kind of measurement — it **composes** the detector
and the estimators over two artefact sets and diffs them. It earns a layer because of one thing only:
metrics that **cannot exist for a single model**. `DecisionFlipCount` is the first citizen — "how many
cars change fate" is undefined unless you hold two models side by side. `OracleValidation` (AUC against
`true_garage_outcome`) also lives here, and its type says out loud what a comment currently only
whispers in `pipeline.cycle()`: **it is synthetic-only and cannot run on real Allianz data.**

**`ShapDiDDelta` — the sharpest re-eval metric, because it re-runs an `EffectEstimator` (added
2026-07-15).** The corruption footprint of § "Positivity is dead at τ" is measured *between baseline
model versions* (v2a → v3a). Nothing stops us measuring the **same DiD between the mitigated
versions** and differencing:

```
footprint_before = ShapDiDEstimator(v2a_baseline,  v3a_baseline )     ( the contaminated world )
footprint_after  = ShapDiDEstimator(v2a_mitigated, v3a_mitigated)     ( the de-contaminated world )
ShapDiDDelta     = footprint_before − footprint_after                 ( how much the corrector erased )
```

This is strictly stronger evidence than `DetectionDelta`. Δ`peak0` says a *score-space* symptom
eased; `ShapDiDDelta` says the corrector actually **collapsed the mechanism** — if mitigation worked,
`footprint_after` should fall toward zero, i.e. the mitigated models should look like models trained
in a world where the forcing never happened. It is the **observational twin of the notebook's positive
control**: 04-02 Phase 2 *injects then removes* corruption by hand and watches the DiD respond; here a
real corrector removes it and the same estimator watches. The tool is reused verbatim — this is exactly
what "`reeval/` composes the estimators over two artefact sets" means. Two cautions, both mechanical
consequences of what it is: (i) it is a DiD, so `falsify()` (the parallel-trends probe) **must re-run on
the mitigated pair** — the ABC makes that automatic; (ii) the mitigated models are trained on
corrected data whose labels above τ are **transported (g(x)), never verified**, so `footprint_after`'s partition B rests on extrapolation
— report the same "rows above τ" diagnostic beside it.

## τ has two sources, and only one of them is tuned

`src/threshold.py` holds three **pure functions** — the policy, in one place. It is *not* where τ is
computed; tuning still *executes* in the version env. Today the rule is duplicated in
`generate/model.py::_tune_threshold` and `export_thresholds.py`, and is **missing from `retrain.py`
altogether**.

```python
tune(y, scores, target=0.985, fallback=0.872)   # the rule: lowest cutoff whose precision ≥ target
apply(scores, tau)                              # decision = score ≥ τ          (universal, env-free)
read_off(scores, decisions)                     # τ as an OBSERVED FACT, from the log
```

| τ | what it is | how it is obtained | env needed |
|---|---|---|---|
| `detection/<v>_tau` (**production**) | **a thing that happened** | **`read_off`** — the boundary in the log. Never recomputed. | no |
| `reeval/<v>_tau` (**mitigated model**) | a model that never ran in production | **must be tuned** → `training/retrain.py`, inside env-vX | yes |

**Why `read_off` rather than "just tune it again".** (a) A production log ships `score` and `decision`
columns, not a τ. (b) A *documented* τ need not be the τ that produced the decisions — 04-01 already
found this (`applied_tau` vs `chosen_tau_oot`, and `applied_tau` is stored to 4 d.p., so it can sit a
rounding hair off the true boundary). (c) **And this is the real reason:** its validation step,
`max{score | decision=0} < min{score | decision=1}`, is a **one-line test of whether treatment
assignment is deterministic** — and that answer decides the entire mitigation strategy (below). Run it
first on real data.

**Why τ_mit must be tuned inside `retrain.py`, and not in the analysis layer.** Tuning needs an
out-of-sample **validation slice**; scoring the whole corrected set and tuning on *that* is in-sample,
gives optimistic precision, lands τ too low, and silently over-scraps. Only the trainer knows the
split. Moreover the split and the tuning rule should be **the version repo's own**, imported dynamically
from that repo — `threshold.tune` is the *fallback* when a repo exposes neither. This extends the existing **re-evaluation invariant** ("reuse the baseline's
fitted `prep` verbatim; only the label may differ") to the split and the cutoff rule.

> **@TODO** `train.py` / `retrain.py` currently make **no validation split at all** and tune **no τ**.
> Both are needed before `DecisionFlipCount` can run on the app path.

**Built 2026-07-31.** `threshold.py` exists and the three functions behave as specified. `apply()`
**refuses** rather than degrades: asked for v1 on a frame with no `mobility` column it raises, because
silently applying one cutoff would mislabel every immobile vehicle in the (0.75, 0.85] band. Same for
v2 without `date` — pooling the two regimes conflates two treatment assignments. `read_off()` returns
`deterministic` alongside τ, which is the one-line positivity test to run first on real data.
Notebooks should stop importing `SCRAP_THRESHOLD` (v2's pre-2026-07 cutoff only) and call
`load(v).tau` or `threshold.apply(v, df)`.

## Positivity is dead at τ — a constraint on the whole mitigation layer

Measured on the DGP (v2a, 2022+, 23,225 rows), and **this is a property of the data, not of any
corrector implementation**:

```
max score among GARAGED rows  = 0.9761
min score among SCRAPPED rows = 0.9762        ← the overlap is exactly zero
rows above τ:  124 total  |  garaged 0  |  scrapped 124
```

The routing rule is a **hard cutoff**, so P(scrap | x) ∈ {0, 1}: **positivity/overlap fails by
construction**, not merely in this sample. Any corrector that keeps only `decision == 0` rows is
therefore **structurally blind above τ** — no reweighting can represent a region with zero retained
rows. Consequences, which the architecture must respect:

- **Never tune τ on IPS-corrected data, and never IPS-weight precision *at* τ** — both are a literal
  0/0. (Weighted **AUC** survives, because it integrates over the whole score range. Build 00
  currently weights AUC but not precision — which is the wrong way round, since precision ≥ 0.985 is
  the business constraint. The fix is **not** "add weighted precision at τ".)
- **The standard overlap diagnostics do not notice.** With the current corrector recipe: max weight
  3.46, zero rows clipped, ESS = 99.7 % of kept rows — a clean bill of health while the estimator is
  undefined. They are population-level; the failure is local to the cutoff. **The diagnostic that
  matters is "how many retained rows lie above τ", and it is zero.**
- **This is why the rest of the thesis exists.** Deterministic assignment is precisely the case in
  which IPS and matching are not identified — and precisely the case RDD was invented for (P7).
  Identification above τ has to come from somewhere else: **continuity at the cutoff** (RDD, 04-01),
  **PU relabelling of the scrapped rows** (P28), or **deliberately injected randomness** (Build 05) —
  exploration is what *manufactures* the positivity IPS needs. **IPS alone cannot fix a
  deterministic-threshold SFP loop.**
- Consequently `DecisionFlipCount` reports **two arms**: **fixed-τ** (hold τ = τ_base — the only
  identified arm; "at the company's current cutoff, how many cars change fate once the model is
  de-contaminated?") and **re-tuned-τ**, which is *not* a second estimate but a **demonstration of the
  failure** — it reports the zero.

## What synthetic cannot rehearse

The DGP gives all versions the **same 86-column feature matrix**. Real v1/v2/v3 genuinely diverge in
preprocessing *and* in feature **set** (real v2's top feature is `location_Home`, absent elsewhere).
So the synthetic data **cannot rehearse the most dangerous real-data property**, and the estimators
must guard against it themselves rather than assume it away. Notebook 04-02 opens with
`assert` that the three feature matrices have identical columns — **that assertion is designed to fire
on real data**, and when it does, the response is to restrict to the intersection of the feature sets
or to freeze preprocessing, never to compare SHAP across differently-preprocessed pipelines.

## Data Flow

```
DataLoader
    │
    ▼
SFPDetector → DetectionReport
                    │
                    ▼ (if sfp_detected)
              SFPMitigator
                    │
          ┌───────expected methods──────┐
          ▼                              ▼
  claim investigation           dataset correction
```

## Scoring Flow (offline) → Analysis Flow

Scoring and analysis are decoupled — they never share a process. See `DESIGN.md` for full rationale (the runtime subprocess loader was superseded 2026-06-25). Paths below use the synthetic source; the real source has the identical `data/real/…` layout.

```
[ Data preparation — run once, upstream of scoring ]

  01_export_v<k>.ipynb   Z: raw + transformed pkl   → raw_dataset / processed_inputs / targets /
                         (inside env-v<k>)            scores, ONE FILE PER SPLIT — the sole
                                                      producer of all four (2026-08-18)
  01_export_v2_logs.ipynb  the live v2 log           → log_raw / log_features / log_scores /
                         (one frame, split by column)   log_targets (event grain)
                                                      + log (claim grain). Only v2 has a log.
  (no baseline-training step at all: all three versions declare paths.model, so the BASELINE is
   each version's own production pickle, loaded — never refitted. Both trainers that produced one
   were deleted 2026-08-19; retrain.py clones its hyperparameters off that pickle.)
                              ▼
[ Scoring — run once per version, each in its OWN env ]

  env-v1  python predict_v1.py --model models/synthetic/baseline/v1.pkl \
                            --features data/synthetic/inputs/features_v1.parquet → data/synthetic/detection/v1_scores.parquet
  env-v2  python predict.py --model models/synthetic/baseline/v2.pkl \
                            --features data/synthetic/inputs/features_v2.parquet → data/synthetic/detection/v2_scores.parquet
  env-v3  python predict.py --model models/synthetic/baseline/v3.pkl \
                            --features data/synthetic/inputs/features_v3.parquet → data/synthetic/detection/v3_scores.parquet
        (each via `uv run --project src/envs/<v>` or that env's own interpreter;
         same version-agnostic script; active env + per-version feature file decide the version)
                              │
                              ▼  detection/*_scores.parquet  (claim_id | model_vX_score)

[ Analysis — single env, NO models loaded ]

  load_scores({...})  →  merge on claim_id  →  SFPDetector
       claim_id | model_v1_score | model_v2_score | model_v3_score
```

### Re-evaluation loop (after mitigation)

The same decoupling drives the "did the mitigation work?" check. Only the **training label** changes between baseline and mitigated; the per-version **features (preprocessing) are held fixed**, so the before→after Δ is attributable to the mitigation (see `problem.md` §2.5 #12).

```
Analysis env   SFPMitigator  → data/synthetic/mitigation/v2_corrected.parquet          (de-contaminated target)
env-v2         retrain.py --features data/synthetic/inputs/features_v2.parquet \
                          --labels   data/synthetic/mitigation/v2_corrected.parquet → models/synthetic/mitigated/v2.pkl
env-v2         predict.py  --model   models/synthetic/mitigated/v2.pkl \
                          --features data/synthetic/inputs/features_v2.parquet   → data/synthetic/reeval/v2_mitigated_scores.parquet
Analysis env   SFPDetector(baseline_scores vs mitigated_scores) → SFP metric dropped?
```

## Environment Map

| Component | Runs in | Notes |
|---|---|---|
| `predict.py` scoring v2/v3 · `predict_v1.py` scoring v1 | `env-v{1,2,3}` (`src/envs/`) | Own uv env; isolated process; writes `detection/*_scores`. Pattern B since 2026-09-02 — the shared file is modern (≥3.10, importable `predict()`), the frozen py3.5 twin reads/writes CSV, see below |
| `training/retrain.py` re-train v2/v3 | `env-v{2,3}` (`src/envs/`) | Own uv env; re-evaluation step; modern (≥3.10, importable `retrain()`) since 2026-09-02 — v1 is never retrained; reuses baseline prep, fits weighted on corrected labels → `models/…/mitigated/v<k>.pkl` |
| pipeline, detector, mitigator, `load_scores.py`, the `*_all.py` drivers | analysis env (`.venv`) | `uv add`/`uv sync` → `pyproject.toml` + `uv.lock`; reads precomputed parquet, no model dependency |

### Three patterns: which interpreter a file runs under (added 2026-09-01)

The table above says *which tier*; this one says which of the four interpreters a given file can
actually be launched with, and — for a file that does not exist yet — which of three patterns to
write it in. Answer three questions in order:

1. **Does it open a model pickle?** No → **pattern C**, the analysis `.venv`. A pkl only unpickles
   where its repo is importable and its xgboost matches, so anything that opens one belongs to a
   version env; everything else belongs to the layer that must stay model-free.
2. **Must it run under v1?** No → write it in ordinary 3.7+ and it serves env-v2 and env-v3
   interchangeably (the right-hand column of **pattern B**). v1 gets a twin later, or never.
3. **It must run under v1 — do its dependencies survive the 3.5 floor?** Yes → **pattern A**, one
   file for all three. No → **pattern B**, a shared v2/v3 file plus a `<name>_v1.py` twin.

Question 3 is the real fork, and the answer is about *dependencies*, not effort. A scorer
needs pandas and a pickle, both of which exist under 3.5; `shap_kit.py` needs matplotlib's modern
plotting layer and shap's `Explanation`, neither of which does. Rewriting the latter to the floor
would mean a worse module for v2/v3 to buy a v1 that still could not draw the same figures.
But dependencies surviving the floor only justifies pattern A while question 2 answers "yes":
`predict.py` was pattern A for one day (2026-09-01) and was re-split to pattern B on 2026-09-02,
because its only v1 duty is baseline scoring — carrying the whole 3.5 discipline in the shared
file bought nothing the frozen twin doesn't provide.

**Pattern A — one file, any of `env-v1/v2/v3`.** Swap the interpreter, same file, same arguments.

| File | |
|---|---|
| `trained_order.py` · `model_io.py` | libraries the version-layer workers import; also imported by the v1 twins — the parts that MUST agree across versions stay one definition |

(`training/retrain.py` and `scoring/predict.py` left this category on 2026-09-02 — both are now
modern ≥3.10 with importable `retrain()`/`predict()` functions whose CLI mains wrap the same
call, and `notebook/real/mitigation/03_03_retrain.ipynb` imports them in-kernel instead of
subprocessing. retrain has no twin: v1 is never retrained.)

Four disciplines, none optional, and each one is a *parse* or *runtime* failure in env-v1 rather
than a style preference:

* **3.5 syntax** — no f-strings (3.6), no variable annotations (3.6), no
  `from __future__ import annotations` (3.7), no builtin generics `list[str]` (3.9), no
  `X | None` unions (3.10). env-v1 is Python 3.5.6 and cannot parse them. **Return annotations
  are not on that list** — function annotations predate 3.0 and `typing` shipped with 3.5, so
  every function here declares its return type like any other file in the repo.
* **ASCII-only** in anything printed or raised — that console is not UTF-8, and an em dash in a
  `SystemExit` message turns a clean error into a `UnicodeEncodeError`.
* **no `import config`** — `config.py` is 3.7+ itself, so importing it excludes env-v1
  transitively. Paths arrive as CLI arguments; a config-aware driver in the analysis env
  (`score_all.py`, `retrain_all.py`) resolves them by KIND.
* **I/O branches on the file extension** — env-v1 has no parquet engine (no pyarrow for 3.5), so
  v1 reads and writes CSV and `v1_csv_to_parquet.py` converts afterwards. `predict_v1.py`'s
  `read_table`/`write_table` are the pattern.

> `score_all.py` resolves the CSV twins via `config.worker_path` (which swaps `.parquet` → `.csv`
> for `config.NO_PARQUET_ENVS`) and dispatches env-v1 to `predict_v1.py`, so a v1 scoring run no
> longer needs hand-typed paths.

**Pattern B — shared v2/v3 file + a `<name>_v1.py` twin.** The v1 file is written last, against
the frozen shared one, and never retrofitted into it. Behaviour changes land in the shared file
FIRST, then are hand-ported to the twin in 3.5 syntax.

| Shared (env-v2 / env-v3) | v1 twin |
|---|---|
| `shap_kit.py` | `shap_kit_v1.py` |
| `features/extract_features.py` | `features/extract_features_v1.py` |
| `scoring/backfill_feature_order.py` | `scoring/backfill_feature_order_v1.py` |
| `scoring/predict.py` | `scoring/predict_v1.py` (frozen 2026-09-02 from the pattern-A body) |
| `training/retrain.py` | **none, by decision** — v1 is never retrained (training data destroyed; 03_03 refuses env-v1) |
| `scoring/attribute.py` | **none, by decision** — `notebook/real/00_SHAP_v1.ipynb` + `shap_kit_v1` already writes the same artefact (as CSV, backend `native`). `attribute_all.NO_HEADLESS_ATTRIBUTION` skips v1 with that instruction instead of failing inside env-v1 |

The twins are not free-standing copies: both import `trained_order` and `model_io`, so the parts
that MUST agree across versions (the `feature_order` vocabulary, the loader ladder) are one
definition, and only the genuinely env-bound half is duplicated.

**Pattern C — analysis `.venv` only.** Opens no model; most import `config`.

`config.py` · `schema.py` · `threshold.py` · `figstyle.py` · `detector/` · `estimator/` ·
`mitigator/` · `pipeline/` · `loaders/` · `scoring/load_scores.py` · the `*_all.py` drivers ·
`features/check_overlap.py` · `data/make_dummy_*.py`

`loaders/version_data.py` reads `.pkl` but is still pattern C: those are pandas *data* pickles
from `Z:`, not models. Opening a model there raises `ModuleNotFoundError: fttl_v1`.

#### Which pattern for NEW work (added 2026-09-01)

**Pattern A is a maintenance category, not a target.** The 3.5 floor exists for exactly one
environment — env-v1, Python 3.5.6, a 2018 stack that cannot be rebuilt. v2 and v3 are already
past it, and **every version after v3 arrives on Python ≥ 3.10**, since the attribution work from
here on is on post-v3 data. So for anything written from now on, question 2 above ("must it run
under v1?") answers **No** unless you are deliberately touching v1's own scoring or retraining
path.

That makes the default concrete:

| Writing a new file that… | Pattern | What you may use |
|---|---|---|
| opens no model | **C** | ordinary modern Python, `import config`, parquet — the analysis `.venv` |
| opens a model, v2/v3/v4+ only | **B**, shared half | ordinary modern Python; **no twin, no CSV branch, no ASCII restriction** |
| opens a model **and v1 must run it** | **A** | the four disciplines above — and say so in the file header, as `retrain.py` and `predict.py` do |

Only four files are in pattern A (`retrain.py`, `predict.py`, `trained_order.py`, `model_io.py`)
and that list should not grow. They are there because v1 still has to be **scored** and
**retrained** for the dissertation's cross-version comparison — not because new code should
imitate them. Writing a new module to the 3.5 floor "to be safe" costs f-strings, real type
syntax and parquet for a compatibility nobody will use.

**When v4 lands**, the two legacy carve-outs must stay one-element lists:

* `config.NO_PARQUET_ENVS` — envs with no parquet engine. v4 has one; do not add it.
* `scoring/attribute_all.NO_HEADLESS_ATTRIBUTION` — versions this driver cannot launch. v4 runs
  `attribute.py` unchanged; do not add it.

If a future version ever *did* need a twin, that is the signal its env is wrong, not that the
pattern should spread.

## Adding a New Model Version

When a new version (e.g., v4) arrives with different dependencies:

1. Create `src/envs/v4/` with its spec (`requirements.txt`, or `pyproject.toml` + `uv.lock`) and build the env (`uv sync` in that dir, or `uv venv` + `uv pip install -r`)
2. Add a `"v4"` entry to `config.VERSIONS` + `VERSION_LABELS` (its `paths`, its `columns` mapping) and its `SPLITS`, then write `notebook/real/01_export_v4.ipynb` (the pattern of `01_export_v{1,2,3}`) to emit `raw_dataset` / `processed_inputs` / `targets` / `scores`, one file per split
3. Declare `paths.model` — v4's own production pickle is the baseline, loaded not refitted. (If v4 ever ships none, a baseline trainer has to be written back; `config.TRAINING_CONFIG` records each version's training call.)
4. Nothing. `score_all.py` loops `config.VERSION_LABELS`, so registering v4 in config already added it
5. Add `"v4": ".../detection/v4_scores.parquet"` to the `score_paths` dict in the analysis

A v4 on Python ≥ 3.10 needs **no `_v1`-style twin and no CSV stage** — it is pattern B's shared half (or C), and neither `config.NO_PARQUET_ENVS` nor `attribute_all.NO_HEADLESS_ATTRIBUTION` should gain an entry. See § "Three patterns".

No new class required. `predict.py`, `attribute.py`, `retrain.py` and `load_scores.py` are all version-agnostic and run unchanged — the active env plus the CLI args (`--model`, `--features`, `--labels`, `--version`) decide which version is trained, scored, or retrained.

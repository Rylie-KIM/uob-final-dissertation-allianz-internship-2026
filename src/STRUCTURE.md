# App Structure

> **Two layers, not three apps (decided 2026-07-01).** The project is split **by concern**, not by model version. **Version Layer — per-version model worker** (`envs/v1│v2│v3`): preprocessing → (re)train → score for ONE version's artefact, each in its own isolated env / `pyproject.toml`. **Analysis Layer** (`pipeline/` + `detector/` + `mitigator/` + re-evaluation): a **single, version-agnostic** app, never triplicated — the SFP signal lives *between* versions and the comparison needs identical code. The layers never share a process (Version Layer writes `inputs/features_<v>.parquet` + `detection/<v>_scores.parquet`; Analysis Layer reads + merges on `claim_id`). Note Version Layer owns **(re)train + score**, not only score: after-mitigation retraining runs in the version's own env on the Analysis-Layer-corrected training set, holding that version's preprocessing fixed across the pre→post comparison. Full rationale + the re-evaluation invariant: `README.md` § "Application Architecture — Split by Concern (Two Layers), Not by Model Version" and `problem.md` §2.5 #12.

> **Directory reorganisation (decided 2026-07-03).** The old `src/data/scores/{features,labels}/` bag-of-artefacts and `src/model/envs/` were reorganised along two clean axes:
> - **`src/model/` → `src/envs/`.** That directory only ever held per-version `.venv`s (no models live there — the pkls are elsewhere), so the name `model/` was misleading. Renamed to `envs/`, which is what it is. Source-agnostic (an env is a library stack), so it is **not** split by synthetic/real.
> - **Data is split by *source* then *pipeline stage*:** `data/{synthetic,real}/{inputs,detection,mitigation,reeval}/`. `inputs/` holds the fixed *materials* (per-version `features_<v>` + `labels_<v>`); the three stage dirs hold pure *outputs* (`detection/` = baseline scores → detector; `mitigation/` = corrected targets ← mitigator; `reeval/` = post-retrain scores). ⚠️ `inputs/features_<v>` is the one file used by **both** scoring and retrain — it is a shared material, not a stage output, hence its own `inputs/` bucket. It is also the likely edit point if feature engineering is revisited.
> - **Model pkls leave `data/` entirely → `src/models/{synthetic,real}/{baseline,mitigated}/`.** A fitted `.pkl` is a model artefact, not data. `baseline/` = the version repo's code retrained on its **original (contaminated)** target = production reproduction; `mitigated/` = the same code retrained on the **corrector's de-contaminated** target. Folder carries the meaning → filename is just `v<k>.pkl`.
>
> **Why baseline pkls are ours to regenerate, not vendored.** Industry convention: code lives in git, **fitted model binaries do not** (they bloat history, don't diff, and are reproducible from code + data + pinned env — they belong in a model registry / object store). The version repos under `model_repos/` are therefore treated as **code only**; every pkl this project scores — baseline *and* mitigated — is **regenerated here** by retraining from that repo's code in its pinned env. Hence both land in `src/models/`, and `model_repos/` is never written to.

> **Legend:** `[now]` = exists on disk today · `[plan]` = target design, not yet created · `[move]` = exists on disk but under the *old* path, pending the reorg above. **The `src/` application code was scaffolded on 2026-07-01 and then deleted the same day** (it was premature); everything under the layer headings is therefore `[plan]` again. The design it describes still stands; only the code is gone.

```
repo/
├── pyproject.toml, uv.lock,
│   .python-version                     [now]  # ANALYSIS env — uv-managed `.venv` (py3.11); `uv sync` to build.
│                                              #   deps: numpy, pandas, pyarrow, scikit-learn, xgboost; grow via `uv add`.
│                                              #   (xgboost needs system libomp → `brew install libomp`)
│
├── model_repos/                        [gitignored]  # version model repos — external, CODE ONLY, NOT vendored in src/
│   │                                                 #   each ships a pickled Pipeline(preprocess+model)'s *code*; to
│   │                                                 #   load a pkl the repo's code must be importable in that version's
│   │                                                 #   env → installed as a git-dependency of env-vX (pinned SHA).
│   │                                                 #   NEVER written to by this project (no generated pkls land here).
│   ├── practice/fttl-v1/               [now]   # ✅ WORKING practice repo (rehearses clone→env→retrain→predict)
│   │   ├── pyproject.toml                      #   installable package spec
│   │   └── fttl_v1/{preprocess.py, train.py}   #   custom transformer + trainer → v1 pipeline
│   ├── practice/fttl-v2, fttl-v3       [now]   # ✅ WORKING SFP-chain practice repos: each trains on the PREV
│   │                                           #   version's emitted log (observed_outcome, w/ forced positives),
│   │                                           #   tunes τ to precision≥0.985, emits log_v{k} for the next link.
│   │                                           #   Per-version preprocessing differs (V2/V3FeatureBuilder) → each
│   │                                           #   pkl needs its OWN repo importable (isolation point, verified).
│   └── v1/ v2/ v3/                     [plan]  # REAL cloned version repos (arrive later)
│
└── src/
    ├── config.py                       [now]  # SCRAP_THRESHOLD, TARGET_PRECISION, FEATURE_COLS, CATEGORICAL_COLS
    ├── STRUCTURE.md / DESIGN.md / ENV_MANAGEMENT.md   [now]  # design docs (kept in sync via hook)
    │
    ├── pipeline/  detector/  mitigator/   [now: EMPTY dirs]   # placeholders; contents planned below
    │
    │   ─── everything below is [plan]/[move] (design + pending reorg — see legend) ───
    │
    │   ══ ANALYSIS LAYER — single shared, version-agnostic app (loads NO model) ══
    │   pipeline/pipeline.py            # SFPPipeline: detect → (if loop) mitigate → re-evaluate
    │   detector/sfp_detector.py        # SFPDetector — cross-version tests (correlation, scrap-rate drift)
    │   detector/algorithm/             # pluggable detection strategies
    │   mitigator/sfp_mitigator.py      # SFPMitigator — emits corrected training labels
    │   mitigator/{policy,corrector}/   # investigation policies · training-data correctors (IPW, …)
    │
    │   ══ SHARED training protocol — imported by BOTH real retrain + synthetic generator ══
    │   training/protocol.py            # train_version() — the Allianz methodology (SAME for every version)
    │   training/spec.py                # VersionTrainSpec + REGISTRY (per-version PARAMS: label/window/hyper/τ)
    │   training/{splits,threshold}.py  # maturation+OOT+80/20 · precision≥0.985 cutoff (port of generate/model.py)
    │
    │   ══ PER-VERSION preprocessing (strategy — genuinely differs, #10) ══
    │   preprocessing/base.py           # Preprocessor ABC (uniform output: claim_id + features)
    │   preprocessing/synthetic.py      # ONE shared recipe → identical features (held-fixed property)
    │   preprocessing/v{1,2,3}.py       # ADAPTER over that version's cloned-repo pipeline; NOT re-implemented (#7)
    │
    │   ══ VERSION LAYER — one isolated env per model version (runs the models) ══
    │   scoring/predict.py              # version-agnostic scorer   (runs INSIDE a version env)
    │   scoring/retrain.py              # version-agnostic retrainer → calls training/protocol.py
    │   scoring/run_all.sh              # orchestrates all versions, each in its own env
    │   envs/v{1,2,3}/                  [move]  # each version's isolated .venv (was src/model/envs/); pins that
    │                                           #   version's stack (git-dep on its model_repos/ repo). Source-agnostic.
    │
    │   ══ MODEL ARTEFACTS — all pkls out of data/; regenerated here, split source→purpose ══
    │   models/                         [plan]  # NEVER in data/. baseline = repo code on ORIGINAL target
    │   ├── synthetic/                          #   (production reproduction); mitigated = same code on the
    │   │   ├── baseline/  v{1,2,3}.pkl         #   corrector's de-contaminated target. Filename = version only.
    │   │   └── mitigated/ v{1,2,3}.pkl         #   (folder carries the meaning). pkls are gitignored.
    │   └── real/{baseline,mitigated}/         #   mirrors synthetic; populated when real version repos arrive.
    │
    │   ══ data-exchange contract — Analysis Layer readers (loaders + score merge) ══
    │   data/scores.py                  # load_scores() — merge per-version score files on claim_id
    │   data/loaders/{base,synthetic,real}.py   # DataLoader ABC + Synthetic/Real loaders
    │                                           #   (own subpackage so module names never collide with the
    │                                           #    data/synthetic/ + data/real/ DIRECTORIES below)
    │
    └── data/                           # ALL source-partitioned data + the synthetic generator live here
        ├── synthetic/                  # synthetic SOURCE = generator + its raw output + staged artefacts (ONE dir)
        │   ├── run.py  evaluate.py     [now]   # generator entry; export_version_features() emits per-version X
        │   ├── generate/               [now]   # base_features, enrichment, imputer, pre_ml, model, garage_outcome, …
        │   ├── script/                 [now]   # 01_schema.sql, 02_seed_data.sql, 03_pipeline_views.sql
        │   ├── csv/  parquet/          [now]   # raw generated datasets (parquet/features/ → migrates into inputs/)
        │   ├── inputs/                 [move]  # features_<v>.parquet (model input X) + labels_<v>.parquet
        │   │                                   #   (original target). Shared material for BOTH scoring & retrain.
        │   ├── detection/              [move]  # <v>_scores.parquet → SFPDetector (baseline scores)
        │   ├── mitigation/             [move]  # <v>_corrected.parquet (label+weight) ← SFPMitigator
        │   └── reeval/                 [move]  # <v>_mitigated_scores.parquet → before/after Δ
        └── real/                       [plan]  # real SOURCE — identical stage layout (arrives later)
            └── inputs/ · detection/ · mitigation/ · reeval/
```

> **Note — generator + staged data share `data/synthetic/`.** Source-first means the synthetic *generator* (`run.py`, `generate/`, `script/`, `csv/`, `parquet/`) and the *staged artefacts* (`inputs/…reeval/`) sit under the **same** `data/synthetic/` node — the generator is what makes this source. The four stage dirs mirror exactly under `data/real/` (which has no generator — real data arrives from Allianz). The Python loaders live in `data/loaders/` (a subpackage) so `synthetic`/`real` module names don't clash with the `data/synthetic/` and `data/real/` directories.

> **Reality vs target (updated 2026-07-03).** What exists on disk **today**: `src/config.py`, the three design docs, three **empty** package dirs (`pipeline/`, `detector/`, `mitigator/`), the whole `src/data/synthetic/` generation tree, the **working practice repos** `model_repos/practice/fttl-v{1,2,3}/`, the built per-version envs (currently `src/model/envs/v{1,2,3}/.venv`, pending rename to `src/envs/`), and practice-chain artefacts currently under `src/data/scores/{features,labels}/` + `*_scores.parquet` (pending the source→stage reorg) with mitigated pkls currently sitting inside the practice repos (pending the move to `src/models/synthetic/mitigated/`). Everything under the Analysis-Layer / training / preprocessing headings is **design-only** (scaffolded then removed 2026-07-01 as premature; written for real once the version repos / real data arrive). The repo-root `pyproject.toml` + `uv.lock` + `.python-version` are the **uv-managed analysis env** (`.venv`, py3.11 — `uv sync`; team standard is uv). `xgboost` needs system OpenMP (`brew install libomp`) to load. There is no `src/main.py`; the analysis entry point is written when the Analysis Layer code lands.

> **Per-version features, not one shared matrix.** Each version is scored on its own `inputs/features_<v>.parquet`. On **real** data each file is built by that version's own preprocessing (which diverged across v1/v2/v3 — see `problem.md` §2.5 #10/#11), so the files genuinely differ. On **synthetic** data a single DGP produces one matrix and `src/data/synthetic/run.py` (`export_version_features`) writes it once per version into `data/synthetic/inputs/features_<tag>.parquet` — identical by construction, preserving the "hold preprocessing fixed, vary only label/window" invariant. Both paths share the same per-version scoring contract. See `DESIGN.md` § "Per-version feature matrices".

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
[ Scoring — run once per version, each in its OWN env ]

  env-v1  python predict.py --model models/synthetic/baseline/v1.pkl \
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
| `predict.py` scoring v1/v2/v3 | `env-v{1,2,3}` (`src/envs/`) | Own uv env; isolated process; writes `detection/*_scores.parquet` |
| `retrain.py` re-train v1/v2/v3 | `env-v{1,2,3}` (`src/envs/`) | Own uv env; re-evaluation step; fits on corrected labels → `models/…/mitigated/v<k>.pkl` |
| `main.py`, pipeline, detector, mitigator, `scores.py` | analysis env (`.venv`) | `uv add`/`uv sync` → `pyproject.toml` + `uv.lock`; reads precomputed parquet, no model dependency |

> **Two environment tiers.** The analysis `.venv` is one evolving env managed by `uv add` (`pyproject.toml` + `uv.lock`). The per-version scoring envs (`env-v1`/`env-v2`/`env-v3`, under `src/envs/`) are frozen, independently pinned, and managed separately — never folded into the analysis `pyproject.toml`. See `ENV_MANAGEMENT.md`.

## Adding a New Model Version

When a new version (e.g., v4) arrives with different dependencies:

1. Create `src/envs/v4/` with its spec (`requirements.txt`, or `pyproject.toml` + `uv.lock`) and build the env (`uv sync` in that dir, or `uv venv` + `uv pip install -r`)
2. Provide `data/<source>/inputs/features_v4.parquet` (v4's own preprocessing on real data; on synthetic, register the version and `export_version_features` emits it)
3. Regenerate the baseline pkl into `models/<source>/baseline/v4.pkl` by retraining v4's repo code in `env-v4`
4. Add one `uv run --project src/envs/v4 python src/scoring/predict.py … --features data/<source>/inputs/features_v4.parquet --version v4` line to `run_all.sh`
5. Add `"v4": ".../detection/v4_scores.parquet"` to the `score_paths` dict in the analysis

No new class required. `predict.py`, `retrain.py`, and `scores.py` are all version-agnostic and run unchanged — the active env plus the CLI args (`--model`, `--features`, `--labels`, `--version`) decide which version is scored or retrained.

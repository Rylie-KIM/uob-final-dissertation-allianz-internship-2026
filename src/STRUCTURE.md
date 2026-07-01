# App Structure

> **Two layers, not three apps (decided 2026-07-01).** The project is split **by concern**, not by model version. **Version Layer — per-version model worker** (`model/envs/v1│v2│v3`): preprocessing → (re)train → score for ONE version's artefact, each in its own isolated env / `pyproject.toml`. **Analysis Layer** (`pipeline/` + `detector/` + `mitigator/` + re-evaluation): a **single, version-agnostic** app, never triplicated — the SFP signal lives *between* versions and the comparison needs identical code. The layers never share a process (Version Layer writes `features_<v>.parquet` + `*_scores.parquet`; Analysis Layer reads + merges on `claim_id`). Note Version Layer owns **(re)train + score**, not only score: after-mitigation retraining runs in the version's own env on the Analysis-Layer-corrected training set, holding that version's preprocessing fixed across the pre→post comparison. Full rationale + the re-evaluation invariant: `README.md` § "Application Architecture — Split by Concern (Two Layers), Not by Model Version" and `problem.md` §2.5 #12.

> **Legend:** `[now]` = fully implemented · `[stub]` = file scaffolded, skeleton only (raises `NotImplementedError` / TODO) · `[plan]` = not yet created. Scaffolded 2026-07-01; `predict.py` + `scores.py` + the detector helper methods are functional, everything else is a stub.

```
repo/
├── pyproject.toml + uv.lock        [now]  # ANALYSIS env (.venv): pandas, numpy, dowhy… (NO xgboost)
├── main.py                         [now]  # Analysis entry point (repo root — there is no src/main.py)
│
├── model_repos/                    [plan]  # GITIGNORED — version model repos live here, NOT in src/
│   │                                       #   each is a pickled Pipeline(preprocess+model); to load it
│   │                                       #   the repo's code must be importable in that version's env
│   │                                       #   → installed as a git-dependency of env-vX (pinned SHA), not a submodule
│   ├── v1/ v2/ v3/                 [plan]  # REAL: cloned version repos (clone → build env → predict_proba(raw))
│   └── practice/                   [plan]  # practice fakes to rehearse the clone→env→pkl→predict loop
│       └── fttl-v1/                        #   fttl-v1/{pyproject.toml, fttl_v1/{preprocess.py, train.py}}
│
└── src/
    ├── config.py                   [now]  # SCRAP_THRESHOLD, TARGET_PRECISION, FEATURE_COLS, CATEGORICAL_COLS
    ├── STRUCTURE.md / DESIGN.md / ENV_MANAGEMENT.md   [now]  # design docs (kept in sync via hook)
    │
    │   ══ ANALYSIS LAYER — single shared, version-agnostic app (loads NO model) ══
    ├── pipeline/
    │   └── pipeline.py             [stub]  # SFPPipeline: detect → (if loop) mitigate → re-evaluate
    ├── detector/
    │   ├── sfp_detector.py         [stub]  # SFPDetector — cross-version tests (corr/scrap-drift helpers work; run() TODO)
    │   └── algorithm/              [plan]  # pluggable detection strategies
    ├── mitigator/
    │   ├── sfp_mitigator.py        [stub]  # SFPMitigator — emits corrected training labels
    │   ├── policy/                 [plan]  # investigation policies
    │   └── corrector/              [plan]  # training-data correction strategies (IPW, …)
    │
    │   ══ SHARED training protocol — imported by BOTH real retrain + synthetic generator ══
    ├── training/
    │   ├── protocol.py             [stub]  # train_version() — the Allianz methodology (SAME for every version)
    │   ├── spec.py                 [stub]  # VersionTrainSpec + REGISTRY (per-version PARAMS: label/window/hyper/τ)
    │   ├── splits.py               [stub]  # maturation buffer + OOT + 80/20  (port of generate/model.py)
    │   └── threshold.py            [stub]  # tune_threshold(): precision ≥ 0.985 cutoff (port of generate/model.py)
    │
    │   ══ PER-VERSION preprocessing (strategy — genuinely differs, #10) ══
    ├── preprocessing/
    │   ├── base.py                 [stub]  # Preprocessor ABC  (uniform output: claim_id + features)
    │   ├── synthetic.py            [stub]  # ONE shared recipe → identical features (held-fixed property)
    │   ├── v1.py / v2.py / v3.py   [stub]  # ADAPTER over that version's cloned-repo pipeline (pipe[:-1].transform);
    │   │                                   #   NOT re-implemented — real preprocessing lives inside the pickle (#7)
    │   └── __init__.py             [stub]  # REGISTRY: {"synthetic","v1","v2","v3"}
    │
    │   ══ VERSION LAYER — one isolated env per model version (runs the models) ══
    ├── scoring/
    │   ├── predict.py              [now]   # version-agnostic scorer    (runs INSIDE a version env)
    │   ├── retrain.py              [stub]  # version-agnostic retrainer → calls training/protocol.py
    │   ├── preprocess.py           [stub]  # build a version's features_<v>.parquet → calls preprocessing/REGISTRY
    │   └── run_all.sh              [now]   # orchestrates all versions, each in its own env
    ├── model/
    │   └── envs/
    │       ├── v1/pyproject.toml   [stub]  # pins v1's stack (xgboost ver TBC) — uv.lock built later
    │       ├── v2/pyproject.toml   [stub]  # pins v2's stack
    │       └── v3/pyproject.toml   [stub]  # pins v3's stack
    │
    └── data/
        ├── base.py                 [stub]  # DataLoader abstract base class
        ├── synthetic.py            [plan]  # SyntheticDataLoader
        ├── real.py                 [plan]  # RealDataLoader
        ├── scores.py               [now]   # load_scores() — merge per-version score files on claim_id
        │
        ├── scores/                 [plan]  # ← the ONLY thing the two layers exchange (REAL data)
        │   ├── features/features_v{1,2,3}.parquet   [plan]  # per-version preprocessed X (held fixed)
        │   ├── labels/                              [plan]  # training targets: original + mitigation-corrected
        │   └── v{1,2,3}_scores.parquet              [plan]  # claim_id | model_vX_score
        │
        └── synthetic/              [now]   # EXISTING — synthetic data generation (unchanged)
            ├── run.py              [now]   # builds datasets; export_version_features() emits per-version X
            ├── evaluate.py         [now]
            ├── generate/           [now]   # base_features, enrichment, imputer, pre_ml, model, garage_outcome, …
            ├── script/            [now]    # 01_schema.sql, 02_seed_data.sql, 03_pipeline_views.sql
            ├── csv/                [now]   # generated datasets (CSV)
            └── parquet/            [now]   # generated datasets (Parquet)
                └── features/       [now]   # features_<tag>.parquet ← per-version feature files live HERE today
                                            #   (v1, v2a, v2b, v3a, v3b — identical by construction on synthetic)
```

> **Reality vs target.** Today only `config.py`, the three design docs, the three empty package dirs (`pipeline/`, `detector/`, `mitigator/`), and the whole `data/synthetic/` generation tree exist. `scoring/`, `model/envs/`, and the `data/*.py` loaders + `data/scores/` are the planned target. The per-version feature files already exist for **synthetic** data at `data/synthetic/parquet/features/features_<tag>.parquet` (written by `export_version_features`); the `data/scores/` path above is where the **real**-data Version-Layer scoring output will land. There is no `src/main.py` — the repo-root `main.py` is the only `main.py`.

> **Per-version features, not one shared matrix.** Each version is scored on its own `features_<v>.parquet`. On **real** data each file is built by that version's own preprocessing (which diverged across v1/v2/v3 — see `problem.md` §2.5 #10/#11), so the files genuinely differ. On **synthetic** data a single DGP produces one matrix and `src/data/synthetic/run.py` (`export_version_features`) writes it once per version under `src/data/synthetic/parquet/features/features_<tag>.parquet` — identical by construction, preserving the "hold preprocessing fixed, vary only label/window" invariant. Both paths share the same per-version scoring contract. See `DESIGN.md` § "Per-version feature matrices".

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

Scoring and analysis are decoupled — they never share a process. See `DESIGN.md` for full rationale (the runtime subprocess loader was superseded 2026-06-25).

```
[ Scoring — run once per version, each in its OWN env ]

  env-v1  python predict.py --model v1.pkl --features features_v1.parquet → v1_scores.parquet
  env-v2  python predict.py --model v2.pkl --features features_v2.parquet → v2_scores.parquet
  env-v3  python predict.py --model v3.pkl --features features_v3.parquet → v3_scores.parquet
        (each via `uv run --project src/model/envs/<v>` or that env's own interpreter;
         same version-agnostic script; active env + per-version feature file decide the version)
                              │
                              ▼  scores/*.parquet  (claim_id | model_vX_score)

[ Analysis — single env, NO models loaded ]

  load_scores({...})  →  merge on claim_id  →  SFPDetector
       claim_id | model_v1_score | model_v2_score | model_v3_score
```

### Re-evaluation loop (after mitigation)

The same decoupling drives the "did the mitigation work?" check. Only the **training label** changes between baseline and mitigated; the per-version **features (preprocessing) are held fixed**, so the before→after Δ is attributable to the mitigation (see `problem.md` §2.5 #12).

```
Analysis env   SFPMitigator  → data/scores/labels/v2_corrected.parquet     (de-contaminated target)
env-v2         retrain.py --features features_v2.parquet --labels v2_corrected.parquet → v2_mitigated.pkl
env-v2         predict.py  --model v2_mitigated.pkl  --features features_v2.parquet → v2_mitigated_scores.parquet
Analysis env   SFPDetector(baseline_scores vs v2_mitigated_scores) → SFP metric dropped?
```

## Environment Map

| Component | Runs in | Notes |
|---|---|---|
| `predict.py` scoring v1/v2/v3 | `env-v{1,2,3}` | Own uv env; isolated process; writes `*_scores.parquet` |
| `retrain.py` re-train v1/v2/v3 | `env-v{1,2,3}` | Own uv env; re-evaluation step; fits on corrected labels → new `.pkl` |
| `main.py`, pipeline, detector, mitigator, `scores.py` | analysis env (`.venv`) | `uv add`/`uv sync` → `pyproject.toml` + `uv.lock`; reads precomputed parquet, no model dependency |

> **Two environment tiers.** The analysis `.venv` is one evolving env managed by `uv add` (`pyproject.toml` + `uv.lock`). The per-version scoring envs (`env-v1`/`env-v2`/`env-v3`) are frozen, independently pinned, and managed separately — never folded into the analysis `pyproject.toml`. See `ENV_MANAGEMENT.md`.

## Adding a New Model Version

When a new version (e.g., v4) arrives with different dependencies:

1. Create `src/model/envs/v4/` with its spec (`requirements.txt`, or `pyproject.toml` + `uv.lock`) and build the env (`uv sync` in that dir, or `uv venv` + `uv pip install -r`)
2. Provide `features_v4.parquet` (v4's own preprocessing on real data; on synthetic, register the version and `export_version_features` emits it)
3. Add one `uv run --project src/model/envs/v4 python src/scoring/predict.py … --features features_v4.parquet --version v4` line to `run_all.sh`
4. Add `"v4": ".../v4_scores.parquet"` to the `score_paths` dict in the analysis

No new class required. `predict.py`, `retrain.py`, and `scores.py` are all version-agnostic and run unchanged — the active env plus the CLI args (`--model`, `--features`, `--labels`, `--version`) decide which version is scored or retrained.

# App Structure



```
repo/
├── pyproject.toml · uv.lock · .python-version
│
├── model_repos/
│   ├── practice/
│   │   ├── fttl-v1/
│   │   ├── fttl-v2/
│   │   └── fttl-v3/
│   └── v1/ · v2/ · v3/
│
└── src/
    ├── config.py
    ├── docs/STRUCTURE.md · DESIGN.md · ENV_MANAGEMENT.md
    │
    ├── pipeline/
    │   └── pipeline.py
    ├── detector/
    │   ├── sfp_detector.py
    │   └── algorithm/
    ├── mitigator/
    │   ├── sfp_mitigator.py
    │   ├── corrector/
    │   └── policy/
    ├── loaders/
    │   └── base.py · synthetic.py · real.py
    │
    ├── scoring/
    │   ├── ingest.py          # source log (manifest) → logs/<v>.parquet
    │   ├── build_inputs.py    # logs/<v>.parquet → inputs/{features,labels}_<v>
    │   ├── predict.py
    │   ├── score_all.sh
    │   └── load_scores.py
    ├── preprocessing/
    │   ├── base.py
    │   └── v1.py · v2.py · v3.py
    ├── training/
    │   ├── train.py           # BASELINE trainer (prep+model fresh on production label)
    │   ├── retrain.py         # MITIGATED retrainer (reuse baseline prep, weighted)
    │   ├── train_all.sh
    │   └── spec.py
    ├── envs/
    │   └── v1/ · v2/ · v3/
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
        ├── synthetic/
        │   ├── run.py · evaluate.py
        │   ├── generate/      # ← the ONLY data synthetic TESTS read (DGP output)
        │   │   └── csv/ · parquet/   (+ parquet/features/)
        │   ├── script/
        │   ├── logs/          #  ┐ staged-artefact dirs: NOT a test-data source —
        │   ├── inputs/        #  │ a synthetic dry-run of the real-data pipeline,
        │   ├── detection/     #  │ built to lock the app structure BEFORE real
        │   ├── mitigation/    #  │ Allianz resources are wired in
        │   └── reeval/        #  ┘ (mirror exactly under data/real/)
        └── real/
            ├── logs/
            ├── inputs/
            ├── detection/
            ├── mitigation/
            └── reeval/
```




> **Two layers, not three apps (decided 2026-07-01).** The project is split **by concern**, not by model version. **Version Layer — per-version model worker** (`envs/v1│v2│v3`): preprocessing → (re)train → score for ONE version's artefact, each in its own isolated env / `pyproject.toml`. **Analysis Layer** (`pipeline/` + `detector/` + `mitigator/` + re-evaluation): a **single, version-agnostic** app, never triplicated — the SFP signal lives *between* versions and the comparison needs identical code. The layers never share a process (Version Layer writes `inputs/features_<v>.parquet` + `detection/<v>_scores.parquet`; Analysis Layer reads + merges on `claim_id`). Note Version Layer owns **(re)train + score**, not only score: after-mitigation retraining runs in the version's own env on the Analysis-Layer-corrected training set, holding that version's preprocessing fixed across the pre→post comparison. Full rationale + the re-evaluation invariant: `README.md` § "Application Architecture — Split by Concern (Two Layers), Not by Model Version" and `problem.md` §2.5 #12.

> **Directory reorganisation (decided 2026-07-03).** The old `src/data/scores/{features,labels}/` bag-of-artefacts and `src/model/envs/` were reorganised along two clean axes:
> - **`src/model/` → `src/envs/`.** That directory only ever held per-version `.venv`s (no models live there — the pkls are elsewhere), so the name `model/` was misleading. Renamed to `envs/`, which is what it is. Source-agnostic (an env is a library stack), so it is **not** split by synthetic/real.
> - **Data is split by *source* then *pipeline stage*:** `data/{synthetic,real}/{inputs,detection,mitigation,reeval}/`. `inputs/` holds the fixed *materials* (per-version `features_<v>` + `labels_<v>`); the three stage dirs hold pure *outputs* (`detection/` = baseline scores → detector; `mitigation/` = corrected targets ← mitigator; `reeval/` = post-retrain scores). ⚠️ `inputs/features_<v>` is the one file used by **both** scoring and retrain — it is a shared material, not a stage output, hence its own `inputs/` bucket. It is also the likely edit point if feature engineering is revisited.
> - **Model pkls leave `data/` entirely → `src/models/{synthetic,real}/{baseline,mitigated}/`.** A fitted `.pkl` is a model artefact, not data. `baseline/` = the version repo's code retrained on its **original (contaminated)** target = production reproduction; `mitigated/` = the same code retrained on the **corrector's de-contaminated** target. Folder carries the meaning → filename is just `v<k>.pkl`.
>
> **Why baseline pkls are ours to regenerate, not vendored.** Industry convention: code lives in git, **fitted model binaries do not** (they bloat history, don't diff, and are reproducible from code + data + pinned env — they belong in a model registry / object store). The version repos under `model_repos/` are therefore treated as **code only**; every pkl this project scores — baseline *and* mitigated — is **regenerated here** by retraining from that repo's code in its pinned env. Hence both land in `src/models/`, and `model_repos/` is never written to.

> **Legend:** `[now]` = exists on disk today · `[plan]` = target design, not yet created · `[move]` = exists on disk but under the *old* path, pending the reorg above. **The `src/` application code was scaffolded on 2026-07-01 and then deleted the same day** (it was premature); everything under the layer headings is therefore `[plan]` again. The design it describes still stands; only the code is gone.

> **Note — `data/` is data-only; code lives in the layer packages.** Source-first means the synthetic *generator* (`run.py`, `generate/` — which holds the DGP output `csv/` + `parquet/` — and `script/`) and the *staged artefacts* (`logs/`, `inputs/…reeval/`) sit under the **same** `data/synthetic/` node — the generator is what makes this source. The stage dirs (`logs/` + the four `inputs/…reeval/`) mirror exactly under `data/real/` (which has no generator — real data arrives from Allianz). Analysis/pipeline **code never lives in `data/`**: log-ingestion (`ingest.py`), the log→inputs split (`build_inputs.py`), and score-ingestion (`load_scores.py`) live under `scoring/`, and the planned `DataLoader` classes under `loaders/` — keeping `data/` purely for stored artefacts.

> **Note — synthetic tests read ONLY `data/synthetic/generate/` (added 2026-07-06).** Every test / experiment / notebook that runs on synthetic data sources its data **exclusively** from `data/synthetic/generate/` — the DGP output produced by `run.py`: `generate/csv/`, `generate/parquet/`, and `generate/parquet/features/`. Nothing else counts as a synthetic test input. The other `data/synthetic/` subtrees — `logs/`, `inputs/`, `detection/`, `mitigation/`, `reeval/` — are **not test data**: they are a synthetic **dry-run of the real-data pipeline**, materialised only to pin down and validate the application structure (the source→stage layout, ingest/build/score/retrain wiring) **before** any real Allianz resource is connected. They exist so the app skeleton is finalised against synthetic stand-ins first; once real data arrives it flows through the identical `data/real/{logs,inputs,detection,mitigation,reeval}/` tree. Treat them as scaffolding for the real-data integration, never as a dataset the synthetic tests consume.

> **Note — log ingestion is the one name-translation point (added 2026-07-06).** Each model application emits its production log under whatever name IT chose (`log_v2.parquet` here; on real data e.g. `v2_prod_scored_2025Q1.parquet`). This pipeline expects one canonical per-version name, `logs/<v>.parquet`. `scoring/ingest.py` is the SINGLE place that translates "their name → our name": it reads the source path from `data/<source>/logs/manifest.json`, validates the required schema (`claim_id`, `observed_outcome`, `true_garage_outcome`, `model_<v>_decision`), and archives a copy as `logs/<v>.parquet`. Everything downstream (`build_inputs.py`) reads only the canonical `logs/`, so when a file name changes you edit `manifest.json` alone — no pipeline code changes.

> **Reality vs target (OO layers built 2026-07-04).** The reorg **and** the Analysis-Layer class hierarchy are **done and verified** — `src/pipeline/pipeline.py` (`SFPPipeline`) runs the full synthetic chain end-to-end (detect → mitigate → re-eval for v1/v2/v3), identical results to the retired procedural `run_cycle.py`. On disk and working: `src/config.py`; the design docs under `src/docs/`; the per-version envs at **`src/envs/v{1,2,3}/`**; the source→stage data tree **`src/data/synthetic/{inputs,detection,mitigation,reeval}/`**; all pkls at **`src/models/synthetic/{baseline,mitigated}/v{1,2,3}.pkl`**; the **Analysis-Layer OO impl** — `pipeline/pipeline.py` (`SFPPipeline`), `detector/sfp_detector.py` (`SFPDetector`) + `detector/algorithm/` (`DetectionAlgorithm` ABC → `ResidualPeakAlgorithm`), `mitigator/sfp_mitigator.py` (`SFPMitigator`) + `mitigator/corrector/` (`TrainingDataCorrector` ABC → `IPSCorrector`) + `mitigator/policy/` (`InvestigationPolicy` ABC); the scoring I/O (`scoring/{ingest,build_inputs,predict}.py`, `score_all.sh`, `load_scores.py`); the **training I/O** (`training/train.py` = baseline trainer, `training/retrain.py` = mitigated retrainer, `train_all.sh`); the canonical **log-ingestion landing zone** `data/synthetic/logs/` (`manifest.json` + `<v>.parquet`); the `src/data/synthetic/` generator tree; and the **working practice repos** `model_repos/practice/fttl-v{1,2,3}/` (code-only — pkls moved out). Still **design-only**: `preprocessing/`, `training/spec.py`, `loaders/`, concrete `InvestigationPolicy` impls, and the whole `data/real/` + `models/real/` side (arrive with the real version repos). The repo-root `pyproject.toml` + `uv.lock` + `.python-version` are the **uv-managed analysis env** (`.venv`, py3.11 — `uv sync`). `xgboost` needs system OpenMP (`brew install libomp`). The entry point is `src/pipeline/pipeline.py`; there is no `src/main.py`.

> **Per-version features, built through each version's own repo (both sources).** Each version is scored on its own `inputs/features_<v>.parquet`, produced by **that version's repo preprocessing** — the `preprocessing/v{1,2,3}.py` adapters run each repo's feature builder. This is now **identical for synthetic and real** (decided 2026-07-03): synthetic is only a temporary stand-in, so it goes through the *same* external-repo path, not a special shared recipe. The single difference is where the raw claims come from — synthetic **generates** them (`data/synthetic/` DGP), real **receives** them from Allianz. Because v1/v2/v3 preprocessing genuinely diverges (`V2/V3FeatureBuilder`; `problem.md` §2.5 #10/#11), `features_<v>` files differ across versions on **both** sources. See `DESIGN.md` § "Per-version feature matrices".

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

  ingest.py        source log (logs/manifest.json)  → data/synthetic/logs/<v>.parquet        (canonical name)
  build_inputs.py  logs/<v>.parquet split           → data/synthetic/inputs/{features,labels}_<v>.parquet
  train_all.sh     features_<v> + production label   → models/synthetic/baseline/<v>.pkl      (BASELINE, each in env-v)
                              │  (train.py fits prep+model FRESH; baseline pkls are gitignored → regenerated here)
                              ▼
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
| `training/train.py` baseline v1/v2/v3 | `env-v{1,2,3}` (`src/envs/`) | Own uv env; fits prep+model FRESH on the production label → `models/…/baseline/v<k>.pkl` |
| `predict.py` scoring v1/v2/v3 | `env-v{1,2,3}` (`src/envs/`) | Own uv env; isolated process; writes `detection/*_scores.parquet` |
| `training/retrain.py` re-train v1/v2/v3 | `env-v{1,2,3}` (`src/envs/`) | Own uv env; re-evaluation step; reuses baseline prep, fits weighted on corrected labels → `models/…/mitigated/v<k>.pkl` |
| `ingest.py`, `build_inputs.py`, pipeline, detector, mitigator, `load_scores.py` | analysis env (`.venv`) | `uv add`/`uv sync` → `pyproject.toml` + `uv.lock`; reads precomputed parquet, no model dependency |

> **Two environment tiers.** The analysis `.venv` is one evolving env managed by `uv add` (`pyproject.toml` + `uv.lock`). The per-version scoring envs (`env-v1`/`env-v2`/`env-v3`, under `src/envs/`) are frozen, independently pinned, and managed separately — never folded into the analysis `pyproject.toml`. See `ENV_MANAGEMENT.md`.

## Adding a New Model Version

When a new version (e.g., v4) arrives with different dependencies:

1. Create `src/envs/v4/` with its spec (`requirements.txt`, or `pyproject.toml` + `uv.lock`) and build the env (`uv sync` in that dir, or `uv venv` + `uv pip install -r`)
2. Add `"v4": "<path to v4's emitted log>"` to `data/<source>/logs/manifest.json`, then `python src/scoring/ingest.py` (→ `logs/v4.parquet`) and `python src/scoring/build_inputs.py` (→ `inputs/{features,labels}_v4.parquet`)
3. Regenerate the baseline pkl into `models/<source>/baseline/v4.pkl` by running `train.py` in `env-v4` (add a v4 line to `train_all.sh`)
4. Add one `uv run --project src/envs/v4 python src/scoring/predict.py … --features data/<source>/inputs/features_v4.parquet --version v4` line to `score_all.sh`
5. Add `"v4": ".../detection/v4_scores.parquet"` to the `score_paths` dict in the analysis

No new class required. `ingest.py`, `build_inputs.py`, `train.py`, `predict.py`, `retrain.py`, and `load_scores.py` are all version-agnostic and run unchanged — the active env plus the CLI args (`--model`, `--features`, `--labels`, `--version`) decide which version is trained, scored, or retrained.

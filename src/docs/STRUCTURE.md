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
    │   ├── predict.py
    │   ├── retrain.py
    │   ├── score_all.sh
    │   ├── build_inputs.py
    │   └── load_scores.py
    ├── preprocessing/
    │   ├── base.py
    │   └── v1.py · v2.py · v3.py
    ├── training/
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
        │   ├── generate/
        │   ├── script/
        │   ├── csv/ · parquet/
        │   ├── inputs/
        │   ├── detection/
        │   ├── mitigation/
        │   └── reeval/
        └── real/
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

> **Note — `data/` is data-only; code lives in the layer packages.** Source-first means the synthetic *generator* (`run.py`, `generate/`, `script/`, `csv/`, `parquet/`) and the *staged artefacts* (`inputs/…reeval/`) sit under the **same** `data/synthetic/` node — the generator is what makes this source. The four stage dirs mirror exactly under `data/real/` (which has no generator — real data arrives from Allianz). Analysis/pipeline **code never lives in `data/`**: score-ingestion (`load_scores.py`) and log-bridging (`build_inputs.py`) live under `scoring/`, and the planned `DataLoader` classes under `loaders/` — keeping `data/` purely for stored artefacts.

> **Reality vs target (OO layers built 2026-07-04).** The reorg **and** the Analysis-Layer class hierarchy are **done and verified** — `src/pipeline/pipeline.py` (`SFPPipeline`) runs the full synthetic chain end-to-end (detect → mitigate → re-eval for v1/v2/v3), identical results to the retired procedural `run_cycle.py`. On disk and working: `src/config.py`; the design docs under `src/docs/`; the per-version envs at **`src/envs/v{1,2,3}/`**; the source→stage data tree **`src/data/synthetic/{inputs,detection,mitigation,reeval}/`**; all pkls at **`src/models/synthetic/{baseline,mitigated}/v{1,2,3}.pkl`**; the **Analysis-Layer OO impl** — `pipeline/pipeline.py` (`SFPPipeline`), `detector/sfp_detector.py` (`SFPDetector`) + `detector/algorithm/` (`DetectionAlgorithm` ABC → `ResidualPeakAlgorithm`), `mitigator/sfp_mitigator.py` (`SFPMitigator`) + `mitigator/corrector/` (`TrainingDataCorrector` ABC → `IPSCorrector`) + `mitigator/policy/` (`InvestigationPolicy` ABC); the scoring I/O (`scoring/{predict,retrain}.py`, `score_all.sh`, `build_inputs.py`, `load_scores.py`); the `src/data/synthetic/` generator tree; and the **working practice repos** `model_repos/practice/fttl-v{1,2,3}/` (code-only — pkls moved out). Still **design-only**: `preprocessing/`, `training/`, `loaders/`, concrete `InvestigationPolicy` impls, and the whole `data/real/` + `models/real/` side (arrive with the real version repos). The repo-root `pyproject.toml` + `uv.lock` + `.python-version` are the **uv-managed analysis env** (`.venv`, py3.11 — `uv sync`). `xgboost` needs system OpenMP (`brew install libomp`). The entry point is `src/pipeline/pipeline.py`; there is no `src/main.py`.

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
| `main.py`, pipeline, detector, mitigator, `load_scores.py` | analysis env (`.venv`) | `uv add`/`uv sync` → `pyproject.toml` + `uv.lock`; reads precomputed parquet, no model dependency |

> **Two environment tiers.** The analysis `.venv` is one evolving env managed by `uv add` (`pyproject.toml` + `uv.lock`). The per-version scoring envs (`env-v1`/`env-v2`/`env-v3`, under `src/envs/`) are frozen, independently pinned, and managed separately — never folded into the analysis `pyproject.toml`. See `ENV_MANAGEMENT.md`.

## Adding a New Model Version

When a new version (e.g., v4) arrives with different dependencies:

1. Create `src/envs/v4/` with its spec (`requirements.txt`, or `pyproject.toml` + `uv.lock`) and build the env (`uv sync` in that dir, or `uv venv` + `uv pip install -r`)
2. Provide `data/<source>/inputs/features_v4.parquet` (v4's own preprocessing on real data; on synthetic, register the version and `export_version_features` emits it)
3. Regenerate the baseline pkl into `models/<source>/baseline/v4.pkl` by retraining v4's repo code in `env-v4`
4. Add one `uv run --project src/envs/v4 python src/scoring/predict.py … --features data/<source>/inputs/features_v4.parquet --version v4` line to `score_all.sh`
5. Add `"v4": ".../detection/v4_scores.parquet"` to the `score_paths` dict in the analysis

No new class required. `predict.py`, `retrain.py`, and `load_scores.py` are all version-agnostic and run unchanged — the active env plus the CLI args (`--model`, `--features`, `--labels`, `--version`) decide which version is scored or retrained.

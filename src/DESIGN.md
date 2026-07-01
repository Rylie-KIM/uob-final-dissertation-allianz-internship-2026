# Design Pattern

> **Implementation status (2026-07-01).** This document describes the **target** design. Of the components below, only `config.py`, the design docs, the empty `pipeline/` `detector/` `mitigator/` package dirs, and the `data/synthetic/` generation tree exist today; `scoring/predict.py`, `scoring/run_all.sh`, and `model/envs/{v1,v2,v3}/` are **not yet created**. On synthetic data the per-version feature files already exist at `data/synthetic/parquet/features/features_<tag>.parquet`; the `data/scores/` paths in `run_all.sh` below are the real-data target. See `STRUCTURE.md` (exists-vs-planned legend) for the authoritative status. The two layers named there map onto this doc as: **Version Layer** = the per-version scoring/(re)train envs; **Analysis Layer** = the pipeline/detector/mitigator that loads no model.

## Strategy Pattern

Each interchangeable component is encapsulated as a separate class behind a common interface. The core classes (`SFPDetector`, `SFPMitigator`) hold references to these strategies and delegate work to them — they never contain the implementation directly.

This means swapping synthetic data for real data, or swapping one detection algorithm for another, requires changing only the injected class — not the core logic.

## Three Strategy Axes

| Axis | Interface | Implementations |
|---|---|---|
| Data loading | `DataLoader` | `SyntheticDataLoader` → `RealDataLoader` |
| Score ingestion | precomputed parquet | one score file per model version, merged on `claim_id` |
| Detection | `DetectionAlgorithm` | TBD (pending research) |
| Investigation policy | `InvestigationPolicy` | TBD (pending research) |
| Training data correction | `TrainingDataCorrector` | TBD (pending research) |

## Class Responsibilities

**`SFPPipeline`** — orchestrates the full run. Calls detector, checks result, calls mitigator if SFP is detected.

**`SFPDetector`** — diagnosis. Loads data and runs detection algorithms. Returns a `DetectionReport`.

**`SFPMitigator`** — prescription. Takes the report and applies mitigation: updates investigation policy and corrects training data.

**`DataLoader`** — abstract base for data ingestion. Concrete implementations differ; callers do not.

**Score ingestion** — the pipeline does **not** load models or run inference. Each model version is scored offline, once, inside its own environment (see below); the resulting per-version score files are merged on `claim_id` and consumed by the detector. This keeps the analysis runtime free of any model dependency.

---

## Model Scoring & Environment Isolation

### The problem

All three model versions (v1, v2, v3) are preserved as serialised files within Allianz's internal systems. However, they **do not share a single set of library dependencies** (exact versions TBC — likely differing XGBoost or scikit-learn releases; v1 clearly diverges, and v2/v3 are not assumed identical either). Loading two versions in the same Python process will cause import conflicts. The project therefore gives **each version its own independently pinned environment** — `env-v1`, `env-v2`, `env-v3`, all three managed separately — so upgrading or retraining one version can never silently mutate another's.

Model versions will also continue to be updated over time, so the design must absorb a new version (v4, v5, …) without code changes or new classes.

### Design decision — offline precompute, not runtime subprocess

> **Decided 2026-06-25 (supervisor meeting).** Running each model under its own environment to produce predictions once, offline, is faster and more efficient than having the analysis app spawn a subprocess per call at runtime. The previously documented runtime `SubprocessModelLoader` design is **superseded** and retained only as design history at the end of this file.

**Scoring and analysis are fully decoupled.**

1. **Scoring stage** — each model version is scored **once**, inside its **own** environment (`env-v1` / `env-v2` / `env-v3`, built with uv — see `ENV_MANAGEMENT.md`), by a single version-agnostic script (`src/scoring/predict.py`). The active environment *is* that version's environment, so there is never a cross-version import conflict in a single process. Output: one parquet score file per version.
2. **Analysis stage** — the pipeline (detector/mitigator) runs in one environment and **never loads a model**. It reads the precomputed per-version score files and merges them on `claim_id`.

Because the two stages never share a process, there is no parent/child coordination, no temp-file marshalling, and no stdout parsing. Scores are cached on disk, so re-running the analysis any number of times does not re-score anything.

### Scoring flow

```
[ Scoring stage — run once per version, each in its own env ]

  uv env: env-v1                          src/scoring/predict.py
  ┌───────────────────────────┐           --model    models/v1.pkl
  │ joblib.load("v1.pkl")     │  ───────▶ --features  features_v1.parquet   ← per-version
  │ predict_proba(X)[:,1]     │           --version   v1
  │ → parquet                 │           --out       scores/v1_scores.parquet
  └───────────────────────────┘

  uv env: env-v2                          (same script, different env + args)
  ┌───────────────────────────┐           --model models/v2.pkl --features features_v2.parquet → v2_scores.parquet
  │ joblib.load("v2.pkl")     │  ───────▶
  └───────────────────────────┘

  uv env: env-v3                          (same script, different env + args)
  ┌───────────────────────────┐           --model models/v3.pkl --features features_v3.parquet → v3_scores.parquet
  │ joblib.load("v3.pkl")     │  ───────▶
  └───────────────────────────┘

                    │  v1_scores.parquet   claim_id | model_v1_score
                    │  v2_scores.parquet   claim_id | model_v2_score
                    ▼  v3_scores.parquet   claim_id | model_v3_score

[ Analysis stage — single env, no models loaded ]

  src/data/scores.py  →  merge on claim_id  →  SFPDetector
       claim_id | model_v1_score | model_v2_score | model_v3_score
```

### Per-version feature matrices (`features_<version>.parquet`)

Each version is scored on its **own** model-ready feature file — `features_v1.parquet`, `features_v2.parquet`, … (`claim_id` + preprocessed features $X$) — never a single shared `features.parquet`. This is the canonical contract for **both** the synthetic and the real data; only *how the files get populated* differs:

| | How the per-version files are produced | Are they identical across versions? |
|---|---|---|
| **Synthetic** | One DGP + one fitted imputer builds a single matrix; `run.py` writes it out once per version (`export_version_features`) | **Yes — identical by construction.** Preprocessing is held fixed and only the training label/window varies. This is the "clean separation" the SFP analysis depends on (`problem.md` §2.5 #10). |
| **Real** | Each version's own preprocessing pipeline rebuilds $X$ from the raw claim | **No — genuinely different.** Each production version re-implemented preprocessing separately, and the FastAPI serving path may preprocess differently from the AML training path within a version (`problem.md` §2.5 #10/#11). |

**Why per-version rather than one shared file.** On the real data, scoring all versions on a single feature matrix would **not reproduce the production scores** and would fold a preprocessing artefact into any cross-version score-drift comparison, masquerading as SFP signal. The per-version contract removes that confound by construction: each model is always scored on the features *it* would actually see. The synthetic path adopts the same contract so that the analysis code, the scoring scripts, and the `RealDataLoader` are structurally identical across both — the only thing that changes is whether the files happen to be equal.

**Producing the files.**
- *Synthetic:* `src/data/synthetic/run.py` → `export_version_features(df, X_all)` writes `parquet/features/features_<tag>.parquet` for every version (`v1, v2a, v2b, v3a, v3b`). Identical by construction; the redundancy is deliberate and documents the invariant.
- *Real:* each version's preprocessing emits its own `features_<version>.parquet` — feasible only if that version's preprocessing code/artefact can be re-run (cf. environment-isolation constraint, `problem.md` §2.5 #7). Where a version's preprocessing cannot be reconstructed, that version is limited to **symptom tracking** (its own emitted scores/decisions) and excluded from the leakage-free quantitative comparison (`problem.md` §2.5 #9).

### Implementation

**`src/scoring/predict.py`** — version-agnostic batch scorer. Run *inside* the target model's own environment; the active env and the CLI args decide which version is scored. The script has no knowledge of which version it is running.

```python
# src/scoring/predict.py
import argparse
import joblib
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--version",  required=True)   # label only, e.g. "v1"
    p.add_argument("--out",      required=True)
    p.add_argument("--id-col",   default="claim_id")
    args = p.parse_args()

    df    = pd.read_parquet(args.features)
    model = joblib.load(args.model)

    X = df.drop(columns=[args.id_col])
    scores = model.predict_proba(X)[:, 1]          # P(total_loss)

    out = pd.DataFrame({
        args.id_col: df[args.id_col],
        f"model_{args.version}_score": scores,
    })
    out.to_parquet(args.out, index=False)
    print(f"[{args.version}] wrote {len(out)} scores → {args.out}")

if __name__ == "__main__":
    main()
```

**`src/scoring/retrain.py`** — version-agnostic **retrainer** for the re-evaluation step, run *inside* the target version's own env exactly like `predict.py`. It fits a fresh model on that version's **fixed** feature file plus a supplied **label** file. The label is the *only* thing that differs between the baseline run (original contaminated target) and the post-mitigation run (the corrector's de-contaminated target); preprocessing is held fixed, so the before→after score difference is attributable to the mitigation, not to a preprocessing change (the re-evaluation invariant, `problem.md` §2.5 #12).

```python
# src/scoring/retrain.py  (runs inside env-vX)
import argparse, joblib, pandas as pd
from xgboost import XGBClassifier

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features",  required=True)   # that version's preprocessed X (held fixed)
    p.add_argument("--labels",    required=True)   # original OR mitigation-corrected target
    p.add_argument("--version",   required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--id-col",    default="claim_id")
    p.add_argument("--label-col", default="label")
    a = p.parse_args()

    X  = pd.read_parquet(a.features)
    y  = pd.read_parquet(a.labels)
    df = X.merge(y, on=a.id_col)
    feats = [c for c in X.columns if c != a.id_col]

    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    model.fit(df[feats], df[a.label_col])
    joblib.dump(model, a.out_model)     # or model.save_model("*.json") for cross-version portability
    print(f"[{a.version}] retrained on {len(df)} rows → {a.out_model}")

if __name__ == "__main__":
    main()
```

> **Re-evaluation loop.** `SFPMitigator` (analysis env) writes `data/scores/labels/<v>_corrected.parquet` → `retrain.py` (version env) fits `<v>_mitigated.pkl` → `predict.py` (version env) emits `<v>_mitigated_scores.parquet` → `SFPDetector` (analysis env) compares baseline vs mitigated. Same file-only decoupling as scoring; no model is ever loaded in the analysis env.

### Where per-version code lives — training (shared) vs preprocessing (per-version)

Two version-specific concerns with **opposite** structures — the single most important rule when adding code:

- **Training methodology → SHARED.** Every version used the *same* Allianz protocol (2-month maturation exclusion → 6-month OOT → 80/20 fit/val → fit → tune τ to precision ≥ 0.985). It lives **once** in `src/training/protocol.py`; the only per-version differences (label column, training window, hyperparameters, τ fallback) are **data** in `src/training/spec.py::VersionTrainSpec`. `retrain.py` *and* the synthetic generator (`generate/model.py`) both call this one protocol, so they can never drift. Adding a version = adding a spec entry, not new code.
- **Preprocessing → PER-VERSION.** On real data each version *re-implemented* preprocessing differently (`problem.md` §2.5 #10), so this is genuinely divergent code behind one interface: `src/preprocessing/base.py::Preprocessor` + `v1.py`/`v2.py`/`v3.py`, with a **uniform output contract** (`claim_id` + features → `features_<v>.parquet`). Synthetic uses ONE shared impl (`synthetic.py`) → identical features by construction. A version whose real preprocessing cannot be reconstructed (§2.5 #7) has no impl and is limited to symptom tracking (§2.5 #9).

> **Rule of thumb:** *same* procedure across versions → shared module + per-version config; *genuinely different* procedure → strategy implementation per version. Training is the former; preprocessing is the latter. Both still **execute in the Version Layer** (the version's own env), because both feed / run the model.

**`src/scoring/run_all.sh`** — orchestrates all versions, each in its **own** uv env, so each scoring run is a separate process (import conflicts are structurally impossible). v1, v2 and v3 each have their own environment. Below uses the **stricter** per-version layout (`uv run --project`); with the standard layout call each env's interpreter directly (`src/model/envs/v1/.venv/bin/python …`). See `ENV_MANAGEMENT.md` for both.

```bash
#!/usr/bin/env bash
# Score every model version, each in its own env, on its OWN feature file.
set -euo pipefail

FEATDIR="src/data/scores/features"   # per-version: features_v1.parquet, features_v2.parquet, …
OUTDIR="src/data/scores"
ENVDIR="src/model/envs"
mkdir -p "$OUTDIR"

uv run --project "$ENVDIR/v1" python src/scoring/predict.py \
    --model models/v1.pkl --features "$FEATDIR/features_v1.parquet" --version v1 \
    --out "$OUTDIR/v1_scores.parquet"

uv run --project "$ENVDIR/v2" python src/scoring/predict.py \
    --model models/v2.pkl --features "$FEATDIR/features_v2.parquet" --version v2 \
    --out "$OUTDIR/v2_scores.parquet"

uv run --project "$ENVDIR/v3" python src/scoring/predict.py \
    --model models/v3.pkl --features "$FEATDIR/features_v3.parquet" --version v3 \
    --out "$OUTDIR/v3_scores.parquet"

echo "All versions scored → $OUTDIR"
```

> Each `--features` path points to that version's own feature file. On synthetic data those files are identical (produced by `export_version_features`); on real data each is built by its version's own preprocessing.

**`src/data/scores.py`** — the analysis-side loader. Reads the precomputed files and merges on `claim_id`. No model dependency, no environment awareness.

```python
# src/data/scores.py
import pandas as pd
from functools import reduce

def load_scores(score_paths: dict[str, str], id_col: str = "claim_id") -> pd.DataFrame:
    """Merge precomputed per-version score files on claim_id.
    score_paths = {"v1": ".../v1_scores.parquet", "v2": ..., "v3": ...}
    """
    frames = [pd.read_parquet(p) for p in score_paths.values()]
    return reduce(lambda l, r: l.merge(r, on=id_col, how="outer"), frames)
```

### env spec files (still required, for reproducibility)

Each version keeps its **own** spec under `src/model/envs/<version>/`, used to *build* the env — not to locate an interpreter at runtime. **v1, v2 and v3 are each a separate environment** (no shared `env-v2v3`). There are two ways to pin each one (see `ENV_MANAGEMENT.md` for full detail):

- **Standard** — a pinned `requirements.txt` (pins with `==`), built via `uv venv` + `uv pip install -r`. Direct deps only, no lockfile.
- **Stricter** — a per-version `pyproject.toml` + `uv.lock`, built via `uv sync`. Captures transitive deps + hashes → byte-for-byte reproducible. Preferred for dissertation-grade reproducibility.

```
src/model/envs/
├── v1/   requirements.txt   (and/or pyproject.toml + uv.lock)
├── v2/   requirements.txt   (and/or pyproject.toml + uv.lock)
└── v3/   requirements.txt   (and/or pyproject.toml + uv.lock)
```

Each version retains an independent spec even when dependencies currently coincide, so retraining or upgrading one version never silently mutates another's pinned environment.

### Adding a new model version (e.g., v4)

1. Create `src/model/envs/v4/` with its spec (`requirements.txt`, or `pyproject.toml` + `uv.lock` for the stricter option) and build the env (`uv sync` in that dir, or `uv venv` + `uv pip install -r`).
2. Produce `features_v4.parquet` (v4's own preprocessing on real data; on synthetic, `export_version_features` already emits it once the version is registered).
3. Add one scoring line for v4 to `run_all.sh` (`uv run --project src/model/envs/v4 python src/scoring/predict.py … --features features_v4.parquet --version v4 --out scores/v4_scores.parquet`).
4. Add `"v4": ".../v4_scores.parquet"` to the `score_paths` dict in the analysis.

No new class. `predict.py` and `scores.py` are unchanged — they are version-agnostic.

### Trade-offs

| Approach | Pro | Con |
|---|---|---|
| **Offline precompute (chosen)** | No runtime process-spawn or marshalling overhead; scores cached on disk and reused; each env runs natively; scoring runs debuggable in isolation; analysis code has zero model dependency | Scores are a build artifact — features must be re-scored if they change (fine here: fixed feature set, cross-version comparison) |
| Runtime subprocess `ModelLoader` | Scoring is transparent to the caller (`predict_proba` looks local) | Per-call spawn + temp-file + stdout-parse overhead; parent/child coupling; superseded (see Design History) |
| Docker containers | Full isolation; fully reproducible | Heavyweight for a research app; requires Docker daemon |
| Env-switching via `importlib` | No subprocess | Unreliable; can corrupt `sys.modules` |

This suits the research workload: a fixed feature set is scored across versions for comparison, not scored in real time.

---

## Design History — Superseded: runtime subprocess `ModelLoader`

> **Status: superseded 2026-06-25** by the offline-precompute design above. Preserved here as a record of the design path and the rationale for the change — not current. The text below describes what *was* proposed.

The original plan exposed a `ModelLoader` abstraction with two implementations: `InProcessModelLoader` (v2/v3, same env as the app) and `SubprocessModelLoader(model_path, env_spec)` (v1, different env). The environment was passed as a parameter (`env_spec`, a YAML file under `src/model/envs/` holding `python_executable`), not encoded in a subclass, so new versions needed no new class.

At call time, `SubprocessModelLoader.predict_proba(X)` would: serialise `X` to a temp `.npy` file → spawn a child process with v1's interpreter (`subprocess.run([...])`) running a version-agnostic `worker.py` → the worker loaded the model, ran `predict_proba`, and printed the scores as JSON to **stdout** → the parent captured stdout (`capture_output=True`), parsed the JSON back into a NumPy array, and deleted the temp file. The caller (`SFPDetector`) saw a uniform `predict_proba` and never knew whether a subprocess was used.

**Why it was dropped:** for this workload v1 is scored a handful of times per run, always over a fixed feature set. Paying a process-spawn + temp-file-write + stdout-parse cost on *every* call — and coupling the analysis runtime to model loading — buys nothing that pre-scoring once to disk does not. The meeting feedback was that running each model under its own env to emit predictions, then analysing the saved predictions, is both faster and simpler. The `base.py` / `inprocess.py` / `subprocess_loader.py` / `worker.py` files were therefore never created.

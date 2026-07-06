# Design Pattern

> **Implementation status (2026-07-04, Analysis-Layer OO built).** This document describes the **target** design; the current synthetic chain runs it end-to-end. Exists and working today: `config.py`, the design docs (under `src/docs/`), `scoring/{ingest,build_inputs,predict}.py` + `score_all.sh` + `load_scores.py`, `training/{train,retrain}.py` + `train_all.sh`, the canonical log landing zone `data/synthetic/logs/` (`manifest.json` + `<v>.parquet`), the per-version envs at `src/envs/{v1,v2,v3}/`, the **Analysis-Layer OO impl** — `pipeline/pipeline.py` (`SFPPipeline`), `detector/sfp_detector.py` (`SFPDetector`) + `detector/algorithm/` (`DetectionAlgorithm` ABC → `ResidualPeakAlgorithm`), `mitigator/sfp_mitigator.py` (`SFPMitigator`) + `mitigator/corrector/` (`TrainingDataCorrector` ABC → `IPSCorrector`) + `mitigator/policy/` (`InvestigationPolicy` ABC) — and the source→stage data tree `data/synthetic/{inputs,detection,mitigation,reeval}/` with all pkls under `src/models/synthetic/{baseline,mitigated}/`. Still design-only: `preprocessing/`, `training/spec.py`, `loaders/`, concrete `InvestigationPolicy` impls, and the whole `data/real/`+`models/real/` side. See `STRUCTURE.md` for the authoritative status. The two layers named there map onto this doc as: **Version Layer** = the per-version scoring/(re)train envs; **Analysis Layer** = the pipeline/detector/mitigator that loads no model.

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

All three model versions (v1, v2, v3) are preserved as serialised files within Allianz's internal systems. **Every model version has a genuinely different, mutually incompatible library environment** — this is confirmed, not a "likely". Each version's pickle is bound to the **exact** third-party library versions it was serialised with (its own XGBoost, scikit-learn, and numpy releases), and those versions **differ across v1, v2, and v3** and cannot coexist in a single Python process. The project therefore gives **each version its own independently pinned environment** — `env-v1`, `env-v2`, `env-v3`, all three managed separately — so upgrading or retraining one version can never silently mutate another's.

#### Why "just install every version's repo into one env" does **not** work

This is the single most misunderstood point, so it is stated explicitly. The blocker is **not** the repos' own source code — that installs fine. The blocker is the **third-party numeric stack** each pickle is bound to.

- **A pickle stores class *references* + fitted *state*, never source code.** `joblib.load("v1.pkl")` reconstructs the pipeline by *importing* both (a) the version repo's custom transformer classes **and** (b) the exact library classes the pipeline was built from (`xgboost.sklearn.XGBClassifier`, `sklearn.pipeline.Pipeline`, numpy dtypes). Every one of those imports must resolve **at a compatible version**, or the load raises (`AttributeError` / `InconsistentVersionWarning` / a booster-format error). Having the repo code present is **necessary but not sufficient** — the exact numeric stack must match the pickle's provenance too.
- **A Python environment is a *flat* library pool: one version of each library, period.** Installing packages does **not** isolate their dependencies. If you `uv pip install -e model_repos/{v1,v2,v3}` into one `.venv`, pip must resolve `xgboost` (and `scikit-learn`, `numpy`) to a **single** version shared by all three. When v1 needs (say) `xgboost==1.7` and v3 needs `xgboost==2.1`, that is physically unsatisfiable — pip either errors on resolution, or installs one and the *other* version's pickle then fails to load at runtime.
- **This is unlike npm.** Node's nested `node_modules` lets two packages carry different versions of the same dependency; Python's flat `site-packages` cannot. So `pip install`-ing the repos gives you their code, **not** dependency isolation. The **only** thing that provides isolation is physically separating the environments.

| Approach | Repo code present? | Different lib versions coexist? | Works when versions incompatible? |
|---|---|---|---|
| One `.venv`, install all three repos as packages | ✅ yes | ❌ no (flat `site-packages`, one xgboost/sklearn) | ❌ **no** |
| One `.venv`, `sys.path` swap per version | ✅ yes | ❌ no (still one installed numeric stack) | ❌ **no** |
| **Separate `env-v1`/`env-v2`/`env-v3`** | ✅ yes | ✅ yes (each env its own stack) | ✅ **yes** |

The first two rows only work in the special case where v1/v2/v3 happen to share a compatible numeric stack — which for this project they **do not**. Hence the per-version environment split is mandatory, not a tidiness choice.

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
  ┌───────────────────────────┐           --model    models/synthetic/baseline/v1.pkl
  │ joblib.load(v1 baseline)  │  ───────▶ --features  data/synthetic/inputs/features_v1.parquet   ← per-version
  │ predict_proba(X)[:,1]     │           --version   v1
  │ → parquet                 │           --out       data/synthetic/detection/v1_scores.parquet
  └───────────────────────────┘

  uv env: env-v2                          (same script, different env + args)
  ┌───────────────────────────┐           --model models/synthetic/baseline/v2.pkl
  │ joblib.load(v2 baseline)  │  ───────▶ --features data/synthetic/inputs/features_v2.parquet → detection/v2_scores.parquet
  └───────────────────────────┘

  uv env: env-v3                          (same script, different env + args)
  ┌───────────────────────────┐           --model models/synthetic/baseline/v3.pkl
  │ joblib.load(v3 baseline)  │  ───────▶ --features data/synthetic/inputs/features_v3.parquet → detection/v3_scores.parquet
  └───────────────────────────┘

                    │  v1_scores.parquet   claim_id | model_v1_score
                    │  v2_scores.parquet   claim_id | model_v2_score
                    ▼  v3_scores.parquet   claim_id | model_v3_score

[ Analysis stage — single env, no models loaded ]

  src/scoring/load_scores.py  →  merge on claim_id  →  SFPDetector
       claim_id | model_v1_score | model_v2_score | model_v3_score
```

### Per-version feature matrices (`features_<version>.parquet`)

Each version is scored on its **own** model-ready feature file — `features_v1.parquet`, `features_v2.parquet`, … (`claim_id` + preprocessed features $X$) — never a single shared `features.parquet`. Each file is built by **that version's own repo preprocessing** (the `preprocessing/v{1,2,3}.py` adapters run each repo's feature builder). This is the canonical contract for **both** the synthetic and the real data; **the mechanism is identical** — only the *source of the raw claims* differs:

| | How the per-version files are produced | Are they identical across versions? |
|---|---|---|
| **Synthetic** | The DGP (`data/synthetic/`) generates raw claims; each version's **repo preprocessing** (`V1/V2/V3FeatureBuilder`) then rebuilds $X$ | **No — genuinely different.** Synthetic is only a temporary stand-in and runs the *same* external-repo path as real (decided 2026-07-03), so per-version preprocessing diverges here too. |
| **Real** | Allianz supplies raw claims; each version's own preprocessing pipeline rebuilds $X$ | **No — genuinely different.** Each production version re-implemented preprocessing separately, and the FastAPI serving path may preprocess differently from the AML training path within a version (`problem.md` §2.5 #10/#11). |

**Why per-version rather than one shared file.** On the real data, scoring all versions on a single feature matrix would **not reproduce the production scores** and would fold a preprocessing artefact into any cross-version score-drift comparison, masquerading as SFP signal. The per-version contract removes that confound by construction: each model is always scored on the features *it* would actually see. Synthetic adopts the **same external-repo path** (not a special shared recipe) so the analysis code, the scoring scripts, and the loaders are structurally identical across both — synthetic exists only because the real dataset is not yet available.

**Producing the files.**
- *Synthetic:* `src/data/synthetic/run.py` generates the raw claims; each version's **repo preprocessing** then emits its own `features_<tag>.parquet` (`v1, v2a, v2b, v3a, v3b`) into `data/synthetic/inputs/` — the *same* per-version external-repo path as real, since synthetic is only a stand-in for the unavailable real data.
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

**`src/training/{train,retrain}.py`** — two version-agnostic trainers, both run *inside* the target version's env exactly like `predict.py`. They are **split by purpose** so baseline and mitigated never share code paths:

- **`train.py` (BASELINE)** — fits a **fresh** `Pipeline(prep + model)` from raw features on the **production** label (`observed_outcome`) → `models/<source>/baseline/v<k>.pkl`. This regenerates the production-reproduction pkl (baseline pkls are gitignored). Nothing is inherited; `prep` is fit here.
- **`retrain.py` (MITIGATED)** — **reuses the baseline pkl's already-fitted `prep`** verbatim (never re-fits it) and fits only a new **weighted** model on the corrector's de-contaminated `--labels` → `models/<source>/mitigated/v<k>.pkl`.

Holding `prep` (and the features) fixed across baseline→mitigated means the only thing that changes is the **label/weight**, so the before→after score difference is attributable to the mitigation (the re-evaluation invariant, `problem.md` §2.5 #12). On real/practice data the fit can instead delegate to the repo's own trainer rather than re-implementing it; the sketch below illustrates the swappable-label contract.

```python
# src/training/retrain.py  (runs inside env-vX)
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

> **Re-evaluation loop.** `SFPMitigator` (analysis env) writes `data/synthetic/mitigation/<v>_corrected.parquet` → `retrain.py` (version env) fits `models/synthetic/mitigated/<v>.pkl` → `predict.py` (version env) emits `data/synthetic/reeval/<v>_mitigated_scores.parquet` → `SFPDetector` (analysis env) compares baseline vs mitigated. Same file-only decoupling as scoring; no model is ever loaded in the analysis env. (Real-data source uses the identical `data/real/…` + `models/real/…` layout.)

### Where per-version code lives — repo-owned logic, `src/` adapters

Both concerns ultimately **execute the version's own repo code** — our `src/` holds thin **adapters**, not re-implementations. This is the single most important rule when adding code:

- **Preprocessing → PER-VERSION adapter.** Each version *re-implemented* preprocessing differently (`problem.md` §2.5 #10), so this is genuinely divergent code owned by the repo. `src/preprocessing/{v1,v2,v3}.py` are **adapters** that call each repo's feature builder behind one interface (`base.py::Preprocessor`) with a **uniform output contract** (`claim_id` + features → `features_<v>.parquet`). Runs in that version's env. Synthetic uses the **same** per-version repo path (decided 2026-07-03) — no special shared recipe. A version whose preprocessing cannot be reconstructed (§2.5 #7) has no adapter and is limited to symptom tracking (§2.5 #9).
- **Training → repo-owned protocol, split baseline/mitigated trainers.** Every version used the *same* Allianz methodology (2-month maturation exclusion → 6-month OOT → 80/20 fit/val → fit → tune τ to precision ≥ 0.985) — implemented in each version's own repo `train.py`, not re-implemented in `src/`. Two version-agnostic trainers sit in `src/training/`: **`train.py`** produces the baseline pkl (fresh `prep`+model on the *original/production* target = production reproduction), and **`retrain.py`** produces the mitigated pkl (reusing the baseline's fixed `prep`, weighted on the *corrector's* target). Features + `prep` are held fixed across the two, so the only difference is the label/weight (the re-evaluation invariant, §2.5 #12). `src/training/spec.py` carries just the per-version **params** (label column, training window, hyperparameters, τ fallback). Adding a version = a spec entry + pointing at its repo, not new training code.

> **Rule of thumb:** the *actual* preprocessing and training logic lives in the per-version repos; `src/` provides **uniform adapters** (`preprocessing/v{1,2,3}.py`, `training/{train,retrain}.py`) so the Analysis Layer sees one contract regardless of version. All **execute in the Version Layer** (the version's own env), because both feed / build the model.

**`src/scoring/score_all.sh`** — orchestrates all versions, each in its **own** uv env, so each scoring run is a separate process (import conflicts are structurally impossible). v1, v2 and v3 each have their own environment. Below uses the **stricter** per-version layout (`uv run --project`); with the standard layout call each env's interpreter directly (`src/envs/v1/.venv/bin/python …`). See `ENV_MANAGEMENT.md` for both. `SOURCE` selects the data source (`synthetic` or `real`) — the whole tree mirrors under `data/<source>/` + `models/<source>/`.

```bash
#!/usr/bin/env bash
# Score every model version, each in its own env, on its OWN feature file.
set -euo pipefail

SOURCE="${1:-synthetic}"                        # synthetic | real
FEATDIR="src/data/$SOURCE/inputs"               # per-version: features_v1.parquet, features_v2.parquet, …
MODELDIR="src/models/$SOURCE/baseline"          # baseline pkls (production reproduction)
OUTDIR="src/data/$SOURCE/detection"             # baseline scores → detector
ENVDIR="src/envs"
mkdir -p "$OUTDIR"

uv run --project "$ENVDIR/v1" python src/scoring/predict.py \
    --model "$MODELDIR/v1.pkl" --features "$FEATDIR/features_v1.parquet" --version v1 \
    --out "$OUTDIR/v1_scores.parquet"

uv run --project "$ENVDIR/v2" python src/scoring/predict.py \
    --model "$MODELDIR/v2.pkl" --features "$FEATDIR/features_v2.parquet" --version v2 \
    --out "$OUTDIR/v2_scores.parquet"

uv run --project "$ENVDIR/v3" python src/scoring/predict.py \
    --model "$MODELDIR/v3.pkl" --features "$FEATDIR/features_v3.parquet" --version v3 \
    --out "$OUTDIR/v3_scores.parquet"

echo "All versions scored → $OUTDIR"
```

> Each `--features` path points to that version's own feature file, built by that version's own repo preprocessing — on **both** synthetic and real (2026-07-03 decision: synthetic runs the same external-repo path, it is only a temporary stand-in).

**`src/scoring/load_scores.py`** — the analysis-side loader. Reads the precomputed files and merges on `claim_id`. No model dependency, no environment awareness.

```python
# src/scoring/load_scores.py
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

Each version keeps its **own** spec under `src/envs/<version>/`, used to *build* the env — not to locate an interpreter at runtime. **v1, v2 and v3 are each a separate environment** (no shared `env-v2v3`). These envs are **source-agnostic** — an env is a library stack, so it is *not* duplicated per synthetic/real. There are two ways to pin each one (see `ENV_MANAGEMENT.md` for full detail):

- **Standard** — a pinned `requirements.txt` (pins with `==`), built via `uv venv` + `uv pip install -r`. Direct deps only, no lockfile.
- **Stricter** — a per-version `pyproject.toml` + `uv.lock`, built via `uv sync`. Captures transitive deps + hashes → byte-for-byte reproducible. Preferred for dissertation-grade reproducibility.

```
src/envs/
├── v1/   requirements.txt   (and/or pyproject.toml + uv.lock)
├── v2/   requirements.txt   (and/or pyproject.toml + uv.lock)
└── v3/   requirements.txt   (and/or pyproject.toml + uv.lock)
```

Each version retains an independent spec even when dependencies currently coincide, so retraining or upgrading one version never silently mutates another's pinned environment.

### Adding a new model version (e.g., v4)

1. Create `src/envs/v4/` with its spec (`requirements.txt`, or `pyproject.toml` + `uv.lock` for the stricter option) and build the env (`uv sync` in that dir, or `uv venv` + `uv pip install -r`).
2. Register v4's emitted log in `data/<source>/logs/manifest.json`, then `ingest.py` → `logs/v4.parquet` and `build_inputs.py` → `inputs/{features,labels}_v4.parquet` (on synthetic, `export_version_features` also emits features once the version is registered). Regenerate `models/<source>/baseline/v4.pkl` by running `train.py` in `env-v4` (add a v4 line to `train_all.sh`).
3. Add one scoring line for v4 to `score_all.sh` (`uv run --project src/envs/v4 python src/scoring/predict.py … --features data/<source>/inputs/features_v4.parquet --version v4 --out data/<source>/detection/v4_scores.parquet`).
4. Add `"v4": ".../detection/v4_scores.parquet"` to the `score_paths` dict in the analysis.

No new class. `predict.py` and `load_scores.py` are unchanged — they are version-agnostic.

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

The original plan exposed a `ModelLoader` abstraction with two implementations: `InProcessModelLoader` (v2/v3, same env as the app) and `SubprocessModelLoader(model_path, env_spec)` (v1, different env). The environment was passed as a parameter (`env_spec`, a YAML file under `src/envs/` holding `python_executable`), not encoded in a subclass, so new versions needed no new class.

At call time, `SubprocessModelLoader.predict_proba(X)` would: serialise `X` to a temp `.npy` file → spawn a child process with v1's interpreter (`subprocess.run([...])`) running a version-agnostic `worker.py` → the worker loaded the model, ran `predict_proba`, and printed the scores as JSON to **stdout** → the parent captured stdout (`capture_output=True`), parsed the JSON back into a NumPy array, and deleted the temp file. The caller (`SFPDetector`) saw a uniform `predict_proba` and never knew whether a subprocess was used.

**Why it was dropped:** for this workload v1 is scored a handful of times per run, always over a fixed feature set. Paying a process-spawn + temp-file-write + stdout-parse cost on *every* call — and coupling the analysis runtime to model loading — buys nothing that pre-scoring once to disk does not. The meeting feedback was that running each model under its own env to emit predictions, then analysing the saved predictions, is both faster and simpler. The `base.py` / `inprocess.py` / `subprocess_loader.py` / `worker.py` files were therefore never created.

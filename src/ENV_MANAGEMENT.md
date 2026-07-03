# Environment Management in Data Science

> **Implementation status (2026-07-01, paths updated 2026-07-03).** The two-tier env model below is the **target**. The analysis `.venv` exists; the per-version scoring envs live under `src/envs/{v1,v2,v3}/` (currently built at `src/model/envs/…`, pending the 2026-07-03 rename to `src/envs/`). These map to the layer names in `STRUCTURE.md`/`DESIGN.md`: the per-version scoring envs are the **Version Layer**, the analysis `.venv` is the **Analysis Layer**. Envs are **source-agnostic** (an env is a library stack) — not duplicated per synthetic/real. See `STRUCTURE.md` for the exists-vs-planned legend and the full directory reorg.

## Why Environments Matter

A Python "environment" is an isolated set of installed packages and their versions. Different projects — or different model versions — often require incompatible package versions. Environments prevent those conflicts from breaking each other.

> **For this project it is not "often" — it is confirmed.** The three FTTL model versions (v1, v2, v3) each have a **genuinely different, mutually incompatible** library stack: every version's pickle is bound to the *exact* XGBoost / scikit-learn / numpy releases it was serialised with, and those releases differ across versions and cannot coexist in one Python process. See `DESIGN.md` § "Model Scoring & Environment Isolation" and `problem.md` §2.5 #7.

## Tooling — the Allianz team standard is `uv`

The team standard is **[uv](https://docs.astral.sh/uv/)**, a fast Rust-based replacement for `pip` + `venv`. Everything below is written for uv. (conda still works and the same two-tier idea applies — the conda equivalents are noted where useful — but uv is the canonical toolchain for this project.)

| Tool | What it does |
|---|---|
| `uv` | Creates environments, resolves + installs packages, and writes a `uv.lock`; drop-in faster `pip`/`venv` |
| `uv add` / `uv sync` | **Project mode** — manages `pyproject.toml` + `uv.lock` for you |
| `uv pip install` | **pip-compatibility mode** — just installs into a venv; writes *no* project file or lock |
| `conda` | Alternative env manager (spec: `environment.yml`); not the team default here |

---

## Two tiers of environment (managed differently, on purpose)

This project runs **two kinds of environment** with **opposite lifecycles**, so they are managed by two different uv mechanisms.

| Tier | What runs in it | uv mechanism | Spec files | Lifecycle |
|---|---|---|---|---|
| **Analysis env** (`.venv`) | The SFP pipeline, detector, mitigator, EDA, notebooks — everything that does **not** load a model | `uv add` / `uv sync` | `pyproject.toml` + `uv.lock` (repo root) | **Evolving** — packages added as research grows |
| **Per-version scoring envs** (`env-v1`, `env-v2`, `env-v3`) | Nothing but `predict.py`, scoring **one** model version's serialised artefact offline | one independent env per version | one **independent** pinned spec per version | **Frozen** — write-once; rebuilt only to reproduce, never casually mutated |

### Why the analysis env uses `uv add` + `pyproject.toml` + `uv.lock`

It is a **single, growing** dependency set. `uv add <pkg>` keeps `pyproject.toml` (what we want) and `uv.lock` (the exact resolved pins, including transitive deps + hashes) in sync in one step. `uv sync` then reproduces a byte-for-byte identical `.venv` on any machine. One source of truth, trivial to extend.

```bash
# build / update the analysis env (repo root)
uv sync                       # reads pyproject.toml + uv.lock → .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv add dowhy econml           # add a dependency (updates pyproject.toml + uv.lock + installs)
```

### Why the per-version envs are managed *separately*, not in that `pyproject.toml`

Each per-version env is a **frozen reproduction** of the libraries its model was serialised with. It must:

1. **Match that version's pins exactly** — a model pickled under XGBoost 1.5 needs XGBoost 1.5 to load and predict identically; a mismatched release raises on load or silently changes behaviour (a pickle is bound to the *exact* library version it was serialised with).
2. **Stay isolated from the other versions** — upgrading v2's stack must not move v1's or v3's. **v1, v2 and v3 are each a fully separate environment** (this project does *not* share one env across v2/v3).
3. **Not be perturbed by analysis work** — running `uv add something` for the pipeline must never touch a scoring env.

Folding the version envs into the analysis `pyproject.toml` would couple all of that together — the exact opposite of isolation. So each version keeps its **own** spec, outside the analysis project, and is scored in its **own process** (offline → parquet; the analysis runtime loads no model). See `DESIGN.md` for the offline-scoring design.

> **Why "just install all three repos as packages into one env" does not achieve this.** A recurring misunderstanding: installing a version's *repo* as a package makes its **code** importable, but it does **not** isolate that code's **dependencies**. A Python environment is a **flat** library pool — **one version of each library, shared by everything installed in it** (unlike npm's nested `node_modules`, which lets two packages carry different versions of the same dependency). Installing v1+v2+v3 into one `.venv` therefore forces `xgboost`/`scikit-learn`/`numpy` to a *single* resolved version; when the versions need incompatible releases (which here they do), pip either errors on resolution or installs one — and the other version's pickle then fails to load at runtime. The repo code is never the blocker; the numeric stack is. **Physically separating the environments is the only thing that provides isolation** — which is exactly what `env-v1`/`env-v2`/`env-v3` do.

---

## Per-version envs — two ways to pin

The per-version separation (v1 / v2 / v3 each isolated) is fixed. What you can still choose is **how tightly each one is pinned**:

| Option | How it's built | Captures | When to use |
|---|---|---|---|
| **Standard — pinned `requirements.txt`** | `uv venv` + `uv pip install -r requirements.txt` (pins written with `==`) | Direct deps only, **no lockfile** | Adequate when the env is frozen and every package is pinned exactly; fewest files |
| **Stricter — per-version `pyproject.toml` + `uv.lock`** | each version is its own tiny uv project; `uv sync` | **Transitive** deps + hashes → byte-for-byte | Dissertation-grade reproducibility, or unstable upstream transitive versions |

Both produce three independent environments; they differ only in the strength of the pin.

### Directory layout (supports either option)

```
src/envs/
├── v1/
│   ├── requirements.txt      ← Standard option (pins with ==)
│   ├── pyproject.toml        ← Stricter option
│   └── uv.lock               ← Stricter option (committed)
├── v2/
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── uv.lock
└── v3/
    ├── requirements.txt
    ├── pyproject.toml
    └── uv.lock
```

Each version directory is self-contained, so retraining or upgrading one version never silently mutates another's pinned environment.

### Build commands

**Standard (pinned requirements):**

```bash
uv venv src/envs/v1/.venv --python 3.11
uv pip install --python src/envs/v1/.venv/bin/python -r src/envs/v1/requirements.txt

uv venv src/envs/v2/.venv --python 3.11
uv pip install --python src/envs/v2/.venv/bin/python -r src/envs/v2/requirements.txt

uv venv src/envs/v3/.venv --python 3.11
uv pip install --python src/envs/v3/.venv/bin/python -r src/envs/v3/requirements.txt
```

**Stricter (per-version project + lock):**

```bash
# one tiny uv project per version; uv sync builds env-vX/.venv from its own lock
( cd src/envs/v1 && uv sync )
( cd src/envs/v2 && uv sync )
( cd src/envs/v3 && uv sync )
```

---

## Scoring with the right env (offline, one process per version)

Each model version is scored **offline, inside its own env**, and the predictions are saved to disk. The analysis then reads the saved scores — no model is loaded in the analysis process, so the environments never meet. (This replaces the older `conda run -n <env>` form; the design rationale is unchanged.)

```bash
# Standard envs — call each env's interpreter directly
src/envs/v1/.venv/bin/python src/scoring/predict.py --model src/models/synthetic/baseline/v1.pkl \
    --features src/data/synthetic/inputs/features_v1.parquet --version v1 \
    --out src/data/synthetic/detection/v1_scores.parquet

# Stricter (project) envs — uv run --project selects that version's env
uv run --project src/envs/v2 python src/scoring/predict.py --model src/models/synthetic/baseline/v2.pkl \
    --features src/data/synthetic/inputs/features_v2.parquet --version v2 \
    --out src/data/synthetic/detection/v2_scores.parquet
```

`src/scoring/run_all.sh` wraps all three versions. The script is **version-agnostic** — the active env plus the CLI args decide which version is scored. See `DESIGN.md` for the full design and the superseded runtime-subprocess alternative.

**Re-training and preprocessing also run in the per-version env.** `src/scoring/retrain.py` (re-evaluation) and `src/scoring/preprocess.py` (build that version's `features_<v>.parquet`) are executed inside `env-v1`/`env-v2`/`env-v3` exactly like `predict.py`. So each per-version env is used for **preprocessing → (re)training → scoring**; only the analysis `.venv` never loads a model. Note the *training protocol* (`src/training/`) is shared across versions and env-agnostic (pure pandas/sklearn + an injected estimator), whereas *preprocessing* (`src/preprocessing/v{1,2,3}.py`) is genuinely per-version — see `DESIGN.md` § "Where per-version code lives" and `STRUCTURE.md`.

---

## Typical workflow (summary)

```
Analysis env (.venv):
  uv sync                         # build from pyproject.toml + uv.lock
  source .venv/bin/activate
  uv add <pkg>                    # extend as research grows

Per-version envs (env-v1 / env-v2 / env-v3):
  build once from each version's own spec (requirements.txt OR pyproject.toml+uv.lock)
  score each version offline in its own process → data/<source>/detection/*_scores.parquet
  analysis reads the parquet; never loads a model
```

Commit `pyproject.toml` + `uv.lock` (analysis), and each version's spec files, so any teammate reproduces every environment exactly.

## Adding a new model version (e.g., v4)

1. Create `src/envs/v4/` with its spec (`requirements.txt`, or `pyproject.toml` + `uv.lock` for the stricter option) and build the env.
2. Produce `data/<source>/inputs/features_v4.parquet` (v4's own preprocessing on real data; on synthetic, `export_version_features` emits it once the version is registered), and regenerate `src/models/<source>/baseline/v4.pkl` by retraining v4's repo code in `env-v4`.
3. Add one scoring line for v4 to `src/scoring/run_all.sh`.
4. Add `"v4": ".../detection/v4_scores.parquet"` to the `score_paths` dict in the analysis.

No new class is required — `predict.py` and `scores.py` are version-agnostic. See `DESIGN.md`.

# Environment Management in Data Science

> **Implementation status (2026-07-03, reorg complete).** The two-tier env model below is live. The analysis `.venv` exists; the per-version scoring envs are built and working at **`src/envs/{v1,v2,v3}/`** (moved from `src/model/envs/` on 2026-07-03 — the move preserves each `.venv` since its base interpreter and editable installs are absolute paths outside the moved dir). These map to the layer names in `STRUCTURE.md`/`DESIGN.md`: the per-version scoring envs are the **Version Layer**, the analysis `.venv` is the **Analysis Layer**. Envs are **source-agnostic** (an env is a library stack) — not duplicated per synthetic/real. See `STRUCTURE.md` for the exists-vs-planned legend and the full directory reorg.

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
| **Per-version model envs** (`env-v1`, `env-v2`, `env-v3`) | Anything that loads or builds **one** version's model: `train.py` (baseline), `retrain.py` (mitigated), `predict.py` (scoring), `attribute.py` (per-row SHAP) — offline | one independent env per version | one **independent** pinned spec per version | **Frozen** — write-once; rebuilt only to reproduce, never casually mutated |

> **`shap` in a frozen env (2026-08-01).** `scoring/attribute.py` is the one Version-Layer script with an optional *extra* dependency, so it is also the one place the "frozen" rule gets tested. Two routes, and the choice is recorded in the run's meta JSON rather than left implicit:
> 1. `uv pip install shap` into `src/envs/v<k>/.venv`. `shap` pins nothing in the numeric stack, so the reproduction survives — but do it for **all three** envs or none, because comparing versions across different SHAP backends is what `estimator.concentration.require_comparable()` refuses.
> 2. `--backend native`, which uses the booster's own `pred_contribs` and adds **nothing** to the env. The cost is methodological, not operational: it is tree-path-dependent, so part of any cross-version difference is a difference in training distributions rather than in the model function.
>
> Whichever route, that rule holds for the headless path.

> **Revised the same day: a version env may host a notebook after all.** The rule above ("keep plotting libraries out") was written for `attribute.py`, which is headless. `notebook/real/00_SHAP.ipynb` is the deliberate exception — the *single-version* SHAP analysis is interactive and must open the pickle, so it runs on a kernel built on the version's own `.venv`:
> ```bash
> src/envs/v2/.venv/bin/python -m pip install ipykernel matplotlib pandas pyarrow
> src/envs/v2/.venv/bin/python -m ipykernel install --user --name fttl-v2 --display-name "FTTL v2 (env-v2)"
> ```
> This is a real, accepted cost: `ipykernel` + `matplotlib` enter a frozen env. It is acceptable because **neither pins anything in the numeric stack** — the pins that define the reproduction are `xgboost` (0.72 / 1.4.2 / 3.2.0), `numpy` and `scikit-learn`, and none of them move. What must still be resisted is `uv add`-ing analysis libraries (statsmodels, dowhy, seaborn, a newer pandas) into a version env to make a notebook cell work: that is where the stack drifts. If a cell needs one, it belongs in the analysis `.venv`, reading the parquet the version env wrote.
>
> `src/shap_kit.py` is what keeps the cost this low — it imports only numpy/pandas/matplotlib and draws every SHAP figure from the raw φ matrix, so no version env needs a `shap` release new enough to have `shap.plots.*`. It also asserts, at the top of every run, that the kernel's xgboost matches `config.VERSIONS[v]["xgboost"]`.
>
> **`shap` remains genuinely optional, with one degraded feature.** Values fall back to the booster's `pred_contribs`; **interaction** values fall back to `pred_interactions`, which arrived in **xgboost 0.81** — so on env-v1 (0.72) there is no native route and the interaction section needs `shap` installed there or it is skipped (the notebook says so and falls back to a correlation proxy for its dependence-plot colours). Installing `shap` into a version env stays preferable to relaxing any numeric pin.
>
> Kernel registration commands for all three envs, macOS and Windows, are in `SETUP.md` § "Jupyter kernels — one per version env", together with what to do when `xgboost==0.72` refuses to install on the platform.

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

### Making the version repo importable — the `fttl.pth` step *(added 2026-08-06)*

The build commands above are generic. The interpreters the **real** envs were actually built on
differ per version: **env-v1 = 3.5.6 (conda `-p`, so `python.exe` sits at the env root, not in
`Scripts\`), env-v2 = 3.10, env-v3 = 3.11** — all confirmed on the company laptop 2026-08-06.
Only v1's deviation is forced (uv does not support < 3.6); v2/v3 follow their repos' own pins.

Installing the pins is only half of building a per-version env. The env must also be able to
`import` the version repo's own modules, because that is what unpickles its model (§ "Scoring
with the right env"). Two distinct facts make this its own step:

- **The version repos are not packages.** They are analysis code with no `[tool.setuptools]`
  declaration and (v1, v2) no `__init__.py` anywhere. At Allianz they were run *from inside* the
  repo, so the running script's own folder was on `sys.path` automatically and nothing needed
  installing. Importing them from outside means putting those folders back on `sys.path`.
- **`uv pip install -e <repo>` does both jobs in one command** — installs `[project.dependencies]`
  *and* registers the repo on `sys.path` — so when the second half fails the pins are not
  installed either. v3 succeeds (one top-level candidate ⇒ setuptools' flat-layout discovery
  resolves it, leaving `__editable__.p146_fttl_product-*.pth` in its site-packages). **v2 fails**:
  its root holds several sibling directories, discovery refuses to choose, and the whole install
  aborts. Adding `[tool.setuptools]` / `py-modules = []` to that repo's `pyproject.toml` disables
  discovery so the dependency half proceeds. That edit lives in the clone, which is gitignored
  (`/model_repos/`), so it does not survive a re-clone — hence this note.

The `sys.path` half is therefore done by hand, per env, and it is **the same procedure for v1 and
v2** — only the interpreter and the repo path differ. Ask that env's interpreter where its
`site-packages` is, then write one file there named `fttl.pth`:

```bash
# ① where does this env's .pth go? (v1: .venv\python.exe — conda puts it at the env root)
src/envs/v2/.venv/Scripts/python.exe -c "import sysconfig;print(sysconfig.get_paths()['purelib'])"
```

`<that dir>/fttl.pth` then holds one **absolute** path per line — the repo root plus every
subdirectory that holds `.py` files:

```
# fttl v2 repo paths
C:\...\model_repos\real\fttl-v2
C:\...\model_repos\real\fttl-v2\<subdir with .py>
```

Verify in a **fresh** process (a `.pth` is read only at interpreter startup):

```bash
src/envs/v2/.venv/Scripts/python.exe -c "import sys;print([p for p in sys.path if 'model_repos' in p])"
```

Five things about that file, each learned the hard way:

- **Subdirectories, not just the root.** Registering only the root makes `import analysis.helpers`
  work but not `import helpers` — and flat sibling imports are exactly the style implied by the
  absence of `__init__.py`. The goal is to **reproduce the `sys.path` shape the pickle was written
  under**, not to restructure the repo into a proper package: a pickle stores its classes'
  module paths as strings, so changing how the repo is imported breaks `load()`. For the same
  reason, **do not add `__init__.py` to the version repos** — it would not fix the `-e` failure
  (5 declared packages still cannot be auto-chosen) and it renames the very modules the pickle
  asks for.
- **One filename, every env.** `fttl.pth` in all three: each env has its own `site-packages`, so
  they never collide, and a fixed name makes rewriting idempotent instead of accumulating stale
  path entries under variant names.
- **A nonexistent path in it is skipped silently** — a typo produces no error and no effect, which
  is why the verification step above is not optional.
- **Name collisions resolve by path order, silently.** If two registered directories both hold
  `utils.py`, `import utils` picks whichever line comes first. Prefer registering only the
  directories actually needed.
- **First line a comment.** If the file is written in Notepad it may gain a UTF-8 BOM, which would
  corrupt the first line; `.pth` ignores lines starting with `#`, so the BOM lands somewhere
  harmless.

### Verifying what a built env actually contains *(added 2026-08-10)*

`src/envs/check_installed.py` reads a `pyproject.toml`'s declared dependencies and reports, for one
or more venvs at once, which are present, at what version, and — via PEP 610 `direct_url.json` —
which came from a git URL or a local path rather than an index. It only inspects; it never installs.

```bash
python src/envs/check_installed.py \
    --pyproject model_repos/real/fttl-v3/pyproject.toml \
    --venv model_repos/real/fttl-v3/.venv \
    --venv src/envs/v3/.venv
```

Two venvs are worth passing together because a version repo cloned from Allianz brings **its own**
`.venv`, built by whoever ran it in production, while `src/envs/v3/.venv` is the one built here from
the pins. Where those two disagree, the production one is the reference — the pickle was written
under it. The `--all` flag additionally lists packages installed but not declared, which is how an
undeclared transitive import (the `pyodbc`-at-load-time class of problem, § v1) surfaces before it
breaks an unpickle.

The driving interpreter needs 3.11+ for `tomllib`; the inspected venvs can be any 3.8+, so the
analysis `.venv` can audit env-v1 (3.5.6) without env-v1 running the script itself.

Two limits, both deliberate: it parses dependency **names** only, not version specifiers (it reports
what is installed rather than judging a constraint), and it resolves `[tool.uv.sources]` only —
Poetry-style source tables are not read, since the version repos are uv/setuptools projects.

---

## Scoring with the right env (offline, one process per version)

Each model version is scored **offline, inside its own env**, and the predictions are saved to disk. The analysis then reads the saved scores — no model is loaded in the analysis process, so the environments never meet. (This replaces the older `conda run -n <env>` form; the design rationale is unchanged.)

```bash
# Standard envs — call each env's interpreter directly
src/envs/v1/.venv/bin/python src/scoring/predict.py --model src/models/real/baseline/v1.pkl \
    --features src/data/real/inputs/features_v1_test.parquet --version v1 \
    --out src/data/real/detection/v1_scores_test.parquet

# Stricter (project) envs — uv run --project selects that version's env
uv run --project src/envs/v2 python src/scoring/predict.py --model src/models/real/baseline/v2.pkl \
    --features src/data/real/inputs/features_v2_test.parquet --version v2 \
    --out src/data/real/detection/v2_scores_test.parquet
```

Note the `_test` suffix: feature matrices, targets and scores exist **only per split** — the export
notebooks write one file each and no unsplit base file. The drivers below take `--split` (one name
for every version, or `v2=test v3=oot`, since split names are each version's own) and resolve those
paths themselves. See `STRUCTURE.md` § "Some kinds exist ONLY per split".

`src/scoring/score_all.py` wraps all three versions (it supersedes `score_all.sh`). The script is **version-agnostic** — the active env plus the CLI args decide which version is scored. It is Python rather than bash for two reasons: it reads `config.py` natively to resolve every path by *kind* (bash cannot import it — see `STRUCTURE.md` § "Paths are resolved by KIND"), and the company laptop is Windows, where `.sh` does not run. It selects each env by calling `config.python_bin(version)`, so the interpreter paths above are declared in one place rather than repeated per call site. See `DESIGN.md` for the full design and the superseded runtime-subprocess alternative.

**There is no baseline-training driver any more** (`train_all.py` deleted 2026-08-19; `train_all.sh` before it, 2026-07-31). It skipped any version whose `config.VERSIONS[v]["paths"]["model"]` is declared — and all three versions now declare one, so every run was a no-op. Its only remaining path, `--force`, resolved `--out-model` to the *declared* path and would have written over the real production pickle. The baseline on real data is each version's own pickle, **loaded, never refitted** — v1 permanently so, its training data having been destroyed. The baseline trainer `training/train.py` went with it the same day, for the same reason — with no version to train it had no caller left. `config.TRAINING_CONFIG` records each version's training call if a refit is ever needed. `training/retrain.py` (the MITIGATED retrainer) is the only trainer that remains, and `training/retrain_all.py` is its config-aware driver — the counterpart of `score_all.py`.

**Why `retrain.py` takes paths instead of reading config.** env-v1 is **Python 3.5.6**, and `config.py` is 3.7+ (future annotations, `dict[str, ...]` variable annotations, ~50 f-strings) — a worker that imported it could not be *parsed* there, let alone run. So `retrain.py` is written to 3.5 (no f-strings, no future annotations, `.format()` throughout) and `retrain_all.py` resolves every path by KIND in the analysis `.venv` before handing over strings. ⚠️ `predict.py` and `attribute.py` have **not** been backported and still cannot run in env-v1.

**Re-training and preprocessing also run in the per-version env.** `src/training/retrain.py` (mitigated pkl, re-evaluation) and `src/preprocessing/v{1,2,3}.py` (build that version's `features_<v>.parquet`) are executed inside `env-v1`/`env-v2`/`env-v3` exactly like `predict.py` — because all load or build that version's model and so need its repo importable. So each per-version env is used for **preprocessing → (re)training → scoring**; only the analysis `.venv` never loads a model. The **log-ingestion** step (`log_source` → `logs/<v>.parquet`) lives in `notebook/real/01_export_v2.ipynb` since 2026-08-09 (`scoring/ingest.py` deleted — only v2 has a production log; the source is a version-bound pandas pickle, so it must be read inside env-v2 anyway). (`scoring/inspect_pickle.py` was DELETED 2026-08-19 — the one-off util that reported a prod pickle's schema and date range. The export notebooks `01_export_v{1,2,3}` now read every `Z:` source inside its own env and write canonical parquet, which is the same job done properly.) Note *preprocessing* (`src/preprocessing/v{1,2,3}.py`) is genuinely per-version — see `DESIGN.md` § "Where per-version code lives" and `STRUCTURE.md`.

### env-v1 has no parquet engine — the CSV stage *(recorded 2026-08-19)*

`src/envs/v1/requirements.txt` pins scikit-learn / pandas / pyodbc / matplotlib 2.2.5 /
ipykernel 4.10.1 / joblib 0.14.1 and **nothing else** — no pyarrow, no fastparquet, and neither can
be added on Python 3.5.6. So env-v1 can neither read nor write parquet. That is *why*
`notebook/real/01_export_v1.ipynb` emits **CSV** while the v2/v3 export notebooks write parquet
directly: an environment limit, not a style choice.

The v1 route is two-stage, and only the first stage is version-bound:

| stage | env | writes |
|---|---|---|
| `notebook/real/01_export_v1.ipynb` | env-v1 (py3.5) | `.csv` under `src/data/real/{inputs,detection}/` |
| `v1_csv_to_parquet.py` (repo root) | analysis `.venv` | `.parquet` at config's fallback paths |

**Status: the conversion has been run and every v1 parquet exists (user-confirmed 2026-08-19),** so
analysis code resolving `config.path(..., "v1", split=...)` finds its files normally. The CSV stage
is upstream history, not something downstream handles.

**The converter derives its paths from `config`** (rewritten 2026-08-19). It walks
`processed_inputs` / `targets` / `scores` × `config.SPLITS["v1"]` plus the single `raw_dataset`,
resolves each destination with `config.path(kind, "v1", split=...)`, and takes the source as that
same path with a `.csv` suffix — the naming `01_export_v1.ipynb` mirrors. Nothing is hand-listed,
so the file names cannot drift from config again. A missing source is reported, not fatal
(`raw_v1.csv` comes from a line that is commented out in the notebook), and every write is checked
by re-reading the parquet's row count.

> It previously carried a hand-written `PAIRS` list of *unsplit* names (`features_v1.csv`) and had
> already drifted from the notebook's per-split output. The company-laptop copy was fixed there
> first; this one was brought into line afterwards. Deriving the names is what removes the class of
> bug rather than one instance of it.

**What this does NOT solve.** Having the parquet files does not let the version-layer *scripts* run
inside env-v1: `scoring/predict.py` and `scoring/attribute.py` call `pd.read_parquet` /
`to_parquet`, which need an engine env-v1 lacks — on top of their Python-3.5 syntax problem
(`from __future__ import annotations`, f-strings, `str | None`; 3 and 21 occurrences respectively).
Running either on v1 needs a 3.5 backport **and** CSV/pkl I/O branches. `training/retrain.py` was
rewritten with both on 2026-08-19 and is the pattern to follow.

### `shap` belongs in the per-version envs too *(added 2026-07-14)*

**`src/scoring/attribute.py`** — the planned sibling of `predict.py`, emitting per-row SHAP attributions to `detection/shap/<v>_attributions_<split>.parquet` for the `estimator/` layer — runs in the **per-version env**, for the same reason `predict.py` does: SHAP needs the model *object*, and a pkl only unpickles where its repo is importable (loading `models/*/baseline/v1.pkl` from the analysis `.venv` raises `ModuleNotFoundError: fttl_v1`). That has a concrete env consequence:

> **`shap` must be installed and pinned in `env-v1`/`env-v2`/`env-v3`**, not only in the analysis `.venv` — and independently per version, because each env's XGBoost differs and `shap`'s tree parser is coupled to the XGBoost model format.

That coupling is not hypothetical: `shap` 0.49 fails on XGBoost 3.x with `ValueError: could not convert string to float: '[1.4605759E-1]'` (the newer format stores `base_score` as a vector), and needs ≥ 0.51. Expect to pin a **different `shap` per version**, exactly as with XGBoost itself — which is precisely the "flat library pool" argument for separate envs, playing out again.

The analysis `.venv` also carries `shap` (added via `uv add shap "numba>=0.60"`), but **only for the notebooks**, which retrain their own models on the DGP and so hold real model objects. The `numba>=0.60` floor is required: without it the resolver selects `numba` 0.53, whose `llvmlite` 0.36 fails to build on Python 3.11 / arm64. No application code in the analysis layer may load a model.

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
2. Add a `"v4"` entry to `config.VERSIONS` (and to `VERSION_LABELS`): its `paths`, its `columns` mapping, its `python`. Run `python src/config.py` to confirm nothing is left as a placeholder.
3. Write a `notebook/real/01_export_v4.ipynb` (the per-version export notebook — the pattern of `01_export_v1/2/3`) to land v4's raw/inputs/targets/scores as canonical parquet, one file per split — the export notebook is the sole producer of all four.
4. Add `"v4": ".../detection/v4_scores.parquet"` to the `score_paths` dict in the analysis.

No new class is required, and **no driver script is edited** — `score_all.py` loops `config.VERSION_LABELS`, so registering the version in config is what adds it to the run. `predict.py`, `attribute.py`, `retrain.py` and `load_scores.py` are all version-agnostic; only the export notebook is per-version, because each repo's artefact layout is. See `DESIGN.md`.

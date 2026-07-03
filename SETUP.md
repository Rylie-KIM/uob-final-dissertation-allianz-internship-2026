# SETUP

Each model version (fttl-v1/v2/v3) needs its **own env** — their xgboost/sklearn stacks are mutually
incompatible and can't share one env. There are **2 ways** to install an external model repo into its
per-version env. Both give the same result: the pkl loads and scores inside env-vX. Everything else
(`predict.py`, `detector`, `mitigator`, …) is **version-agnostic → no changes needed**.

## The two methods

| | **A. local clone + editable** | **B. git-dependency** |
|---|---|---|
| Where the code lives | `model_repos/` original (live link) | copy in site-packages (frozen) |
| Local folder | required (deleting it breaks the env) | not needed |
| Which files are used | working tree (no commit needed) | **only committed + pushed files** |
| Use for | development / editing | reproducibility / deploy / onboarding |

## 0. Prerequisites

```bash
brew install libomp     # needed for xgboost to load (macOS). uv must already be installed.
```

## Method A — local clone + editable

Keep the repo on disk and **point** at it (no copy). Source edits are reflected live.
**Deleting the folder breaks the env.**

```bash
for V in 1 2 3; do
  git clone git@github.com:Rylie-KIM/fttl-v$V.git model_repos/real/fttl-v$V
  uv venv src/model/envs/v$V/.venv --python 3.11
  uv pip install --python src/model/envs/v$V/.venv/bin/python -e model_repos/real/fttl-v$V
done
```

## Method B — git-dependency (pinned SHA)

Fetch a specific commit online **once** and copy it into site-packages (frozen snapshot). No local
clone needed. Note: the repo must have its code + `pyproject.toml` **committed and pushed** (a
README-only commit fails).

```bash
for V in 1 2 3; do
  uv venv src/model/envs/v$V/.venv --python 3.11
  uv pip install --python src/model/envs/v$V/.venv/bin/python \
    "fttl-v$V @ git+ssh://git@github.com/Rylie-KIM/fttl-v$V.git@<SHA>"
done
```

## Verify

```bash
# importing its own package in that env = success (it must FAIL in the analysis .venv = isolation works)
src/model/envs/v2/.venv/bin/python -c "import fttl_v2; print('env-v2 ok')"
```

## Analysis env + run

```bash
uv sync                                             # build the root analysis .venv
.venv/bin/python src/data/build_scoring_inputs.py   # generate features/labels
.venv/bin/python src/run_cycle.py                   # detect -> mitigate -> re-evaluate
```

> Pipeline details: `src/test.cycle.md` · architecture: `src/STRUCTURE.md`, `src/DESIGN.md`

# SETUP

Each model version needs its **own env** — their xgboost/sklearn stacks are mutually
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

### macOS / Linux

```bash
for V in 1 2 3; do
  git clone git@github.com:Rylie-KIM/fttl-v$V.git model_repos/real/fttl-v$V
  uv venv src/envs/v$V/.venv --python 3.11
  uv pip install --python src/envs/v$V/.venv/bin/python -e model_repos/real/fttl-v$V
done
```

### Windows (PowerShell)

```powershell
foreach ($V in 1,2,3) {
  git clone git@github.com:Rylie-KIM/fttl-v$V.git model_repos\real\fttl-v$V
  uv venv "src\envs\v$V\.venv" --python 3.11
  uv pip install --python "src\envs\v$V\.venv\Scripts\python.exe" -e "model_repos\real\fttl-v$V"
}
```

## Method B — git-dependency (pinned SHA)

Fetch a specific commit online **once** and copy it into site-packages (frozen snapshot). No local
clone needed. Note: the repo must have its code + `pyproject.toml` **committed and pushed** (a
README-only commit fails).

### macOS / Linux

```bash
for V in 1 2 3; do
  uv venv src/envs/v$V/.venv --python 3.11
  uv pip install --python src/envs/v$V/.venv/bin/python \
    "fttl-v$V @ git+ssh://git@github.com/Rylie-KIM/fttl-v$V.git@<SHA>"
done
```

### Windows (PowerShell)

```powershell
foreach ($V in 1,2,3) {
  uv venv "src\envs\v$V\.venv" --python 3.11
  uv pip install --python "src\envs\v$V\.venv\Scripts\python.exe" `
    "fttl-v$V @ git+ssh://git@github.com/Rylie-KIM/fttl-v$V.git@<SHA>"
}
```

## Verify

### macOS / Linux

```bash
# importing its own package in that env = success (it must FAIL in the analysis .venv = isolation works)
src/envs/v2/.venv/bin/python -c "import fttl_v2; print('env-v2 ok')"
```

### Windows (PowerShell)

```powershell
& "src\envs\v2\.venv\Scripts\python.exe" -c "import fttl_v2; print('env-v2 ok')"
```

## Analysis env + run

### macOS / Linux

```bash
uv sync                                             # build the root analysis .venv
.venv/bin/python src/data/build_scoring_inputs.py   # generate features/labels
.venv/bin/python src/run_cycle.py                   # detect -> mitigate -> re-evaluate
```

### Windows (PowerShell)

```powershell
uv sync
.\.venv\Scripts\python.exe src\data\build_scoring_inputs.py
.\.venv\Scripts\python.exe src\run_cycle.py
```

> Pipeline details: `src/test.cycle.md` · architecture: `src/STRUCTURE.md`, `src/DESIGN.md`

---

# Jupyter kernels — one per version env (added 2026-08-01)

`notebook/real/00_SHAP.ipynb` opens a version's model pickle, so it must run **inside that version's
env** — a pickle only unpickles under the release it was serialised with, and these are far apart:

| env | xgboost | kernel name suggested |
|---|---|---|
| `src/envs/v1/.venv` | **0.72.1** | `fttl-v1` |
| `src/envs/v2/.venv` | **1.4.2** | `fttl-v2` |
| `src/envs/v3/.venv` | **3.2.0** | `fttl-v3` |

One notebook, three kernels — not three notebooks. It detects which version it is from the
interpreter path and **raises** if the running xgboost is not the one `config.VERSIONS[v]["xgboost"]`
declares. That check is the point: a mismatched load produces plausible figures, not an error.

## What to install into a version env — and what not to

```
ipykernel      required   the kernel itself
matplotlib     required   src/shap_kit.py draws every figure
pandas         required   almost certainly already there (the pickle needs it)
pyarrow        required   to write the attributions parquet
shap           optional   better SHAP values (interventional) + interaction values;
                          without it shap_kit falls back to the booster's own TreeSHAP
```

**Do not** `uv add` analysis libraries (statsmodels, dowhy, seaborn, a newer pandas) into a version
env to make a cell work. None of the four above move `xgboost` / `numpy` / `scikit-learn` — the pins
that *are* the reproduction — but analysis packages will. Anything that needs them belongs in the
analysis `.venv`, reading the parquet the version env wrote.

## macOS / Linux

```bash
for V in 2 3; do
  uv pip install --python src/envs/v$V/.venv/bin/python ipykernel matplotlib pandas pyarrow
  src/envs/v$V/.venv/bin/python -m ipykernel install --user \
    --name fttl-v$V --display-name "FTTL v$V (env-v$V)"
done
# v1 is Python 3.5 — outside uv. Its packages (ipykernel 4.x, matplotlib) come from
# src/envs/v1/requirements.txt via its own pip; only the registration step remains:
src/envs/v1/.venv/bin/python -m ipykernel install --user \
  --name fttl-v1 --display-name "FTTL v1 (env-v1)"
jupyter kernelspec list          # expect fttl-v1, fttl-v2, fttl-v3
```

(The kernel is registered by calling the env's interpreter **directly** — `uv run` would try to
resolve a project env of its own, which is exactly what a frozen version env must not be part of.)

`uv pip install --python <interpreter>` is pip-compatibility mode: it installs into that env and
writes **no** `pyproject.toml` and **no** lock entry, which is what a frozen env wants.

## Windows (PowerShell) — uv

The interpreter lives in `Scripts\` rather than `bin/`, and `shap_kit.detect_version()` handles both
(it normalises `\` to `/` before looking for `/envs/v2/`).

```powershell
foreach ($V in 2,3) {
  uv pip install --python "src\envs\v$V\.venv\Scripts\python.exe" ipykernel matplotlib pandas pyarrow
  & "src\envs\v$V\.venv\Scripts\python.exe" -m ipykernel install --user `
      --name "fttl-v$V" --display-name "FTTL v$V (env-v$V)"
}
# v1 is Python 3.5 — outside uv (see "The v1 env" below); only the registration step here.
# NB: if env-v1 was built with conda, its interpreter is src\envs\v1\.venv\python.exe (env root,
# no Scripts\) — use that path instead:
& "src\envs\v1\.venv\python.exe" -m ipykernel install --user `
    --name "fttl-v1" --display-name "FTTL v1 (env-v1)"
jupyter kernelspec list
```

Building the v2/v3 envs themselves on Windows, if they do not exist yet (v1: next section):

```powershell
foreach ($V in 2,3) {
  uv venv "src\envs\v$V\.venv" --python 3.11
  uv pip install --python "src\envs\v$V\.venv\Scripts\python.exe" -r "src\envs\v$V\requirements.txt"
}
```

One Windows-specific note: **no `libomp` step.** The `brew install libomp` prerequisite above is
macOS-only; the Windows xgboost wheels bundle their OpenMP runtime.

## Using the kernels

Start Jupyter from the **analysis** env — the kernels are registered user-wide, so one Jupyter
serves all of them:

```bash
uv run jupyter lab          # then pick the kernel per notebook
```

| notebook | kernel |
|---|---|
| `notebook/real/00_SHAP.ipynb` | `fttl-v1` / `fttl-v2` / `fttl-v3` — run it once on each |
| `notebook/real/00_shap_attribution.ipynb` | `sfp-detection` (the analysis `.venv`) |
| everything in `notebook/` | `sfp-detection` |

Confirm the kernel is the one you think it is — the first cell prints it, but as a manual check:

```python
import sys, xgboost
print(sys.executable, xgboost.__version__)   # must contain envs/v2 and print 1.4.2
```

## The v1 env — Python 3.5.2, no uv (facts confirmed 2026-08-04)

The v1 model was built under **Python 3.5.2**, and its repo vendors the xgboost binary directly —
`Dependencies/packages/xgboost-0.72-cp35*-win_amd64.whl` — because PyPI had no Windows xgboost
wheel in that era. Two consequences:

- **uv cannot build or manage this env**: "uv does not work with Python versions prior to 3.6"
  (https://docs.astral.sh/uv/reference/policies/python/), and v1 is 3.5.2. v1 alone is built with
  Python 3.5's own `venv` + `pip` — every other env stays on uv.
- **xgboost comes from the vendored wheel, not PyPI**, and the wheel says `0.72`: after install,
  check `xgboost.__version__` and set `config.VERSIONS["v1"]["xgboost"]` to whatever it prints
  (the strict check in `shap_kit` compares against it).

**The route that worked (2026-08-05) — conda for python+pip only, everything else via pip.**
The repo's `conda_dependencies_local.yml` is not executed: the company network blocks its channel
(conda-forge) and its exact python (3.5.2); conda traffic goes through the internal Artifactory
mirror instead, which carries python 3.5.6 (same cp35m ABI — the vendored wheel fits). The yml
stays as the *source of the version pins*. Full sequence and the recorded deviations:
**`src/envs/v1/requirements.txt`**. Short form:

```powershell
conda create -p src\envs\v1\.venv python=3.5 pip
src\envs\v1\.venv\python.exe -m pip install "pip==20.3.4" "setuptools==50.3.2"
src\envs\v1\.venv\python.exe -m pip install <v1-repo>\Dependencies\packages\xgboost-0.72-cp35-cp35m-win_amd64.whl
src\envs\v1\.venv\python.exe -m pip install -r src\envs\v1\requirements.txt
```

In a conda env on Windows `python.exe` sits at the env ROOT (no `Scripts\` for the interpreter);
`config.python_bin` knows all three layouts.

**Fallback — python.org installer + venv**, if conda is unavailable. The requirements template
with what to copy from the yml lives in **`src/envs/v1/requirements.txt`**. Short form
(PowerShell; use the full path to python.exe if the `py` launcher is not installed):

```powershell
py -3.5 -m venv src\envs\v1\.venv
src\envs\v1\.venv\Scripts\python.exe -m pip install "pip==20.3.4" "setuptools==50.3.2" "wheel==0.36.2"
src\envs\v1\.venv\Scripts\python.exe -m pip install <v1-repo>\Dependencies\packages\xgboost-0.72-cp35.cp36-win_amd64.whl
src\envs\v1\.venv\Scripts\python.exe -m pip install -r src\envs\v1\requirements.txt
```

(`pip==20.3.4` is the last pip supporting 3.5. The kernel registers the same way as the others,
with this interpreter; `ipykernel` is pinned to the 4.x series in the requirements file because
modern ipykernel does not run on 3.5.)

If Python 3.5.2 cannot be installed on the machine at all, fall back in order:

1. **WSL / a Linux box** — build a 3.5/3.6 env there; the pickle and the parquet it produces are
   portable, the env is not.
2. **Last resort — migrate the model, and verify it.** Load the v1 pickle in the *v2* env (1.4.2 can
   usually read a 0.72-era booster) and re-export a portable model:
   ```python
   est.get_booster().save_model("v1_booster.json")
   ```
   Then explain that JSON under a newer stack. ⚠️ This is a different object from the deployed one
   until proven otherwise: score the same claims through both and confirm the outputs match before
   any figure derived from it enters the thesis. If v1 cannot be loaded anywhere, say so as a
   limitation — v1's training data was already destroyed, so an unreproducible v1 is a documented
   gap, not a silent one.

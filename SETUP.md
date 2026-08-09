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

> The loops below are the **generic** form. On the real repos only v3 ran as written — v1 needs
> conda + Python 3.5 and v2 needs a `pyproject.toml` edit plus a hand-written `.pth`. Read
> § "What actually happened" before running these.

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

> `fttl_v2` is a placeholder from the synthetic era. The **real** repos expose no package of that
> name — what each version actually imports is recorded in the next section.

## What actually happened — real envs, company laptop (built 2026-08-05/06)

Method A ran verbatim for **v3 only**. v1 and v2 each diverged, for different reasons, and both
ended up needing a hand-written `.pth`. Current state:

| | **v1** | **v2** | **v3** |
|---|---|---|---|
| Python | 3.5.6 (conda `-p`) | **3.10** (`uv venv`) | 3.11 (`uv venv`) |
| `python.exe` | `.venv\python.exe` (env ROOT) | `.venv\Scripts\python.exe` | `.venv\Scripts\python.exe` |
| Dependencies from | `-r src\envs\v1\requirements.txt` + vendored xgboost wheel | `-e <repo>` (pyproject) | `-e <repo>` (pyproject) |
| `-e` installed the repo? | never attempted | **no** — deps only | **yes**, editable |
| Repo on `sys.path` via | hand-written `fttl.pth` | hand-written `fttl.pth` | the editable install |
| `.pth` filename | `fttl.pth` | `fttl.pth` | `__editable__.p146_fttl_product-0.1.0.post300+git.060e3615.pth` |

**Why v2 could not use Method A as written.** `uv pip install -e <repo>` does two jobs in one
command — install `[project.dependencies]` *and* register the repo on `sys.path` — so a failure in
the second half aborts the first as well. v2's repo root holds several sibling directories
(`outputs`, `analysis`, `lookup_tables`, `reconciliation`, …), setuptools' flat-layout discovery
cannot choose a package among them, and the whole install fails with *"multiple top-level packages
discovered in a flat-layout"*. v3 has a single top-level candidate, so discovery resolves and
Method A completes. The fix for v2 is to disable discovery in **the clone's** `pyproject.toml`:

```toml
[tool.setuptools]
py-modules = []
```

Then the original command succeeds and installs the pins. **That edit does not survive a
re-clone** — `/model_repos/` is gitignored, so re-running Method A on a fresh clone will hit the
same error. Re-apply it.

### The `.pth` step (v1 and v2)

`fttl.pth` goes in that env's own `site-packages` and holds one **absolute** path per line — the
repo root plus each subdirectory containing `.py` files. Two routes; both produce the same file.

**Route 1 — one command per env** (run from the **repo root**; `cmd`, quotes as shown):

```
:: v2  — venv layout, interpreter in Scripts\
src\envs\v2\.venv\Scripts\python.exe -c "import os,sysconfig,pathlib;r=os.path.abspath('model_repos/real/fttl-v2');d=[r]+[str(p) for p in pathlib.Path(r).iterdir() if p.is_dir() and list(p.glob('*.py'))];f=pathlib.Path(sysconfig.get_paths()['purelib'])/'fttl.pth';f.write_text('\n'.join(d));print(f);print(*d,sep='\n')"

:: v1  — conda layout, interpreter at the env ROOT (no Scripts\)
src\envs\v1\.venv\python.exe -c "import os,sysconfig,pathlib;r=os.path.abspath('model_repos/real/v1');d=[r]+[str(p) for p in pathlib.Path(r).iterdir() if p.is_dir() and list(p.glob('*.py'))];f=pathlib.Path(sysconfig.get_paths()['purelib'])/'fttl.pth';f.write_text('\n'.join(d));print(f);print(*d,sep='\n')"
```

Only two things differ between the two lines: **which `python.exe`**, and **which repo path**.
Everything else is identical — that is the point of ⑤ below. Note the repo path is written with
forward slashes because it is a **Python string**, where `\r`, `\f`, `\v`, `\t` would be escape
sequences (`'model_repos\real\fttl-v2'` silently contains a carriage return); the shell arguments
around it keep backslashes.

Unfolded, that one-liner is (`-c` runs a string as a program, and `cmd` cannot carry newlines, so
statements are joined with `;`):

```python
import os, sysconfig, pathlib

r = os.path.abspath('model_repos/real/fttl-v2')                    # ①
d = [r] + [str(p)                                                  # ④
           for p in pathlib.Path(r).iterdir()                      # ②
           if p.is_dir() and list(p.glob('*.py'))]                 # ③
f = pathlib.Path(sysconfig.get_paths()['purelib']) / 'fttl.pth'    # ⑤
f.write_text('\n'.join(d))                                         # ⑥
print(f); print(*d, sep='\n')                                      # ⑦
```

- **① `os.path.abspath(...)`** — relative → absolute, resolved against the **current working
  directory**, which is why this must be run from the repo root. Absolute is mandatory: see ⑥.
- **② `.iterdir()`** — direct children only (files + dirs), not recursive. Takes no arguments.
- **③ `p.is_dir() and list(p.glob('*.py'))`** — keep directories that hold at least one `.py`
  *directly* inside (`glob` looks one level down; `rglob` would recurse). `glob` returns a
  generator, so `list()` forces it — an empty list is falsy, a non-empty one truthy, which is the
  filter.
- **④ `[r] + [...]`** — repo root first. `sys.path` is searched front to back, so on a name clash
  the root wins.
- **⑤ `sysconfig.get_paths()['purelib']`** — the `site-packages` of **the interpreter that is
  running this**. Because it is invoked as `src\envs\v2\...\python.exe`, it resolves to v2's env.
  This is what makes one command reusable across v1/v2/v3: swap the `python.exe` and it writes to
  the right place by itself. `pathlib`'s `/` is path join.
- **⑥ `write_text('\n'.join(d))`** — one path per line, overwriting any existing file, so
  re-running is idempotent. **How a `.pth` works:** at interpreter startup the `site` module reads
  every `.pth` in `site-packages` and appends each line to `sys.path`. Paths must be absolute
  because a relative line is resolved against `site-packages` and would point somewhere else. The
  trap: **a nonexistent path is skipped silently** — a typo raises nothing and does nothing.
- **⑦** prints the file written and the paths registered, so ⑥'s silent failure mode is visible.

**Route 2 — no typing** (if the long line is awkward to enter by hand, or gets mangled in transit).
Ask for the directory, then create the file in Notepad:

```powershell
& "src\envs\v2\.venv\Scripts\python.exe" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])"
& "src\envs\v1\.venv\python.exe"          -c "import sysconfig;print(sysconfig.get_paths()['purelib'])"
```

```
# fttl v2 repo paths          <- first line a comment on purpose: absorbs a Notepad UTF-8 BOM
C:\...\model_repos\real\fttl-v2
C:\...\model_repos\real\fttl-v2\<subdir holding .py>
```

**Verify** in a **fresh** process — a `.pth` is only read at interpreter startup:

```powershell
& "src\envs\v2\.venv\Scripts\python.exe" -c "import sys;print([p for p in sys.path if 'model_repos' in p])"
```

An empty list means the file was not read: check for a typo'd path, or for Notepad having saved it
as `fttl.pth.txt` (`dir /b ...\*.pth`).

**Why the same filename in every env.** Each env has its own `site-packages`, so the three
`fttl.pth` files never meet and cannot see each other — which is the whole point of separate envs:

```
src\envs\v1\.venv\Lib\site-packages\fttl.pth   <- v1 repo paths
src\envs\v2\.venv\Lib\site-packages\fttl.pth   <- v2 repo paths
```

A fixed name is better than a per-version one: rewriting simply overwrites, so the operation is
repeatable. Variant names (`fttl_v2.pth`, `fttl_v2_fixed.pth`, …) leave older files behind, stacking
dead paths onto one env's `sys.path` with nothing recording why they are there. If an earlier
attempt left one, delete it — otherwise the repo is registered twice:

```
dir /b "src\envs\v2\.venv\Lib\site-packages\*.pth"
del "src\envs\v2\.venv\Lib\site-packages\fttl_v2.pth"
```

Why subdirectories and not just the root, why **not** to add `__init__.py`, and the name-collision
trap: `src/docs/ENV_MANAGEMENT.md` § "Making the version repo importable".

### v3 stays on the editable install

Not converted to the hand-written form for uniformity, decided 2026-08-06. Where `-e` works it is
the better route: one command does deps + path together, the result is recorded in installed
metadata (`uv pip list` shows it, `RECORD` tracks the file, uninstall is clean), and it is
reproducible from the command alone. The hand-written `.pth` is the **fallback** for envs where
discovery cannot resolve a package — flattening v3 onto it would trade a tool-managed registration
for a hand-managed one and, worse, erase the record of *why* v2 needed different treatment. The
three envs are meant to match in **effect**, not in filename; that is what the table above records.

**And do not rename v3's `.pth`.** That file belongs to the editable install and
is listed in its `RECORD`; renaming it orphans the entry, so a later reinstall or `uv pip uninstall`
recreates or fails to remove the original and the env ends up with the repo on `sys.path` twice.
The three envs are consistent in *effect*, not in filename. If a v3 import turns out to need a
subdirectory the editable install did not register, **add** a `fttl.pth` alongside it rather than
touching the `__editable__` file.

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
| `src/envs/v1/.venv` | **0.72** *(vendored wheel; confirmed in-env 2026-08-08)* | `fttl-v1` |
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

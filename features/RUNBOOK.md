# Runbook — what to run now (company laptop)
## 2026-08-20: 실행 결국 안하고 model pkl 파일만 feature 추출해서, 아직 module 인지 못하는 문제는 해결 못함 

Written 2026-08-20. Ordered: each step unblocks the next. Everything is PowerShell, run from the
**repo root**. Commands are one line each, so no continuation character is involved — if you split
one over lines, PowerShell's continuation character is a backtick `` ` ``, **not** `^` (that is
cmd's, and PowerShell passes it to the script as a literal argument).

Two interpreters appear below and they are not interchangeable:

| what | interpreter | why |
|---|---|---|
| analysis env | `.venv\Scripts\python.exe` | reads parquet, imports `config` (Python 3.11) |
| env-v1 | `src\envs\v1\.venv\python.exe` | the only place v1's pickles unpickle (Python 3.5.6, conda layout — interpreter at the env **root**, not `Scripts\`) |

---

## Step 1 — Unblock v1: make `LVanalytics` importable

**Blocking everything v1.** v1's model pickle is a `sklearn.pipeline.Pipeline` holding custom
transformers defined in `LVanalytics`, a package directly under the v1 repo root. A pickle stores
*references* to classes, not the classes, so the module must be importable or the load fails:

```
ModuleNotFoundError: No module named 'LVanalytics'
```

The fix is to put the v1 repo root on env-v1's `sys.path` via `fttl.pth`.

**1a. Confirm the cause** (temporary, this shell only):

```powershell
$env:PYTHONPATH = "C:\...\model_repos\real\<v1-repo>"
src\envs\v1\.venv\python.exe -c "import LVanalytics; print('OK', LVanalytics.__file__)"
```

**1b. Get the repo path from config** (analysis env — `repo_dir` is filled in on this machine):

```powershell
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');import config;print(config.repo('v1'))"
```

**1c. Write `fttl.pth`** — paste 1b's output as the argument:

```powershell
src\envs\v1\.venv\python.exe -c "import sysconfig,pathlib,sys;r=sys.argv[1];d=[r]+[str(p) for p in pathlib.Path(r).iterdir() if p.is_dir() and list(p.glob('*.py'))];f=pathlib.Path(sysconfig.get_paths()['purelib'])/'fttl.pth';f.write_text('\n'.join(d));print(f);print(*d,sep='\n')" "<path from 1b>"
```

> **Let Python write the file.** `Set-Content` and `>` redirection add a UTF-8 BOM, which corrupts
> the **first** line — and the first line is the repo root. The symptom is exactly what we saw:
> modules deeper in the tree import fine while the top-level package does not. `write_text()` adds
> no BOM.

**1d. Verify in a FRESH process** (a `.pth` is read only at interpreter startup) and clear the
temporary override so you are testing the real thing:

```powershell
$env:PYTHONPATH = ""
src\envs\v1\.venv\python.exe -c "import LVanalytics; print('OK', LVanalytics.__file__)"
```

**If another `No module named X` appears:** `fttl.pth` covers the repo root plus each subdirectory
that *directly* contains `.py` files. A module nested deeper needs its parent appended:

```powershell
src\envs\v1\.venv\python.exe -c "import sysconfig,pathlib,sys;f=pathlib.Path(sysconfig.get_paths()['purelib'])/'fttl.pth';f.write_text(f.read_text().rstrip()+'\n'+sys.argv[1]+'\n');print(f.read_text())" "C:\...\<parent dir of that module>"
```

**Why this matters beyond this runbook:** every script that opens a v1 model fails the same way
until Step 1 passes — `scoring/predict.py`, `scoring/attribute.py`, `training/retrain.py`,
`notebook/real/00_SHAP.ipynb`.

---

## Step 2 — Build `features/registry/v1.json`

v2 and v3 are already done (`extract_features.py` ran successfully). Only v1 is missing, and it
needs a separate script because env-v1 is Python 3.5: `extract_features.py` uses f-strings, future
annotations and `list[str]`, and it falls back to importing `config`, which is 3.7+ itself. Both
are SyntaxErrors there.

Two routes. **They produce the same JSON schema as v2/v3** — the same five keys, no extras.

### Route A — from the transformed matrix (no model pickle, no version env)

This is how `01_export_v1.ipynb` already works: it never opens v1's model, only the DataFrames, and
its own comment records that the transformed table is "39 model-ready cols include claimnumber +
target". So the 37 feature names are already in the exported parquet's columns.

```powershell
.venv\Scripts\python.exe features\extract_features_v1.py --from-matrix src\data\real\inputs\features_v1_train.parquet
```

**Read the `excluded N: [...]` line it prints.** It should exclude exactly two columns — the id and
the target. If fewer than two are excluded the feature list is wrong; pass the real spellings:

```powershell
.venv\Scripts\python.exe features\extract_features_v1.py --from-matrix src\data\real\inputs\features_v1_train.parquet --id-col claimnumber --target-col target
```

Expect **37** model features. `raw_features` is written empty — raw claim columns are a property of
the preprocessor, which this route never sees. Only `check_overlap.py`'s raw-side validation uses
that half; `model_features` is the load-bearing one.

### Route B — from the fitted pickles (needs Step 1)

Gives both name sets. Add `--dry-run` first to see what it found without writing.

```powershell
src\envs\v1\.venv\python.exe features\extract_features_v1.py --model "C:\...\outputs\fasstacker_xgb.pkl" --preprocessor "C:\...\outputs\fttl_pipeline.pkl" --dry-run
```

Verify the filename against `dir` first — `fasstacker` vs `fasttracker` was transcribed from the
training script, not confirmed. A typo fails immediately with a clear message.

The script tries four loaders in writer-first order (`sklearn.externals.joblib` → `joblib` →
`joblib.load_compatibility` → `pickle`) and prints which one worked. env-v1 carries scikit-learn
0.19.1, which bundles its own joblib, so the bundled loader is tried first.

### Check the result either way

```powershell
.venv\Scripts\python.exe -c "import json,pathlib;p=pathlib.Path('features/registry');[print(f.name, len(json.loads(f.read_text())['model_features']), 'model /', len(json.loads(f.read_text())['raw_features']), 'raw') for f in sorted(p.glob('*.json'))]"
```

---

## Step 3 — Validate the hand-confirmed mapping

With all three registries present, this checks every name in the Excel sheet against the names the
pickles actually carry — the typo check. It writes `features/feature_overlap.json`, which the
cross-version SHAP notebook reads.

```powershell
.venv\Scripts\python.exe features\check_overlap.py --excel features\common_features_260804.xlsx --sheet ALL_SORTED
```

Defaults are already that Excel and that sheet, so bare `check_overlap.py` does the same. It
refuses to write when a mapped name is absent from a registry; fix the sheet (or the registry)
rather than forcing past it.

---

## Step 4 — SHAP attributions

Runs each version inside its own env; every path is resolved from `config`. Add `--dry-run` to any
of these to print the commands without running them.

```powershell
.venv\Scripts\python.exe src\scoring\attribute_all.py --split train --rows 5000 --background 500
.venv\Scripts\python.exe src\scoring\attribute_all.py --split v1=val2 v2=test v3=oot --rows 5000 --background 500
.venv\Scripts\python.exe src\scoring\attribute_all.py --split v1=val2 v2=test v3=oot --backend native --out-suffix _native
```

Those are the three configurations `notebook/real/00_shap_attribution.ipynb` reads: concentration
on data the fit saw, on each version's own out-of-time holdout, and the same holdout under the
tree-path-dependent backend so the backend is isolated as the only variable.

**v1 will fail here even after Step 1** — see Known blockers below. v2 and v3 will run; the
notebook reports v1 as "NOT attributed yet" and continues.

---

## Step 5 — Notebooks

**`notebook/real/00_SHAP.ipynb`** — one version, in that version's own kernel. Set `SPLIT` in §0
before running; it defaults to `config.SPLITS[VERSION][0]` (that version's `"train"`). For the
holdout use `config.OOT_SPLIT[VERSION]` (v1 `val2` · v2 `test` · v3 `oot`).

Kernel per env, once:

```powershell
src\envs\v2\.venv\Scripts\python.exe -m ipykernel install --user --name fttl-v2 --display-name "FTTL v2 (env-v2)"
```

**`notebook/real/00_shap_attribution.ipynb`** — the cross-version comparison, in the analysis
kernel (`ml-sfp-detection`). Reads the parquet Step 4 wrote; opens no model. Run Step 4 first or
every run reports nothing on disk and stops with a clear assertion.

---

## Known blockers

| blocker | affects | status |
|---|---|---|
| `LVanalytics` not importable | every v1 model load | **Step 1** fixes it |
| `attribute.py` / `predict.py` are Python 3.6+/3.7+ syntax (21 and 3 occurrences) | running either in env-v1 | not fixed — needs a 3.5 backport, `training/retrain.py` is the worked example |
| env-v1 has **no parquet engine** (no pyarrow, no fastparquet, and neither installs on 3.5) | same two scripts, which call `read_parquet` / `to_parquet` | not fixed — needs CSV/pkl I/O branches alongside the backport |
| — consequence — | **v1 has no SHAP φ**, so cross-version concentration is v2 vs v3 only | open |

The v1 CSV route exists for the same reason: `01_export_v1.ipynb` writes CSV and
`v1_csv_to_parquet.py` converts it in the analysis env. That conversion has already been run and
every v1 parquet exists, which is why Step 2 Route A works today.

---

## If something fails

Report the **full traceback**, not just the last line. This session lost time to a `KeyError: 0`
that looked like a corrupt pickle and was actually `ModuleNotFoundError: No module named
'LVanalytics'` two frames up. The loader chain in `extract_features_v1.py` now prints all four
failures side by side for exactly that reason.

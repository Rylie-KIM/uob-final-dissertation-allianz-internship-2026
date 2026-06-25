# Environment Management in Data Science

## Why Environments Matter

A Python "environment" is an isolated set of installed packages and their versions. Different projects — or different model versions — often require incompatible package versions. Environments prevent those conflicts from breaking each other.

## Common Tools

| Tool | What it does |
|---|---|
| `conda` | Creates isolated environments; manages Python version + packages; widely used in data science |
| `venv` | Python's built-in lightweight environment tool; no conda required |
| `pip` | Installs packages into the active environment |
| `poetry` | Manages dependencies and environments together; stricter version locking |

Most data science teams use **conda** (environment spec: `environment.yml`) or **venv + pip** (spec: `requirements.txt`).

## Typical Workflow

```
1. Create environment
   conda create -n my-project python=3.11
   conda activate my-project

2. Install dependencies
   pip install -r requirements.txt
   # or: conda env create -f environment.yml

3. Pin exact versions (for reproducibility)
   pip freeze > requirements.txt
   # or: conda env export > environment.yml

4. Share the spec file
   Anyone else runs: pip install -r requirements.txt
   → identical environment, guaranteed
```

## Handling Multiple Model Versions with Different Dependencies

When two model versions require conflicting packages (e.g., v1 was trained with XGBoost 1.5, v2 with XGBoost 2.1), they cannot coexist in one environment. The standard approach:

```
env-v1/          ← conda env or venv for v1 only
  xgboost==1.5
  scikit-learn==1.0

env-v2v3/        ← shared env for v2 and v3
  xgboost==2.1
  scikit-learn==1.4
```

Each env has its own Python interpreter (`env-v1/bin/python`, `env-v2v3/bin/python`). To run v1 inference from within the v2/v3 environment, a **subprocess** is spawned using v1's interpreter — the two environments never share a process.

## Spec Files in This Project

```
src/model/envs/
├── v1.yml                   ← conda env spec for v1
├── v1_requirements.txt      ← pip-pinned v1 dependencies
├── v2.yml                   ← conda env spec for v2
├── v2_requirements.txt      ← pip-pinned v2 dependencies
├── v3.yml                   ← conda env spec for v3
└── v3_requirements.txt      ← pip-pinned v3 dependencies
```

Each version has its own spec file even if two versions currently have identical dependencies. This ensures that updating or retraining one version does not silently affect another. `SubprocessModelLoader` reads the relevant spec to locate the correct Python interpreter before spawning a child process. See `DESIGN.md` for full detail.

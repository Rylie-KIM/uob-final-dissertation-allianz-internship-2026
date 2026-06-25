# Design Pattern

## Strategy Pattern

Each interchangeable component is encapsulated as a separate class behind a common interface. The core classes (`SFPDetector`, `SFPMitigator`) hold references to these strategies and delegate work to them — they never contain the implementation directly.

This means swapping synthetic data for real data, or swapping one detection algorithm for another, requires changing only the injected class — not the core logic.

## Three Strategy Axes

| Axis | Interface | Implementations |
|---|---|---|
| Data loading | `DataLoader` | `SyntheticDataLoader` → `RealDataLoader` |
| Model loading | `ModelLoader` | `InProcessModelLoader`, `SubprocessModelLoader` |
| Detection | `DetectionAlgorithm` | TBD (pending research) |
| Investigation policy | `InvestigationPolicy` | TBD (pending research) |
| Training data correction | `TrainingDataCorrector` | TBD (pending research) |

## Class Responsibilities

**`SFPPipeline`** — orchestrates the full run. Calls detector, checks result, calls mitigator if SFP is detected.

**`SFPDetector`** — diagnosis. Loads data and runs detection algorithms. Returns a `DetectionReport`.

**`SFPMitigator`** — prescription. Takes the report and applies mitigation: updates investigation policy and corrects training data.

**`DataLoader`** — abstract base for data ingestion. Concrete implementations differ; callers do not.

**`ModelLoader`** — abstract base for loading a serialised model file and running inference. Hides environment differences from the rest of the pipeline. See below.

---

## Model Environment Isolation

### The problem

All three model versions (v1, v2, v3) are preserved as serialised files within Allianz's internal systems. However, **v1 has different library dependencies from v2 and v3** (exact versions TBC — likely a different XGBoost or scikit-learn release). v2 and v3 share the same environment. Loading v1 and v2/v3 in the same Python process will cause import conflicts.

Model versions will also continue to be updated over time. A design that hardcodes a separate loader class per version (e.g., `V1SubprocessModelLoader`, `V2V3ModelLoader`) would require a new class for each new version — that is the wrong level of abstraction.

### Design decision — env_spec as parameter, not as subclass

The `ModelLoader` interface has exactly two concrete implementations:

```
ModelLoader (abstract)
├── InProcessModelLoader(model_path)
│     └── loads model in the current process — use when env matches
└── SubprocessModelLoader(model_path, env_spec)
      └── spawns a child process using the Python interpreter
          specified in env_spec — use when env differs
```

The **environment** is not encoded in the class — it is passed as a parameter (`env_spec`), which is a path to a version-specific config file under `src/model/envs/`. When a new model version arrives with yet another dependency set, no new class is needed: add an env spec file, point `SubprocessModelLoader` at it.

### env_spec file format

Each env spec is a small YAML file in `src/model/envs/`:

```yaml
# src/model/envs/v1.yml
env_name: env-v1
python_executable: /path/to/env-v1/bin/python   # resolved at deploy time
requirements_file: src/model/envs/v1_requirements.txt
```

```yaml
# src/model/envs/v2.yml
env_name: env-v2
python_executable: /path/to/env-v2/bin/python
requirements_file: src/model/envs/v2_requirements.txt
```

```yaml
# src/model/envs/v3.yml
env_name: env-v3
python_executable: /path/to/env-v3/bin/python
requirements_file: src/model/envs/v3_requirements.txt
```

Each version has its own spec even if two versions currently share identical dependencies. This keeps each version's environment independently modifiable — retraining or upgrading v3 will not affect v2's pinned dependencies. `SubprocessModelLoader` reads `python_executable` from the spec to spawn the worker. `requirements_file` is the source of truth for reproducibility. New version → new `.yml` + new `_requirements.txt`; no code changes required.

### Implementation

**`src/model/base.py`** — abstract interface

```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class ModelLoader(ABC):
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(total_loss) for each row. Shape: (n,)."""
```

---

**`src/model/inprocess.py`** — for models in the same env as the app

```python
import joblib
import numpy as np
import pandas as pd
from .base import ModelLoader

class InProcessModelLoader(ModelLoader):
    def __init__(self, model_path: str):
        self._model = joblib.load(model_path)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]
```

---

**`src/model/subprocess_loader.py`** — for models in a different env

```python
import json
import subprocess
import tempfile
import os
import numpy as np
import pandas as pd
import yaml
from .base import ModelLoader

class SubprocessModelLoader(ModelLoader):
    def __init__(self, model_path: str, env_spec: str):
        self._model_path = model_path
        with open(env_spec) as f:
            spec = yaml.safe_load(f)
        self._python = spec["python_executable"]   # e.g. /path/to/env-v1/bin/python

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            X_path = tmp.name
        try:
            np.save(X_path, X.values)
            result = subprocess.run(
                [self._python, "src/model/worker.py", self._model_path, X_path],
                capture_output=True, text=True, check=True
            )
            return np.array(json.loads(result.stdout))
        finally:
            os.unlink(X_path)
```

---

**`src/model/worker.py`** — version-agnostic; runs inside the target env

```python
import sys
import json
import numpy as np
import joblib

# Called as: <python> worker.py <model_path> <X_path>
model_path = sys.argv[1]
X_path     = sys.argv[2]

model  = joblib.load(model_path)
X      = np.load(X_path, allow_pickle=False)
scores = model.predict_proba(X)[:, 1]

print(json.dumps(scores.tolist()))   # parent reads this from stdout
```

This script has no knowledge of which version it is running — it just loads whatever model file it is given. It never needs to be edited when a new version is added.

---

### Subprocess data flow

```
Main process (env-v2v3)
│
│  1. Serialise X to a temp file                         [subprocess_loader.py]
│     np.save("/tmp/X_abc123.npy", X.values)
│        └─ .npy: NumPy's binary format — faster and more compact than
│             CSV or JSON; preserves array shape and dtype exactly
│
│  2. Spawn child process (blocking — main process waits) [subprocess_loader.py]
│     subprocess.run([
│         "/env-v1/bin/python",    ← v1-specific Python interpreter
│         "src/model/worker.py",
│         "models/v1.pkl",         ← model file path (passed as argument)
│         "/tmp/X_abc123.npy",     ← feature matrix path (passed as argument)
│     ])
│
│         ┌──────────────────────────────────────┐
│         │  Child process (env-v1)              │  [worker.py]
│         │  — separate memory space,            │
│         │    no shared state with parent       │
│         │                                      │
│         │  model = joblib.load("v1.pkl")       │
│         │  X     = np.load("X_abc123.npy")     │  ← reads the temp file
│         │  scores = model.predict_proba(X)[:,1]│     written by parent
│         │  print(json.dumps(scores.tolist()))  │
│         └──────────────────────────────────────┘
│                          │
│                     stdout (JSON string)
│                          │
│  3. Parse stdout into array                            [subprocess_loader.py]
│     scores = np.array(json.loads(result.stdout))
│
│  4. Clean up temp file                                 [subprocess_loader.py]
│     os.unlink("/tmp/X_abc123.npy")
│
└── Return scores (np.ndarray) — caller has no idea a subprocess was used
```

**Why v2/v3 use InProcess instead of subprocess:**

v2 and v3 run in the same environment as the main app (`env-v2v3`), so there is no library conflict. Subprocess isolation is only justified when environments differ — using it unnecessarily adds overhead with no benefit.

| | InProcessModelLoader (v2, v3) | SubprocessModelLoader (v1) |
|---|---|---|
| Process spawn cost | None | Yes |
| File I/O (write + read X.npy) | None | Yes |
| stdout parsing | None | Yes |
| Isolation needed | No (same env) | Yes (different env) |

Use InProcess whenever possible. Subprocess is reserved for cases where environment isolation is unavoidable — as with v1.

---

### How the caller uses them

The `SFPDetector` (or pipeline setup) instantiates loaders and calls `predict_proba` identically regardless of which env each model lives in:

```python
# pipeline.py or main.py
from src.model.inprocess import InProcessModelLoader
from src.model.subprocess_loader import SubprocessModelLoader

v1_loader = SubprocessModelLoader(
    model_path="models/v1.pkl",
    env_spec="src/model/envs/v1.yml",   # points to env-v1 python
)
v2_loader = InProcessModelLoader(model_path="models/v2.pkl")
v3_loader = InProcessModelLoader(model_path="models/v3.pkl")

# Uniform interface — SFPDetector does not know or care about envs
v1_scores = v1_loader.predict_proba(X)   # runs in subprocess (env-v1)
v2_scores = v2_loader.predict_proba(X)   # runs in current process
v3_scores = v3_loader.predict_proba(X)   # runs in current process
```

Adding v4 with a new dependency set requires no code changes — only:

```python
v4_loader = SubprocessModelLoader(
    model_path="models/v4.pkl",
    env_spec="src/model/envs/v4.yml",   # new file; new env
)
```

### Adding a new model version (e.g., v4)

1. Add `src/model/envs/v4.yml` (point `python_executable` at the v4 env)
2. Add `src/model/envs/v4_requirements.txt` (pin the v4 dependencies)
3. Instantiate: `SubprocessModelLoader(model_path="v4.pkl", env_spec="src/model/envs/v4.yml")`

No new class. No changes to existing code.

### Trade-offs

| Approach | Pro | Con |
|---|---|---|
| Parameterised env_spec (chosen) | Scales to any number of versions without new classes; env spec files are version-controllable | Requires env Python paths to be configured at deploy time |
| Hardcoded subclass per version | Simple for two versions | Requires a new class for every new version; logic duplicated |
| Docker containers | Full isolation; fully reproducible | Heavyweight for a research app; requires Docker daemon |
| Env-switching via `importlib` | No subprocess | Unreliable; can corrupt sys.modules |

The subprocess + env_spec approach is sufficient for this research application where v1 inference is called at most a few times per pipeline run (cross-version score comparison, not real-time scoring).

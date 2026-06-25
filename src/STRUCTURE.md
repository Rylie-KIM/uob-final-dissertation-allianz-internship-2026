# App Structure

```
src/
├── main.py                  # Entry point
├── pipeline.py              # SFPPipeline — top-level orchestrator
│
├── detector/
│   ├── sfp_detector.py      # SFPDetector — runs detection algorithms
│   └── algorithm/           # Detection strategies (TBD)
│
├── mitigator/
│   ├── sfp_mitigator.py     # SFPMitigator — runs mitigation strategies
│   ├── policy/              # Investigation policies (TBD)
│   └── corrector/           # Training data correction strategies (TBD)
│
├── model/
│   ├── base.py              # ModelLoader abstract base class
│   ├── inprocess.py         # InProcessModelLoader — direct load, same env as app
│   ├── subprocess_loader.py # SubprocessModelLoader — spawns isolated child process
│   ├── worker.py            # Version-agnostic worker; runs inside target env
│   └── envs/                # One env spec file per model version
│       ├── v1.yml           # env-v1: python_executable + requirements_file path
│       ├── v1_requirements.txt
│       ├── v2v3.yml         # env-v2v3: shared by v2 and v3
│       └── v2v3_requirements.txt
│
└── data/
    ├── base.py              # DataLoader abstract base class
    ├── synthetic.py         # SyntheticDataLoader
    └── real.py              # RealDataLoader (to be implemented)
```

## Data Flow

```
DataLoader
    │
    ▼
SFPDetector → DetectionReport
                    │
                    ▼ (if sfp_detected)
              SFPMitigator
                    │
          ┌───────expected methods──────┐
          ▼                              ▼
  claim investigation           dataset correction
```

## Model Loading Flow

```
SFPDetector needs model scores
        │
        ▼
   ModelLoader.predict_proba(X)
        │
        ├── InProcessModelLoader          SubprocessModelLoader(env_spec="envs/v1.yml")
        │   (v2, v3 — same env as app)         (v1 — isolated env)
        │         │                                     │
        │   direct pickle load              reads python_executable from v1.yml
        │   predict_proba(X)               spawns: /env-v1/bin/python worker.py
        │         │                                     │
        │         │                         worker loads model, runs predict_proba
        │         │                         returns scores via stdout
        │         │                                     │
        └─────────────── np.ndarray ──────────────────┘
                              │
                        SFPDetector
              (uniform interface — unaware of env difference)
```

## Environment Map

| Component | Runs in | Notes |
|---|---|---|
| `main.py`, pipeline, detector, mitigator | `env-v2v3` | Primary app runtime |
| `InProcessModelLoader` (v2, v3) | `env-v2v3` | Direct import; same env as app |
| `SubprocessModelLoader` (v1) | `env-v2v3` | Spawns child; parent stays in v2v3 env |
| `worker.py` (child process) | env from `env_spec` | e.g. `env-v1` for v1 — isolated |

## Adding a New Model Version

When a new version (e.g., v4) arrives with different dependencies:

1. Add `src/model/envs/v4.yml` — set `python_executable` and `requirements_file`
2. Add `src/model/envs/v4_requirements.txt` — pin the v4 dependencies
3. Instantiate `SubprocessModelLoader(model_path="...", env_spec="src/model/envs/v4.yml")`

No new class required. `worker.py` is version-agnostic and runs unchanged.

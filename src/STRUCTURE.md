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

# Design Pattern

## Strategy Pattern

Each interchangeable component is encapsulated as a separate class behind a common interface. The core classes (`SFPDetector`, `SFPMitigator`) hold references to these strategies and delegate work to them — they never contain the implementation directly.

This means swapping synthetic data for real data, or swapping one detection algorithm for another, requires changing only the injected class — not the core logic.

## Three Strategy Axes

| Axis | Interface | Implementations |
|---|---|---|
| Data loading | `DataLoader` | `SyntheticDataLoader` → `RealDataLoader` |
| Detection | `DetectionAlgorithm` | TBD (pending research) |
| Investigation policy | `InvestigationPolicy` | TBD (pending research) |
| Training data correction | `TrainingDataCorrector` | TBD (pending research) |

## Class Responsibilities

**`SFPPipeline`** — orchestrates the full run. Calls detector, checks result, calls mitigator if SFP is detected.

**`SFPDetector`** — diagnosis. Loads data and runs detection algorithms. Returns a `DetectionReport`.

**`SFPMitigator`** — prescription. Takes the report and applies mitigation: updates investigation policy and corrects training data.

**`DataLoader`** — abstract base for data ingestion. Concrete implementations differ; callers do not.

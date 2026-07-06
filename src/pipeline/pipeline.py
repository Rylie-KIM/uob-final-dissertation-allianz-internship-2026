"""SFPPipeline — orchestrates the full run: detect -> (if loop) mitigate -> re-evaluate.

Runs in the ANALYSIS env and never loads a model. For the two model-touching steps (baseline
scoring, mitigated retrain + re-scoring) it shells out to that version's OWN env (src/envs/vX), so
the analysis process stays model-free and no cross-version import conflict is ever possible.
Detection and mitigation happen inline via the injected `SFPDetector` / `SFPMitigator` strategies.

Per version v in {v1, v2, v3}:
  1. predict  (env-v)     baseline scores            src/models/synthetic/baseline/v.pkl
  2. detect   (analysis)  SFPDetector  BEFORE
  3. mitigate (analysis)  SFPMitigator -> corrected labels+weights
  4. retrain  (env-v)     src/models/synthetic/mitigated/v.pkl  (weighted; preprocessing held fixed)
  5. predict  (env-v)     mitigated scores
  6. detect   (analysis)  SFPDetector  AFTER
  7. report   metric before vs after (+ oracle-validation AUC, kept clearly separate)

  .venv/bin/python -m pipeline.pipeline        # (run from repo root, src on sys.path)
  .venv/bin/python src/pipeline/pipeline.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from detector import ResidualPeakAlgorithm, SFPDetector          # noqa: E402
from detector.sfp_detector import DetectionReport                # noqa: E402
from mitigator import IPSCorrector, SFPMitigator                 # noqa: E402

# source-partitioned artefacts (synthetic): inputs = fixed materials, stages = pure outputs
DATA = ROOT / "src/data/synthetic"
INPUTS = DATA / "inputs"              # features_<v> + labels_<v>
DETECTION = DATA / "detection"       # <v>_scores           (baseline scores -> detector)
MITIGATION = DATA / "mitigation"     # <v>_corrected        (corrected target <- mitigator)
REEVAL = DATA / "reeval"             # <v>_mitigated_scores (post-retrain scores)
MODELS = ROOT / "src/models/synthetic"   # baseline/ + mitigated/ pkls (out of data/)


def _env_py(v: str) -> str:
    return str(ROOT / f"src/envs/{v}/.venv/bin/python")


def _baseline_pkl(v: str) -> str:
    return str(MODELS / "baseline" / f"{v}.pkl")


def _mitigated_pkl(v: str) -> str:
    return str(MODELS / "mitigated" / f"{v}.pkl")


def _run(env_python: str, script: str, *args: str) -> None:
    # `script` is a repo-root-relative path (scoring and training live in separate packages).
    subprocess.run([env_python, str(ROOT / script), *args], check=True, cwd=ROOT)


@dataclass
class CycleResult:
    version: str
    before: DetectionReport
    after: DetectionReport
    ips_diag: dict
    auc_before: float          # oracle validation (NOT part of detection) — see note in cycle()
    auc_after: float


class SFPPipeline:
    """Detect -> mitigate -> re-evaluate for one or more versions, using injected strategies."""

    def __init__(self, detector: SFPDetector, mitigator: SFPMitigator) -> None:
        self.detector = detector
        self.mitigator = mitigator
        for d in (DETECTION, MITIGATION, REEVAL, MODELS / "mitigated"):
            d.mkdir(parents=True, exist_ok=True)

    def cycle(self, v: str) -> CycleResult:
        feats = str(INPUTS / f"features_{v}.parquet")
        labels = pd.read_parquet(INPUTS / f"labels_{v}.parquet")

        # 1. baseline scoring (version env)
        base_scores_p = str(DETECTION / f"{v}_scores.parquet")
        _run(_env_py(v), "src/scoring/predict.py", "--model", _baseline_pkl(v),
             "--features", feats, "--version", v, "--out", base_scores_p)
        base_scores = pd.read_parquet(base_scores_p)

        # 2. detect BEFORE (analysis, oracle-free)
        before = self.detector.run(base_scores, labels, v)

        # 3. mitigate — corrected labels + weights (analysis); only if a loop was flagged
        corrected, ips_diag = self.mitigator.correct_training_data(pd.read_parquet(feats), labels)
        corr_p = str(MITIGATION / f"{v}_corrected.parquet")
        corrected.to_parquet(corr_p, index=False)

        # 4. retrain mitigated model (version env; preprocessing reused/fixed)
        mit_pkl = _mitigated_pkl(v)
        _run(_env_py(v), "src/training/retrain.py", "--baseline", _baseline_pkl(v),
             "--features", feats, "--labels", corr_p, "--version", v, "--out-model", mit_pkl)

        # 5. re-score with mitigated model (version env)
        mit_scores_p = str(REEVAL / f"{v}_mitigated_scores.parquet")
        _run(_env_py(v), "src/scoring/predict.py", "--model", mit_pkl,
             "--features", feats, "--version", v, "--out", mit_scores_p)
        mit_scores = pd.read_parquet(mit_scores_p)

        # 6. detect AFTER (analysis)
        after = self.detector.run(mit_scores, labels, v)

        # 7. oracle validation — NOT part of the detector: do scores track the TRUE outcome better?
        truth = labels["true_garage_outcome"].to_numpy()
        sc = f"model_{v}_score"
        return CycleResult(
            version=v, before=before, after=after, ips_diag=ips_diag,
            auc_before=roc_auc_score(truth, base_scores[sc]),
            auc_after=roc_auc_score(truth, mit_scores[sc]),
        )

    def run(self, versions: tuple[str, ...] = ("v1", "v2", "v3")) -> list[CycleResult]:
        return [self.cycle(v) for v in versions]


def _report(results: list[CycleResult]) -> None:
    print("\n" + "=" * 78)
    print("SFP CYCLE REPORT  —  residual-zero peak (detector, oracle-free)  ·  detect->mitigate->reeval")
    print("=" * 78)
    print(f"{'ver':<4} {'peak0_before':>12} {'sfp?':>5} {'peak0_after':>12} {'Δpeak0':>8}"
          f" {'IPS drop/keep':>14} {'AUC (oracle val)':>20}")
    for r in results:
        b, a, d = r.before.metrics, r.after.metrics, r.ips_diag
        dpeak = a["peak0"] - b["peak0"]
        dropkeep = f"{d['n_scrapped_dropped']}/{d['n_kept_honest']}"
        auc = f"{r.auc_before:.3f}->{r.auc_after:.3f}"
        print(f"{r.version:<4} {b['peak0']:>12.3f} {str(r.before.sfp_detected):>5} {a['peak0']:>12.3f}"
              f" {dpeak:>+8.3f} {dropkeep:>14} {auc:>20}")
    print("=" * 78)
    print("peak0 lower after mitigation  = loop loosened (model stops echoing forced labels).")
    print("AUC higher after (oracle validation) = mitigated scores track the TRUE outcome better.")


def main() -> None:
    pipeline = SFPPipeline(
        detector=SFPDetector(ResidualPeakAlgorithm()),
        mitigator=SFPMitigator(IPSCorrector()),
    )
    _report(pipeline.run())


if __name__ == "__main__":
    main()

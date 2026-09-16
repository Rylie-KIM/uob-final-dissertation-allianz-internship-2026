"""SFPPipeline — orchestrates the full run: detect -> (if loop) mitigate -> re-evaluate.

Runs in the ANALYSIS env and never loads a model. For the two model-touching steps (baseline
scoring, mitigated retrain + re-scoring) it shells out to that version's OWN env (src/envs/vX), so
the analysis process stays model-free and no cross-version import conflict is ever possible.
Detection and mitigation happen inline via the injected `DetectionAlgorithm` / `TrainingDataCorrector`
strategies, called directly — there is no `SFPDetector`/`SFPMitigator` wrapper (both deleted
2026-09-16: neither was imported by any real-data notebook, and this pipeline was their only
caller, so it now holds the strategies itself, exactly like the notebooks already do).

Per version v in {v1, v2, v3}:
  1. predict  (env-v)     baseline scores        config.path("model", v)
  2. detect   (analysis)  DetectionAlgorithm.detect()  BEFORE
  3. mitigate (analysis)  TrainingDataCorrector.correct() -> corrected labels+weights
  4. retrain  (env-v)     config.path("mitigated", v)  (weighted; preprocessing untouched)
  5. predict  (env-v)     mitigated scores
  6. detect   (analysis)  DetectionAlgorithm.detect()  AFTER
  7. report   metric before vs after

Every path comes from config.path(kind, version) — nothing here names a file. See
src/docs/STRUCTURE.md "Paths are resolved by KIND".

Oracle validation (AUC against an independently verified outcome) is OPTIONAL and off by default,
because real Allianz data has no such column: scrapped cars are never sent to a garage, so the
verified outcome does not exist to be measured against. Pass --oracle-col only where one genuinely
exists.

--split is required: features / targets / scores exist only per split, and the whole cycle —
detect, mitigate, retrain, re-score — has to happen on one population or the before/after
comparison is between two different samples.

  .venv/bin/python -m pipeline.pipeline --split test         # (from repo root, src on sys.path)
  .venv/bin/python src/pipeline/pipeline.py --split test --versions v2
  .venv/bin/python src/pipeline/pipeline.py --split v2=test v3=oot
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import config                                                    # noqa: E402
import schema                                                    # noqa: E402
from detector import DetectionAlgorithm, ResidualPeakAlgorithm   # noqa: E402
from mitigator import ReweightCorrector, TrainingDataCorrector   # noqa: E402
from mitigator.corrector.reweight import SCHEMES                 # noqa: E402


def _run(env_python: str, script: str, *args: str) -> None:
    # `script` is a repo-root-relative path (scoring and training live in separate packages).
    subprocess.run([env_python, str(ROOT / script), *args], check=True, cwd=ROOT)


@dataclass
class DetectionReport:
    """Result of running one detection algorithm on one model version.

    Was `detector.sfp_detector.DetectionReport` until `SFPDetector` (its only producer) was
    deleted 2026-09-16 — this pipeline is the sole caller, so the type moved here with it.
    """

    version: str
    metrics: dict
    sfp_detected: bool

    @property
    def score(self) -> float:
        """The headline metric value (e.g. peak0), for convenient before/after comparison."""
        return self.metrics.get(self.metrics.get("_metric_name", ""), float("nan"))


def _detect(algorithm: DetectionAlgorithm, scores: pd.DataFrame, labels: pd.DataFrame,
            version: str) -> DetectionReport:
    """Run `algorithm` and package the outcome as a `DetectionReport` — was `SFPDetector.run()`."""
    metrics = algorithm.detect(scores, labels, score_col=f"model_{version}_score")
    metrics["_metric_name"] = algorithm.metric_name
    return DetectionReport(version=version, metrics=metrics, sfp_detected=bool(metrics["sfp_detected"]))


@dataclass
class CycleResult:
    version: str
    before: DetectionReport
    after: DetectionReport
    corrector_diag: dict
    auc_before: float | None   # oracle validation — None unless an oracle column was supplied
    auc_after: float | None


class SFPPipeline:
    """Detect -> mitigate -> re-evaluate for one or more versions, using injected strategies."""

    def __init__(
        self,
        algorithm: DetectionAlgorithm,
        corrector: TrainingDataCorrector,
        source: str = "real",
        oracle_col: str | None = None,
        splits: dict[str, str] | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.corrector = corrector
        self.source = source
        self.oracle_col = oracle_col
        self.splits = splits or {}

    def _path(self, kind: str, v: str) -> str:
        """Resolve by kind, carrying this run's split for the kinds that have one.

        The split is looked up per version because the names differ (v3's holdout is "oot",
        v2's is "test"); kinds outside config.SPLIT_KINDS — model, log — take none.
        """
        split = self.splits.get(v) if kind in config.SPLIT_KINDS else None
        if kind in config.SPLIT_KINDS and split is None:
            raise ValueError(
                f"[{v}] no split chosen, but {kind!r} exists only per split. Construct "
                f"SFPPipeline(..., splits={{{v!r}: <one of {config.SPLITS[v]}>}})."
            )
        p = config.path(kind, v, self.source, split=split)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    def cycle(self, v: str) -> CycleResult:
        env_py = str(config.python_bin(v))
        print(f"[{v}] cycle on split {self.splits.get(v)!r}")
        feats = self._path("processed_inputs", v)
        targets = pd.read_parquet(self._path("targets", v))

        # 1. baseline scoring (version env). --features-json is the fallback column list for an
        #    estimator that exposes no feature names; the booster answers first when it can.
        registry = str(config.registry_path(v))
        base_scores_p = self._path("scores", v)
        _run(env_py, "src/scoring/predict.py", "--model", self._path("model", v),
             "--features", feats, "--version", v, "--out", base_scores_p,
             "--features-json", registry)
        base_scores = pd.read_parquet(base_scores_p)

        # 2. detect BEFORE (analysis, oracle-free)
        before = _detect(self.algorithm, base_scores, targets, v)

        # 3. mitigate — corrected labels + weights (analysis); only if a loop was flagged
        #    The corrector needs the recorded treatment (`decision`), which plain `targets` does
        #    not carry — 03_01_corrector_inputs joins it on and writes `corrector_targets`.
        #    feature_cols: the matrix also carries the target (v3's its own predictions too), so the
        #    transport model is told which columns are model inputs rather than inferring them.
        corr_targets_p = pathlib.Path(self._path("corrector_targets", v))
        if not corr_targets_p.is_file():
            raise SystemExit(
                f"\n[{v}] no corrector_targets at\n    {corr_targets_p}\n"
                f"Run notebook/real/mitigation/03_01_corrector_inputs.ipynb for this split first "
                f"— it joins the recorded treatment (decision) onto the targets.\n"
            )
        corrected, corrector_diag = self.corrector.correct(
            pd.read_parquet(feats), pd.read_parquet(corr_targets_p),
            feature_cols=config.model_features(v))
        corr_p = self._path("corrected", v)
        corrected.to_parquet(corr_p, index=False)

        # 4. retrain mitigated model (version env; the preprocessor is never touched)
        mit_pkl = self._path("mitigated", v)
        _run(env_py, "src/training/retrain.py", "--baseline", self._path("model", v),
             "--features", feats, "--labels", corr_p, "--version", v, "--out-model", mit_pkl,
             "--features-json", registry)

        # 5. re-score with mitigated model (version env)
        mit_scores_p = self._path("reeval_scores", v)
        _run(env_py, "src/scoring/predict.py", "--model", mit_pkl,
             "--features", feats, "--version", v, "--out", mit_scores_p,
             "--features-json", registry)
        mit_scores = pd.read_parquet(mit_scores_p)

        # 6. detect AFTER (analysis)
        after = _detect(self.algorithm, mit_scores, targets, v)

        # 7. oracle validation — NOT detection, and NOT available on real data. Only runs when an
        #    independently verified outcome column was explicitly named.
        auc_before = auc_after = None
        if self.oracle_col:
            if self.oracle_col not in targets.columns:
                raise KeyError(
                    f"[{v}] oracle column {self.oracle_col!r} is not in "
                    f"{self._path('targets', v)}. Real Allianz data has no verified outcome for "
                    f"scrapped cars — omit --oracle-col."
                )
            truth = targets[self.oracle_col].to_numpy()
            sc = f"model_{v}_score"
            auc_before = roc_auc_score(truth, base_scores[sc])
            auc_after = roc_auc_score(truth, mit_scores[sc])

        return CycleResult(version=v, before=before, after=after, corrector_diag=corrector_diag,
                           auc_before=auc_before, auc_after=auc_after)

    def run(self, versions=config.VERSION_LABELS) -> list[CycleResult]:
        return [self.cycle(v) for v in versions]


def _report(results: list[CycleResult]) -> None:
    print("\n" + "=" * 78)
    print("SFP CYCLE REPORT  —  residual-zero peak (detector, oracle-free)  ·  detect->mitigate->reeval")
    print("=" * 78)
    print(f"{'ver':<4} {'peak0_before':>12} {'sfp?':>5} {'peak0_after':>12} {'Δpeak0':>8}"
          f" {'garage/U':>14} {'w_pos':>6} {'AUC (oracle val)':>20}")
    for r in results:
        b, a, d = r.before.metrics, r.after.metrics, r.corrector_diag
        dpeak = a["peak0"] - b["peak0"]
        garage_u = f"{d['n_garage']}/{d['n_scrapped']}"
        auc = "n/a" if r.auc_before is None else f"{r.auc_before:.3f}->{r.auc_after:.3f}"
        print(f"{r.version:<4} {b['peak0']:>12.3f} {str(r.before.sfp_detected):>5} {a['peak0']:>12.3f}"
              f" {dpeak:>+8.3f} {garage_u:>14} {d['weighted_pos_share']:>6.3f} {auc:>20}")
    print("=" * 78)
    print("peak0 lower after mitigation  = loop loosened (model stops echoing forced labels).")
    if all(r.auc_before is None for r in results):
        print("AUC 'n/a': no verified outcome exists for scrapped cars — this is the whole problem,")
        print("not a gap in the run. Detection above is oracle-free and does not need one.")
    print("garage/U: verified vs forced-label (model-scrapped) rows — the corrector keeps both and")
    print("reweights; w_pos = weighted share of label 1 the retrain sees (scheme in the _meta.json).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="real", choices=("real", "synthetic"))
    p.add_argument("--versions", nargs="+", default=list(config.VERSION_LABELS))
    p.add_argument("--oracle-col", default=None,
                   help="independently verified outcome column; real data has none")
    p.add_argument("--split", nargs="+", default=None,
                   help="split to run on: one name for all versions, or v2=test v3=oot")
    p.add_argument("--scheme", default="transport", choices=SCHEMES,
                   help="ReweightCorrector scheme; naive = the drop-only refit baseline arm")
    a = p.parse_args()

    pipeline = SFPPipeline(
        algorithm=ResidualPeakAlgorithm(),
        corrector=ReweightCorrector(scheme=a.scheme),
        source=a.source,
        oracle_col=a.oracle_col,
        splits=config.resolve_splits(a.split, a.versions),
    )
    _report(pipeline.run(a.versions))


if __name__ == "__main__":
    main()

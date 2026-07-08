from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

from detector.algorithm.base import DetectionAlgorithm

# --------------------------------------------------------------------------------------------------
# Oracle-free reading primitives (shared by notebooks + detector.report).
#
# The SFP residual is (genuine 0/1 label − model score) on the common garage-verified rows. Its KDE
# height exactly at 0, ψ = peak0, is the threshold-free loop signal: a reinforcing loop piles mass on
# residual≈0 (score already matches the forced label), pushing the peak up. classification_metrics /
# residual_peak_report add the operating-point view (AUC + precision/recall + confusion matrix) so a
# single table shows both the loop signal and what precision≥0.985 actually buys out-of-time.
# --------------------------------------------------------------------------------------------------


def residual(y_true: ArrayLike, scores: ArrayLike) -> np.ndarray:
    """SFP residual = genuine 0/1 label − model score, elementwise."""
    return np.asarray(y_true, dtype=float) - np.asarray(scores, dtype=float)


def peak0(r: ArrayLike, *, bw_method=None) -> float:
    """ψ — KDE height of the residual distribution exactly at 0 (the residual-peak loop signal)."""
    return float(gaussian_kde(np.asarray(r, dtype=float), bw_method=bw_method)(0.0)[0])


def classification_metrics(y_true: ArrayLike, scores: ArrayLike, thr: float) -> dict:
    """AUC (threshold-free) + precision/recall + confusion matrix at decision threshold ``thr``.

    FP = repairable car scrapped (the costly error precision≥0.985 guards against);
    TP = true total loss correctly scrapped. Returns keys AUC, precision, recall, TN, FP, FN, TP.
    """
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores, dtype=float) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    try:
        auc = float(roc_auc_score(y, np.asarray(scores, dtype=float)))
    except ValueError:  # only one class present in y_true
        auc = float("nan")
    return {
        "AUC": auc,
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def residual_peak_report(
    scores_by_version: dict[str, ArrayLike],
    y_true: ArrayLike,
    *,
    tau: float,
    round_to: int = 3,
) -> pd.DataFrame:
    """One row per version: ψ = peak0 + AUC/precision/recall/CM at ``tau`` on the common rows.

    ``scores_by_version`` = {version label: scores on the shared garage-verified rows};
    ``y_true`` = genuine 0/1 outcome on those same rows. Index = version labels, so
    ``report["peak0"]`` is a per-version Series ready for plot_metric_trajectory.
    """
    y = np.asarray(y_true).astype(int)
    rows = {}
    for name, scores in scores_by_version.items():
        cm = classification_metrics(y, scores, tau)
        rows[name] = {
            "peak0": round(peak0(residual(y, scores)), round_to),
            "AUC": round(cm["AUC"], round_to),
            "precision": round(cm["precision"], round_to),
            "recall": round(cm["recall"], round_to),
            "TN": cm["TN"],
            "FP": cm["FP"],
            "FN": cm["FN"],
            "TP": cm["TP"],
        }
    return pd.DataFrame(rows).T


class ResidualPeakAlgorithm(DetectionAlgorithm):
    # measure KDE height of the residual (observed - score) exactly at 0 
    metric_name = "peak0"

    # alert - peak0 threshold above which the loop is called "reinforcing". 
    # eps - half-width for the backup "fraction near zero" statistic.

    def __init__(self, alert: float = 1.5, eps: float = 0.1) -> None:
        self.alert = alert
        self.eps = eps

    def detect(
        self,
        scores: pd.DataFrame,
        labels: pd.DataFrame,
        score_col: str,
        id_col: str = "claim_id",
        outcome_col: str = "observed_outcome",
    ) -> dict:
        m = scores.merge(labels[[id_col, outcome_col]], on=id_col)
        r = residual(m[outcome_col], m[score_col])

        peak = peak0(r)  # density of the residual exactly at 0
        frac_near_zero = float((np.abs(r) < self.eps).mean())
        return {
            "n": int(len(r)),
            "peak0": round(peak, 4),
            "frac_near_zero": round(frac_near_zero, 4),
            "residual_std": round(float(r.std()), 4),
            "sfp_detected": bool(peak >= self.alert),
            "alert_threshold": self.alert,
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--score-col", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--id-col", default="claim_id")
    p.add_argument("--out")  # optional: write metrics as JSON
    a = p.parse_args()

    metrics = ResidualPeakAlgorithm().detect(
        pd.read_parquet(a.scores), pd.read_parquet(a.labels), a.score_col, a.id_col
    )
    print(json.dumps(metrics))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()

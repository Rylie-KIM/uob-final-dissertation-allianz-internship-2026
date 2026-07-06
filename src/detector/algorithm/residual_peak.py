from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from detector.algorithm.base import DetectionAlgorithm


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
        r = (m[outcome_col] - m[score_col]).to_numpy()

        peak0 = float(gaussian_kde(r)(0.0)[0])  # density of the residual exactly at 0
        frac_near_zero = float((np.abs(r) < self.eps).mean())
        return {
            "n": int(len(r)),
            "peak0": round(peak0, 4),
            "frac_near_zero": round(frac_near_zero, 4),
            "residual_std": round(float(r.std()), 4),
            "sfp_detected": bool(peak0 >= self.alert),
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

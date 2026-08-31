"""
Synthetic-only SFP verification against the oracle (garage_outcome).

⚠️ THIS IS A DGP-VERIFICATION TOOL, NOT A BUILD. Builds 00–06 and the real pipeline are
oracle-free by construction: operationally there is NO ground-truth label for scrapped cars,
and `garage_outcome` is never persisted to the shared datasets. This script exists ONLY to
prove that the self-fulfilling-prophecy loop is genuinely baked into the synthetic DGP — it
re-runs generation deterministically (fixed SEED), keeps `garage_outcome` in memory, and
measures what the contaminated-label evaluation (evaluate.py) structurally cannot see.

What it shows per model version, on each version's OOT window:
  - contaminated precision : decision vs the version's forced-label training target
                             (what tuning/evaluation sees — stays high because scrap→label 1)
  - oracle precision       : decision vs garage_outcome (the truth)
  - gap = contaminated − oracle : the hidden SFP false-positive inflation ("second signal")
  - true_FP_among_scrap    : fraction of scrapped cars that were actually repairable

The headline SFP signature in this DGP is a DEGRADING oracle precision (and a widening gap)
while the contaminated precision the business monitors stays pinned near 0.985 — i.e. the
threshold re-tuning absorbs the score drift, so the loop hides in the clean-label gap rather
than in precision/scrap-rate. Run: `python verify_sfp_oracle.py`.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from generate.base_features import generate_base_features
from generate.enrichment import build_enrichment_table, join_enrichment
from generate.pre_ml import generate_pre_ml_era
from generate.model import train_and_apply
from generate.config import SEED, V1_SPEC, V2A_SPEC, V2B_SPEC, V3A_SPEC, V3B_SPEC
from generate.logger import get_logger

log = get_logger(__name__)

# OOT windows + each version's forced-label training target (mirrors evaluate.py).
OOT_WINDOWS = {
    "v1":  ("2021-05-01", "2021-10-31"),
    "v2a": ("2024-05-01", "2024-10-31"),
    "v2b": ("2024-05-01", "2024-10-31"),
    "v3a": ("2024-05-01", "2024-10-31"),
    "v3b": ("2024-05-01", "2024-10-31"),
}
TRAIN_LABEL = {
    "v1":  "pre_ml_label",
    "v2a": "model_v1_observed_outcome",
    "v2b": "model_v1_observed_outcome",
    "v3a": "model_v2a_observed_outcome",
    "v3b": "model_v2b_observed_outcome",
}


def _generate_with_oracle() -> tuple[pd.DataFrame, pd.Series]:
    """Re-run the generation flow (deterministic) and return (df, garage_outcome)."""
    rng = np.random.default_rng(SEED)
    df = generate_base_features()
    enrichment = build_enrichment_table()
    df = join_enrichment(df, enrichment)
    df, garage_outcome = generate_pre_ml_era(df, rng)

    df, X_all, _ = train_and_apply(df, V1_SPEC, garage_outcome, enrichment_table=enrichment)
    for spec in [V2A_SPEC, V2B_SPEC, V3A_SPEC, V3B_SPEC]:
        df, _, _ = train_and_apply(df, spec, garage_outcome, X_all=X_all)
    return df, garage_outcome.reindex(df.index)


def verify(df: pd.DataFrame, oracle: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(df["claim_date"])
    rows = []
    for v, (a, b) in OOT_WINDOWS.items():
        m = (dates >= a) & (dates <= b)
        dec = df.loc[m, f"model_{v}_decision"].astype(int).to_numpy()
        y_oracle = oracle[m].astype(int).to_numpy()
        lab = df.loc[m, TRAIN_LABEL[v]].astype("Int64")
        ok = lab.notna().to_numpy()
        scrapped = dec == 1

        cont_prec   = precision_score(lab[ok].astype(int), dec[ok], zero_division=0)
        oracle_prec = precision_score(y_oracle, dec, zero_division=0)
        rows.append({
            "version":             v,
            "oot_rows":            int(m.sum()),
            "scrap_rate":          round(float(dec.mean()), 4),
            "cont_prec":           round(float(cont_prec), 4),
            "oracle_prec":         round(float(oracle_prec), 4),
            "gap":                 round(float(cont_prec - oracle_prec), 4),
            "true_FP_among_scrap": round(float(1 - y_oracle[scrapped].mean()), 4) if scrapped.sum() else None,
            "oracle_recall":       round(float(recall_score(y_oracle, dec, zero_division=0)), 4),
            "true_TL_rate":        round(float(y_oracle.mean()), 4),
        })
    return pd.DataFrame(rows).set_index("version")


def main() -> None:
    df, oracle = _generate_with_oracle()
    summary = verify(df, oracle)
    log.info("=== SFP oracle verification (SYNTHETIC-ONLY ground truth) ===")
    log.info("\n" + summary.to_string())
    log.info(
        "SFP signature = oracle_prec degrading + gap widening across v1→v2→v3 while "
        "cont_prec (the monitored, contaminated precision) stays pinned near 0.985."
    )


if __name__ == "__main__":
    main()

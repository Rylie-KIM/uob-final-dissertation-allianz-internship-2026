import numpy as np
import pandas as pd
from pathlib import Path

from generate.base_features import generate_base_features
from generate.enrichment import build_enrichment_table, join_enrichment
from generate.garage_outcome import compute_garage_outcome
from generate.pre_ml import simulate_pre_ml_era
from generate.model import train_and_apply
from generate.config import SEED, V1_SPEC, V2A_SPEC, V2B_SPEC, V3A_SPEC, V3B_SPEC
from generate.logger import get_logger

BASE_DIR    = Path(__file__).parent
CSV_DIR     = BASE_DIR / "csv"
PARQUET_DIR = BASE_DIR / "parquet"

log = get_logger(__name__)


def save(df: pd.DataFrame, name: str) -> None:
    CSV_DIR.mkdir(exist_ok=True)
    PARQUET_DIR.mkdir(exist_ok=True)

    df.to_csv(CSV_DIR / f"{name}.csv", index=False)
    df.to_parquet(PARQUET_DIR / f"{name}.parquet", index=False)
    log.info(f"saved {name}.csv + {name}.parquet  ({len(df):,} rows)")


def main() -> None:
    rng = np.random.default_rng(SEED)

    log.info("Step 1   — generating base features")
    df = generate_base_features()

    log.info("Step 1b  — building enrichment table and joining")
    enrichment = build_enrichment_table()
    df = join_enrichment(df, enrichment)

    log.info("Step 2   — computing garage_outcome")
    garage_outcome = compute_garage_outcome(df, rng)

    log.info("Step 3   — simulating pre-ML era decisions (2016–2021)")
    df = simulate_pre_ml_era(df, garage_outcome)

    pre_ml_mask = pd.to_datetime(df["claim_date"]).dt.year <= 2021
    save(df[pre_ml_mask].reset_index(drop=True), "claims_pre_v1")
    save(enrichment, "vehicle_enrichment")

    log.info("Steps 4–5 — training v1 and applying policy")
    df, X_all, imputer = train_and_apply(df, V1_SPEC, garage_outcome, enrichment_table=enrichment)

    for spec, label in [
        (V2A_SPEC, "v2 option A (v1 log only)"),
        (V2B_SPEC, "v2 option B (pre-ML + v1 log)"),
        (V3A_SPEC, "v3 option A (v2a log only)"),
        (V3B_SPEC, "v3 option B (v2b log)"),
    ]:
        log.info(f"training {label}")
        df, _, _ = train_and_apply(df, spec, garage_outcome, X_all=X_all)

    save(df[~pre_ml_mask].reset_index(drop=True), "claims_v1_log")
    save(df.reset_index(drop=True), "claims_all")

if __name__ == "__main__":
    main()

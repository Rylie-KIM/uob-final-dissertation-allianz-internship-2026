import numpy as np
import pandas as pd
from pathlib import Path

from generate.base_features import generate_base_features
from generate.enrichment import build_enrichment_table, join_enrichment
from generate.garage_outcome import compute_garage_outcome
from generate.pre_ml import simulate_pre_ml_era
from generate.model import train_and_apply_v1, train_and_apply_v2

SEED = 42
BASE_DIR = Path(__file__).parent
CSV_DIR     = BASE_DIR / "csv"
PARQUET_DIR = BASE_DIR / "parquet"


def save(df: pd.DataFrame, name: str) -> None:
    CSV_DIR.mkdir(exist_ok=True)
    PARQUET_DIR.mkdir(exist_ok=True)
    df.to_csv(CSV_DIR / f"{name}.csv", index=False)
    df.to_parquet(PARQUET_DIR / f"{name}.parquet", index=False)
    print(f"  saved {name}.csv + {name}.parquet  ({len(df):,} rows)")


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Step 1   — generating base features")
    df = generate_base_features()

    print("Step 1b  — building enrichment table and joining")
    enrichment = build_enrichment_table()
    df = join_enrichment(df, enrichment)

    print("Step 2   — computing garage_outcome")
    garage_outcome = compute_garage_outcome(df, rng)

    print("Step 3   — simulating pre-ML era decisions (2016–2021)")
    df = simulate_pre_ml_era(df, garage_outcome)

    pre_ml_mask = pd.to_datetime(df["claim_date"]).dt.year <= 2021
    save(df[pre_ml_mask].reset_index(drop=True), "claims_pre_v1")
    save(enrichment, "vehicle_enrichment")

    print("Steps 4–5 — training v1 and applying policy")
    df, X_all = train_and_apply_v1(df, garage_outcome)

    print("Steps 6–7 — training v2 option A (v1 log only) and applying policy")
    df = train_and_apply_v2(df, X_all, option="A")

    print("Steps 6–7 — training v2 option B (pre-ML + v1 log) and applying policy")
    df = train_and_apply_v2(df, X_all, option="B")

    save(df[~pre_ml_mask].reset_index(drop=True), "claims_v1_log")

if __name__ == "__main__":
    main()

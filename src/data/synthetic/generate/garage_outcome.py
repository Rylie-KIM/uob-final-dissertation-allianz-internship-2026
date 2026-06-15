# Step 2 — Internal Data Generating Process.

import numpy as np
import pandas as pd

# Simulate garage engineer decision (human agent)
def compute_garage_outcome(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    logit = (
        -2.0
        + 2.5 * (df["repair_to_value_ratio"] > 0.8).astype(float)
        + 1.5 * (df["damage_severity"] == "severe").astype(float)
        + 1.0 * (df["damage_location"] == "multiple").astype(float)
        + 0.8 * (df["vehicle_age_years"] > 12).astype(float)
        + 0.6 * (df["mileage"] > 120_000).astype(float)
        + 0.4 * (df["damage_type"] == "fire").astype(float)
        + rng.normal(0, 0.3, size=len(df))
    )

    prob = 1 / (1 + np.exp(-logit))  # sigmoid
    return pd.Series(rng.binomial(1, prob), index=df.index, name="garage_outcome")

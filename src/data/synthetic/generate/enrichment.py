# Step 1b — Build enrichment table and join onto base claims.
import numpy as np
import pandas as pd

SEED = 42

BASE_REPAIR_COST = {
    ("minor",    "front"):    500,
    ("minor",    "rear"):     450,
    ("minor",    "side"):     400,
    ("minor",    "roof"):     600,
    ("minor",    "multiple"): 800,
    ("moderate", "front"):   2_000,
    ("moderate", "rear"):    1_800,
    ("moderate", "side"):    1_500,
    ("moderate", "roof"):    2_500,
    ("moderate", "multiple"):3_500,
    ("severe",   "front"):   5_000,
    ("severe",   "rear"):    4_500,
    ("severe",   "side"):    4_000,
    ("severe",   "roof"):    6_000,
    ("severe",   "multiple"):8_000,
}

# Typical new-car market value (GBP) and part cost index per make/model
MAKE_MODEL_SPECS = {
    ("Ford",       "Fiesta"):   (12_000, 0.75),
    ("Ford",       "Focus"):    (18_000, 0.80),
    ("Ford",       "Mondeo"):   (22_000, 0.85),
    ("Ford",       "Kuga"):     (28_000, 0.90),
    ("Ford",       "Transit"):  (30_000, 0.95),
    ("BMW",        "1 Series"): (28_000, 1.70),
    ("BMW",        "3 Series"): (35_000, 1.80),
    ("BMW",        "5 Series"): (45_000, 1.90),
    ("BMW",        "X3"):       (48_000, 1.85),
    ("BMW",        "X5"):       (65_000, 2.00),
    ("Toyota",     "Yaris"):    (15_000, 0.80),
    ("Toyota",     "Corolla"):  (22_000, 0.85),
    ("Toyota",     "RAV4"):     (32_000, 0.90),
    ("Toyota",     "Camry"):    (30_000, 0.88),
    ("Volkswagen", "Polo"):     (16_000, 0.90),
    ("Volkswagen", "Golf"):     (25_000, 1.00),
    ("Volkswagen", "Passat"):   (30_000, 1.05),
    ("Volkswagen", "Tiguan"):   (35_000, 1.10),
    ("Vauxhall",   "Corsa"):    (13_000, 0.70),
    ("Vauxhall",   "Astra"):    (18_000, 0.75),
    ("Vauxhall",   "Mokka"):    (22_000, 0.80),
    ("Vauxhall",   "Insignia"): (25_000, 0.85),
    ("Audi",       "A3"):       (30_000, 1.75),
    ("Audi",       "A4"):       (38_000, 1.80),
    ("Audi",       "A6"):       (48_000, 1.90),
    ("Audi",       "Q3"):       (35_000, 1.75),
    ("Audi",       "Q5"):       (45_000, 1.85),
    ("Honda",      "Jazz"):     (16_000, 0.85),
    ("Honda",      "Civic"):    (22_000, 0.88),
    ("Honda",      "HR-V"):     (25_000, 0.90),
    ("Honda",      "CR-V"):     (30_000, 0.92),
    ("Mercedes",   "A-Class"):  (30_000, 1.80),
    ("Mercedes",   "C-Class"):  (40_000, 1.90),
    ("Mercedes",   "E-Class"):  (52_000, 2.00),
    ("Mercedes",   "GLC"):      (48_000, 1.95),
    ("Nissan",     "Micra"):    (13_000, 0.78),
    ("Nissan",     "Juke"):     (20_000, 0.85),
    ("Nissan",     "Qashqai"):  (28_000, 0.90),
    ("Nissan",     "Leaf"):     (30_000, 1.00),
    ("Hyundai",    "i10"):      (12_000, 0.75),
    ("Hyundai",    "i20"):      (16_000, 0.78),
    ("Hyundai",    "i30"):      (20_000, 0.82),
    ("Hyundai",    "Tucson"):   (28_000, 0.88),
}

YEAR_BANDS = list(range(2004, 2025))


def build_enrichment_table() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)

    for (make, model), (base_value, part_index) in MAKE_MODEL_SPECS.items():
        for year in YEAR_BANDS:
            # Small random variation per year band to mimic real market fluctuation
            value_noise = rng.uniform(0.95, 1.05)
            index_noise = rng.uniform(0.97, 1.03)
            rows.append({
                "vehicle_make":           make,
                "vehicle_model":          model,
                "manufacture_year":       year,
                "typical_market_value_gbp": round(base_value * value_noise, 2),
                "part_cost_index":        round(part_index * index_noise, 3),
            })

    return pd.DataFrame(rows)


def join_enrichment(df: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(enrichment, on=["vehicle_make", "vehicle_model", "manufacture_year"], how="left")

    age = df["vehicle_age_years"]
    depreciation = (1.0 - 0.08 * age).clip(lower=0.10)

    df["vehicle_value"] = (df["typical_market_value_gbp"] * depreciation).round(2)

    base_cost = df.apply(
        lambda r: BASE_REPAIR_COST.get((r["damage_severity"], r["damage_location"]), 1_000),
        axis=1,
    )
    df["repair_estimate_gbp"] = (df["part_cost_index"] * base_cost).round(2)

    df["repair_to_value_ratio"] = (
        df["repair_estimate_gbp"] / df["vehicle_value"]
    ).clip(upper=2.0).round(4)

    # Relative high-value flag (above 75th percentile of this dataset)
    threshold = df["vehicle_value"].quantile(0.75)
    df["is_high_value_vehicle"] = df["vehicle_value"] > threshold

    return df

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

DAMAGE_SEVERITIES = sorted({sev for sev, _ in BASE_REPAIR_COST})
DAMAGE_LOCATIONS  = sorted({loc for _, loc in BASE_REPAIR_COST})

# Typical new-car specs per make/model.
# Tuple: (market_value_gbp, part_cost_index, bhp, accel_0_60_sec, num_gears, kerb_weight_kg, height_mm)
# Physical dimensions (bhp, weight, height) are static per model; small noise added per year
# band in build_enrichment_table() to reflect trim-level variation across model years.
# num_gears: 1 for single-speed electric (Leaf); CVT models use closest fixed-ratio equivalent.
MAKE_MODEL_SPECS = {
    #                                 value    pci   bhp  0-60   gears  kg    mm
    ("Ford",       "Fiesta"):   (12_000, 0.75,  100, 10.5,  5,  1100, 1460),
    ("Ford",       "Focus"):    (18_000, 0.80,  125,  9.5,  6,  1300, 1470),
    ("Ford",       "Mondeo"):   (22_000, 0.85,  165,  9.0,  6,  1500, 1490),
    ("Ford",       "Kuga"):     (28_000, 0.90,  150,  9.5,  6,  1700, 1680),
    ("Ford",       "Transit"):  (30_000, 0.95,  130, 14.0,  6,  2000, 1980),
    ("BMW",        "1 Series"): (28_000, 1.70,  136,  8.5,  8,  1380, 1436),
    ("BMW",        "3 Series"): (35_000, 1.80,  184,  7.1,  8,  1500, 1440),
    ("BMW",        "5 Series"): (45_000, 1.90,  252,  6.3,  8,  1700, 1470),
    ("BMW",        "X3"):       (48_000, 1.85,  184,  7.5,  8,  1800, 1660),
    ("BMW",        "X5"):       (65_000, 2.00,  340,  5.5,  8,  2200, 1745),
    ("Toyota",     "Yaris"):    (15_000, 0.80,   72, 14.0,  5,  1000, 1500),
    ("Toyota",     "Corolla"):  (22_000, 0.85,  122, 10.9,  6,  1300, 1435),
    ("Toyota",     "RAV4"):     (32_000, 0.90,  222,  8.4,  6,  1800, 1685),
    ("Toyota",     "Camry"):    (30_000, 0.88,  218,  8.3,  6,  1700, 1455),
    ("Volkswagen", "Polo"):     (16_000, 0.90,   95, 10.8,  5,  1100, 1453),
    ("Volkswagen", "Golf"):     (25_000, 1.00,  130,  9.0,  7,  1300, 1456),
    ("Volkswagen", "Passat"):   (30_000, 1.05,  150,  8.9,  7,  1500, 1456),
    ("Volkswagen", "Tiguan"):   (35_000, 1.10,  150,  9.6,  7,  1700, 1673),
    ("Vauxhall",   "Corsa"):    (13_000, 0.70,  100, 10.8,  5,  1100, 1470),
    ("Vauxhall",   "Astra"):    (18_000, 0.75,  110, 10.1,  6,  1300, 1465),
    ("Vauxhall",   "Mokka"):    (22_000, 0.80,  130, 10.7,  6,  1450, 1616),
    ("Vauxhall",   "Insignia"): (25_000, 0.85,  165,  9.3,  6,  1600, 1490),
    ("Audi",       "A3"):       (30_000, 1.75,  150,  8.4,  7,  1350, 1416),
    ("Audi",       "A4"):       (38_000, 1.80,  190,  7.3,  7,  1500, 1424),
    ("Audi",       "A6"):       (48_000, 1.90,  245,  6.1,  7,  1700, 1458),
    ("Audi",       "Q3"):       (35_000, 1.75,  150,  8.5,  7,  1600, 1615),
    ("Audi",       "Q5"):       (45_000, 1.85,  204,  7.1,  7,  1900, 1659),
    ("Honda",      "Jazz"):     (16_000, 0.85,   98, 11.5,  5,  1100, 1525),
    ("Honda",      "Civic"):    (22_000, 0.88,  122,  9.5,  6,  1300, 1415),
    ("Honda",      "HR-V"):     (25_000, 0.90,  130, 10.3,  6,  1430, 1590),
    ("Honda",      "CR-V"):     (30_000, 0.92,  193,  8.8,  6,  1700, 1680),
    ("Mercedes",   "A-Class"):  (30_000, 1.80,  136,  9.3,  7,  1370, 1420),
    ("Mercedes",   "C-Class"):  (40_000, 1.90,  184,  7.5,  9,  1560, 1440),
    ("Mercedes",   "E-Class"):  (52_000, 2.00,  245,  6.5,  9,  1770, 1455),
    ("Mercedes",   "GLC"):      (48_000, 1.95,  197,  7.5,  9,  1925, 1640),
    ("Nissan",     "Micra"):    (13_000, 0.78,   90, 12.5,  5,  1050, 1455),
    ("Nissan",     "Juke"):     (20_000, 0.85,  114, 10.8,  6,  1250, 1570),
    ("Nissan",     "Qashqai"):  (28_000, 0.90,  140, 10.1,  7,  1500, 1625),
    ("Nissan",     "Leaf"):     (30_000, 1.00,  150,  7.9,  1,  1640, 1545),
    ("Hyundai",    "i10"):      (12_000, 0.75,   67, 15.8,  5,   900, 1480),
    ("Hyundai",    "i20"):      (16_000, 0.78,  100, 11.1,  5,  1100, 1450),
    ("Hyundai",    "i30"):      (20_000, 0.82,  120, 10.3,  6,  1300, 1455),
    ("Hyundai",    "Tucson"):   (28_000, 0.88,  180,  9.1,  7,  1700, 1647),
}

# UK used car market index by claim year (baseline = 2016).
# Reflects demand/supply shocks independent of a vehicle's make/model/age:
#   same ABI code + same registration year → different ACV depending on claim year.
# Sources: CAP HPI used car price trends; ONS used vehicle price sub-index.
USED_CAR_PRICE_INDEX: dict[int, float] = {
    2016: 1.00,  # baseline
    2017: 1.02,
    2018: 1.03,
    2019: 1.05,
    2020: 0.97,  # COVID lockdowns → demand shock, auction volumes collapse
    2021: 1.18,  # semiconductor shortage → new car supply crunch, used car surge
    2022: 1.28,  # peak used car prices across all segments
    2023: 1.18,  # price normalisation begins as supply recovers
    2024: 1.10,  # further normalisation; still above pre-COVID baseline
}

YEAR_BANDS = list(range(2004, 2025))

VEHICLE_MODELS: dict[str, list[str]] = {}
for _make, _model in MAKE_MODEL_SPECS:
    VEHICLE_MODELS.setdefault(_make, []).append(_model)


def build_enrichment_table() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)

    for (make, model), (base_value, part_index, bhp, accel, gears, weight, height) in MAKE_MODEL_SPECS.items():
        for year in YEAR_BANDS:
            # Price/cost columns fluctuate with market conditions each year
            value_noise = rng.uniform(0.95, 1.05)
            index_noise = rng.uniform(0.97, 1.03)
            # Physical specs vary slightly across year bands (trim-level and minor refresh variation)
            # num_gears is deterministic — gearbox spec doesn't change within a generation
            bhp_noise    = rng.uniform(0.97, 1.03)
            accel_noise  = rng.uniform(0.98, 1.02)
            weight_noise = rng.uniform(0.99, 1.01)
            height_noise = rng.uniform(0.99, 1.01)
            rows.append({
                "vehicle_make":             make,
                "vehicle_model":            model,
                "manufacture_year":         year,
                "typical_market_value_gbp": round(base_value * value_noise, 2),
                "part_cost_index":          round(part_index * index_noise, 3),
                "bhp":                      round(bhp * bhp_noise),
                "acceleration_0_60_sec":    round(accel * accel_noise, 1),
                "num_gears":                gears,
                "kerb_weight_kg":           round(weight * weight_noise),
                "vehicle_height_mm":        round(height * height_noise),
            })

    return pd.DataFrame(rows)


def join_enrichment(df: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(enrichment, on=["vehicle_make", "vehicle_model", "manufacture_year"], how="left")

    age = df["vehicle_age_years"]
    depreciation = (1.0 - 0.08 * age).clip(lower=0.10)

    claim_year = pd.to_datetime(df["claim_date"]).dt.year
    df["used_car_price_index"] = claim_year.map(USED_CAR_PRICE_INDEX).fillna(1.0).round(3)

    df["vehicle_value"] = (
        df["typical_market_value_gbp"] * depreciation * df["used_car_price_index"]
    ).round(2)

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

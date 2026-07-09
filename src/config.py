SCRAP_THRESHOLD = 0.872
TARGET_PRECISION = 0.985
SEED = 42

FEATURE_COLS = [
    "vehicle_age_years", "mileage", "driver_age", "prior_claims_count",
    "customer_tenure_years", "repair_to_value_ratio", "vehicle_value",
    "repair_estimate_gbp", "report_delay_days", "used_car_price_index",
    "is_weekend_claim", "has_prior_claims", "is_high_value_vehicle", "car_driveable",
]

CATEGORICAL_COLS = [
    "vehicle_make", "vehicle_model", "vehicle_type",
    "damage_type", "damage_location", "damage_severity",
    "agent_channel", "coverage_type",
]

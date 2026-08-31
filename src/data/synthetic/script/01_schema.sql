-- =============================================================
-- 01_schema.sql
-- Synthetic Dataset: UK Motor Insurance Claims
-- SFP Loop Research — Allianz Motor Total Loss Prediction (FTTL)
--
-- Tables created here:
--   base_repair_cost    static lookup: repair cost by severity × location
--   vehicle_enrichment  static lookup: market values & part-cost indices
--   claims_pre_v1       2016–2021 human-era claims   (33 columns)
--   claims_v1_log       2022–2024 ML-era claims       (48 columns)
--
-- Parquet equivalents: claims_pre_v1.parquet, claims_v1_log.parquet,
--                      vehicle_enrichment.parquet  (src/data/synthetic/parquet/)
-- =============================================================


-- -------------------------------------------------------------
-- LOOKUP 1: Base repair cost
--   Raw cost before brand multiplier (part_cost_index) applied.
--   15 rows: 3 severities × 5 locations.
--   Seed data: see 02_seed_data.sql
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS base_repair_cost (
    damage_severity  TEXT    NOT NULL,
    damage_location  TEXT    NOT NULL,
    base_cost_gbp    INTEGER NOT NULL CHECK (base_cost_gbp > 0),

    PRIMARY KEY (damage_severity, damage_location),

    CHECK (damage_severity IN ('minor', 'moderate', 'severe')),
    CHECK (damage_location IN ('front', 'rear', 'side', 'roof', 'multiple'))
);


-- -------------------------------------------------------------
-- LOOKUP 2: Vehicle enrichment
--   Join key: (vehicle_make, vehicle_model, manufacture_year)
--   903 rows: 43 make/model combinations × 21 year bands (2004–2024),
--   each with small random market-value noise to mimic real fluctuation.
--   Updated independently of model retraining cycle.
--
--   Derived at JOIN time (not stored here):
--     age_depreciation_factor = MAX(0.10, 1.0 - 0.08 * vehicle_age_years)
--     vehicle_value           = typical_market_value_gbp * age_depreciation_factor
--     repair_estimate_gbp     = part_cost_index * base_repair_cost(severity, location)
--     repair_to_value_ratio   = repair_estimate_gbp / vehicle_value  [clipped at 2.0]
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicle_enrichment (
    vehicle_make              TEXT    NOT NULL,
    vehicle_model             TEXT    NOT NULL,
    manufacture_year          INTEGER NOT NULL,
    typical_market_value_gbp  REAL    NOT NULL CHECK (typical_market_value_gbp > 0),
    part_cost_index           REAL    NOT NULL CHECK (part_cost_index > 0),
    --  > 1.0 = luxury (BMW, Audi, Mercedes)
    --  < 1.0 = economy (Ford, Vauxhall, Hyundai)
    --  = 1.0 = mid-market (Volkswagen Golf)
    bhp                       INTEGER NOT NULL,
    -- engine output in brake horsepower; small year-band noise (±3%) simulates trim variation
    acceleration_0_60_sec     REAL    NOT NULL,
    -- 0–60 mph time in seconds; lower = faster; inversely correlated with bhp (±2% noise)
    num_gears                 INTEGER NOT NULL,
    -- number of forward gears; deterministic per model year; 1 for single-speed EV (Nissan Leaf)
    kerb_weight_kg            INTEGER NOT NULL,
    -- unladen weight in kg; heavier → higher structural repair cost (±1% noise)
    vehicle_height_mm         INTEGER NOT NULL,
    -- overall height in mm; taller vehicles (SUVs, vans) → more bodywork exposure (±1% noise)

    PRIMARY KEY (vehicle_make, vehicle_model, manufacture_year),
    CHECK (manufacture_year BETWEEN 2004 AND 2024)
);


-- -------------------------------------------------------------
-- CLAIMS 1: Pre-ML era  (2016–2021)
--   Saved BEFORE v1 is trained → carries NO model columns.
--   Decisions made by human handlers using two hard rules.
--   v1 training target: pre_ml_label
--
--   Human decision rules:
--     Rule A: repair_to_value_ratio > 0.9
--          → handler scraps immediately (pre_ml_decision = 1)
--          → pre_ml_label = 1  [self-fulfilling — garage never checked]
--     Rule B: damage_severity = 'severe' AND vehicle_age_years > 15
--          → handler scraps immediately (pre_ml_decision = 1)
--          → pre_ml_label = 1  [self-fulfilling]
--     Otherwise:
--          → sent to garage (pre_ml_decision = 0)
--          → pre_ml_label = actual garage repair result  [0 or 1]
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS claims_pre_v1 (

    -- ── Identifiers (2) ──────────────────────────────────────
    claim_id               TEXT    NOT NULL PRIMARY KEY,
    -- Format: 'CLM000001', sequential, zero-padded
    policy_id              TEXT    NOT NULL,
    -- Format: 'POL000001'; ~7,000 distinct for 70,000 rows
    -- ~75% appear once, ~20% twice, ~5% three or more times

    -- ── Dates (5) ────────────────────────────────────────────
    claim_date             DATE    NOT NULL,
    -- 2016-01-01 → 2021-12-31 for pre-ML era
    incident_date          DATE    NOT NULL,
    -- always ≤ claim_date
    policy_start_date      DATE    NOT NULL,
    -- start of current annual policy period (not customer's first ever)
    policy_expiry_date     DATE    NOT NULL,
    -- policy_start_date + 365 days
    report_delay_days      INTEGER NOT NULL,
    -- claim_date − incident_date, 0–30

    -- ── Categorical (8) ──────────────────────────────────────
    vehicle_make           TEXT    NOT NULL,
    -- Ford | BMW | Toyota | Volkswagen | Vauxhall | Audi | Honda | Mercedes | Nissan | Hyundai
    vehicle_model          TEXT    NOT NULL,
    -- model within make (e.g. Fiesta, 3 Series, Corolla…)
    vehicle_type           TEXT    NOT NULL,
    -- Sedan | SUV | Hatchback | Estate | Van | Motorbike
    damage_type            TEXT    NOT NULL,
    -- collision | flood | fire | vandalism | theft_damage
    -- 'fire' is strongest non-ratio DGP predictor (+0.4)
    damage_location        TEXT    NOT NULL,
    -- front | rear | side | roof | multiple
    -- 'multiple' is highest total-loss risk (+1.0 in DGP)
    damage_severity        TEXT    NOT NULL,
    -- minor | moderate | severe
    -- handler's subjective intake assessment BEFORE garage inspection (+1.5 in DGP)
    agent_channel          TEXT    NOT NULL,
    -- online | broker | direct | app | phone
    coverage_type          TEXT    NOT NULL,
    -- third_party | third_party_fire_theft | comprehensive

    -- ── Boolean (4) ──────────────────────────────────────────
    is_weekend_claim       INTEGER NOT NULL,  -- 0 / 1
    has_prior_claims       INTEGER NOT NULL,  -- 0 / 1  (prior_claims_count > 0)
    is_high_value_vehicle  INTEGER NOT NULL,  -- 0 / 1  (vehicle_value > 75th pctile of training set)
    car_driveable          INTEGER NOT NULL,  -- 0 / 1

    -- ── Numerical — user-reported (7) ────────────────────────
    manufacture_year       INTEGER NOT NULL,
    -- 2004–2024, from policyholder's V5C logbook; enrichment join key
    registration_year      INTEGER NOT NULL,
    -- DVLA plate issue date; 0–2 years after manufacture_year
    vehicle_age_years      INTEGER NOT NULL,
    -- claim_date.year − registration_year; KEY DGP feature (+0.8 at > 12 yrs)
    mileage                INTEGER NOT NULL,
    -- 0–200,000; KEY DGP feature (+0.6 at > 120,000)
    driver_age             INTEGER NOT NULL,
    -- 17–85, normal(μ=42, σ=15)
    prior_claims_count     INTEGER NOT NULL,
    -- 0–10, Poisson(λ=0.8)
    customer_tenure_years  INTEGER NOT NULL,
    -- 0–20; years with insurer across all renewals (not reset at renewal unlike policy_start_date)

    -- ── Enrichment-derived (5) ───────────────────────────────
    -- Computed after JOIN on (vehicle_make, vehicle_model, manufacture_year)
    typical_market_value_gbp  REAL NOT NULL,
    -- base market value from enrichment table (before age depreciation)
    part_cost_index           REAL NOT NULL,
    -- brand composite repair multiplier (luxury > 1.0, economy < 1.0)
    vehicle_value             REAL NOT NULL,
    -- typical_market_value_gbp × MAX(0.10, 1.0 - 0.08 × vehicle_age_years)
    -- denominator of repair_to_value_ratio; errors here propagate directly to DGP
    repair_estimate_gbp       REAL NOT NULL,
    -- part_cost_index × base_repair_cost(damage_severity, damage_location)
    -- numerator of repair_to_value_ratio
    repair_to_value_ratio     REAL NOT NULL,
    -- repair_estimate_gbp / vehicle_value, clipped at 2.0
    -- STRONGEST DGP predictor (+2.5 at > 0.8); industry total-loss threshold is 0.7–0.8

    -- ── Pre-ML era decisions (2) ─────────────────────────────
    pre_ml_decision        INTEGER NOT NULL,
    -- 1 = handler scrapped  →  label forced to 1 (self-fulfilling)
    -- 0 = sent to garage    →  label = actual repair outcome
    pre_ml_label           INTEGER NOT NULL,
    -- v1 training target
    -- 1 if scrapped (forced); 0 or 1 if garage (actual outcome)
    -- NOTE: no NULLs within 2016–2021 rows — every claim received a decision

    -- ── Constraints ──────────────────────────────────────────
    CHECK (claim_date     BETWEEN '2016-01-01' AND '2021-12-31'),
    CHECK (incident_date  <= claim_date),
    CHECK (report_delay_days BETWEEN 0 AND 30),
    CHECK (vehicle_type   IN ('Sedan','SUV','Hatchback','Estate','Van','Motorbike')),
    CHECK (damage_type    IN ('collision','flood','fire','vandalism','theft_damage')),
    CHECK (damage_location IN ('front','rear','side','roof','multiple')),
    CHECK (damage_severity IN ('minor','moderate','severe')),
    CHECK (agent_channel  IN ('online','broker','direct','app','phone')),
    CHECK (coverage_type  IN ('third_party','third_party_fire_theft','comprehensive')),
    CHECK (is_weekend_claim      IN (0, 1)),
    CHECK (has_prior_claims      IN (0, 1)),
    CHECK (is_high_value_vehicle IN (0, 1)),
    CHECK (car_driveable         IN (0, 1)),
    CHECK (manufacture_year      BETWEEN 2004 AND 2024),
    CHECK (vehicle_age_years     BETWEEN 0 AND 20),
    CHECK (mileage               BETWEEN 0 AND 200000),
    CHECK (driver_age            BETWEEN 17 AND 85),
    CHECK (prior_claims_count    BETWEEN 0 AND 10),
    CHECK (customer_tenure_years BETWEEN 0 AND 20),
    CHECK (repair_to_value_ratio BETWEEN 0 AND 2.0),
    CHECK (pre_ml_decision       IN (0, 1)),
    CHECK (pre_ml_label          IN (0, 1))
);


-- -------------------------------------------------------------
-- CLAIMS 2: v1 production log  (2022–2024)
--   All 15 model columns included.
--   pre_ml_decision / pre_ml_label are NULL (data disposed under
--   data retention policy before v2 was retrained).
--
--   SFP loop core:
--     v1 → v2a training target = model_v1_observed_outcome
--     model_v1_decision = 1  →  car scrapped  →  outcome forced to 1 (self-fulfilling)
--     model_v1_decision = 0  →  garage         →  outcome = actual repair result
--     SFP signal: mean(model_v2a_score) > mean(model_v1_score)
--                 rate(model_v2a_decision=1) > rate(model_v1_decision=1)
--
--   Scrap threshold: score ≥ 0.872 → decision = 1  (same for ALL model versions)
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS claims_v1_log (

    -- ── Identifiers (2) ──────────────────────────────────────
    claim_id               TEXT    NOT NULL PRIMARY KEY,
    policy_id              TEXT    NOT NULL,

    -- ── Dates (5) ────────────────────────────────────────────
    claim_date             DATE    NOT NULL,
    -- 2022-01-01 → 2024-12-31 for v1 production era
    incident_date          DATE    NOT NULL,
    policy_start_date      DATE    NOT NULL,
    policy_expiry_date     DATE    NOT NULL,
    report_delay_days      INTEGER NOT NULL,

    -- ── Categorical (8) ──────────────────────────────────────
    vehicle_make           TEXT    NOT NULL,
    vehicle_model          TEXT    NOT NULL,
    vehicle_type           TEXT    NOT NULL,
    damage_type            TEXT    NOT NULL,
    damage_location        TEXT    NOT NULL,
    damage_severity        TEXT    NOT NULL,
    agent_channel          TEXT    NOT NULL,
    coverage_type          TEXT    NOT NULL,

    -- ── Boolean (4) ──────────────────────────────────────────
    is_weekend_claim       INTEGER NOT NULL,
    has_prior_claims       INTEGER NOT NULL,
    is_high_value_vehicle  INTEGER NOT NULL,
    car_driveable          INTEGER NOT NULL,

    -- ── Numerical — user-reported (7) ────────────────────────
    manufacture_year       INTEGER NOT NULL,
    registration_year      INTEGER NOT NULL,
    vehicle_age_years      INTEGER NOT NULL,
    mileage                INTEGER NOT NULL,
    driver_age             INTEGER NOT NULL,
    prior_claims_count     INTEGER NOT NULL,
    customer_tenure_years  INTEGER NOT NULL,

    -- ── Enrichment-derived (5) ───────────────────────────────
    typical_market_value_gbp  REAL NOT NULL,
    part_cost_index           REAL NOT NULL,
    vehicle_value             REAL NOT NULL,
    repair_estimate_gbp       REAL NOT NULL,
    repair_to_value_ratio     REAL NOT NULL,

    -- ── Pre-ML columns (NULL — data disposed) ────────────────
    pre_ml_decision        INTEGER,  -- always NULL in this table
    pre_ml_label           INTEGER,  -- always NULL in this table

    -- ── Model v1 (3) — real deployed model ───────────────────
    -- Trained on pre_ml_label (2016–2021); deployed 2022
    model_v1_score             REAL    NOT NULL,
    -- P(total_loss) ∈ [0.00, 1.00]
    model_v1_decision          INTEGER NOT NULL,
    -- 1 if score ≥ 0.872 (scrap); 0 = send to garage
    model_v1_observed_outcome  INTEGER NOT NULL,
    -- 1 if scrapped (forced); actual garage outcome if decision = 0
    -- v2a training target — contains SFP contamination

    -- ── Model v2a (3) — REAL v2 (currently deployed) ─────────
    -- Trained on model_v1_observed_outcome ONLY (2022–2024)
    -- pre_ml_label was disposed before v2 retraining
    model_v2a_score             REAL    NOT NULL,
    model_v2a_decision          INTEGER NOT NULL,
    model_v2a_observed_outcome  INTEGER NOT NULL,
    -- SFP deepened: scraps more than v1, score distribution shifted upward

    -- ── Model v2b (3) — RESEARCH COMPARISON ONLY ─────────────
    -- NOT a real Allianz model; pre_ml_label was never available at v2 retraining
    -- Trained on: pre_ml_label (2020–2021) + model_v1_observed_outcome (2022–2024)
    -- Purpose: baseline showing SFP dilution when unbiased prior exists
    model_v2b_score             REAL    NOT NULL,
    model_v2b_decision          INTEGER NOT NULL,
    model_v2b_observed_outcome  INTEGER NOT NULL,

    -- ── Model v3a (3) — SFP deepening demo (NOT DEPLOYED) ────
    -- Trained on model_v2a_observed_outcome (2023+)
    -- Real v3 attempt (2025): recall collapsed at precision ≥ 0.985 — loop deepened
    model_v3a_score             REAL    NOT NULL,
    model_v3a_decision          INTEGER NOT NULL,
    model_v3a_observed_outcome  INTEGER NOT NULL,

    -- ── Model v3b (3) — SFP-diluted counterfactual ───────────
    -- Trained on model_v2b_observed_outcome (2023+)
    -- Research path: what if SFP had been partially mitigated at v2?
    model_v3b_score             REAL    NOT NULL,
    model_v3b_decision          INTEGER NOT NULL,
    model_v3b_observed_outcome  INTEGER NOT NULL,

    -- ── Constraints ──────────────────────────────────────────
    CHECK (claim_date     BETWEEN '2022-01-01' AND '2024-12-31'),
    CHECK (incident_date  <= claim_date),
    CHECK (report_delay_days BETWEEN 0 AND 30),
    CHECK (vehicle_type   IN ('Sedan','SUV','Hatchback','Estate','Van','Motorbike')),
    CHECK (damage_type    IN ('collision','flood','fire','vandalism','theft_damage')),
    CHECK (damage_location IN ('front','rear','side','roof','multiple')),
    CHECK (damage_severity IN ('minor','moderate','severe')),
    CHECK (agent_channel  IN ('online','broker','direct','app','phone')),
    CHECK (coverage_type  IN ('third_party','third_party_fire_theft','comprehensive')),
    CHECK (is_weekend_claim      IN (0, 1)),
    CHECK (has_prior_claims      IN (0, 1)),
    CHECK (is_high_value_vehicle IN (0, 1)),
    CHECK (car_driveable         IN (0, 1)),
    CHECK (manufacture_year      BETWEEN 2004 AND 2024),
    CHECK (vehicle_age_years     BETWEEN 0 AND 20),
    CHECK (mileage               BETWEEN 0 AND 200000),
    CHECK (driver_age            BETWEEN 17 AND 85),
    CHECK (prior_claims_count    BETWEEN 0 AND 10),
    CHECK (customer_tenure_years BETWEEN 0 AND 20),
    CHECK (repair_to_value_ratio BETWEEN 0 AND 2.0),
    CHECK (model_v1_decision         IN (0, 1)),
    CHECK (model_v1_observed_outcome IN (0, 1)),
    CHECK (model_v2a_decision        IN (0, 1)),
    CHECK (model_v2a_observed_outcome IN (0, 1)),
    CHECK (model_v2b_decision        IN (0, 1)),
    CHECK (model_v2b_observed_outcome IN (0, 1)),
    CHECK (model_v3a_decision        IN (0, 1)),
    CHECK (model_v3a_observed_outcome IN (0, 1)),
    CHECK (model_v3b_decision        IN (0, 1)),
    CHECK (model_v3b_observed_outcome IN (0, 1))
);


-- =============================================================
-- Column count summary
-- =============================================================
--  Group                        claims_pre_v1  claims_v1_log
--  Identifiers          (2)          ✓              ✓
--  Dates                (5)          ✓              ✓
--  Categorical          (8)          ✓              ✓
--  Boolean              (4)          ✓              ✓
--  Numerical            (7)          ✓              ✓
--  Enrichment-derived   (5)          ✓              ✓
--  Pre-ML era           (2)       NOT NULL        NULL
--  Model v1             (3)          –              ✓
--  Model v2a            (3)          –              ✓
--  Model v2b            (3)          –              ✓
--  Model v3a            (3)          –              ✓
--  Model v3b            (3)          –              ✓
--  ─────────────────────────────────────────────────────
--  TOTAL                             33             48
-- =============================================================

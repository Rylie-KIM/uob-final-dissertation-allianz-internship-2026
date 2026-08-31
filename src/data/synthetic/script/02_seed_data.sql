-- =============================================================
-- 02_seed_data.sql
-- Static lookup / reference data
--
-- Populates:
--   base_repair_cost     15 rows  (3 severities × 5 locations)
--   vehicle_make_model   43 rows  (make/model base specs)
--
-- Note: vehicle_enrichment (903 rows) is generated from vehicle_make_model
--   by expanding across manufacture_year 2004–2024 with ±5% market noise.
--   See: src/data/synthetic/generate/enrichment.py → build_enrichment_table()
-- =============================================================


-- -------------------------------------------------------------
-- BASE REPAIR COST
--   Raw GBP cost before brand multiplier (part_cost_index).
--   Represents a mid-market UK vehicle (part_cost_index = 1.0).
--
--   Applied formula:
--     repair_estimate_gbp = part_cost_index × base_cost_gbp
--
--   Example:
--     BMW (index ≈ 1.80) + severe/multiple → £8,000 × 1.80 = £14,400
--     Vauxhall (index ≈ 0.70) + minor/rear → £450  × 0.70 = £315
-- -------------------------------------------------------------
INSERT INTO base_repair_cost (damage_severity, damage_location, base_cost_gbp) VALUES

    -- minor severity
    ('minor', 'front',    500),
    ('minor', 'rear',     450),
    ('minor', 'side',     400),
    ('minor', 'roof',     600),   -- roof > front/rear: full panel replacement (hail/rollover)
    ('minor', 'multiple', 800),   -- multiple = most expensive at every severity

    -- moderate severity (~×4 of minor)
    ('moderate', 'front',    2000),
    ('moderate', 'rear',     1800),
    ('moderate', 'side',     1500),
    ('moderate', 'roof',     2500),
    ('moderate', 'multiple', 3500),

    -- severe severity (~×4–5 of minor)
    ('severe', 'front',    5000),
    ('severe', 'rear',     4500),
    ('severe', 'side',     4000),
    ('severe', 'roof',     6000),
    ('severe', 'multiple', 8000);


-- -------------------------------------------------------------
-- VEHICLE MAKE/MODEL BASE SPECS
--   Reference: 43 make/model combinations used in enrichment table.
--   Columns: make, model, new_car_value_gbp (typical), part_cost_index
--
--   part_cost_index interpretation:
--     > 1.0 = premium brand  (scarce/expensive parts)
--     = 1.0 = mid-market     (VW Golf as reference)
--     < 1.0 = economy brand  (widely available parts)
--
--   The enrichment table expands each row across manufacture_year 2004–2024
--   with ±5% random noise on both value and index.
-- -------------------------------------------------------------
-- Columns: make, model, value (GBP), part_cost_index, bhp, 0-60 (sec), gears, kerb_weight (kg), height (mm)
-- Physical specs are base values before per-year noise is applied in build_enrichment_table().
-- num_gears is deterministic (no noise); all others get ±1–3% random noise per year band.
CREATE TABLE IF NOT EXISTS vehicle_make_model (
    vehicle_make          TEXT    NOT NULL,
    vehicle_model         TEXT    NOT NULL,
    new_car_value_gbp     INTEGER NOT NULL,
    part_cost_index       REAL    NOT NULL,
    bhp                   INTEGER NOT NULL,
    acceleration_0_60_sec REAL    NOT NULL,
    num_gears             INTEGER NOT NULL,
    kerb_weight_kg        INTEGER NOT NULL,
    vehicle_height_mm     INTEGER NOT NULL,
    PRIMARY KEY (vehicle_make, vehicle_model)
);

INSERT INTO vehicle_make_model
    (vehicle_make, vehicle_model, new_car_value_gbp, part_cost_index,
     bhp, acceleration_0_60_sec, num_gears, kerb_weight_kg, vehicle_height_mm)
VALUES

    -- Ford (economy–mid; index 0.75–0.95)
    ('Ford',       'Fiesta',    12000, 0.75,  100, 10.5, 5, 1100, 1460),
    ('Ford',       'Focus',     18000, 0.80,  125,  9.5, 6, 1300, 1470),
    ('Ford',       'Mondeo',    22000, 0.85,  165,  9.0, 6, 1500, 1490),
    ('Ford',       'Kuga',      28000, 0.90,  150,  9.5, 6, 1700, 1680),
    ('Ford',       'Transit',   30000, 0.95,  130, 14.0, 6, 2000, 1980),

    -- BMW (premium; index 1.70–2.00)
    ('BMW',        '1 Series',  28000, 1.70,  136,  8.5, 8, 1380, 1436),
    ('BMW',        '3 Series',  35000, 1.80,  184,  7.1, 8, 1500, 1440),
    ('BMW',        '5 Series',  45000, 1.90,  252,  6.3, 8, 1700, 1470),
    ('BMW',        'X3',        48000, 1.85,  184,  7.5, 8, 1800, 1660),
    ('BMW',        'X5',        65000, 2.00,  340,  5.5, 8, 2200, 1745),

    -- Toyota (economy–mid; index 0.80–0.90)
    ('Toyota',     'Yaris',     15000, 0.80,   72, 14.0, 5, 1000, 1500),
    ('Toyota',     'Corolla',   22000, 0.85,  122, 10.9, 6, 1300, 1435),
    ('Toyota',     'RAV4',      32000, 0.90,  222,  8.4, 6, 1800, 1685),
    ('Toyota',     'Camry',     30000, 0.88,  218,  8.3, 6, 1700, 1455),

    -- Volkswagen (mid-market reference; index 0.90–1.10)
    ('Volkswagen', 'Polo',      16000, 0.90,   95, 10.8, 5, 1100, 1453),
    ('Volkswagen', 'Golf',      25000, 1.00,  130,  9.0, 7, 1300, 1456),  -- reference vehicle (index = 1.0)
    ('Volkswagen', 'Passat',    30000, 1.05,  150,  8.9, 7, 1500, 1456),
    ('Volkswagen', 'Tiguan',    35000, 1.10,  150,  9.6, 7, 1700, 1673),

    -- Vauxhall (economy; index 0.70–0.85)
    ('Vauxhall',   'Corsa',     13000, 0.70,  100, 10.8, 5, 1100, 1470),
    ('Vauxhall',   'Astra',     18000, 0.75,  110, 10.1, 6, 1300, 1465),
    ('Vauxhall',   'Mokka',     22000, 0.80,  130, 10.7, 6, 1450, 1616),
    ('Vauxhall',   'Insignia',  25000, 0.85,  165,  9.3, 6, 1600, 1490),

    -- Audi (premium; index 1.75–1.90)
    ('Audi',       'A3',        30000, 1.75,  150,  8.4, 7, 1350, 1416),
    ('Audi',       'A4',        38000, 1.80,  190,  7.3, 7, 1500, 1424),
    ('Audi',       'A6',        48000, 1.90,  245,  6.1, 7, 1700, 1458),
    ('Audi',       'Q3',        35000, 1.75,  150,  8.5, 7, 1600, 1615),
    ('Audi',       'Q5',        45000, 1.85,  204,  7.1, 7, 1900, 1659),

    -- Honda (economy–mid; index 0.85–0.92)
    ('Honda',      'Jazz',      16000, 0.85,   98, 11.5, 5, 1100, 1525),
    ('Honda',      'Civic',     22000, 0.88,  122,  9.5, 6, 1300, 1415),
    ('Honda',      'HR-V',      25000, 0.90,  130, 10.3, 6, 1430, 1590),
    ('Honda',      'CR-V',      30000, 0.92,  193,  8.8, 6, 1700, 1680),

    -- Mercedes (luxury; index 1.80–2.00)
    ('Mercedes',   'A-Class',   30000, 1.80,  136,  9.3, 7, 1370, 1420),
    ('Mercedes',   'C-Class',   40000, 1.90,  184,  7.5, 9, 1560, 1440),
    ('Mercedes',   'E-Class',   52000, 2.00,  245,  6.5, 9, 1770, 1455),
    ('Mercedes',   'GLC',       48000, 1.95,  197,  7.5, 9, 1925, 1640),

    -- Nissan (economy–mid; index 0.78–1.00)
    ('Nissan',     'Micra',     13000, 0.78,   90, 12.5, 5, 1050, 1455),
    ('Nissan',     'Juke',      20000, 0.85,  114, 10.8, 6, 1250, 1570),
    ('Nissan',     'Qashqai',   28000, 0.90,  140, 10.1, 7, 1500, 1625),
    ('Nissan',     'Leaf',      30000, 1.00,  150,  7.9, 1, 1640, 1545),  -- EV: single-speed, parts ~mid-market

    -- Hyundai (economy; index 0.75–0.88)
    ('Hyundai',    'i10',       12000, 0.75,   67, 15.8, 5,  900, 1480),
    ('Hyundai',    'i20',       16000, 0.78,  100, 11.1, 5, 1100, 1450),
    ('Hyundai',    'i30',       20000, 0.82,  120, 10.3, 6, 1300, 1455),
    ('Hyundai',    'Tucson',    28000, 0.88,  180,  9.1, 7, 1700, 1647);


-- -------------------------------------------------------------
-- DEPRECIATION REFERENCE TABLE
--   Shows age_depreciation_factor at key ages.
--   Formula: MAX(0.10, 1.0 - 0.08 × vehicle_age_years)
--   Applied: vehicle_value = typical_market_value_gbp × factor
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS depreciation_reference (
    vehicle_age_years        INTEGER NOT NULL PRIMARY KEY,
    age_depreciation_factor  REAL    NOT NULL,
    note                     TEXT
);

INSERT INTO depreciation_reference (vehicle_age_years, age_depreciation_factor, note) VALUES
    ( 0,  1.00, 'new vehicle'),
    ( 1,  0.92, NULL),
    ( 2,  0.84, NULL),
    ( 3,  0.76, NULL),
    ( 4,  0.68, NULL),
    ( 5,  0.60, NULL),
    ( 6,  0.52, NULL),
    ( 7,  0.44, NULL),
    ( 8,  0.36, NULL),
    ( 9,  0.28, NULL),
    (10,  0.20, NULL),
    (11,  0.12, NULL),
    (12,  0.10, '10% floor kicks in (0.04 → floored to 0.10)'),
    (13,  0.10, '10% floor'),
    (14,  0.10, '10% floor'),
    (15,  0.10, '10% floor'),
    (16,  0.10, '10% floor'),
    (17,  0.10, '10% floor'),
    (18,  0.10, '10% floor'),
    (19,  0.10, '10% floor'),
    (20,  0.10, '10% floor — prevents vehicle_value hitting zero');


-- -------------------------------------------------------------
-- MODEL VERSION SUMMARY
--   Quick reference for all model versions in claims_v1_log.
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_version_summary (
    model_tag          TEXT    NOT NULL PRIMARY KEY,
    real_world_status  TEXT    NOT NULL,
    training_target    TEXT    NOT NULL,
    training_window    TEXT    NOT NULL,
    oot_holdout        TEXT    NOT NULL,
    maturation_buffer  TEXT    NOT NULL,
    notes              TEXT
);

INSERT INTO model_version_summary VALUES
    ('v1',
     'Superseded by v2',
     'pre_ml_label',
     '2016-01 → 2021-04  (80/20 random split)',
     '2021-05 → 2021-10  (6 months)',
     '2021-11 → 2021-12  (2 months)',
     'Trained on human handler decisions (2016–2021). Deployed 2022.'),

    ('v2a',
     'CURRENTLY DEPLOYED (real v2)',
     'model_v1_observed_outcome',
     '2022-01 → 2024-04  (80/20 random split)',
     '2024-05 → 2024-10  (6 months, SFP-contaminated)',
     '2024-11 → 2024-12  (2 months)',
     'Trained on v1 log ONLY. pre_ml_label was disposed before retraining. SFP deepened.'),

    ('v2b',
     'RESEARCH COMPARISON ONLY — not a real Allianz model',
     'pre_ml_label (2020–2021) + model_v1_observed_outcome (2022–2024)',
     '2020-01 → 2024-04  (80/20 random split)',
     '2024-05 → 2024-10  (same OOT as v2a)',
     '2024-11 → 2024-12  (2 months)',
     'Shows SFP dilution when unbiased prior is mixed in. Analytical baseline only.'),

    ('v3a',
     'NOT DEPLOYED — SFP deepening demo',
     'model_v2a_observed_outcome',
     '2023-01 → 2024-04  (80/20 random split)',
     '2024-05 → 2024-10  (same OOT)',
     '2024-11 → 2024-12  (2 months)',
     'Real v3 attempt (2025): recall collapsed at precision ≥ 0.985 threshold. Loop deepened.'),

    ('v3b',
     'RESEARCH COMPARISON ONLY — SFP-diluted counterfactual',
     'model_v2b_observed_outcome',
     '2023-01 → 2024-04  (80/20 random split)',
     '2024-05 → 2024-10  (same OOT)',
     '2024-11 → 2024-12  (2 months)',
     'Counterfactual path: shows what v3 might look like if v2 had been SFP-mitigated.');


-- =============================================================
-- SCRAP THRESHOLD
--   Single constant applied identically across ALL model versions.
--   Absolute probability cutoff (NOT a percentile rank).
--   Scrap rate floats with score distribution → SFP signal visible
--   as rising scrap rate from v1 → v2a → v3a.
-- =============================================================
--  SCRAP_THRESHOLD = 0.872
--  decision = 1  iff  model_{tag}_score >= 0.872
--  Tuned for precision >= 0.985 on v1 training data.
-- =============================================================

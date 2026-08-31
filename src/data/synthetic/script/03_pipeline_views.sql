-- =============================================================
-- 03_pipeline_views.sql
-- Data Generation Pipeline — expressed as SQL views
--
-- Steps:
--   Step 1   base_features           raw claim attributes
--   Step 1b  enrichment_join         vehicle value & repair cost
--   Step 2   dgp_logit               internal oracle (NOT saved)
--   Step 3   pre_ml_decisions        human-era label simulation
--   Step 4   v1_training_window      rows used to train model v1
--   Step 5   v1_label_mechanism      SFP loop begins (v1 production)
--   Step 6   v2_training_windows     v2a and v2b training row sets
--   Step 7   sfp_verification        observable SFP signals (no oracle needed)
--
-- These views assume claims_pre_v1 and claims_v1_log have been
-- loaded from parquet. They use UNION ALL where cross-era queries
-- are needed.
-- =============================================================


-- -------------------------------------------------------------
-- STEP 1b — Enrichment join derivations
--   Confirms the computed columns against the lookup formula.
--   Useful for auditing after data generation.
-- -------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_enrichment_derived AS
SELECT
    c.claim_id,
    c.vehicle_make,
    c.vehicle_model,
    c.manufacture_year,
    c.vehicle_age_years,

    -- formula: MAX(0.10, 1.0 - 0.08 * vehicle_age_years)
    MAX(0.10, 1.0 - 0.08 * c.vehicle_age_years)  AS age_depreciation_factor,

    e.typical_market_value_gbp,
    e.typical_market_value_gbp
        * MAX(0.10, 1.0 - 0.08 * c.vehicle_age_years) AS vehicle_value_expected,
    c.vehicle_value                                     AS vehicle_value_actual,

    b.base_cost_gbp,
    e.part_cost_index,
    ROUND(e.part_cost_index * b.base_cost_gbp, 2)      AS repair_estimate_expected,
    c.repair_estimate_gbp                               AS repair_estimate_actual,

    ROUND(c.repair_estimate_gbp / NULLIF(c.vehicle_value, 0), 4) AS repair_to_value_check,

    -- physical spec columns (pass-through from enrichment; year-band noise already applied)
    e.bhp,
    e.acceleration_0_60_sec,
    e.num_gears,
    e.kerb_weight_kg,
    e.vehicle_height_mm

FROM claims_pre_v1 c
JOIN vehicle_enrichment e
    ON  e.vehicle_make     = c.vehicle_make
    AND e.vehicle_model    = c.vehicle_model
    AND e.manufacture_year = c.manufacture_year
JOIN base_repair_cost b
    ON  b.damage_severity = c.damage_severity
    AND b.damage_location = c.damage_location

UNION ALL

SELECT
    c.claim_id,
    c.vehicle_make,
    c.vehicle_model,
    c.manufacture_year,
    c.vehicle_age_years,
    MAX(0.10, 1.0 - 0.08 * c.vehicle_age_years),
    e.typical_market_value_gbp,
    e.typical_market_value_gbp
        * MAX(0.10, 1.0 - 0.08 * c.vehicle_age_years),
    c.vehicle_value,
    b.base_cost_gbp,
    e.part_cost_index,
    ROUND(e.part_cost_index * b.base_cost_gbp, 2),
    c.repair_estimate_gbp,
    ROUND(c.repair_estimate_gbp / NULLIF(c.vehicle_value, 0), 4),
    e.bhp,
    e.acceleration_0_60_sec,
    e.num_gears,
    e.kerb_weight_kg,
    e.vehicle_height_mm

FROM claims_v1_log c
JOIN vehicle_enrichment e
    ON  e.vehicle_make     = c.vehicle_make
    AND e.vehicle_model    = c.vehicle_model
    AND e.manufacture_year = c.manufacture_year
JOIN base_repair_cost b
    ON  b.damage_severity = c.damage_severity
    AND b.damage_location = c.damage_location;


-- -------------------------------------------------------------
-- STEP 2 — Internal DGP logit (documentation view; no data saved)
--   garage_outcome is NEVER stored as a column.
--   Shown here to make the generation logic readable.
--
--   logit = -2.0
--         + 2.5 × (repair_to_value_ratio > 0.80)   ← strongest predictor
--         + 1.5 × (damage_severity = 'severe')
--         + 1.0 × (damage_location = 'multiple')
--         + 0.8 × (vehicle_age_years > 12)
--         + 0.6 × (mileage > 120,000)
--         + 0.4 × (damage_type = 'fire')
--         + N(0, 0.3) noise term
--
--   garage_outcome = Bernoulli(sigmoid(logit))   ~10% positive rate
--
--   The view below computes the DETERMINISTIC part of the logit
--   (without the noise term) as a diagnostic reference.
-- -------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_dgp_logit_base AS
SELECT
    claim_id,
    repair_to_value_ratio,
    damage_severity,
    damage_location,
    vehicle_age_years,
    mileage,
    damage_type,

    -- deterministic logit components
    -2.0
    + CASE WHEN repair_to_value_ratio > 0.80 THEN 2.5 ELSE 0 END
    + CASE WHEN damage_severity = 'severe'         THEN 1.5 ELSE 0 END
    + CASE WHEN damage_location  = 'multiple'      THEN 1.0 ELSE 0 END
    + CASE WHEN vehicle_age_years > 12             THEN 0.8 ELSE 0 END
    + CASE WHEN mileage > 120000                   THEN 0.6 ELSE 0 END
    + CASE WHEN damage_type = 'fire'               THEN 0.4 ELSE 0 END
    AS logit_deterministic,

    -- sigmoid(logit) without noise — approximate P(total_loss)
    1.0 / (1.0 + EXP(-(
        -2.0
        + CASE WHEN repair_to_value_ratio > 0.80 THEN 2.5 ELSE 0 END
        + CASE WHEN damage_severity = 'severe'         THEN 1.5 ELSE 0 END
        + CASE WHEN damage_location  = 'multiple'      THEN 1.0 ELSE 0 END
        + CASE WHEN vehicle_age_years > 12             THEN 0.8 ELSE 0 END
        + CASE WHEN mileage > 120000                   THEN 0.6 ELSE 0 END
        + CASE WHEN damage_type = 'fire'               THEN 0.4 ELSE 0 END
    ))) AS p_total_loss_approx

FROM claims_pre_v1

UNION ALL

SELECT
    claim_id,
    repair_to_value_ratio,
    damage_severity,
    damage_location,
    vehicle_age_years,
    mileage,
    damage_type,
    -2.0
    + CASE WHEN repair_to_value_ratio > 0.80 THEN 2.5 ELSE 0 END
    + CASE WHEN damage_severity = 'severe'         THEN 1.5 ELSE 0 END
    + CASE WHEN damage_location  = 'multiple'      THEN 1.0 ELSE 0 END
    + CASE WHEN vehicle_age_years > 12             THEN 0.8 ELSE 0 END
    + CASE WHEN mileage > 120000                   THEN 0.6 ELSE 0 END
    + CASE WHEN damage_type = 'fire'               THEN 0.4 ELSE 0 END,
    1.0 / (1.0 + EXP(-(
        -2.0
        + CASE WHEN repair_to_value_ratio > 0.80 THEN 2.5 ELSE 0 END
        + CASE WHEN damage_severity = 'severe'         THEN 1.5 ELSE 0 END
        + CASE WHEN damage_location  = 'multiple'      THEN 1.0 ELSE 0 END
        + CASE WHEN vehicle_age_years > 12             THEN 0.8 ELSE 0 END
        + CASE WHEN mileage > 120000                   THEN 0.6 ELSE 0 END
        + CASE WHEN damage_type = 'fire'               THEN 0.4 ELSE 0 END
    )))

FROM claims_v1_log;


-- -------------------------------------------------------------
-- STEP 3 — Pre-ML human decision rules (2016–2021)
--   Replicates the Python simulation logic for auditing.
--
--   Rule A: repair_to_value_ratio > 0.9    → scrap (self-fulfilling)
--   Rule B: severe AND age > 15            → scrap (self-fulfilling)
--   Else:                                  → garage (actual outcome)
-- -------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_pre_ml_decision_rules AS
SELECT
    claim_id,
    claim_date,
    repair_to_value_ratio,
    damage_severity,
    vehicle_age_years,

    CASE
        WHEN repair_to_value_ratio > 0.9                          THEN 'Rule A: rtv > 0.9'
        WHEN damage_severity = 'severe' AND vehicle_age_years > 15 THEN 'Rule B: severe + age > 15'
        ELSE                                                            'No rule: sent to garage'
    END AS decision_rule,

    CASE
        WHEN repair_to_value_ratio > 0.9                          THEN 1
        WHEN damage_severity = 'severe' AND vehicle_age_years > 15 THEN 1
        ELSE                                                            0
    END AS expected_decision,

    pre_ml_decision AS actual_decision,
    pre_ml_label

FROM claims_pre_v1;


-- -------------------------------------------------------------
-- STEP 4 — v1 training window
--   Train: 2016-01 → 2021-04  (before 6-month OOT + 2-month buffer)
--   OOT:   2021-05 → 2021-10
--   Excl:  2021-11 → 2021-12  (maturation buffer)
-- -------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_v1_training_window AS
SELECT
    claim_id,
    claim_date,
    pre_ml_label              AS training_target,
    CASE
        WHEN claim_date < '2021-05-01'              THEN 'train_test'  -- 80/20 random split
        WHEN claim_date BETWEEN '2021-05-01'
                            AND '2021-10-31'        THEN 'oot'
        WHEN claim_date >= '2021-11-01'             THEN 'excluded_buffer'
    END                       AS split,
    pre_ml_label IS NOT NULL  AS has_label

FROM claims_pre_v1

WHERE strftime('%Y', claim_date) BETWEEN '2016' AND '2021';


-- -------------------------------------------------------------
-- STEP 5 — v1 label mechanism (SFP loop begins)
--   Verifies the self-fulfilling prophecy mechanism:
--     decision = 1 → outcome always = 1 (scrapped, never verified)
--     decision = 0 → outcome = actual garage result (0 or 1)
-- -------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_v1_label_mechanism AS
SELECT
    claim_id,
    claim_date,
    model_v1_score,
    model_v1_decision,
    model_v1_observed_outcome,
    CASE
        WHEN model_v1_decision = 1 THEN 'scrapped (forced label = 1)'
        ELSE                            'garage   (label = repair outcome)'
    END AS outcome_source,
    CASE
        WHEN model_v1_decision = 1 AND model_v1_observed_outcome != 1
        THEN 'DATA ERROR: scrapped car must have outcome = 1'
        ELSE 'OK'
    END AS integrity_check

FROM claims_v1_log;


-- -------------------------------------------------------------
-- STEP 5 (continued) — SFP label contamination audit
--   rate(outcome=1 | decision=1) should be exactly 1.0 (self-fulfilling)
--   rate(outcome=1 | decision=0) should be much lower (~10% = base DGP rate)
-- -------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_v1_sfp_audit AS
SELECT
    model_v1_decision                              AS decision,
    COUNT(*)                                       AS n_claims,
    SUM(model_v1_observed_outcome)                 AS n_positive_labels,
    ROUND(
        100.0 * SUM(model_v1_observed_outcome) / COUNT(*), 2
    )                                              AS positive_label_rate_pct,
    CASE
        WHEN model_v1_decision = 1 THEN 'Expected: 100% (self-fulfilling)'
        ELSE                            'Expected: ~10% (DGP base rate)'
    END AS interpretation

FROM claims_v1_log
GROUP BY model_v1_decision;


-- -------------------------------------------------------------
-- STEP 6 — v2a training window  (real v2 scenario)
--   Train: 2022-01 → 2024-04  (target = model_v1_observed_outcome)
--   OOT:   2024-05 → 2024-10  (SFP-contaminated — scrapped cars in OOT forced to 1)
--   Excl:  2024-11 → 2024-12  (maturation buffer)
-- -------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_v2a_training_window AS
SELECT
    claim_id,
    claim_date,
    model_v1_observed_outcome  AS training_target,
    CASE
        WHEN claim_date < '2024-05-01'              THEN 'train_test'
        WHEN claim_date BETWEEN '2024-05-01'
                            AND '2024-10-31'        THEN 'oot_sfp_contaminated'
        WHEN claim_date >= '2024-11-01'             THEN 'excluded_buffer'
    END                        AS split,
    model_v1_decision          AS v1_decision_at_time_of_labelling

FROM claims_v1_log;


-- -------------------------------------------------------------
-- STEP 6 (continued) — v2b training window  (research comparison)
--   Mixes pre_ml_label (2020–2021) with model_v1_observed_outcome (2022–2024).
--   NOT a real Allianz scenario (pre_ml_label was disposed before v2 retraining).
-- -------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_v2b_training_window AS
SELECT
    claim_id,
    claim_date,
    pre_ml_label               AS training_target,
    'pre_ml_era'               AS data_source,
    CASE
        WHEN claim_date < '2024-05-01'  THEN 'train_test'
        ELSE                                 'excluded_oot_or_buffer'
    END AS split

FROM claims_pre_v1
WHERE strftime('%Y', claim_date) >= '2020'

UNION ALL

SELECT
    claim_id,
    claim_date,
    model_v1_observed_outcome,
    'v1_log',
    CASE
        WHEN claim_date < '2024-05-01'  THEN 'train_test'
        WHEN claim_date BETWEEN '2024-05-01' AND '2024-10-31' THEN 'oot_sfp_contaminated'
        ELSE 'excluded_buffer'
    END

FROM claims_v1_log;


-- -------------------------------------------------------------
-- STEP 7 — SFP verification queries
--   Observable checks that require NO oracle (ground truth).
--   These are the four headline signals the SFP detection framework uses.
-- -------------------------------------------------------------

-- Signal 1: Score drift (v1 → v2a mean score should increase)
CREATE VIEW IF NOT EXISTS v_sfp_score_drift AS
SELECT
    'model_v1'  AS model_version,
    ROUND(AVG(model_v1_score),  4) AS mean_score,
    COUNT(*)                        AS n_rows
FROM claims_v1_log

UNION ALL

SELECT
    'model_v2a',
    ROUND(AVG(model_v2a_score), 4),
    COUNT(*)
FROM claims_v1_log

UNION ALL

SELECT
    'model_v3a',
    ROUND(AVG(model_v3a_score), 4),
    COUNT(*)
FROM claims_v1_log;


-- Signal 2: Decision rate inflation (v1 → v2a scrap rate should increase)
CREATE VIEW IF NOT EXISTS v_sfp_decision_rate AS
SELECT
    'model_v1'  AS model_version,
    SUM(model_v1_decision)                              AS n_scrapped,
    COUNT(*)                                             AS n_total,
    ROUND(100.0 * SUM(model_v1_decision) / COUNT(*), 2) AS scrap_rate_pct
FROM claims_v1_log

UNION ALL

SELECT
    'model_v2a',
    SUM(model_v2a_decision),
    COUNT(*),
    ROUND(100.0 * SUM(model_v2a_decision) / COUNT(*), 2)
FROM claims_v1_log

UNION ALL

SELECT
    'model_v3a',
    SUM(model_v3a_decision),
    COUNT(*),
    ROUND(100.0 * SUM(model_v3a_decision) / COUNT(*), 2)
FROM claims_v1_log;


-- Signal 3: Label mechanism bias
-- rate(outcome=1 | decision=1) = 1.0 (tautological — self-fulfilling)
-- rate(outcome=1 | decision=0) << 1.0 (actual garage survival rate)
CREATE VIEW IF NOT EXISTS v_sfp_label_mechanism AS
SELECT
    model_tag,
    decision,
    n_claims,
    n_outcome_1,
    ROUND(100.0 * n_outcome_1 / n_claims, 2) AS outcome_1_rate_pct
FROM (
    SELECT 'v1' AS model_tag, model_v1_decision AS decision,
           COUNT(*) AS n_claims, SUM(model_v1_observed_outcome) AS n_outcome_1
    FROM claims_v1_log GROUP BY model_v1_decision

    UNION ALL

    SELECT 'v2a', model_v2a_decision,
           COUNT(*), SUM(model_v2a_observed_outcome)
    FROM claims_v1_log GROUP BY model_v2a_decision
)
ORDER BY model_tag, decision;


-- Signal 4 (synthetic only): pre-ML vs v1 observed rate comparison
-- Shows whether ML-era labels are more contaminated than human-era labels.
-- ⚠ This check is NOT available on real Allianz data (pre_ml_label disposed).
CREATE VIEW IF NOT EXISTS v_sfp_era_comparison AS
SELECT
    'pre_ml_era (2016–2021)' AS era,
    'pre_ml_label'           AS label_column,
    COUNT(*)                  AS n_rows,
    SUM(pre_ml_label)         AS n_positive,
    ROUND(100.0 * SUM(pre_ml_label) / COUNT(*), 2) AS positive_rate_pct
FROM claims_pre_v1

UNION ALL

SELECT
    'v1_production_era (2022–2024)',
    'model_v1_observed_outcome',
    COUNT(*),
    SUM(model_v1_observed_outcome),
    ROUND(100.0 * SUM(model_v1_observed_outcome) / COUNT(*), 2)
FROM claims_v1_log;


-- =============================================================
-- SFP LOOP SUMMARY VIEW
--   One-row-per-version summary for dissertation reporting.
-- =============================================================
CREATE VIEW IF NOT EXISTS v_sfp_summary AS
SELECT
    dr.model_tag,
    dr.scrap_rate_pct,
    sd.mean_score,
    m.real_world_status
FROM v_sfp_decision_rate dr
JOIN (
    SELECT 'model_v1'  AS model_version, mean_score FROM v_sfp_score_drift WHERE model_version = 'model_v1'
    UNION ALL
    SELECT 'model_v2a', mean_score FROM v_sfp_score_drift WHERE model_version = 'model_v2a'
    UNION ALL
    SELECT 'model_v3a', mean_score FROM v_sfp_score_drift WHERE model_version = 'model_v3a'
) sd ON sd.model_version = dr.model_version
JOIN model_version_summary m ON m.model_tag = REPLACE(dr.model_version, 'model_', '')
ORDER BY dr.model_version;

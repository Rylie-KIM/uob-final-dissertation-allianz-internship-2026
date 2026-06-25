# Real Data Experiment Design

> This document records design decisions, assumptions, and constraints for experiments run on **real Allianz UK production data**, once access is granted. The synthetic framework (`src/data/synthetic/`) is a placeholder — when real data arrives, swap the parquet files and this document governs how experiments are set up.

---

## Key Data Assumptions

### Threshold — constant at 0.872

The production scrap threshold was briefly changed away from 0.872 at some point during deployment (exact value and dates not confirmed). Performance degraded and the threshold was promptly reverted to 0.872.

**For all real-data experiments, the threshold is treated as constant at 0.872 throughout the entire production period.** The brief deviation is not modelled separately and is not reflected in the decision columns in the production log (`model_v1_decision`, `model_v2a_decision`). Any anomalous decision patterns in the log that appear inconsistent with a 0.872 threshold should be noted but not used to infer a different threshold value.

This assumption is consistent with Allianz's operational understanding of the dataset.

### Labels — contaminated, not missing

Scrapped cars (`decision = 1`) have `observed_outcome = 1` by construction — this is a forced label, not a missing label. Do not treat these rows as unlabelled or impute their outcomes. The correct analytical frame is **selective labelling / label contamination** (see `problem.md` §2.2).

### No oracle

There is no ground-truth label for any claim in the real data:
- `pre_ml_label` (handler/engineer decisions, pre-2022) has been disposed of under data retention policy
- Scrapped cars were never garage-verified — their true repairability is permanently unknown
- All detection and evaluation methods must operate without oracle access

---

## Data Access Constraints

| Constraint | Detail |
|---|---|
| Format | Parquet files + database tables (not CSV) |
| PII | Customer data columns require PII handling once NDA is signed |
| Retention | GDPR constraint: data older than 8 years at model build time excluded (internal policy) |
| Pre-ML labels | `pre_ml_label` disposed — unavailable for real-data analysis |
| Exact dates | v1 deployment date, v2 deployment date, and log start date (~2018) are approximate — to be confirmed when data is provided |

---

## Experiment Mapping (Builds → Real Data)

| Build | What changes on real data |
|---|---|
| **00 — EDA** | Run on real log; check ENOL/FNOL split, missing value patterns, score distributions per channel |
| **02 — Loop Detection** | Same `SFPDetector` logic; real scrap rates replace synthetic ~19%/~21.5% figures |
| **03 — Unbiased Evaluation** | IPS correction applied to real OOT holdout; note OOT is also SFP-contaminated |
| **04 — Intervention Analysis** | DiD / RDD on real score drift; ENOL/FNOL mix shift is the primary confound to control |
| **05 — Randomisation** | Cost-benefit uses real garage assessment cost + hire car cost (to be confirmed with ops team) |
| **06 — Causal Mitigation** | DoWhy DAG unchanged; real propensity scores replace synthetic |

---

## Outstanding Clarifications Needed

- [ ] Exact v1 and v2 deployment dates
- [ ] Real log start date (approximate: ~2018)
- [ ] Threshold change — exact value it was changed to, and the date range it was active (even if brief)
- [ ] Garage assessment cost and hire car cost (for Build 05 cost-benefit)
- [ ] ENOL introduction date (needed to separate channel-mix shift from SFP signal)
- [ ] **Enrichment table update mechanics** — when the table is refreshed (~6/9/12 months), what actually changes?
  - Are existing per-ABI-code value fields (`typical_market_value_gbp`, `part_cost_index`) revised to reflect current market prices?
  - Or are only new make/model/year rows appended, with existing rows left unchanged?
  - Or is there a combination (physical specs static; value fields periodically refreshed)?
  - This matters because if values are refreshed, `repair_to_value_ratio` can drift for the same vehicle across model training windows purely due to enrichment changes — a confound for SFP score drift detection that must be controlled for.

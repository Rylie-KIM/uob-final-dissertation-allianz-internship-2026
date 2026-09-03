# Design Pattern

> **Implementation status (2026-07-04, Analysis-Layer OO built).** This document describes the **target** design; the current synthetic chain runs it end-to-end. Exists and working today: `config.py`, the design docs (under `src/docs/`), `scoring/predict.py` + `score_all.py` (superseding `score_all.sh`) + `load_scores.py` (`scoring/ingest.py` deleted 2026-08-09 — log ingestion absorbed into `notebook/real/01_export_v2.ipynb`, since the log source is a version-bound pickle that must be read inside env-v2), `training/retrain.py` (`train.py` + `train_all.py` deleted 2026-08-19 — every version declares `paths.model`, so baseline training had no caller and the driver's `--force` path overwrote the real pickle), `schema.py`, the canonical log landing zone `data/<source>/logs/<v>.parquet` (its source path now declared in `config`, not `manifest.json`), the per-version envs at `src/envs/{v1,v2,v3}/`, the **Analysis-Layer OO impl** — `pipeline/pipeline.py` (`SFPPipeline`), `detector/sfp_detector.py` (`SFPDetector`) + `detector/algorithm/` (`DetectionAlgorithm` ABC → `ResidualPeakAlgorithm`), `mitigator/sfp_mitigator.py` (`SFPMitigator`) + `mitigator/corrector/` (`TrainingDataCorrector` ABC → `ReweightCorrector`; `IPSCorrector` deleted 2026-09-03 — the thesis uses the reweighting corrector only) + `mitigator/policy/` (`InvestigationPolicy` ABC) — and the source→stage data tree `data/synthetic/{inputs,detection,mitigation,reeval}/` with all pkls under `src/models/synthetic/{baseline,mitigated}/`. Also built 2026-07-31: `schema.py`, `threshold.py`, `loaders/`. Still design-only: `preprocessing/`, `training/spec.py`, concrete `InvestigationPolicy` impls, and the whole `data/real/`+`models/real/` side. See `STRUCTURE.md` for the authoritative status. The two layers named there map onto this doc as: **Version Layer** = the per-version scoring/(re)train envs; **Analysis Layer** = the pipeline/detector/mitigator that loads no model.
>
> **Added to the target design 2026-07-14, all design-only:** `estimator/` (`EffectEstimator` ABC → `RDDEstimator` · `ShapDiDEstimator`), `reeval/` (`ReEvalMetric` ABC → `DecisionFlipCount` · `DetectionDelta` · `OracleValidation`), `scoring/attribute.py` (per-row SHAP, Version Layer), and `src/threshold.py` (`tune` · `apply` · `read_off`). The methods they formalise **already exist and run in the notebooks** (04-01 RDD, 04-02 SHAP-DiD) on the DGP; promoting them into the app is what these packages are for.
>
> **Built since (2026-08-01):** `src/threshold.py`, `scoring/attribute.py` + `attribute_all.py`, `estimator/concentration.py` (the Hill / Shannon / Simpson family on mean|φ|, with a comparability guard), and `src/shap_kit.py` + `notebook/real/00_SHAP.ipynb` — the interactive counterpart, one notebook run once per version *inside* that version's env. Still design-only: the `EffectEstimator` ABC and its subclasses, and all of `reeval/`.
>
> `shap_kit` also draws a line the analysis will otherwise blur: **SHAP interaction values** (what the model does jointly with a pair — exact TreeSHAP, and the thing that explains vertical dispersion in a dependence plot) and **column association in X** (pearson / spearman / mutual information — a property of the data) are separate functions with separate names, because they routinely disagree: correlated one-hot columns typically show no interaction at all, and independent features can interact strongly. Interaction values are O(rows × p²), so they are computed on a subsample to *rank* pairs; magnitudes are reported from the full φ matrix.

The interactive route does not weaken the layer rule, it instantiates it: the notebook is Version-Layer code (it opens a pkl, so it runs where that pkl unpickles) and it emits the same `detection/shap/<v>/<v>_attributions.parquet` the headless script does. The Analysis Layer still reads only parquet. What is new is the **library floor**: because that notebook must execute under xgboost 0.72 as well as 3.2.0, `shap_kit` restricts itself to numpy/pandas/matplotlib and draws every figure from raw φ instead of calling shap's plotting layer. See `STRUCTURE.md` § "One notebook, three environments". (Amended 2026-08-24: the floor could not stretch to env-v1's Python 3.5 — `shap_kit.py` is 3.7+ syntax — so v1 runs its own pair, `src/shap_kit_v1.py` + `notebook/real/00_SHAP_v1.ipynb`, per the `<name>_v1.py` twin rule; it emits the same attributions artefact as CSV for the analysis env to convert.)

## Strategy Pattern

Each interchangeable component is encapsulated as a separate class behind a common interface. The core classes (`SFPDetector`, `SFPMitigator`) hold references to these strategies and delegate work to them — they never contain the implementation directly.

This means swapping synthetic data for real data, or swapping one detection algorithm for another, requires changing only the injected class — not the core logic.

## Five Strategy Axes  *(was three; `EffectEstimator` and `ReEvalMetric` added 2026-07-14)*

| Axis | Interface | Implementations |
|---|---|---|
| Data loading | `loaders.load(v)` → `VersionData` | one class, config-driven (see below) |
| Score ingestion | precomputed parquet | one score file per model version, merged on `claim_id` |
| Detection | `DetectionAlgorithm` | `ResidualPeakAlgorithm` |
| **Effect estimation** ★ | **`EffectEstimator`** | `RDDEstimator` (nb 04-01) · `ShapDiDEstimator` (nb 04-02) · `LogitAdjustEstimator` (nb 04-03) |
| Investigation policy | `InvestigationPolicy` | TBD (pending research) |
| Training data correction | `TrainingDataCorrector` | `ReweightCorrector` (nb 03-02's naive/rarity/transport/pnu; transport/pnu emit duplicated `claim_id` split rows, which `retrain.py`'s join expands) |
| **Re-evaluation** ★ | **`ReEvalMetric`** | `DecisionFlipCount` · `DetectionDelta` · `ShapDiDDelta` · `OracleValidation` (synthetic-only) |

## Class Responsibilities

**`SFPPipeline`** — orchestrates the full run. Calls detector, checks result, calls mitigator if SFP is detected.

**`SFPDetector`** — diagnosis. *Is there a loop?* Runs a `DetectionAlgorithm` over (scores, labels). Oracle-free; opens no model. Returns a `DetectionReport`.

**`EffectEstimator`** ★ — *how harmful is it, and by what mechanism?* Distinct from detection: the detector returns a yes/no, an estimator returns a **causal quantity** (04-01: the effect of the scrap decision, in £; 04-02: the label-corruption footprint in the model's feature-dependence structure). Its ABC deliberately mandates three methods, not one:

```python
class EffectEstimator(ABC):
    @abstractmethod
    def assumptions(self) -> list[Assumption]: ...          # named, not buried in a docstring
    @abstractmethod
    def falsify(self, data) -> FalsificationReport: ...     # gates that CAN fail
    @abstractmethod
    def estimate(self, data) -> EffectReport: ...           # refuses to report if the gates failed
```

RDD and DiD are identified only under assumptions that are in general **untestable** (continuity at the cutoff; parallel trends). The discipline that makes them usable — state them, test their observable implications, bound the damage when they fail (Imbens & Wooldridge, P7) — is here encoded in the **type system**: an estimator that has not run its falsification gates cannot emit a number. This is the project's methodological claim, made structural.

`LogitAdjustEstimator` (nb 04-03) is the third implementation, and it is the one where the ABC earns its keep as a **guard against a negative result being mis-reported as positive**. It fits `logit(Y) ~ γ₀ + γ₁·T + γ₂·X` and would read the FTTL effect off `exp(γ₁)`; `statsmodels.Logit` hands back `exp(γ₁)`, the Wald p-value, and the likelihood-ratio p-value directly, so `estimate()` is a few lines. Its `assumptions()` are **unconfoundedness + overlap**, and its `falsify()` gate **fails on this problem by construction**: reading T as the scrap *decision* destroys overlap (`T=1 ⇒ Y=1`, zero honest labels in the treated region — the positivity-dead-at-τ fact), and reading T as the FTTL *era* destroys unconfoundedness (the market confound of 04-02, plus the outcome label changing meaning across the regime). The value is exactly that the gate turns "the most familiar estimator doesn't work here" into an automatic, typed refusal rather than something a reviewer has to catch — and the *way* it fails re-confirms the SFP structure. This is what `statsmodels` is in the analysis env for (04-02 used a bootstrap because HHI has no coefficient to test).

**`SFPMitigator`** — prescription. Takes the report and applies mitigation: updates investigation policy and corrects training data.

**`ReEvaluator`** ★ — *what changed?* The only component that reads **two** artefact sets (before-mitigation and after-mitigation). It **composes** the detector and the estimators over both and diffs them — it re-implements nothing. It earns its own layer for one reason: metrics that **cannot exist for a single model**. `DecisionFlipCount` is the first — "how many cars change fate" is undefined unless two models are held side by side. `OracleValidation` (AUC vs `true_garage_outcome`) also lives here, and its type states what `pipeline.cycle()` currently only says in a comment: **synthetic-only; it cannot run on real Allianz data.**

**`loaders.VersionData`** ★ *(built 2026-07-31; replaces the planned `DataLoader` ABC)* — the one way a notebook or analysis script reaches a version's artefacts. `load("v2").frame` returns targets + scores joined on `claim_id`; `.tau` is read off the production log; `.decisions` applies that version's own rule.

The planned `SyntheticDataLoader` / `RealDataLoader` subclasses were **not built, and should not be**. The synthetic/real difference is a *path* difference, and paths are already resolved by `config.path(kind, version, source)` — a subclass per source would re-introduce, in the type system, exactly the branching the config layer exists to remove. One class, one `source` argument.

What earns the class its place is not loading files but **failing before a guess reaches a figure**. A notebook reading parquet paths directly hard-codes two things it cannot verify — where a version's files live, and what its columns are really called — and a wrong column name does not raise: it selects a different column and produces a plausible chart. That is not hypothetical (`pre_ml_label`, a synthetic column name, sat in `config` marked "confirmed"). `VersionData` resolves both through config and reports which config entry to fix when one is missing.

Only `v1`/`v2`/`v3` exist here. The synthetic **a/b arms** (`v2a`/`v2b`/`v3a`/`v3b`) were retired 2026-07-31: `v2b` was a hand-built counterfactual (pre-ML + v1 log mixed) with no real counterpart, since Allianz trained v2 exactly once. `load("v2a")` raises and says so.

**Score ingestion** — the pipeline does **not** load models or run inference. Each model version is scored offline, once, inside its own environment (see below); the resulting per-version score files are merged on `claim_id` and consumed by the detector. This keeps the analysis runtime free of any model dependency.

**Attribution ingestion** ★ — the same rule, extended. SHAP needs the model *function*, not merely its scores, which would naively force `EffectEstimator` to open a pkl — and **it must not**, because a pkl only unpickles inside its own version env (loading `models/*/baseline/v1.pkl` from the analysis `.venv` raises `ModuleNotFoundError: fttl_v1`, and the real pkls will behave identically). Attribution therefore runs in the **Version Layer** — `scoring/attribute.py`, the sibling of `predict.py` — and emits `detection/shap/<v>/<v>_attributions_<split>.parquet` (per-row φ). `estimator/` reads that parquet and never touches a model. See `STRUCTURE.md` § "Five layers".

> **As built (2026-08-01; split recorded 2026-08-18).** `attribute.py` writes a sidecar `<v>_attributions_<split>_meta.json` alongside the φ parquet, and that file is load-bearing, not documentation: it records the **split** these φ describe (concentration on train is a different statement from concentration on a holdout — see `STRUCTURE.md` § "Some kinds exist ONLY per split"), the SHAP **backend** (interventional TreeSHAP against a shared background, or the booster's own tree-path-dependent `pred_contribs` for a frozen env that cannot take the `shap` dependency), the **claim set** explained, and the estimator's **hyperparameters**. `estimator.concentration.require_comparable()` reads it and *raises* when versions were attributed under different backends. The reasoning is the same one that put attribution in the Version Layer: the analysis layer cannot re-derive any of this from the parquet, and a mixed-backend comparison fails silently — it produces a number that looks entirely reasonable. `attribute_all.py` is the driver, and additionally fixes one shared claim set across versions so a concentration difference cannot be case-mix.

---

## Model Scoring & Environment Isolation

### The problem

All three model versions (v1, v2, v3) are preserved as serialised files within Allianz's internal systems. **Every model version has a genuinely different, mutually incompatible library environment** — this is confirmed, not a "likely". Each version's pickle is bound to the **exact** third-party library versions it was serialised with (its own XGBoost, scikit-learn, and numpy releases), and those versions **differ across v1, v2, and v3** and cannot coexist in a single Python process. The project therefore gives **each version its own independently pinned environment** — `env-v1`, `env-v2`, `env-v3`, all three managed separately — so upgrading or retraining one version can never silently mutate another's.

#### Why "just install every version's repo into one env" does **not** work

This is the single most misunderstood point, so it is stated explicitly. The blocker is **not** the repos' own source code — that installs fine. The blocker is the **third-party numeric stack** each pickle is bound to.

- **A pickle stores class *references* + fitted *state*, never source code.** `joblib.load("v1.pkl")` reconstructs the pipeline by *importing* both (a) the version repo's custom transformer classes **and** (b) the exact library classes the pipeline was built from (`xgboost.sklearn.XGBClassifier`, `sklearn.pipeline.Pipeline`, numpy dtypes). Every one of those imports must resolve **at a compatible version**, or the load raises (`AttributeError` / `InconsistentVersionWarning` / a booster-format error). Having the repo code present is **necessary but not sufficient** — the exact numeric stack must match the pickle's provenance too.
- **A Python environment is a *flat* library pool: one version of each library, period.** Installing packages does **not** isolate their dependencies. If you `uv pip install -e model_repos/{v1,v2,v3}` into one `.venv`, pip must resolve `xgboost` (and `scikit-learn`, `numpy`) to a **single** version shared by all three. When v1 needs (say) `xgboost==1.7` and v3 needs `xgboost==2.1`, that is physically unsatisfiable — pip either errors on resolution, or installs one and the *other* version's pickle then fails to load at runtime.
- **This is unlike npm.** Node's nested `node_modules` lets two packages carry different versions of the same dependency; Python's flat `site-packages` cannot. So `pip install`-ing the repos gives you their code, **not** dependency isolation. The **only** thing that provides isolation is physically separating the environments.

| Approach | Repo code present? | Different lib versions coexist? | Works when versions incompatible? |
|---|---|---|---|
| One `.venv`, install all three repos as packages | ✅ yes | ❌ no (flat `site-packages`, one xgboost/sklearn) | ❌ **no** |
| One `.venv`, `sys.path` swap per version | ✅ yes | ❌ no (still one installed numeric stack) | ❌ **no** |
| **Separate `env-v1`/`env-v2`/`env-v3`** | ✅ yes | ✅ yes (each env its own stack) | ✅ **yes** |

The first two rows only work in the special case where v1/v2/v3 happen to share a compatible numeric stack — which for this project they **do not**. Hence the per-version environment split is mandatory, not a tidiness choice.

Model versions will also continue to be updated over time, so the design must absorb a new version (v4, v5, …) without code changes or new classes.

### Design decision — offline precompute, not runtime subprocess

> **Decided 2026-06-25 (supervisor meeting).** Running each model under its own environment to produce predictions once, offline, is faster and more efficient than having the analysis app spawn a subprocess per call at runtime. The previously documented runtime `SubprocessModelLoader` design is **superseded** and retained only as design history at the end of this file.

**Scoring and analysis are fully decoupled.**

1. **Scoring stage** — each model version is scored **once**, inside its **own** environment (`env-v1` / `env-v2` / `env-v3`, built with uv — see `ENV_MANAGEMENT.md`), by a version-agnostic script (`src/scoring/predict.py`; env-v1 runs `predict_v1.py`, its frozen py3.5 twin — same flags, same behaviour). The active environment *is* that version's environment, so there is never a cross-version import conflict in a single process. Output: one parquet score file per version. **`scoring/attribute.py` is its sibling** and runs in exactly the same way, emitting per-row SHAP attributions instead of scores — because SHAP needs the model object, and the model object only exists inside this env.
2. **Analysis stage** — the pipeline (detector / **estimator** / mitigator / **reeval**) runs in one environment and **never loads a model**. It reads the precomputed per-version score **and attribution** files and merges them on `claim_id`.

Because the two stages never share a process, there is no parent/child coordination, no temp-file marshalling, and no stdout parsing. Scores are cached on disk, so re-running the analysis any number of times does not re-score anything.

### Scoring flow

```
[ Scoring stage — run once per version, each in its own env ]

  uv env: env-v1                          src/scoring/predict_v1.py   (the py3.5 twin)
  ┌───────────────────────────┐           --model    models/synthetic/baseline/v1.pkl
  │ joblib.load(v1 baseline)  │  ───────▶ --features  data/synthetic/inputs/features_v1.parquet   ← per-version
  │ predict_proba(X)[:,1]     │           --version   v1
  │ → parquet                 │           --out       data/synthetic/detection/v1_scores.parquet
  └───────────────────────────┘

  uv env: env-v2                          (same script, different env + args)
  ┌───────────────────────────┐           --model models/synthetic/baseline/v2.pkl
  │ joblib.load(v2 baseline)  │  ───────▶ --features data/synthetic/inputs/features_v2.parquet → detection/v2_scores.parquet
  └───────────────────────────┘

  uv env: env-v3                          (same script, different env + args)
  ┌───────────────────────────┐           --model models/synthetic/baseline/v3.pkl
  │ joblib.load(v3 baseline)  │  ───────▶ --features data/synthetic/inputs/features_v3.parquet → detection/v3_scores.parquet
  └───────────────────────────┘

                    │  v1_scores.parquet   claim_id | model_v1_score
                    │  v2_scores.parquet   claim_id | model_v2_score
                    ▼  v3_scores.parquet   claim_id | model_v3_score

[ Analysis stage — single env, no models loaded ]

  src/scoring/load_scores.py  →  merge on claim_id  →  SFPDetector
       claim_id | model_v1_score | model_v2_score | model_v3_score
```

### Per-version feature matrices (`features_<version>.parquet`)

Each version is scored on its **own** model-ready feature file — `features_v1.parquet`, `features_v2.parquet`, … (`claim_id` + preprocessed features $X$) — never a single shared `features.parquet`. Each file is built by **that version's own repo preprocessing** (the `preprocessing/v{1,2,3}.py` adapters run each repo's feature builder). This is the canonical contract for **both** the synthetic and the real data; **the mechanism is identical** — only the *source of the raw claims* differs:

| | How the per-version files are produced | Are they identical across versions? |
|---|---|---|
| **Synthetic** | The DGP (`data/synthetic/`) generates raw claims; each version's **repo preprocessing** (`V1/V2/V3FeatureBuilder`) then rebuilds $X$ | **No — genuinely different.** Synthetic is only a temporary stand-in and runs the *same* external-repo path as real (decided 2026-07-03), so per-version preprocessing diverges here too. |
| **Real** | Allianz supplies raw claims; each version's own preprocessing pipeline rebuilds $X$ | **No — genuinely different.** Each production version re-implemented preprocessing separately, and the FastAPI serving path may preprocess differently from the AML training path within a version (`problem.md` §2.5 #10/#11). |

**Why per-version rather than one shared file.** On the real data, scoring all versions on a single feature matrix would **not reproduce the production scores** and would fold a preprocessing artefact into any cross-version score-drift comparison, masquerading as SFP signal. The per-version contract removes that confound by construction: each model is always scored on the features *it* would actually see. Synthetic adopts the **same external-repo path** (not a special shared recipe) so the analysis code, the scoring scripts, and the loaders are structurally identical across both — synthetic exists only because the real dataset is not yet available.

**Producing the files.**
- *Synthetic:* `src/data/synthetic/run.py` generates the raw claims; each version's **repo preprocessing** then emits its own `features_<tag>.parquet` (`v1, v2a, v2b, v3a, v3b`) into `data/synthetic/inputs/` — the *same* per-version external-repo path as real, since synthetic is only a stand-in for the unavailable real data.
- *Real:* each version's preprocessing emits its own `features_<version>.parquet` — feasible only if that version's preprocessing code/artefact can be re-run (cf. environment-isolation constraint, `problem.md` §2.5 #7). Where a version's preprocessing cannot be reconstructed, that version is limited to **symptom tracking** (its own emitted scores/decisions) and excluded from the leakage-free quantitative comparison (`problem.md` §2.5 #9).

### Implementation

**`src/scoring/predict.py`** — version-agnostic batch scorer. Run *inside* the target model's own environment; the active env and the args decide which version is scored — the script has no knowledge of which version it is running. Since 2026-09-02 it is the modern (≥3.10) shared v2/v3 worker exposing an importable `predict()` (03_03 calls it in-kernel; the CLI main wraps the same function), and `predict_v1.py` is its frozen py3.5 twin for env-v1.

```python
# src/scoring/predict.py
import argparse
import joblib
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--version",  required=True)   # label only, e.g. "v1"
    p.add_argument("--out",      required=True)
    p.add_argument("--id-col",   default="claim_id")
    args = p.parse_args()

    df    = pd.read_parquet(args.features)
    model = joblib.load(args.model)

    X = df.drop(columns=[args.id_col])
    scores = model.predict_proba(X)[:, 1]          # P(total_loss)

    out = pd.DataFrame({
        args.id_col: df[args.id_col],
        f"model_{args.version}_score": scores,
    })
    out.to_parquet(args.out, index=False)
    print(f"[{args.version}] wrote {len(out)} scores → {args.out}")

if __name__ == "__main__":
    main()
```

**`src/training/retrain.py`** — the one remaining trainer, run *inside* the target version's env exactly like `predict.py`.

- **`retrain.py` (MITIGATED)** — **clones the baseline estimator's hyperparameters** (`sklearn.base.clone`) and fits on the corrector's de-contaminated `--labels` → `config.path("mitigated", v)`.
- **BASELINE training is gone** (`train.py` + `train_all.py`, deleted 2026-08-19). The design had two trainers split by purpose; on real data the baseline half never runs, because all three versions declare `paths.model` and the baseline is that measured pickle, **loaded, never refitted**. `clone()` reads the real hyperparameters straight off it, so the mitigated half needs no fresh-fit sibling. `config.TRAINING_CONFIG` records each version's training call should a refit ever be required.

> **Revised 2026-07-31 — preprocessing is a separate pickle.** The real repos pickle the fitted preprocessing pipeline **separately** from the model, and `predict_proba` takes the **already-preprocessed** matrix. So `features_<v>.parquet` is post-preprocessing, and neither trainer loads, re-fits or even reads a preprocessor: `retrain.py` fits on the *same matrix file* the baseline used, and the re-evaluation invariant therefore holds **by construction** rather than by careful code. Two hard-codings died with it — `base.named_steps["prep"]` (real step names are not `"prep"`; nothing addresses a step by name any more) and the literal `XGBClassifier(n_estimators=80, max_depth=3)` (real v2 uses `reg_alpha=20` / `scale_pos_weight=4.5`, so hand-written hyperparameters would have changed the model between baseline and mitigated — the exact confound the invariant exists to kill). Note `sample_weight` and `scale_pos_weight` **multiply** in XGBoost: with v2's cloned `spw=4.5`, a corrector weight `w` acts as `4.5·w` on positives. The baseline carries the same factor, so the comparison stays fair, but report effective weights when interpreting the corrector's diagnostics.

Holding the features fixed across baseline→mitigated means the only thing that changes is the **label/weight**, so the before→after score difference is attributable to the mitigation (the re-evaluation invariant, `problem.md` §2.5 #12).

> **The invariant extends to the split and to τ (added 2026-07-14).** A decision is `score ≥ τ`, so a *decision*-level comparison (`DecisionFlipCount`) needs a τ for the mitigated model — and that model never ran in production, so its τ **must be tuned**. Two things follow.
> **(1) Tuning belongs here, not in the analysis layer.** It needs an out-of-sample **validation slice**; tuning on the model's own re-scored output is in-sample, gives optimistic precision, lands τ too low, and silently over-scraps. Only the trainer knows the split.
> **(2) The split and the tuning rule must be the version repo's own**, imported dynamically from that repo. Inventing our own split would mean baseline and mitigated tune τ on *different* slices — which alone would move the decision count, re-introducing the confound the invariant exists to kill. `src/threshold.py::tune` (the project's canonical rule: lowest cutoff whose precision ≥ 0.985) is the **fallback**, for when a repo exposes neither.
>
> **@TODO** — `train.py` and `retrain.py` today make **no validation split at all** and tune **no τ**. Both are prerequisites for `DecisionFlipCount` on the app path. See `STRUCTURE.md` § "τ has two sources".

```python
# src/training/retrain.py  (runs inside env-vX)
base = joblib.load(a.baseline)                # the version's own estimator
X    = pd.read_parquet(a.features)            # ALREADY preprocessed — the same file the baseline used
df   = X.merge(pd.read_parquet(a.labels), on=a.id_col)
w    = df[a.weight_col] if a.weight_col in df.columns else None

model = clone(base)                           # hyperparameters copied, fitted state discarded
model.fit(df[feats], df[a.label_col], sample_weight=w)
joblib.dump(model, a.out_model)               # estimator only; the preprocessor is untouched
```

> **Re-evaluation loop.** `SFPMitigator` (analysis env) writes `data/synthetic/mitigation/<v>_corrected.parquet` → `retrain.py` (version env) fits `models/synthetic/mitigated/<v>.pkl` **and tunes `reeval/<v>_tau` on the repo's own validation split** → `predict.py` (version env) emits `data/synthetic/reeval/<v>_mitigated_scores.parquet` → **`ReEvaluator` (analysis env)** compares baseline vs mitigated. Same file-only decoupling as scoring; no model is ever loaded in the analysis env. (Real-data source uses the identical `data/real/…` + `models/real/…` layout.)
>
> **What the comparison consists of (revised 2026-07-14).** Previously this step was just "run `SFPDetector` again and diff `peak0`". That is one metric among several, and it lives entirely in **score space** — the *policy* is never re-derived, so it cannot say whether any car's fate actually changed. `ReEvaluator` therefore holds a list of `ReEvalMetric`s:
> - **`DetectionDelta`** — the old behaviour: re-run the detector on both artefact sets, report Δ`peak0`. Composition, not re-implementation.
> - **`DecisionFlipCount`** — `apply(τ)` to both score sets and count `1→0` (**rescued**: would have been scrapped, now goes to a garage) and `0→1` (**newly scrapped**). This is the first metric that needs a **policy**, and the first that is **undefined for a single model**. It is also the only quantity in the project with a unit a business reads directly: *cars per year*.
> - **`ShapDiDDelta`** ★ — re-runs `ShapDiDEstimator` on the **mitigated** version pair and differences it against the baseline pair: `footprint_before − footprint_after`. A pure **composition** of an `EffectEstimator` over the two artefact sets — the layer's defining move. Strictly stronger than `DetectionDelta`: Δ`peak0` says a score-space symptom eased, `ShapDiDDelta` says the corrector **collapsed the mechanism** (`footprint_after → 0` ⇒ the mitigated models resemble ones trained where the forcing never happened). It is the observational twin of notebook 04-02's positive control. Because it is a DiD, `falsify()` (the parallel-trends probe) re-runs on the mitigated pair automatically, and its partition B rests on corrected data whose labels above τ are transported, never verified — carry the "rows above τ" diagnostic. See `STRUCTURE.md` § "Why `reeval/` is its own layer".
> - **`OracleValidation`** — AUC against `true_garage_outcome`. **Synthetic-only, by type**: no such column exists in real Allianz data. It currently sits in `pipeline.cycle()` guarded only by a comment.
>
> **`DecisionFlipCount` reports two arms, and only one of them is identified.** **fixed-τ** (hold τ = τ_base) asks *"at the company's current cutoff, how many cars change fate once the model is de-contaminated?"* — no extra assumption. **re-tuned-τ** asks what the company *would* have chosen in an uncontaminated world — and on FTTL data that arm **cannot be computed**, because an IPS-corrected training set retains **zero rows above τ** (the routing rule is a hard cutoff, so overlap there is exactly zero). It is reported not as a second estimate but as a **demonstration of that failure**. See `STRUCTURE.md` § "Positivity is dead at τ".

### Where per-version code lives — repo-owned logic, `src/` adapters

Both concerns ultimately **execute the version's own repo code** — our `src/` holds thin **adapters**, not re-implementations. This is the single most important rule when adding code:

- **Preprocessing → PER-VERSION adapter.** Each version *re-implemented* preprocessing differently (`problem.md` §2.5 #10), so this is genuinely divergent code owned by the repo. `src/preprocessing/{v1,v2,v3}.py` are **adapters** that call each repo's feature builder behind one interface (`base.py::Preprocessor`) with a **uniform output contract** (`claim_id` + features → `features_<v>.parquet`). Runs in that version's env. Synthetic uses the **same** per-version repo path (decided 2026-07-03) — no special shared recipe. A version whose preprocessing cannot be reconstructed (§2.5 #7) has no adapter and is limited to symptom tracking (§2.5 #9).
- **Training → repo-owned protocol, split baseline/mitigated trainers.** Every version used the *same* Allianz methodology (2-month maturation exclusion → 6-month OOT → 80/20 fit/val → fit → tune τ to precision ≥ 0.985) — implemented in each version's own repo `train.py`, not re-implemented in `src/`. Two version-agnostic trainers sit in `src/training/`: **`train.py`** produces the baseline pkl (fresh `prep`+model on the *original/production* target = production reproduction), and **`retrain.py`** produces the mitigated pkl (reusing the baseline's fixed `prep`, weighted on the *corrector's* target). Features + `prep` are held fixed across the two, so the only difference is the label/weight (the re-evaluation invariant, §2.5 #12). `src/training/spec.py` carries just the per-version **params** (label column, training window, hyperparameters, τ fallback). Adding a version = a spec entry + pointing at its repo, not new training code.

> **Rule of thumb:** the *actual* preprocessing and training logic lives in the per-version repos; `src/` provides **uniform adapters** (`preprocessing/v{1,2,3}.py`, `training/{train,retrain}.py`) so the Analysis Layer sees one contract regardless of version. All **execute in the Version Layer** (the version's own env), because both feed / build the model.

**`src/scoring/score_all.py`** — orchestrates all versions, each in its **own** uv env, so each scoring run is a separate process (import conflicts are structurally impossible). v1, v2 and v3 each have their own environment. `--source` selects the data source (`synthetic` or `real`) — the whole tree mirrors under `data/<source>/` + `models/<source>/`.

> **Superseded `score_all.sh` on 2026-07-31.** The bash version assembled every path itself (`MODELDIR="src/models/$SOURCE/baseline"`, `--model "$MODELDIR/v1.pkl"`, …), which only holds while all three versions share a directory layout — and the real repos do not. The driver is Python for two reasons: bash cannot import `config.py`, and the company laptop is Windows, where `.sh` does not run. Both `score_all.sh` and `train_all.sh` were deleted the same day.

```python
# src/scoring/score_all.py  (analysis env; spawns each version env)
def command(version: str, source: str) -> list[str]:
    """The one place a scoring invocation is assembled — every path comes from config."""
    return [
        str(config.python_bin(version)),
        str(PREDICT),
        "--model",    str(config.path("model",    version, source)),
        "--features", str(config.path("features", version, source)),
        "--version",  version,
        "--out",      str(config.path("scores",   version, source)),
    ]

for v in args.versions:                      # default: config.VERSION_LABELS
    subprocess.run(command(v, args.source), check=True)
```

> Nothing here names a repo, a pickle file or an interpreter. It asks `config` for a **kind** and a **version** — see `STRUCTURE.md` § "Paths are resolved by KIND". The consequence that matters: `config.path("model", v)` returns the **real production pickle** when `VERSIONS[v]["paths"]["model"]` is declared, and the **baseline pkl we retrained** when it is blank. All three declare it today, so all three take the first route. (v1's production *log* is gone, but its *training* data is not — `inputs_transformed.pkl` survives on `Z:`, user-confirmed 2026-08-19 — so v1 is retrainable like the others.) That choice is now a config edit, not a code edit.

> Each `--features` path points to that version's own feature file, built by that version's own repo preprocessing — on **both** synthetic and real (2026-07-03 decision: synthetic runs the same external-repo path, it is only a temporary stand-in).

**`src/scoring/load_scores.py`** — the analysis-side loader. Reads the precomputed files and merges on `claim_id`. No model dependency, no environment awareness.

```python
# src/scoring/load_scores.py
import pandas as pd
from functools import reduce

def load_scores(score_paths: dict[str, str], id_col: str = "claim_id") -> pd.DataFrame:
    """Merge precomputed per-version score files on claim_id.
    score_paths = {"v1": ".../v1_scores.parquet", "v2": ..., "v3": ...}
    """
    frames = [pd.read_parquet(p) for p in score_paths.values()]
    return reduce(lambda l, r: l.merge(r, on=id_col, how="outer"), frames)
```

### env spec files (still required, for reproducibility)

Each version keeps its **own** spec under `src/envs/<version>/`, used to *build* the env — not to locate an interpreter at runtime. **v1, v2 and v3 are each a separate environment** (no shared `env-v2v3`). These envs are **source-agnostic** — an env is a library stack, so it is *not* duplicated per synthetic/real. There are two ways to pin each one (see `ENV_MANAGEMENT.md` for full detail):

- **Standard** — a pinned `requirements.txt` (pins with `==`), built via `uv venv` + `uv pip install -r`. Direct deps only, no lockfile.
- **Stricter** — a per-version `pyproject.toml` + `uv.lock`, built via `uv sync`. Captures transitive deps + hashes → byte-for-byte reproducible. Preferred for dissertation-grade reproducibility.

```
src/envs/
├── v1/   requirements.txt   (and/or pyproject.toml + uv.lock)
├── v2/   requirements.txt   (and/or pyproject.toml + uv.lock)
└── v3/   requirements.txt   (and/or pyproject.toml + uv.lock)
```

Each version retains an independent spec even when dependencies currently coincide, so retraining or upgrading one version never silently mutates another's pinned environment.

`src/envs/<version>/` holds **specs and build notes only — no code**, and `src/envs/` itself holds no script either (`check_installed.py`, an inspector of built envs, was deleted 2026-08-31 without ever being used), so the no-code-per-version rule stands unqualified. And installing the pins is only half of building the env: the version repo's own modules must also be importable, because that is what unpickles the model. The real repos are not installable packages (no package declaration, no `__init__.py`), so that half is a hand-written `fttl.pth` in each env's `site-packages` rather than an editable install — see `ENV_MANAGEMENT.md` § "Making the version repo importable", which also records the interpreter each real env was actually built on (v1 3.5.6 conda, v2 3.10, v3 3.11).

### Adding a new model version (e.g., v4)

1. Create `src/envs/v4/` with its spec (`requirements.txt`, or `pyproject.toml` + `uv.lock` for the stricter option) and build the env (`uv sync` in that dir, or `uv venv` + `uv pip install -r`).
2. Add a `"v4"` entry to `config.VERSIONS` (`repo_dir`, `python`, its `paths` for the kinds v4 already ships, and its `columns` mapping) and to `config.VERSION_LABELS`. Run `python src/config.py` to confirm nothing is left as a placeholder. (The `builder`/`adapter`/`package` keys were removed 2026-08-09 — preprocessing is always loaded from `paths.preprocessor`, and repo importability comes from the env build, not a config declaration.)
3. Land v4's emitted log via a per-version export notebook (`notebook/real/01_export_v4.ipynb`, the pattern of `01_export_v1/2/3`) → `logs/v4.parquet`, which emits `inputs/{features,targets}_v4_<split>.parquet` directly. Declare `paths.model` and score v4's own production pickle — there is no baseline trainer any more (deleted 2026-08-19).
4. Add `"v4": ".../detection/v4_scores.parquet"` to the `score_paths` dict in the analysis.

No new class, and **no edit to `score_all.py`** — it loops `config.VERSION_LABELS`, so registering the version in config is what adds it to the run. `predict.py` and `load_scores.py` are unchanged; they are version-agnostic.

### Trade-offs

| Approach | Pro | Con |
|---|---|---|
| **Offline precompute (chosen)** | No runtime process-spawn or marshalling overhead; scores cached on disk and reused; each env runs natively; scoring runs debuggable in isolation; analysis code has zero model dependency | Scores are a build artifact — features must be re-scored if they change (fine here: fixed feature set, cross-version comparison) |
| Runtime subprocess `ModelLoader` | Scoring is transparent to the caller (`predict_proba` looks local) | Per-call spawn + temp-file + stdout-parse overhead; parent/child coupling; superseded (see Design History) |
| Docker containers | Full isolation; fully reproducible | Heavyweight for a research app; requires Docker daemon |
| Env-switching via `importlib` | No subprocess | Unreliable; can corrupt `sys.modules` |

This suits the research workload: a fixed feature set is scored across versions for comparison, not scored in real time.

---

## Design History — Superseded: runtime subprocess `ModelLoader`

> **Status: superseded 2026-06-25** by the offline-precompute design above. Preserved here as a record of the design path and the rationale for the change — not current. The text below describes what *was* proposed.

The original plan exposed a `ModelLoader` abstraction with two implementations: `InProcessModelLoader` (v2/v3, same env as the app) and `SubprocessModelLoader(model_path, env_spec)` (v1, different env). The environment was passed as a parameter (`env_spec`, a YAML file under `src/envs/` holding `python_executable`), not encoded in a subclass, so new versions needed no new class.

At call time, `SubprocessModelLoader.predict_proba(X)` would: serialise `X` to a temp `.npy` file → spawn a child process with v1's interpreter (`subprocess.run([...])`) running a version-agnostic `worker.py` → the worker loaded the model, ran `predict_proba`, and printed the scores as JSON to **stdout** → the parent captured stdout (`capture_output=True`), parsed the JSON back into a NumPy array, and deleted the temp file. The caller (`SFPDetector`) saw a uniform `predict_proba` and never knew whether a subprocess was used.

**Why it was dropped:** for this workload v1 is scored a handful of times per run, always over a fixed feature set. Paying a process-spawn + temp-file-write + stdout-parse cost on *every* call — and coupling the analysis runtime to model loading — buys nothing that pre-scoring once to disk does not. The meeting feedback was that running each model under its own env to emit predictions, then analysing the saved predictions, is both faster and simpler. The `base.py` / `inprocess.py` / `subprocess_loader.py` / `worker.py` files were therefore never created.

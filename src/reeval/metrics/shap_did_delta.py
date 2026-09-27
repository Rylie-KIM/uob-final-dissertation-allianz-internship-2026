"""ShapDiDDelta — the `ReEvalMetric` that re-runs the region(A/B) x version DiD on SHAP
concentration a second time, between the MITIGATED retrains of a version pair, and differences it
against the same DiD on the baseline (contaminated) pair (`subsec:shapdiddelta`,
`eq:shapdiddelta`):

    ShapDiDDelta = corruption_footprint(baseline) - corruption_footprint(mitigated axis)

Promoted from `notebook/real/shap_did/04_03_shap_did_mitigation_delta.ipynb`'s §1-§4 machinery
(`region_effect` / `shap_did_table` / the Step-2 `corruption_footprint` loop / `select_local_h`);
that notebook is left as the presentation layer and should import this module rather than
redefine the logic — the same relationship `notebook/real/shap_did/04_02_shap_did_concentration.
ipynb` has to `estimator/effect/shap_did.py`.

**Composition, not reimplementation** (`src/docs/DESIGN.md`'s "Four Strategy Axes" table calls
this "the layer's defining move" for this exact metric). The region/version/Simpson mechanics
already live in `estimator.effect.shap_did` (`version_pair_table`, `cell_simpson`, `tau_by_date`,
`local_region`) and are reused here unchanged rather than re-derived. The one thing that module
did not need before this class existed was reading a MITIGATED model's attributions instead of
the baseline ones, so `version_pair_table()` gained an optional `tag` keyword for exactly this
caller (2026-09-23) — every other caller keeps passing none and gets byte-identical behaviour.
This class does not call `ShapDiDEstimator.estimate()` itself: that method's `falsify()` step
(dose + `attribution_meta`) is specific to a baseline-vs-baseline comparison, and threading a tag
through it too would touch far more of that module for a benefit `compute()` does not need — the
DiD arithmetic itself is two lines, so composing at the level of the shared primitives is the
right grain here.

**What `tag` means.** `"baseline"` is the unmitigated model's own `attributions` kind. Any other
string is one `mitigator.corrector.reweight` axis (`"naive"`, `"transport"`, `"rarity_regime"`,
`"pnu_fixed"`, ...) — the `mitigated_attributions` kind, produced by re-running `attribute.py`
against that axis's retrained pickle inside `03_03_retrain.ipynb`, with the axis name appended to
the file stem. Which axes actually exist on disk for a given (version, split) is presentation-
layer bookkeeping (which mitigation runs have been done so far) — `tag_available()` below only
checks that one file exists; the notebook keeps its own axis registry for discovery.

**`region_effect()`** is the within-(version, split, tag) primitive: Simpson(B) - Simpson(A),
optionally restricted to `corrector_targets` — the population every corrected model in that
(version, split) was actually fit on, see `restrict_to_corrector_targets()`'s own docstring for
why this matters only on TRAIN — and/or banded to `|score - tau| <= h` first via
`estimator.effect.shap_did.local_region`, the same boundary idea `ShapDiDEstimator.
estimate_local()` uses, applied to one (version, split, tag) rather than to a before/after pair.

**`corruption_footprint()`** is `region_effect(after, tag) - region_effect(before, tag)` for ONE
tag across a version pair — algebraically the same DiD `ShapDiDEstimator.estimate()` computes,
just reading `tag`'s attributions on both sides instead of always "baseline". Calling it with
`tag="baseline"` reproduces `ShapDiDEstimator.estimate(v_before, split_before, v_after,
split_after).estimate` for that pair.

**`compute()`** is `eq:shapdiddelta` itself, for one correction axis `tag`. Its `h` argument bands
both footprints identically before differencing — the notebook's Step 3 (a robustness check on
`eq:shapdiddelta`, not a separate metric).

v1 is excluded from every local (`h is not None`) call, the same reason
`ShapDiDEstimator.estimate_local()` excludes it: v1's decision rule is mobility-segmented, not a
scalar tau, and mobility is permanently unrecoverable (`project_v1_mobility_not_a_feature`).
"""
from __future__ import annotations

import pandas as pd

import config
import schema
from estimator.effect.shap_did import cell_simpson, local_region, tau_by_date, version_pair_table
from reeval.metrics.base import ReEvalMetric, ReEvalReport

ID_COL = schema.CLAIM_ID

#: local-band grid-selection defaults — identical to `ShapDiDEstimator`'s own, re-derived
#: independently rather than imported (same reasoning that module's docstring gives for not
#: importing `mitigator.corrector.reweight`'s `band_h` selection either: conceptually similar,
#: not the same computation).
LOCAL_H_GRID = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1)
LOCAL_MIN_CELL_N = 100
LOCAL_H_MAX = 0.05
LOCAL_H_FALLBACK = 0.01


class ShapDiDDelta(ReEvalMetric):
    """See the module docstring. Loads its own inputs from canonical artefacts (`config`,
    `estimator.effect.shap_did`) rather than taking a pre-built frame — the same "reads its own
    artefacts" rule `ShapDiDEstimator` and `estimator.concentration` follow.
    """

    def tag_available(self, version: str, split: str, tag: str) -> bool:
        """Whether `tag`'s attributions exist on disk for (version, split). Every other method
        here asserts `tag="baseline"` exists; this is for checking a MITIGATION axis before
        calling `compute()`/`corruption_footprint()` with it."""
        if tag == "baseline":
            return config.path("attributions", version, split=split).exists()
        base = config.path("mitigated_attributions", version, split=split)
        return base.with_name(f"{base.stem}_{tag}{base.suffix}").exists()

    def restrict_to_corrector_targets(self, table: pd.DataFrame, version: str, split: str) -> pd.DataFrame:
        """Restrict to claims `corrector_targets` covers — the population every corrected model
        for this (version, split) actually trained on (03_02 fits every axis on the SAME
        `corrector_targets` file). Only meaningful for TRAIN: baseline was fit on the FULL
        population but corrected models were fit on this filtered subset, so comparing baseline's
        SHAP on the full population against a corrected model's SHAP on the same full population
        is partly out-of-fit-sample for the corrected side. OOT needs no such restriction — no
        model, baseline or corrected, ever trained on OOT rows, so every tag is equally
        out-of-sample there already. A no-op (returns `table` unchanged) if no `corrector_targets`
        file exists yet for this (version, split).
        """
        ctp = config.split_path("corrector_targets", version, split)
        if not ctp.is_file():
            return table
        covered = pd.read_parquet(ctp, columns=[ID_COL])[ID_COL]
        return table[table[ID_COL].isin(covered)]

    def region_effect(self, version: str, split: str, tag: str,
                       restrict_population: bool = False, h: float | None = None) -> dict:
        """Simpson(B) - Simpson(A) for ONE (version, split, tag). `h` bands to
        `|score - tau| <= h` first via `estimator.effect.shap_did.local_region` (adds
        `region_local`, keeps `region` — see that function's own docstring)."""
        table = version_pair_table(version, split, tag)
        if restrict_population:
            table = self.restrict_to_corrector_targets(table, version, split)
        if h is not None:
            table = local_region(table, version, h)
            sA = cell_simpson(table, drop_extra=["region_local"], region_local="A")
            sB = cell_simpson(table, drop_extra=["region_local"], region_local="B")
        else:
            sA = cell_simpson(table, region="A")
            sB = cell_simpson(table, region="B")
        return {"n_claims": len(table), "simpson_A": sA, "simpson_B": sB, "region_effect": sB - sA}

    def corruption_footprint(self, v_before: str, split_before: str, v_after: str, split_after: str,
                              tag: str, restrict_population: bool = False,
                              h: float | None = None) -> dict:
        """`region_effect(after, tag) - region_effect(before, tag)` — the same DiD arithmetic
        `ShapDiDEstimator.estimate()` uses, reading `tag`'s attributions on both sides. `tag=
        "baseline"` reproduces `ShapDiDEstimator.estimate(v_before, split_before, v_after,
        split_after).estimate` for this pair."""
        before = self.region_effect(v_before, split_before, tag, restrict_population, h)
        after = self.region_effect(v_after, split_after, tag, restrict_population, h)
        return {
            "tag": tag,
            "before": before,
            "after": after,
            "corruption_footprint": after["region_effect"] - before["region_effect"],
        }

    def compute(self, v_before: str, split_before: str, v_after: str, split_after: str,
                tag: str, baseline_tag: str = "baseline", h: float | None = None) -> ReEvalReport:
        """`eq:shapdiddelta`: `corruption_footprint(baseline_tag) - corruption_footprint(tag)`,
        for one correction axis `tag` on this version pair. `before`/`after` on the returned
        `ReEvalReport` are the two corruption_footprint NUMBERS (contaminated, mitigated) — not
        to be confused with `corruption_footprint()`'s own before/after, which are the two
        VERSIONS behind one footprint."""
        contaminated = self.corruption_footprint(v_before, split_before, v_after, split_after,
                                                   baseline_tag, h=h)
        mitigated = self.corruption_footprint(v_before, split_before, v_after, split_after,
                                                tag, h=h)
        delta = contaminated["corruption_footprint"] - mitigated["corruption_footprint"]
        pct_erased = (delta / contaminated["corruption_footprint"]
                      if contaminated["corruption_footprint"] else float("nan"))

        label = f"{baseline_tag}->{tag}" + (f" (h={h})" if h is not None else "")
        print(f"ShapDiDDelta({label}) = {contaminated['corruption_footprint']:+.4f} - "
              f"{mitigated['corruption_footprint']:+.4f} = {delta:+.4f}"
              + (f"  (~{pct_erased:.0%} erased)" if pd.notna(pct_erased) else ""))

        return ReEvalReport(
            before=contaminated["corruption_footprint"],
            after=mitigated["corruption_footprint"],
            delta=delta,
            diagnostics={
                "pair": f"{v_before}({split_before})->{v_after}({split_after})",
                "tag": tag, "baseline_tag": baseline_tag, "h": h, "pct_erased": pct_erased,
                "contaminated": contaminated, "mitigated": mitigated,
            },
        )

    def select_local_h(self, v_before: str, split_before: str, v_after: str, split_after: str,
                        baseline_tag: str = "baseline", grid: tuple[float, ...] = LOCAL_H_GRID,
                        min_n: int = LOCAL_MIN_CELL_N, h_max: float = LOCAL_H_MAX,
                        fallback: float = LOCAL_H_FALLBACK) -> tuple[float, pd.DataFrame]:
        """Grid search on CELL COUNTS ONLY (baseline attributions of both versions) — same
        power-gate discipline `ShapDiDEstimator.select_local_h` uses, re-derived independently
        rather than imported (same module docstring reasoning)."""
        table_before = version_pair_table(v_before, split_before, baseline_tag)
        table_after = version_pair_table(v_after, split_after, baseline_tag)
        dist_before = table_before["score"] - tau_by_date(table_before, v_before)
        dist_after = table_after["score"] - tau_by_date(table_after, v_after)

        rows = []
        for h_candidate in sorted(grid):
            in_before, in_after = dist_before.abs() <= h_candidate, dist_after.abs() <= h_candidate
            n = {"A_before": int((dist_before[in_before] <= 0).sum()),
                 "B_before": int((dist_before[in_before] > 0).sum()),
                 "A_after": int((dist_after[in_after] <= 0).sum()),
                 "B_after": int((dist_after[in_after] > 0).sum())}
            rows.append({"h": h_candidate, **n, "all_pass": min(n.values()) >= min_n})
        table = pd.DataFrame(rows).set_index("h")

        ok = table.index[table["all_pass"] & (table.index <= h_max)]
        if len(ok):
            chosen = float(ok[0])
            print(f"selected h={chosen} (smallest with every region x version cell >= {min_n}, "
                  f"h<={h_max})")
        else:
            chosen = fallback
            print(f"!! no h <= {h_max} reaches {min_n} claims in every cell -- keeping fallback "
                  f"h={fallback}; read compute(..., h={fallback}) as LOW-POWER, not a clean "
                  f"estimate.")
        return chosen, table

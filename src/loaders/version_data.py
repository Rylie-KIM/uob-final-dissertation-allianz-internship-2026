"""VersionData — the one way a notebook gets at a version's artefacts.

WHY THIS EXISTS. A notebook that reads parquet paths and column names directly has hard-coded two
things it cannot verify: where a version's files live, and what its columns are really called.
Both differ per version, and a wrong guess does not raise — it silently selects a different column
and produces a plausible figure. (That is not hypothetical: `pre_ml_label` sat in config marked
"confirmed" for weeks; it is a synthetic column name.)

So notebooks ask for a KIND and a VERSION, and everything real is resolved here:

    from loaders import load
    d = load("v2")
    d.frame          # features + targets + scores, joined on claim_id
    d.tau            # read off the production log — an observed fact, not a constant
    d.decisions      # that version's own rule applied (v1 segmented, v2 piecewise, v3 global)

When a path or column is still a placeholder, loading FAILS with the config entry to fix named.
That failure is the point: it is what stops a guess from reaching a figure.

Only v1/v2/v3 exist here. The synthetic a/b arms (v2a/v2b/v3a/v3b) were a DGP construct — v2b was
a hand-built counterfactual with no real counterpart — and are gone.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config      # noqa: E402
import schema      # noqa: E402
import threshold   # noqa: E402


def _read(kind: str, version: str, source: str, split: str | None = None) -> pd.DataFrame:
    """Resolve a kind to a path and read it, reporting WHICH config entry to fix on failure.

    ``split`` is passed straight to config.path() and validated there, with one addition:
    ``config.ALL_SPLITS`` reads every split of this version and concatenates them, adding a
    ``split`` column. Pooling is never implicit — see VersionData.split.
    """
    if split == config.ALL_SPLITS:
        parts = []
        for one in config.SPLITS[version]:
            part = _read(kind, version, source, one)
            parts.append(part.assign(split=one))
        return pd.concat(parts, ignore_index=True)

    try:
        path = config.path(kind, version, source, split=split)
    except ValueError as exc:
        raise FileNotFoundError(f"[{version}] cannot resolve {kind!r}: {exc}") from None

    if not path.exists():
        raise FileNotFoundError(
            f"[{version}] {kind!r} does not exist at\n    {path}\n"
            f"Either declare config.VERSIONS['{version}']['paths']['{kind}'] (if the real repo "
            f"already ships it) or run the step that produces it."
        )
    # The real repos' data artefacts are pandas pickles (Z: drive), not parquet. A pandas pickle
    # only reliably opens under the pandas that wrote it — so a .pkl declared here should be read
    # inside that version's own env (00_SHAP / attribute.py) and re-exported as parquet for this
    # analysis env. The branch exists so the SAME loader works in both places.
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_parquet(path)


def _join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Inner join on claim_id — and on the split marker too when both sides carry one.

    Under split="all" every per-split frame gains a `split` column. Joining on claim_id alone
    would leave `split_x` / `split_y` and, for a claim that appears in two splits, cross rows
    that never existed. Joining on both keeps one row per (claim, split).
    """
    keys = [schema.CLAIM_ID]
    if "split" in left.columns and "split" in right.columns:
        keys.append("split")
    return left.merge(right, on=keys, how="inner")


@dataclass
class VersionData:
    """One version's artefacts. Every attribute is lazy — you pay only for what you touch.

    ``split`` names WHICH split the per-split artefacts come from (features / targets / scores /
    attributions — config.SPLIT_KINDS). It is required for those, because the export notebooks
    write one file per split and no unsplit base file: there is no "the" feature matrix to
    default to, and silently picking one would hide a case-mix choice under every number
    downstream. `log` and `raw_dataset` are single artefacts and ignore it.

    Pass ``split=config.ALL_SPLITS`` ("all") to pool every split, which adds a ``split`` column
    so the pooling stays visible in the frame itself. Use it where an analysis genuinely needs
    the whole population — cross-version joins like 02_error_inheritance — and say so in the
    write-up, because v1 splits 4 ways and v2/v3 split 3.
    """

    version: str
    source: str = "real"
    split: str | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _require_split(self, kind: str) -> str:
        if self.split is None:
            raise ValueError(
                f"[{self.version}] {kind!r} exists only per split. Load with a split:\n"
                f"    load({self.version!r}, split=...)   one of {config.SPLITS[self.version]}\n"
                f"    load({self.version!r}, split={config.ALL_SPLITS!r})  every split pooled, "
                f"with a `split` column"
            )
        return self.split

    # ---- raw artefacts ---------------------------------------------------------------

    @cached_property
    def features(self) -> pd.DataFrame:
        """claim_id + the POST-preprocessing matrix this version's model consumes.

        Feature columns keep their encoded feature names on purpose — they are the measurement itself.
        """
        return _read("processed_inputs", self.version, self.source,
                     self._require_split("processed_inputs"))

    @cached_property
    def targets(self) -> pd.DataFrame:
        """claim_id + date + decision + observed, in canonical names (see src/schema.py)."""
        return _read("targets", self.version, self.source, self._require_split("targets"))

    @cached_property
    def log(self) -> pd.DataFrame:
        """The full canonical log — targets plus whatever else the production log carried."""
        return _read("log", self.version, self.source)

    @cached_property
    def scores(self) -> pd.DataFrame:
        """claim_id + model_<v>_score, as recomputed by scoring/predict.py."""
        return _read("scores", self.version, self.source, self._require_split("scores"))

    @property
    def score_col(self) -> str:
        return f"model_{self.version}_score"

    @cached_property
    def attributions(self) -> pd.DataFrame:
        """claim_id + one per-row SHAP column per feature + `_base_value`.

        Written by scoring/attribute.py inside that version's env — the notebook reads the parquet
        and never opens a pkl. Feature columns keep their encoded feature names on purpose (see .features).
        """
        return _read("attributions", self.version, self.source,
                     self._require_split("attributions"))

    @cached_property
    def attribution_meta(self) -> dict:
        """The sidecar JSON: backend, perturbation, background size, the id files, feature names.

        Carry it into any cross-version comparison — `estimator.concentration.require_comparable`
        uses it to refuse mixed backends.
        """
        import json

        split = self._require_split("attributions")
        if split == config.ALL_SPLITS:
            raise ValueError(
                f"[{self.version}] each split was attributed in its own run, so there is no one "
                f"meta for {config.ALL_SPLITS!r}. Compare versions one split at a time — "
                f"require_comparable() has nothing to check against a pooled frame."
            )
        path = config.path("attributions", self.version, self.source, split=split)
        meta = path.with_name(path.stem + "_meta.json")
        if not meta.exists():
            raise FileNotFoundError(
                f"[{self.version}] attributions exist but {meta.name} does not. Re-run "
                f"src/scoring/attribute_all.py --split {split} — without the meta there is no "
                f"record of which SHAP backend produced these numbers, and versions must not be "
                f"compared blind."
            )
        return json.loads(meta.read_text(encoding="utf-8"))

    @cached_property
    def mean_abs_shap(self) -> pd.Series:
        """mean|phi| per feature, descending — the input to the concentration measures."""
        from estimator import concentration

        return concentration.mean_abs(self.attributions, id_col=schema.CLAIM_ID)

    # ---- derived ---------------------------------------------------------------------

    @cached_property
    def frame(self) -> pd.DataFrame:
        """targets + scores joined on claim_id. The frame most analysis actually wants.

        Uses an inner join, so `len(frame)` below `len(targets)` means some claims were never
        scored — check that before reading anything into a shrunken sample.
        """
        return _join(self.targets, self.scores)

    @cached_property
    def full(self) -> pd.DataFrame:
        """frame + the feature matrix. Only build this when features are genuinely needed."""
        return _join(self.frame, self.features)

    @cached_property
    def tau(self) -> dict:
        """tau read OFF the production log, plus the determinism check.

        `tau["deterministic"]` is the single most consequential number in the project: True means
        assignment is a hard cutoff, positivity above tau is exactly zero, and propensity-based
        correction is not identified there. See src/threshold.py.
        """
        return threshold.read_off(self.log)

    @cached_property
    def decisions(self):
        """This version's OWN rule applied to its scores — segmented / piecewise / global."""
        df = self.frame
        if self.version == "v1" and schema.MOBILITY in self.log.columns:
            df = df.merge(self.log[[schema.CLAIM_ID, schema.MOBILITY]], on=schema.CLAIM_ID)
        return threshold.apply(self.version, df, score_col=self.score_col)


def load(version: str, source: str = "real", split: str | None = None) -> VersionData:
    """Get one version's data. Nothing is read until an attribute is touched.

    `split` is validated here rather than at first use, so a typo fails on the load() line
    instead of three cells later. Valid names are config.SPLITS[version] — each version's OWN
    (v1 and v2 invert "test"; only v3 has "oot") — or config.ALL_SPLITS to pool them.
    """
    if version not in config.VERSION_LABELS:
        raise KeyError(
            f"unknown version {version!r}; expected one of {config.VERSION_LABELS}. "
            f"(The synthetic a/b arms v2a/v2b/v3a/v3b were retired — v2b had no real counterpart.)"
        )
    if split is not None and split != config.ALL_SPLITS and split not in config.SPLITS[version]:
        raise KeyError(
            f"unknown split {split!r} for {version}; expected one of {config.SPLITS[version]} "
            f"or {config.ALL_SPLITS!r}."
        )
    return VersionData(version=version, source=source, split=split)


def _per_version(split, versions: list[str]) -> dict[str, str | None]:
    """One split name for every version, or a {version: split} mapping, normalised.

    A mapping is not a nicety: only v3 has "oot" and only v1 has "val1"/"val2", so there is no
    single name that means "the holdout" across versions. Saying which one each version uses is
    the whole point of naming splits at the call site.
    """
    if isinstance(split, dict):
        missing = [v for v in versions if v not in split]
        if missing:
            raise KeyError(f"no split given for {', '.join(missing)}")
        return {v: split[v] for v in versions}
    return {v: split for v in versions}


def load_all(versions=None, source: str = "real", split=None) -> dict[str, VersionData]:
    """All versions, keyed by label — for the cross-version comparisons the thesis is about.

    `split` is one name for every version, or {version: split} when they differ (v3's holdout is
    "oot", v2's is "test").
    """
    versions = list(versions or config.VERSION_LABELS)
    chosen = _per_version(split, versions)
    return {v: load(v, source, chosen[v]) for v in versions}


def scores_wide(versions=None, source: str = "real", split=None) -> pd.DataFrame:
    """claim_id + one score column per version, merged. The detector's input shape.

    Outer join: a claim scored by v2 but not v3 keeps a row with NaN, rather than vanishing. An
    inner join here would silently restrict every cross-version comparison to the intersection.

    The per-version `split` marker is dropped: a claim can sit in v2's train and v3's oot, so
    there is no one split to attach to a merged row. If which split a claim came from matters,
    read the versions separately.
    """
    versions = list(versions or config.VERSION_LABELS)
    chosen = _per_version(split, versions)
    frames = [load(v, source, chosen[v]).scores.drop(columns="split", errors="ignore")
              for v in versions]
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=schema.CLAIM_ID, how="outer")
    return out

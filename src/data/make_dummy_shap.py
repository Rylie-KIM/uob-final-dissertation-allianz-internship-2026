"""Write FAKE per-row SHAP attributions into src/data/real/detection/shap/, one dir per version.

WHY THIS EXISTS. `make_dummy_real.py` gives the chain matrices, targets and scores, but it stops
short of the one kind that cannot be generated the same way: attributions come out of
`src/scoring/attribute.py`, which must OPEN A PICKLE inside that version's own env. No pickle
exists on this laptop, so `00_shap_attribution.ipynb` — the notebook the whole concentration
chapter runs through — had nothing to read for v1/v2 and a mislabelled pair for v3.

This generates the shape that notebook consumes: for every version x split x backend, one
`<v>_attributions_<split>[_native].parquet` plus its `_meta.json` sidecar, laid out exactly where
`config.path("attributions", ...)` resolves and named exactly the way `attribute_all.py
--out-suffix` names them.

    python src/data/make_dummy_shap.py                 # every version x split x both backends
    python src/data/make_dummy_shap.py --versions v2   # just one version
    python src/data/make_dummy_shap.py --splits train  # just the splits named
    python src/data/make_dummy_shap.py --clean         # delete what this script wrote

WHAT IS FAITHFUL, AND WHAT IS NOT.

Faithful — these are the properties the notebook actually exercises:
  * feature columns are each version's OWN raw names, taken from features/registry/<v>.json (the
    same authority §2 of the notebook cross-checks against), never invented here and never
    collapsed across versions;
  * the two backends are labelled the way `shap_kit.compute` labels them — no-suffix is
    `shap`/`interventional` with a background, `_native` is `native`/`tree_path_dependent` with
    none — so `concentration.require_comparable` passes WITHIN a run and would raise if the
    labels were ever mixed;
  * phi is additive: `phi.sum(axis=1) + base_value == logit(score)` for the row's dummy score, to
    ~1e-9, so an additivity check downstream has something real to check;
  * `estimator_params` differ across versions the way problem.md §1.4c says the real ones do
    (v2's `reg_alpha=20` against v1's `0`, v2's eta/learning_rate clash, `eval_metric=auc`), so
    §3b prints its confound warning instead of a falsely clean table.

NOT faithful, and deliberately so:
  * every number is a seeded RNG draw. No Allianz value, name or claim is involved.
  * per-feature magnitudes are drawn from ONE distribution with only the seed differing per
    version. No version is made more concentrated than another — baking the SFP direction into
    the stand-in would let a figure "confirm" the hypothesis before any real data arrives.
  * the two backends differ by a correlated perturbation (rho ~ 0.85), not by any real difference
    in reference distribution. A large `interventional` vs `path_dependent` gap here means
    nothing.

Writes only into a tree already carrying `make_dummy_real.py`'s `_DUMMY_DATA` marker, and appends
its files to that marker's list so `make_dummy_real.py --clean` removes them too.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import schema  # noqa: E402

MARKER = "_DUMMY_DATA"
SEED = 20260825
BASE_COL = "_base_value"        # same sentinel column attribute.py writes

#: (out_suffix, backend, perturbation, background_n) — the two rows RUN_SPEC in
#: notebook/real/00_shap_attribution.ipynb reads. The no-suffix file is the CANONICAL one and is
#: interventional; "_native" is the tree-path-dependent sibling. Never "_tree_path_dependent":
#: the suffix names the BACKEND, and attribute_all.py's flag is `--backend native`.
BACKENDS = (
    ("",        "shap",   "interventional",      500),
    ("_native", "native", "tree_path_dependent",   0),
)

#: Plausible library stacks per version, for the `env` block. The xgboost release is the one
#: config declares (authoritative — shap_kit compares that string for equality); the rest are
#: stand-ins consistent with what each env is known to hold.
ENVS: dict[str, dict[str, str]] = {
    "v1": {"python": "3.5.6",  "shap": "0.35.0", "numpy": "1.14.6", "pandas": "0.24.2",
           "sklearn": "0.19.1", "joblib": "0.14.1", "matplotlib": "2.2.5"},
    "v2": {"python": "3.9.13", "shap": "0.41.0", "numpy": "1.21.6", "pandas": "1.3.5",
           "sklearn": "1.0.2", "joblib": "1.1.0", "matplotlib": "3.5.3"},
    "v3": {"python": "3.11.15", "shap": "0.49.1", "numpy": "2.4.6", "pandas": "3.0.5",
           "sklearn": "1.9.0", "joblib": "1.5.3", "matplotlib": "3.11.1"},
}

#: Hyperparameters, per version. These are NOT read off the real repos — they are stand-ins
#: chosen so the DIFFERENCES problem.md §1.4c names are present: v2 regularises hard where v1
#: does not, v2 carries both `eta` and `learning_rate`, and its `eval_metric` is auc while the
#: precision constraint is what the model is judged on. §3b of the notebook reads these.
PARAMS: dict[str, dict] = {
    "v1": {"objective": "binary:logistic", "n_estimators": 300, "max_depth": 6,
           "min_child_weight": 1, "learning_rate": 0.1, "reg_alpha": 0, "reg_lambda": 1,
           "gamma": 0, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1,
           "eval_metric": "logloss", "base_score": 0.5, "silent": True, "seed": 0},
    "v2": {"objective": "binary:logistic", "n_estimators": 500, "max_depth": 4,
           "min_child_weight": 10, "learning_rate": 0.05, "eta": 0.05, "reg_alpha": 20,
           "reg_lambda": 1, "gamma": 0.1, "subsample": 0.9, "colsample_bytree": 0.7,
           "scale_pos_weight": 4.5, "eval_metric": "auc", "base_score": 0.5,
           "use_label_encoder": False, "random_state": 42},
    "v3": {"objective": "binary:logistic", "n_estimators": 802, "max_depth": 3,
           "max_leaves": 18, "min_child_weight": 44, "learning_rate": 0.09973689,
           "reg_alpha": None, "reg_lambda": 1.182373785, "gamma": 0.0004847861,
           "subsample": 0.98046, "colsample_bytree": 0.887008, "scale_pos_weight": 5.552301,
           "grow_policy": "depthwise", "max_delta_step": 61, "n_jobs": -1,
           "enable_categorical": False, "random_state": 123},
}


# ======================================================================================
# the pieces
# ======================================================================================

def _tau_for(version: str) -> tuple[float, str]:
    """The same cutoff make_dummy_real.py drew the scores against, plus how to read it.

    Kept in step with that file on purpose: the scores these attributions are made additive to
    were generated around this value, so a different one here would put the fast-track region in
    a different place than every other dummy artefact says it is.
    """
    rule = config.DECISION_RULES[version]
    if rule["shape"] == "global":
        return float(rule["threshold"]), "single global cutoff"
    if rule["shape"] == "piecewise_global":
        return (float(rule["regimes"][0]["threshold"]),
                f"piecewise in time ({len(rule['regimes'])} regimes) — first regime's cutoff")
    lo, hi = min(rule["thresholds"].values()), max(rule["thresholds"].values())
    return float(hi), (f"segmented on {rule['segment_by']} ({lo} immobile / {hi} mobile) — "
                       f"the higher cutoff; the overlap band is {tuple(rule['overlap_band'])}")


def _feature_cols(version: str, matrix: pd.DataFrame) -> list[str]:
    """The model's own feature columns, in the model's own order.

    The registry is the authority — it is the pickle-extracted record of what the booster
    consumes, and §2 of 00_shap_attribution.ipynb cross-checks the phi columns against exactly
    this file. Deriving them here as "every column except claim_id" would reproduce the bug that
    rule is there to catch: the matrix CARRIES THE TARGET in all three versions.
    """
    target = config.column(version, "observed")
    fallback = [c for c in matrix.columns if c not in (schema.CLAIM_ID, target)]

    reg_path = config.registry_path(version)
    if not reg_path.exists():
        print(f"  no registry ({reg_path.name}) — falling back to matrix columns minus "
              f"{target!r}")
        return fallback

    listed = [str(c) for c in (json.loads(reg_path.read_text(encoding="utf-8"))
                               .get("model_features") or [])]
    if not listed:
        print(f"  registry lists no model_features — falling back to matrix columns minus "
              f"{target!r}")
        return fallback

    absent = [c for c in listed if c not in matrix.columns]
    if absent:
        raise SystemExit(
            f"\n[{version}] {reg_path.name} names {len(absent)} feature(s) the matrix does not "
            f"have: {absent[:8]}\nRegenerate one or the other — do not write attributions "
            f"against a matrix the registry disagrees with.\n")
    return listed


def _margins(version: str, split: str, n: int, ids: np.ndarray) -> np.ndarray:
    """logit(score) for the rows being explained — the value phi must sum to.

    `model_output` is "raw" for both backends (shap_kit passes it explicitly), so the additive
    quantity is the log-odds margin, not the probability. Missing scores are not an error here:
    without them the phi still has the right shape, it just is not tied to anything.
    """
    path = config.path("scores", version, "real", split=split)
    if not path.exists():
        print(f"  no scores at {path.name} — phi will be additive to a drawn margin instead")
        return np.full(n, np.nan)
    df = pd.read_parquet(path)
    col = next(c for c in df.columns if c != schema.CLAIM_ID)
    s = df.set_index(schema.CLAIM_ID)[col].reindex(ids).to_numpy(dtype=float)
    return np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))


def _phi(rng: np.random.RandomState, n: int, p: int, margins: np.ndarray,
         other: np.ndarray | None) -> tuple[np.ndarray, float]:
    """(n, p) attributions that sum, with the base value, to `margins`.

    `other` is the sibling backend's phi when there is one: the second backend is drawn as a
    correlated perturbation of the first rather than independently, because two INDEPENDENT
    draws would make the backend comparison in §5 of the notebook a comparison of noise, and two
    IDENTICAL ones (what the current v3 pair is) would make it a tautology.
    """
    scale = rng.lognormal(mean=-1.2, sigma=0.9, size=p)     # per-feature magnitude, power-law-ish
    z = rng.standard_normal((n, p))
    if other is not None:
        # Standardise by the sibling's own per-column SD (not the SD of its absolute values —
        # that runs ~0.6x smaller for a symmetric draw and would inflate this backend's whole
        # magnitude scale, showing up as a fake mean|phi| gap between the two backends).
        rho = 0.85
        z = rho * (other / np.maximum(other.std(axis=0), 1e-9)) + np.sqrt(1 - rho ** 2) * z
    raw = z * scale

    m = margins if np.isfinite(margins).all() else rng.standard_normal(n) * 1.5 - 1.0
    base = float(np.mean(m))

    # Push the whole per-row residual onto the features, split in proportion to |phi| so the
    # correction lands on the features that were already doing the work rather than smearing
    # evenly over the tail (which would flatten every concentration measure downstream).
    w = np.abs(raw)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return raw + (m - base - raw.sum(axis=1))[:, None] * w, base


def _meta(version: str, split: str, cols: list[str], phi: np.ndarray, base: float,
          backend: str, perturbation: str, background_n: int, seed: int,
          features_path: pathlib.Path) -> dict:
    tau, tau_note = _tau_for(version)
    xgb_version = config.VERSIONS[version]["xgboost"]
    env = dict(ENVS[version])
    return {
        "version": version,
        "split": split,
        "model_path": "<dummy — src/data/make_dummy_shap.py, no pickle was opened>",
        "features_path": str(features_path),
        "features_provenance": "processed_inputs file (parquet)",
        "estimator": "XGBClassifier",
        "estimator_params": PARAMS[version],
        "feature_order": "exact",
        "backend": backend,
        "perturbation": perturbation,
        "model_output": "raw",
        "note": (f"shap {env['shap']}" if backend == "shap" else f"xgboost {xgb_version}"),
        "n_rows": int(phi.shape[0]),
        "n_features": len(cols),
        "feature_names": cols,
        "background_n": int(background_n),
        "explain_ids_file": None,
        "background_ids_file": None,
        "seed": int(seed),
        "tau_used": tau,
        "tau_note": tau_note,
        "base_value": base,
        "env": {"version": version, "python": env["python"],
                "executable": str(config.ROOT / "src" / "envs" / version / ".venv" / "bin"
                                  / "python3"),
                "xgboost": xgb_version, "numpy": env["numpy"], "pandas": env["pandas"],
                "matplotlib": env["matplotlib"], "shap": env["shap"], "sklearn": env["sklearn"],
                "joblib": env["joblib"], "xgboost_expected": xgb_version},
        "dummy": True,
    }


# ======================================================================================
# writing
# ======================================================================================

def build(version: str, splits: list[str], written: list[pathlib.Path]) -> None:
    tau, tau_note = _tau_for(version)
    print(f"\n[{version}]  splits {tuple(splits)}  ·  tau {tau}  ·  {tau_note}")

    for split in splits:
        features_path = config.path("processed_inputs", version, "real", split=split)
        if not features_path.exists():
            print(f"  {split}: no {features_path.name} — run make_dummy_real.py first; skipping")
            continue

        matrix = pd.read_parquet(features_path)
        cols = _feature_cols(version, matrix)

        # One RNG per (version, split): the row sample and both backends come out of it in a
        # fixed order, so re-running reproduces the file byte for byte.
        rng = np.random.RandomState(SEED + 1000 * int(version[1:]) + len(split) * 17
                                    + sum(map(ord, split)))

        # attribute_all.py explains a SAMPLE (--rows). Every dummy split is smaller than the 5000
        # the notebook's command block passes, so all rows are explained — but the order is
        # shuffled, as a sample's would be, so nothing downstream may assume phi rows line up
        # positionally with the matrix.
        order = rng.permutation(len(matrix))
        ids = matrix[schema.CLAIM_ID].to_numpy()[order]
        margins = _margins(version, split, len(ids), ids)

        previous = None
        for out_suffix, backend, perturbation, background_n in BACKENDS:
            phi, base = _phi(rng, len(ids), len(cols), margins, previous)
            previous = phi

            out = pd.DataFrame(phi, columns=cols)
            out.insert(0, schema.CLAIM_ID, ids)
            out[BASE_COL] = base

            path = config.path("attributions", version, "real", split=split)
            if out_suffix:
                path = path.with_name(path.stem + out_suffix + path.suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(path, index=False)
            written.append(path)

            meta_path = path.with_name(path.stem + "_meta.json")
            meta_path.write_text(json.dumps(
                _meta(version, split, cols, phi, base, backend, perturbation, background_n,
                      SEED, features_path), indent=2), encoding="utf-8")
            written.append(meta_path)

            gap = float(np.abs(phi.sum(axis=1) + base - margins).max()) \
                if np.isfinite(margins).all() else float("nan")
            top = sorted(zip(cols, np.abs(phi).mean(axis=0)), key=lambda kv: -kv[1])[:3]
            print(f"  {path.name:<46} {len(out):>5} rows x {len(cols):>3} feats  "
                  f"{backend}/{perturbation}  additivity gap {gap:.2e}")
            print(f"      top mean|phi|: " + ", ".join(f"{n}={v:.4f}" for n, v in top))


# ======================================================================================
# guards + entry point
# ======================================================================================

def _root() -> pathlib.Path:
    return config.ROOT / "src" / "data" / "real"


def _marker(root: pathlib.Path) -> pathlib.Path:
    marker = root / MARKER
    if not marker.exists():
        raise SystemExit(
            f"\nno {MARKER} in {root.relative_to(config.ROOT)} — this is not a dummy tree, so "
            f"refusing to write fake attributions into it.\nOn a machine that has the real "
            f"artefacts, produce these with src/scoring/attribute_all.py instead.\n")
    return marker


def _record(marker: pathlib.Path, written: list[pathlib.Path]) -> None:
    """Add what we wrote to the marker's file list, so make_dummy_real.py --clean removes it."""
    payload = json.loads(marker.read_text(encoding="utf-8"))
    listed = set(payload.get("files", []))
    listed.update(str(p.relative_to(config.ROOT)) for p in written)
    payload["files"] = sorted(listed)
    payload.setdefault("also", []) 
    note = "attributions from src/data/make_dummy_shap.py (seed %d)" % SEED
    if note not in payload["also"]:
        payload["also"].append(note)
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clean(root: pathlib.Path) -> None:
    marker = _marker(root)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    shap_dir = root / "detection" / "shap"
    removed = [f for f in payload.get("files", []) if "/detection/shap/" in f]
    for rel in removed:
        (config.ROOT / rel).unlink(missing_ok=True)
    payload["files"] = sorted(set(payload.get("files", [])) - set(removed))
    payload["also"] = [a for a in payload.get("also", []) if "make_dummy_shap" not in a]
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for d in sorted((p for p in shap_dir.rglob("*") if p.is_dir()), reverse=True) if shap_dir.exists() else []:
        if not any(d.iterdir()):
            d.rmdir()
    print(f"removed {len(removed)} attribution files listed in {MARKER}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--versions", nargs="+", default=list(config.VERSION_LABELS))
    p.add_argument("--splits", nargs="+", default=None,
                   help="split names to write (default: every split each version has)")
    p.add_argument("--clean", action="store_true", help="delete the dummy attributions and stop")
    a = p.parse_args()

    root = _root()
    if a.clean:
        _clean(root)
        return

    marker = _marker(root)
    written: list[pathlib.Path] = []
    for v in a.versions:
        if v not in config.VERSION_LABELS:
            raise SystemExit(f"\nunknown version {v!r}; expected one of {config.VERSION_LABELS}\n")
        splits = list(config.SPLITS[v]) if a.splits is None else [
            s for s in a.splits if s in config.SPLITS[v]]
        skipped = [] if a.splits is None else [s for s in a.splits if s not in config.SPLITS[v]]
        for s in skipped:
            print(f"[{v}] has no split {s!r} (its splits are {config.SPLITS[v]}) — skipped")
        if splits:
            build(v, splits, written)

    _record(marker, written)
    print(f"\nwrote {len(written)} files ({len(written) // 2} attribution parquets + sidecars)")
    print("delete with: python src/data/make_dummy_shap.py --clean")


if __name__ == "__main__":
    main()

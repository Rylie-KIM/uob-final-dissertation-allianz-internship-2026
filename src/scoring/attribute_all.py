"""Attribute every model version, each inside its OWN env. The sibling of score_all.py.

EXPLAIN ROWS vs BACKGROUND ROWS — two different things, and the flags are separate:

  explain     the claims whose phi is computed. These become the rows of the output parquet.
  background  the REFERENCE population for interventional TreeSHAP. phi is a difference, not an
              absolute: base_value (the mean output over the background) + sum(phi) = f(x). Change
              the background and every phi changes, which is why it is recorded in the meta.

SHARED CLAIMS ARE THE EXCEPTION, NOT THE DEFAULT (corrected 2026-08-18). Explaining every version
on the SAME claims would remove case-mix from a cross-version difference — but the versions' data
windows make it impossible in general: v2 runs 2018-01 to 2020-09 and v3 runs 2023-06 to 2026-05,
so they share no claim at all, and there is no set of claims common to all three. Per-version
sampling is therefore what this driver does unless told otherwise, and the case-mix caveat is
STANDING, not conditional. `--shared-claims` opts into the intersection for a pair that genuinely
overlaps in time (v1 and v2 do — v2's window sits inside v1's production era, which is what
02_error_inheritance relies on); it fails loudly when the intersection is empty.

Note what a shared claim set would and would not fix. It removes case-mix. It does NOT make the
phi vectors directly comparable: the versions consume different feature matrices under different
column names, so any cross-version reading still goes through the hand-confirmed mapping in
features/feature_overlap.json, restricted to the mapped features and renormalised within them.

Two jobs:

1. Choose the explain/background rows — per version by default, shared with `--shared-claims`.
   Reading a parquet needs no model, so this part runs in the analysis env.
2. Resolve model / features / output by KIND from config and launch each version's interpreter.

--split is required, and it is not bookkeeping: SHAP concentration measured on the train split
is a statement about the fitted function on data it saw, and on a holdout it is a different
statement. The chosen split lands in the output filename and the sidecar meta, so the number can
never be reported without it.

    python src/scoring/attribute_all.py --split test          # same name for every version
    python src/scoring/attribute_all.py --split v2=test v3=oot
    python src/scoring/attribute_all.py --split test --versions v2 v3
    python src/scoring/attribute_all.py --split test --rows 5000 --background 500
    python src/scoring/attribute_all.py --split test --backend native   # env without `shap`
    python src/scoring/attribute_all.py --split test --dry-run          # print, run nothing

    # tree-path-dependent SHAP on the SAME split already attributed under the default
    # interventional backend — --out-suffix keeps both parquets on disk instead of one
    # overwriting the other:
    python src/scoring/attribute_all.py --split test --backend native --out-suffix _native

    # only where the windows actually overlap:
    python src/scoring/attribute_all.py --split test --versions v1 v2 --shared-claims
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import schema  # noqa: E402

ATTRIBUTE = pathlib.Path(__file__).parent / "attribute.py"


def _ids(version: str, source: str, split: str) -> pd.Index:
    """Just the claim_id column of a version's feature matrix — no model involved."""
    path = config.path("processed_inputs", version, source, split=split)
    if not path.exists():
        raise SystemExit(
            f"\n[{version}] processed_inputs for split {split!r} do not exist at\n    {path}\n"
            f"Run the export notebook / build step first (or declare "
            f"config.VERSIONS['{version}']['paths']['processed_inputs'] if the repo ships it).\n"
        )
    return pd.Index(pd.read_parquet(path, columns=[schema.CLAIM_ID])[schema.CLAIM_ID].unique())


def common_sample(versions: list[str], source: str, splits: dict[str, str], n_explain: int,
                  n_background: int, seed: int,
                  dry_run: bool = False) -> tuple[pathlib.Path, pathlib.Path, int]:
    """Write the shared explain/background id files. Returns (explain, background, n_common).

    The intersection is taken WITHIN the chosen splits: a claim in v2's test split and v3's train
    split is not a shared row for this run, because the two attributions would then describe
    different populations. That is the same reason the split is named at all.
    """
    shared = _ids(versions[0], source, splits[versions[0]])
    for v in versions[1:]:
        shared = shared.intersection(_ids(v, source, splits[v]))
    if len(shared) == 0:
        raise SystemExit(
            f"\nversions {versions} share no claim_id at all, so they cannot be explained on the "
            f"same rows. This is the expected answer whenever v3 is involved: its window "
            f"(2023-06..2026-05) does not meet v2's (2018-01..2020-09). Either restrict "
            f"--versions to a pair that overlaps in time (v1 v2), or drop --shared-claims and "
            f"carry the case-mix caveat.\n"
        )

    ordered = pd.Series(sorted(shared))
    n_take = min(n_explain + n_background, len(ordered))
    drawn = ordered.sample(n=n_take, random_state=seed).reset_index(drop=True)
    explain = drawn.iloc[:min(n_explain, len(drawn))]
    background = drawn.iloc[len(explain):]        # disjoint from explain, by construction

    out_dir = config.ROOT / "src" / "data" / source / "inputs"
    e_path = out_dir / "shap_explain_ids.parquet"
    b_path = out_dir / "shap_background_ids.parquet"
    if not dry_run:                    # a dry run must not touch the filesystem either
        out_dir.mkdir(parents=True, exist_ok=True)
        explain.rename(schema.CLAIM_ID).to_frame().to_parquet(e_path, index=False)
        background.rename(schema.CLAIM_ID).to_frame().to_parquet(b_path, index=False)

    suffix = "  (dry run — not written)" if dry_run else ""
    print(f"common claims across {', '.join(v + '/' + splits[v] for v in versions)}: {len(shared)}")
    print(f"  explain    {len(explain):>6}  -> {e_path}{suffix}")
    print(f"  background {len(background):>6}  -> {b_path}{suffix}")
    if len(background) == 0:
        print("  NOTE: no rows left for the background — the shap backend will fall back to "
              "tree-path-dependent. Lower --rows or raise the overlap.")
    return e_path, b_path, len(shared)


def command(version: str, source: str, split: str, backend: str, seed: int, explain_ids,
            background_ids, rows: int, background: int, out_suffix: str = "") -> list[str]:
    """The one place an attribution invocation is assembled — every path comes from config.

    `out_suffix` exists for ONE reason: config.path("attributions", ...) is keyed only by
    (version, split), not by backend. Attributing the SAME split under two backends — e.g. the
    default interventional `shap` and the tree-path-dependent `native`, to see whether the
    reference/perturbation choice itself moves the concentration numbers — would otherwise have
    the second run silently overwrite the first at the identical path. Passing e.g. "_native"
    keeps both on disk side by side; the default "" reproduces today's filename exactly.
    """
    out = config.path("attributions", version, source, split=split)
    if out_suffix:
        out = out.with_name(out.stem + out_suffix + out.suffix)
    cmd = [
        str(config.python_bin(version)),
        str(ATTRIBUTE),
        "--model",    str(config.path("model", version, source)),
        "--features", str(config.path("processed_inputs", version, source, split=split)),
        "--version",  version,
        "--split",    split,
        "--out",      str(out),
        "--backend",  backend,
        "--seed",     str(seed),
    ]
    if explain_ids:
        cmd += ["--explain-ids", str(explain_ids)]
    else:
        cmd += ["--rows", str(rows)]
    if background_ids:
        cmd += ["--background-ids", str(background_ids)]
    else:
        cmd += ["--background", str(background)]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="real", choices=("real", "synthetic"))
    p.add_argument("--versions", nargs="+", default=list(config.VERSION_LABELS))
    p.add_argument("--split", nargs="+", default=None,
                   help="split to attribute: one name for all versions, or v2=test v3=oot")
    p.add_argument("--rows", type=int, default=5000, help="claims to explain")
    p.add_argument("--background", type=int, default=500, help="interventional background size")
    p.add_argument("--backend", default="auto", choices=("auto", "shap", "native"))
    p.add_argument("--out-suffix", default="",
                   help="appended to the output filename's stem, e.g. '_native' — use this to "
                        "attribute the SAME split under a second --backend without overwriting "
                        "the first run's parquet (config.path() is keyed by (version, split) "
                        "only, not by backend).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shared-claims", action="store_true",
                   help="explain every version on the INTERSECTION of their claim_ids. Only "
                        "possible where the windows overlap (v1/v2); v3 shares no claim with "
                        "either, so this fails loudly rather than silently sampling.")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    splits = config.resolve_splits(a.split, a.versions)

    e_path = b_path = None
    if a.shared_claims:
        e_path, b_path, _ = common_sample(a.versions, a.source, splits, a.rows, a.background,
                                          a.seed, dry_run=a.dry_run)
    else:
        print("Per-version sampling (the default): each version is explained on its OWN rows.\n"
              "  A cross-version difference is therefore confounded with case-mix. That caveat is\n"
              "  standing, not optional — the data windows leave no claim common to all three\n"
              "  versions (v2 2018-01..2020-09, v3 2023-06..2026-05). Pass --shared-claims only\n"
              "  for a pair that genuinely overlaps, e.g. --versions v1 v2.")

    for v in a.versions:
        try:
            cmd = command(v, a.source, splits[v], a.backend, a.seed, e_path, b_path, a.rows,
                          a.background, out_suffix=a.out_suffix)
        except ValueError as exc:          # a placeholder config needs is still blank
            raise SystemExit(f"\n[{v}] {exc}\n")

        print(f"\n[{v}] split={splits[v]}\n      " + " \\\n      ".join(cmd))
        if not a.dry_run:                  # a dry run must not touch the filesystem either
            pathlib.Path(cmd[cmd.index("--out") + 1]).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True)

    print(f"\nDone ({'dry run' if a.dry_run else 'attributed'}): {', '.join(a.versions)}")


if __name__ == "__main__":
    main()

"""Attribute every model version, each inside its OWN env. The sibling of score_all.py.

Two jobs, and the first one is the reason this file exists rather than three shell calls:

1. **Pick ONE set of claims and give it to every version.** SHAP values are only comparable across
   versions if they describe the same rows — otherwise a shift in the concentration statistic can
   be a shift in case-mix. This driver intersects the versions' `claim_id`s, takes a seeded sample,
   and writes it to `inputs/shap_explain_ids.parquet` + `inputs/shap_background_ids.parquet`, which
   every worker then reads. Reading a parquet needs no model, so this part runs in the analysis env.
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

If the versions share no `claim_id`s at all (their windows are disjoint — v2 and v3 nearly are; see
README on the ~2yr gap), pass `--per-version-sample`: each version is then explained on its own
rows, and any cross-version comparison must carry that as a caveat.
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
            f"same rows. Either restrict --versions to an overlapping pair, or pass "
            f"--per-version-sample and carry the non-comparability as a caveat.\n"
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
            background_ids, rows: int, background: int) -> list[str]:
    """The one place an attribution invocation is assembled — every path comes from config."""
    cmd = [
        str(config.python_bin(version)),
        str(ATTRIBUTE),
        "--model",    str(config.path("model", version, source)),
        "--features", str(config.path("processed_inputs", version, source, split=split)),
        "--version",  version,
        "--split",    split,
        "--out",      str(config.path("attributions", version, source, split=split)),
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-version-sample", action="store_true",
                   help="skip the shared id files; each version samples its own rows (NOT comparable)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    splits = config.resolve_splits(a.split, a.versions)

    e_path = b_path = None
    if not a.per_version_sample:
        e_path, b_path, _ = common_sample(a.versions, a.source, splits, a.rows, a.background,
                                          a.seed, dry_run=a.dry_run)
    else:
        print("--per-version-sample: each version is explained on its OWN rows. A cross-version "
              "difference is then confounded with case-mix — say so wherever it is reported.")

    for v in a.versions:
        try:
            cmd = command(v, a.source, splits[v], a.backend, a.seed, e_path, b_path, a.rows,
                          a.background)
        except ValueError as exc:          # a placeholder config needs is still blank
            raise SystemExit(f"\n[{v}] {exc}\n")

        print(f"\n[{v}] split={splits[v]}\n      " + " \\\n      ".join(cmd))
        if not a.dry_run:                  # a dry run must not touch the filesystem either
            pathlib.Path(cmd[cmd.index("--out") + 1]).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True)

    print(f"\nDone ({'dry run' if a.dry_run else 'attributed'}): {', '.join(a.versions)}")


if __name__ == "__main__":
    main()

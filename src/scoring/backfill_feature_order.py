"""Add the `feature_order` field to attribution metas written before the check existed.

WHY THIS INSTEAD OF RE-RUNNING. The phi values on disk are fine: both 00_SHAP notebooks call
`shap_kit.align()`, so their matrices WERE in the booster's trained order — the order was correct,
it was just never recorded. What is missing is one field in the sidecar JSON, and recomputing SHAP
to recover a field that can be derived from the file itself would be hours for nothing.

`meta["feature_names"]` is the column order the phi were computed against. Comparing it to the
model's own trained names answers exactly what `feature_order` reports, without touching a phi
value. Only the JSON is rewritten; the parquet is never opened.

RUNS INSIDE THAT VERSION'S ENV — it loads the pickle, so it needs the version repo importable:

    src/envs/v2/.venv/bin/python src/scoring/backfill_feature_order.py --version v2
    src/envs/v3/.venv/bin/python src/scoring/backfill_feature_order.py --version v3

    --dry-run          report what would change, write nothing
    --meta <path> ...  specific files instead of every meta under that version's shap directory

v1 is NOT covered: env-v1 is Python 3.5 and cannot run this file. It gets its own
`backfill_feature_order_v1.py`, written separately.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import joblib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config     # noqa: E402
import shap_kit   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True, help="our label: v2 or v3 (v1 has its own script)")
    p.add_argument("--source", default="real")
    p.add_argument("--model", default=None, help="override; defaults to config.path('model', ...)")
    p.add_argument("--meta", nargs="*", default=None, help="specific _meta.json files")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.version == "v1":
        raise SystemExit("v1 runs on Python 3.5 — use backfill_feature_order_v1.py instead.")

    model_path = pathlib.Path(a.model) if a.model else config.path("model", a.version, a.source)
    est = joblib.load(model_path)          # needs that version's repo importable in THIS env
    trained = shap_kit.model_feature_names(est)
    print(f"[{a.version}] model {model_path}")
    print(f"        trained feature names: {len(trained)}"
          + ("" if trained else "  <- none exposed; every file will read 'unverified'"))

    if a.meta:
        metas = [pathlib.Path(m) for m in a.meta]
    else:
        # every backend/split under this version's shap directory, found off the kind's own path
        base = config.path("attributions", a.version, a.source, split=config.SPLITS[a.version][0])
        metas = sorted(base.parent.glob(f"{a.version}_attributions_*_meta.json"))
    if not metas:
        raise SystemExit(f"[{a.version}] no _meta.json found — nothing to backfill.")

    changed = skipped = flagged = 0
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        names = meta.get("feature_names")
        if not names:
            print(f"  {meta_path.name:<48} SKIP — no feature_names to check against")
            skipped += 1
            continue

        status = shap_kit.feature_order(est, names)
        previous = meta.get("feature_order")
        if previous == status:
            print(f"  {meta_path.name:<48} already {status!r}")
            skipped += 1
            continue

        if status in ("reordered", "set_mismatch"):
            # Not a formatting problem: SHAP is positional, so these phi are attributed to the
            # wrong features and the file cannot be used. Record it and say so — do not repair.
            print(f"  {meta_path.name:<48} ** {status.upper()} — these phi are WRONG, "
                  f"recompute this file **")
            flagged += 1

        meta["feature_order"] = status
        if not a.dry_run:
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  {meta_path.name:<48} {'would set' if a.dry_run else 'set'} -> {status!r}"
              + (f"   (was {previous!r})" if previous is not None else ""))
        changed += 1

    verb = "would be written" if a.dry_run else "written"
    print(f"\n[{a.version}] {changed} {verb}, {skipped} unchanged, {flagged} FLAGGED"
          + ("   (dry run — nothing on disk changed)" if a.dry_run else ""))
    if flagged:
        raise SystemExit(f"{flagged} file(s) have misordered features — recompute them.")


if __name__ == "__main__":
    main()

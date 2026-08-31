"""v1's backfill of the `feature_order` meta field -- the py3.5 twin of backfill_feature_order.py.

WHY A SEPARATE FILE (three reasons, all hard):

  1. env-v1 is Python 3.5.6. The shared script is full of f-strings.
  2. The shared script does `import config`, and src/config.py cannot even be IMPORTED under
     3.5 -- it opens with `from __future__ import annotations` (3.7+) and uses PEP 585
     annotations (`dict[str, ...]`, 3.9+). So paths here come from shap_kit_v1 instead, which
     is v1's own stand-in for config.
  3. v1's attributions are CSV, not parquet (env-v1 has no parquet engine). The meta name is
     the same either way, but the sibling data file is not.

AND ONE THAT IS NOT ABOUT SYNTAX. v1 routinely has no trained names in the pickle: xgboost 0.72
often exposes only f0/f1/..., which shap_kit_v1.model_feature_names() rejects on purpose. So
00_SHAP_v1.ipynb falls back to features/registry/v1.json for the column order, and this script
must resolve the order the SAME way or it would report "unverified" for runs whose order was in
fact checked -- against the registry. The status then carries "(via registry)" so the source of
the verdict travels with the file.

Nothing is recomputed: phi are never read, the CSV/parquet is never opened. Only the JSON.

env-v1 is a CONDA env (Python 3.5 predates uv), so its interpreter sits directly in the env
directory -- no `Scripts\\`, unlike env-v2/env-v3:

    src\\envs\\v1\\.venv\\python.exe src\\scoring\\backfill_feature_order_v1.py ^
        --model model_repos\\real\\<v1-repo>\\outputs\\fasttracker_xgb.pkl --dry-run

Drop --dry-run to write.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shap_kit_v1 as sk    # noqa: E402  -- v1's config stand-in AND the shared vocabulary


def resolve_trained(est):
    """The column order, from the pickle if it has one, else from the registry.

    Returns (names, source) where source is None when the pickle answered, and the registry's
    own `model_features_source` when it did not. Mirrors 00_SHAP_v1.ipynb exactly -- if the two
    ever drift, this script reports a verdict about an order the notebook did not use.
    """
    names = sk.model_feature_names(est)
    if names:
        return names, None
    try:
        return sk.registry_features(), sk.registry_features_source()
    except RuntimeError as exc:
        print("  registry unavailable: {}".format(exc))
        return [], None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="path to v1's estimator pickle (outputs/fasttracker_xgb.pkl)")
    p.add_argument("--meta", nargs="*", default=None,
                   help="specific _meta.json files; default is every one in v1's shap directory")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    est = sk.load_estimator(a.model)
    trained, trained_source = resolve_trained(est)
    print("[v1] model {}".format(a.model))
    print("     trained feature names: {}{}".format(
        len(trained),
        "  (from {}, order source: {})".format(
            os.path.basename(sk.REGISTRY_PATH), trained_source) if trained_source
        else ("" if trained else "  <- none available; every file will read 'unverified'")))

    if a.meta:
        metas = list(a.meta)
    else:
        shap_dir = os.path.dirname(sk.attributions_csv_path(sk.SPLITS[0]))
        metas = sorted(glob.glob(os.path.join(shap_dir, "v1_attributions_*_meta.json")))
    if not metas:
        raise SystemExit("[v1] no _meta.json found -- nothing to backfill.")

    changed = skipped = flagged = 0
    for meta_path in metas:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        names = meta.get("feature_names")
        if not names:
            print("  {:<48} SKIP -- no feature_names to check against".format(
                os.path.basename(meta_path)))
            skipped += 1
            continue

        # `trained=` only when the registry answered: passing it unconditionally would tag a
        # pickle-confirmed order as "(via registry)". The source travels with it, so the verdict
        # says which rung of extract_features_v1.py produced the registry's order.
        status = (sk.feature_order(est, names, trained=trained, trained_source=trained_source)
                  if trained_source else sk.feature_order(est, names))
        previous = meta.get("feature_order")
        if previous == status:
            print("  {:<48} already {!r}".format(os.path.basename(meta_path), status))
            skipped += 1
            continue

        if status.startswith("reordered") or status.startswith("set_mismatch"):
            # SHAP is positional: these phi are attributed to the wrong features. Record it,
            # do not repair it.
            print("  {:<48} ** {} -- these phi are WRONG, recompute this file **".format(
                os.path.basename(meta_path), status.upper()))
            flagged += 1

        meta["feature_order"] = status
        if not a.dry_run:
            with open(meta_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, indent=2))
        print("  {:<48} {} -> {!r}{}".format(
            os.path.basename(meta_path),
            "would set" if a.dry_run else "set", status,
            "   (was {!r})".format(previous) if previous is not None else ""))
        changed += 1

    print("\n[v1] {} {}, {} unchanged, {} FLAGGED{}".format(
        changed, "would be written" if a.dry_run else "written", skipped, flagged,
        "   (dry run -- nothing on disk changed)" if a.dry_run else ""))
    if flagged:
        raise SystemExit("{} file(s) have misordered features -- recompute them.".format(flagged))


if __name__ == "__main__":
    main()

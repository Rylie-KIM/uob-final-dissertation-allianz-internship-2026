"""Retrain the mitigated model for every version, each inside its OWN env. Sibling of score_all.py.

Nothing here names a file — it asks config for a KIND and a VERSION and gets back a real path.
This is the config-aware half of the pair; `retrain.py` is the worker that runs inside env-vX and
takes those paths as arguments.

WHY THE SPLIT, rather than having retrain.py read config itself: env-v1 is **Python 3.5.6** and
`config.py` is 3.7+ (future annotations, `dict[str, ...]` variable annotations, f-strings), so a
worker that imported config could not even be PARSED there. The driver runs in the analysis
`.venv`, resolves everything, and hands over strings. Same shape as score_all.py -> predict.py and
attribute_all.py -> attribute.py.

WHAT IT RESOLVES, per version:

    model              -> --baseline    the version's own production pickle (hyperparameters are
                                        CLONED off it; its fitted trees are discarded)
    processed_inputs   -> --features    that split's post-preprocessing matrix, held fixed
    corrected          -> --labels      the corrector's de-contaminated target + weights
    mitigated          -> --out-model
    (registry)         -> --features-json  features/registry/<v>.json — which columns of the
                                        matrix are model inputs, when the booster cannot say

--split is required, and it is not bookkeeping: the corrector runs per split, and pairing one
split's matrix with another's labels joins to nothing. It must be the SAME split throughout the
detect -> mitigate -> retrain -> re-score cycle, or the before/after comparison is between two
different populations.

  python src/training/retrain_all.py --split train              # same name for every version
  python src/training/retrain_all.py --split v2=train v3=train
  python src/training/retrain_all.py --split train --versions v2
  python src/training/retrain_all.py --split train --dry-run    # print the commands, run nothing
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RETRAIN = pathlib.Path(__file__).parent / "retrain.py"


def command(version: str, source: str, split: str) -> list[str]:
    """The one place a retrain invocation is assembled — every path comes from config."""
    return [
        str(config.python_bin(version)),
        str(RETRAIN),
        "--baseline",  str(config.path("model", version, source)),
        "--features",  str(config.path("processed_inputs", version, source, split=split)),
        "--labels",    str(config.path("corrected", version, source, split=split)),
        "--version",   version,
        "--out-model", str(config.path("mitigated", version, source, split=split)),
        # Only consulted when the baseline booster carries no feature names. Written by
        # features/extract_features.py inside this same version env.
        "--features-json", str(config.registry_path(version)),
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="real", choices=("real", "synthetic"))
    p.add_argument("--versions", nargs="+", default=list(config.VERSION_LABELS))
    p.add_argument("--split", nargs="+", default=None,
                   help="split to retrain on: one name for all versions, or v2=train v3=train")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    splits = config.resolve_splits(a.split, a.versions)

    for v in a.versions:
        try:
            cmd = command(v, a.source, splits[v])
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise SystemExit("\n[{}] {}\n".format(v, exc))

        labels = pathlib.Path(cmd[cmd.index("--labels") + 1])
        if not labels.exists():
            raise SystemExit(
                f"\n[{v}] no corrected labels at\n    {labels}\n"
                f"Run the corrector for this split first (mitigator/corrector/reweight.py, or "
                f"notebook 03_02_reweight_mitigation) — retraining is the step AFTER it.\n"
            )

        print(f"\n[{v}] split={splits[v]}\n      " + " \\\n      ".join(cmd))
        if not a.dry_run:                  # a dry run must not touch the filesystem either
            pathlib.Path(cmd[cmd.index("--out-model") + 1]).parent.mkdir(parents=True,
                                                                        exist_ok=True)
            subprocess.run(cmd, check=True)

    print(f"\nDone ({'dry run' if a.dry_run else 'retrained'}): {', '.join(a.versions)}")


if __name__ == "__main__":
    main()

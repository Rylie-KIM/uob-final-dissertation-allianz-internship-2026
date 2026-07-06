#!/usr/bin/env bash
# Train every version's BASELINE pkl locally, each in its OWN env, on its OWN inputs.
# The baseline pkls are not in git; this rebuilds them so predict.py / pipeline.py can score.
# Version-agnostic: the active env + CLI args decide which version is trained. Run from repo root.
#
#   bash src/training/train_all.sh            # synthetic (default)
#   bash src/training/train_all.sh real       # real inputs, when present
#
# For retraining the MITIGATED model instead, see src/training/retrain.py (driven by pipeline.py).
set -euo pipefail

SOURCE="${1:-synthetic}"                        # synthetic | real
ENVDIR="src/envs"
INPUTS="src/data/$SOURCE/inputs"                # per-version features_<v> + labels_<v>
OUTDIR="src/models/$SOURCE/baseline"           # baseline pkls (production reproduction)
mkdir -p "$OUTDIR"

for V in v1 v2 v3; do
    "$ENVDIR/$V/.venv/bin/python" src/training/train.py \
        --features  "$INPUTS/features_$V.parquet" \
        --labels    "$INPUTS/labels_$V.parquet" \
        --version   "$V" \
        --out-model "$OUTDIR/$V.pkl"
done

echo "All baseline models trained -> $OUTDIR"

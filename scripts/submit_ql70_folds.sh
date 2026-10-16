#!/bin/bash
# LOGIN-NODE script. Run it, do NOT sbatch it:
#   bash scripts/submit_ql70_folds.sh                       # Llama-70B lane
#   TAG=q72 MODEL=Qwen/Qwen2.5-72B-Instruct bash scripts/submit_ql70_folds.sh
#
# Submits folds 0 1 2 4 of a 70B-class QLoRA risk member. Fold 3 is always the
# pilot and is trained separately, because 48 GPU-hours should not be committed
# to a model nobody has seen a single number from.
#
# IDEMPOTENT: rerun whenever a slot frees. Skips folds already trained, skips
# folds already queued, and stops at the 5-submitted cap instead of collecting
# rejections. The TAG AND FOLD both go in the job name (ql70f0, q72f1 ...) so
# the queued-check works and the two lanes never confuse each other.
#
# THE TAG IS LOAD-BEARING. risk_qlora70_pilot.sh writes to
# runs/risk_${TAG}_f${FOLD} with NO skip-if-exists guard, so two lanes sharing a
# TAG would silently overwrite each other's adapters and pool two different
# models into one OOF file. Always pass MODEL and TAG together.
#
# When all five folds of a lane are down:  TAG=<tag> bash scripts/pool_ql70.sh

set -u
cd "$(dirname "$0")/.." || exit 1

TAG="${TAG:-ql70}"
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
FOLDS="${FOLDS:-0 1 2 4}"
MAXSUB="${MAXSUB:-5}"                       # cluster cap: 2 running / 5 submitted
BATCH="scripts/risk_qlora70_pilot.sh"

[ -f "$BATCH" ] || { echo "ERROR: $BATCH missing"; exit 1; }
[ -d "$MODEL" ] || { echo "ERROR: MODEL $MODEL missing"; exit 1; }
mkdir -p logs

# --- gate: the pilot must have finished AND left per-post probabilities ------
# Without them fold 3 cannot join the pool, the OOF file covers 4/5 of the data,
# and the comparison against oof_base stops being paired.
P3="runs/risk_${TAG}_f3/val_predictions.csv"
if [ ! -f "$P3" ]; then
    echo "[${TAG}] waiting on the fold-3 pilot -- ${P3} not present yet."
    echo "[${TAG}] nothing submitted. (ls -la runs/risk_${TAG}_f3/ to inspect)"
    exit 1
fi
echo "[${TAG}] pilot probs OK: ${P3}"
echo "[${TAG}] model ${MODEL}"

QNAMES="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"
NSUB="$(printf '%s\n' "$QNAMES" | grep -c .)"
echo "[${TAG}] jobs submitted right now: ${NSUB} (cap ${MAXSUB})"

for f in $FOLDS; do
    if [ -f "runs/risk_${TAG}_f${f}/val_predictions.csv" ]; then
        echo "[${TAG}] fold ${f}: already trained, skipping"; continue
    fi
    if printf '%s\n' "$QNAMES" | grep -qx "${TAG}f${f}"; then
        echo "[${TAG}] fold ${f}: already queued, skipping"; continue
    fi
    if [ "$NSUB" -ge "$MAXSUB" ]; then
        echo "[${TAG}] fold ${f}: NOT submitted, at the ${MAXSUB}-job cap"
        continue
    fi
    OUT="$(sbatch -J "${TAG}f${f}" \
             --export=ALL,FOLD="${f}",TAG="${TAG}",MODEL="${MODEL}" "$BATCH")" \
        && { echo "[${TAG}] fold ${f}: ${OUT}"; NSUB=$((NSUB + 1)); } \
        || echo "[${TAG}] fold ${f}: SUBMIT FAILED"
done

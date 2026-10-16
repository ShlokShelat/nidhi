#!/bin/bash
#SBATCH --job-name=riskfull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Rebuild the risk column from members trained on ALL 1635 gold posts instead
# of the 1263-post train_split.
#
# THE DEFECT IT FIXES. Every deployed risk member -- runs/risk_v1{,_s14,_s15}
# and both risk-NLI runs -- has `"train": "artifacts/train_split.csv"` in its
# run_config.json. train_split is 1263 posts / 121 authors; val_split is 372
# posts / 32 authors; together they are the full 1635. So 23% of the labelled
# data has never produced a gradient in any deployed model, and for `attempt` --
# the rarest class -- that is 89 training examples out of 114 available, a 28%
# shortfall on the label with the thinnest supervision.
#
# runs/risk_full_s13/14/15 already exist, trained by scripts/fulldata.sh on
# artifacts/full_train.csv with the SAME recipe (deberta-v3-large, 5 epochs).
#
# WHY THE CHECKPOINTS ARE CLEAN DESPITE `--val artifacts/full_train.csv`.
# train_risk.py selects the best epoch by val weighted-F1, and here val IS the
# training data -- which would pick whichever epoch memorised hardest. It did
# not happen: all three run_config.json record `best_val_weighted_f1: -1.0`,
# meaning the selection branch never fired and what was saved is the LAST epoch
# of a fixed 5-epoch run. That is a principled fixed-epoch model, and arguably
# cleaner than the deployed members, which ARE best-by-val selected.
#
# WHAT THIS IS NOT. It does not reweight the loss or fit anything new: same
# recipe, same downstream chain, more supervision. Nothing here is tuned on the
# data it is scored on.
#
# NOT VALIDATABLE OFFLINE, and that is stated rather than hidden: once a model
# trains on all 1635, val_split is inside its training data, so any val or OOF
# number computed from it is contaminated. The epoch count is therefore
# transferred from the train_split runs, which peaked at 4-5.
#
# Stage 1 output already exists from an interactive run; it is reused unless
# FORCE=1. Only stages 2-3 need the GPU (blend_risk_saved.py scores both NLI
# models on val and test).
#
# Usage:
#   sbatch scripts/risk_fulldata_submission.sh
#   sbatch --export=ALL,FORCE=1 scripts/risk_fulldata_submission.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

BASE="${BASE:-submissions/NIDHI_cluster.csv}"
TAG="${TAG:-full}"
ENS="artifacts/ensembled_risk_${TAG}.csv"

for p in "${BASE}" data/test.csv artifacts/val_split.csv \
         artifacts/friend_risk_probs.csv; do
    [ -f "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done
for s in 13 14 15; do
    [ -f "runs/risk_full_s${s}/model_final.pt" ] || {
        echo "ERROR: runs/risk_full_s${s}/model_final.pt missing"; exit 1; }
done

echo "================================================================"
echo "STAGE 1  3 full-data seeds + friend  ->  ${ENS}"
echo "================================================================"
if [ -f "${ENS}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "--- ${ENS} exists, reusing (FORCE=1 to rebuild) ---"
else
    python src/ensemble_risk.py \
        --test data/test.csv \
        --our-risk-run runs/risk_full_s13,runs/risk_full_s14,runs/risk_full_s15 \
        --friend-probs artifacts/friend_risk_probs.csv \
        --out "${ENS}" || { echo "FAILED stage 1"; exit 1; }
fi

echo ""
echo "================================================================"
echo "STAGE 2  blend the 2 risk-NLI members at the FORCED 0.30"
echo "================================================================"
# --ens-val deliberately points at the TRAIN_SPLIT runs' val predictions, not
# the full-data ones: those are honest (val was held out for them), whereas the
# full-data models have val inside training. Since the weight is FORCED, the
# val side only drives the printed sweep -- IGNORE those numbers, they are
# diagnostic and, for this build, partly contaminated.
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
    --ens-test "${ENS}" \
    --force-weight 0.30 \
    --out "artifacts/risk_blend_${TAG}.csv" || { echo "FAILED stage 2"; exit 1; }

echo ""
echo "================================================================"
echo "STAGE 3  the same two symbolic rules"
echo "================================================================"
python src/apply_rules_to_test.py \
    --test data/test.csv \
    --risk-blend "artifacts/risk_blend_${TAG}.csv" \
    --rules urgency,completed_act \
    --out "artifacts/risk_blend_${TAG}_rules.csv" || { echo "FAILED stage 3"; exit 1; }

echo ""
echo "================================================================"
echo "STAGE 4  patch the risk column onto the champion"
echo "================================================================"
python src/patch_submission_risk.py \
    --base "${BASE}" \
    --risk "artifacts/risk_blend_${TAG}_rules.csv" \
    --out "submissions/NIDHI_risk${TAG}.csv" || { echo "FAILED stage 4"; exit 1; }

echo ""
echo "================================================================"
echo "DONE -> submissions/NIDHI_risk${TAG}.csv"
echo "================================================================"
echo "Champion risk marginals: indicator 137 / ideation 123 / behavior 101 /"
echo "attempt 17. Stage 1 alone gives 132 / 126 / 96 / 24, before the NLI"
echo "blend and the rules."
echo "CHECK: only the risk column is patched, so the evidence and factor"
echo "columns must be byte-identical to the base. If the factor score moves,"
echo "the patch was not clean and the risk reading is void."

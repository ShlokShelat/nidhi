#!/bin/bash
#SBATCH --job-name=nlildr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Train the `ladder` phrasing of the risk-level definitions as a 5-fold NLI arm
# and pool its OOF against the base and gaur arms, per class.
#
# THE DESIGN CONSTRAINT. The four risk levels are a severity ladder scored by one
# softmax over four hypotheses, so the hypotheses must partition the ladder. Two
# properties are required and are easy to violate when definitions are written
# one class at a time:
#   1. UPPER BOUNDS. A `behavior` definition that admits past self-injury (scars,
#      "has harmed themselves") with no upper bound also describes gold ATTEMPT
#      posts, so `behavior` absorbs its neighbour above it.
#   2. MATCHED CUE DENSITY. Four hypotheses feeding one softmax compete on
#      surface richness as well as on meaning. A 90-word, 20-cue `behavior`
#      hypothesis beside a terse 15-word `attempt` hypothesis wins on richness
#      alone, independent of which class the post belongs to.
# RISK_DEFS=ladder satisfies both: every rung explicitly excludes the rungs above
# it, and all four are written at comparable length and cue density. Its
# `attempt` rung names completed-act markers, which occur on attempt spans at
# 0.480 against 0.021 on behavior spans -- a 23x separation available in the
# annotated text itself.
#
# ONLY THE LADDER ARM TRAINS HERE. The base and gaur fold models are reused if
# present, so this is 5 trainings rather than 15, and the comparison stays
# like-for-like on identical folds.
#
# PRE-STATED ACCEPTANCE BAR (recorded beside RISK_DEFINITIONS_LADDER):
#   attempt recall >= 0.620 (base 0.640, i.e. within 0.02) AND pooled
#   weighted-F1 > 0.7985 (base). Attempt recall is the guarded quantity because
#   it is what a poorly bounded neighbouring rung destroys; weighted-F1 guards
#   against buying that recall with everything else.
#
# Usage:
#   sbatch scripts/risk_nli_ladder.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

EPOCHS="${EPOCHS:-4}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"
[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }

for f in 0 1 2 3 4; do
    [ -f "runs/risknli_base_f${f}/model.pt" ] || {
        echo "ERROR: runs/risknli_base_f${f}/model.pt missing (run scripts/risk_nli_gaur.sh first)"; exit 1; }
done

for f in 0 1 2 3 4; do
    d="runs/risknli_ladder_f${f}"
    if [ -f "${d}/model.pt" ]; then
        echo "--- ladder fold ${f}: exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "ladder fold ${f}   RISK_DEFS='ladder'"
    echo "================================================================"
    RISK_DEFS=ladder python src/train_risk_nli.py \
        --train "artifacts/folds/train_f${f}.csv" \
        --val   "artifacts/folds/val_f${f}.csv" \
        --model "${MODEL}" --outdir "${d}" --epochs "${EPOCHS}" \
        || { echo "FAILED ladder fold ${f}"; exit 1; }
done

echo ""
echo "================================================================"
echo "POOLED OOF -- base vs gaur vs ladder, per class"
echo "================================================================"
python src/risk_nli_pool_oof.py --arms base,gaur,ladder --outdir artifacts/nligaur

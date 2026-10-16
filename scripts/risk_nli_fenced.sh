#!/bin/bash
#SBATCH --job-name=nlifnc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Symmetric include/exclude definitions -- the ladder idea done properly.
#
# WHAT LADDER STILL GETS WRONG. Its fences are ABSTRACT while its inclusions are
# CONCRETE:
#     includes  "cutting, burning, hitting themselves ... pills, gun, rope"
#     excludes  "but the author has NOT already made an attempt"
# The model matches surface cues, so ten concrete nouns fire hard and a bare
# negation of an abstract noun barely fires. The include side wins by
# construction -- which is exactly how "swallowed two packs of Zoloft" and
# "attempted suicide and was in the hospital" kept landing on `behavior`
# (26 of 86 broken posts, the single largest cell in risk_arm_transitions.py).
#
# RISK_DEFS=fenced gives both sides the same concreteness: every class names
# what it covers AND names, with the same cue vocabulary, what belongs to its
# neighbours.
#
# COST, measured against full_train.csv (median post 38 words, p90 150).
# train_risk_nli.py truncates "only_first", so the POST pays for hypothesis
# length:
#     shipped   19-word hyp -> 262-word post budget ->  3.9% of posts cut
#     ladder    78           -> 203                  ->  6.5%
#     fenced   ~120          -> ~155                 -> ~9.5%
# Acceptable: long posts were separately measured to score BETTER than short
# ones, so losing their tails is not the risk it looks like.
#
# ONLY THE FENCED ARM TRAINS HERE. base/gaur (job 23135) and ladder (job 23144)
# are reused, so this is 5 trainings and the pool compares all four.
#
# PRE-STATED FALSIFIER, same bar as ladder:
#   attempt recall >= 0.620 (base 0.640) AND pooled weighted-F1 > 0.7985 (base).
# If BOTH ladder and fenced miss it, the four definitions are too entangled to
# tune independently and definition engineering closes for risk.
#
# Usage:
#   sbatch scripts/risk_nli_fenced.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

EPOCHS="${EPOCHS:-4}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"
[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }

for f in 0 1 2 3 4; do
    [ -f "runs/risknli_base_f${f}/model.pt" ] || {
        echo "ERROR: runs/risknli_base_f${f}/model.pt missing (job 23135)"; exit 1; }
done

for f in 0 1 2 3 4; do
    d="runs/risknli_fenced_f${f}"
    if [ -f "${d}/model.pt" ]; then
        echo "--- fenced fold ${f}: exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "fenced fold ${f}   RISK_DEFS='fenced'"
    echo "================================================================"
    RISK_DEFS=fenced python src/train_risk_nli.py \
        --train "artifacts/folds/train_f${f}.csv" \
        --val   "artifacts/folds/val_f${f}.csv" \
        --model "${MODEL}" --outdir "${d}" --epochs "${EPOCHS}" \
        || { echo "FAILED fenced fold ${f}"; exit 1; }
done

echo ""
echo "================================================================"
echo "POOLED OOF -- every arm we have, per class"
echo "================================================================"
ARMS="base,gaur,fenced"
for f in 0 1 2 3 4; do
    [ -f "runs/risknli_ladder_f${f}/model.pt" ] || { ARMS="base,gaur,fenced"; break; }
    ARMS="base,gaur,ladder,fenced"
done
echo "arms: ${ARMS}"
python src/risk_nli_pool_oof.py --arms "${ARMS}" --outdir artifacts/nligaur

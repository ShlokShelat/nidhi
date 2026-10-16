#!/bin/bash
#SBATCH --job-name=rerankoof
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Attempt 5 at evidence SELECTION -- the first one trained on the distribution
# it is actually deployed on.
#
# WHY THE CLOSED LANE RE-OPENS. Attempt 4 (scripts/span_reranker.sh) built its
# negatives with `perturb(gold, post)` and `sentence_fragments(post)`, then
# scored REAL BIO-tagger candidates at inference. The model never saw a tagger
# output during training. That is the configuration the stacking literature
# identifies as broken -- a second-stage model must be trained on FIRST-STAGE
# OUTPUTS produced out-of-fold -- and every symptom matched: training loss
# collapsed to 0.048 while val went nowhere, and a 4.6x smaller model overfit
# identically. The closure was pre-registered as needing "new evidence, not a
# new intuition"; a verified distribution mismatch inside the experiment is
# new evidence.
#
# WHAT IS DIFFERENT. src/build_reranker_pairs_oof.py labels every real OOF
# tagger candidate with the OFFICIAL matcher against gold. Hard negatives are
# now genuine tagger near-misses -- semantically plausible, conventionally
# wrong -- which is exactly the discrimination the failed model lacked
# (recall 0.8575 at precision 0.608: it found the right regions and could not
# tell a match from a near-miss).
#
# TEMPER EXPECTATIONS -- MEASURED, NOT ASSUMED. The union dedupes to only 2660
# pairs (2150 after removing val posts), a THIRD of attempt 4's 7087. The
# taggers agree heavily: ~2.3 unique candidates per post. And 509 of 1137 posts
# have every candidate matching gold, so they teach nothing. The genuinely
# discriminative subset is roughly 500 posts. The distribution is fixed; the
# volume is worse.
#
# ABORT CONDITION, unchanged from attempt 4: if mean loss drops BELOW 0.05 the
# model memorised again, and with distribution-matched data that means volume
# was never the fixable part -- evidence selection then closes for good.
#
# BAR: cross-fitted F1 must beat the deployed vote on FULL val AND on BOTH
# author halves.
#
# Usage:
#   sbatch scripts/span_reranker_oof.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PAIRS="${PAIRS:-artifacts/reranker_pairs_oof.csv}"

echo "================================================================"
echo "STEP 1  build pairs from REAL out-of-fold tagger candidates (CPU)"
echo "================================================================"
if [ -f "${PAIRS}" ]; then
    echo "--- ${PAIRS} exists, reusing ---"
else
    python src/build_reranker_pairs_oof.py --out "${PAIRS}" || exit 1
fi

echo ""
echo "================================================================"
echo "STEP 2a  deberta-v3-large, 3 epochs"
echo "================================================================"
python src/train_span_reranker.py --pairs "${PAIRS}" \
    --model microsoft/deberta-v3-large --epochs 3 \
    --outdir runs/span_reranker_oof_large || exit 1

echo ""
echo "================================================================"
echo "STEP 2b  deberta-v3-base, 4 epochs  (4.6x fewer parameters)"
echo "================================================================"
# Attempt 4's base variant overfit IDENTICALLY to large, which is what showed
# capacity was not the constraint. Repeat it here: if base and large again land
# on the same number, the remaining gap is data volume and no reranker of this
# family will close it.
python src/train_span_reranker.py --pairs "${PAIRS}" \
    --model microsoft/deberta-v3-base --epochs 4 \
    --outdir runs/span_reranker_oof_base || exit 1

echo ""
echo "================================================================"
echo "DONE"
echo "================================================================"
echo "Read the per-epoch mean loss FIRST:"
echo "  below 0.05  -> memorised again; selection closes permanently"
echo "  0.15-0.30   -> the harder negatives are doing their job"
echo "Then the cross-fitted F1: it must beat the deployed vote on FULL val"
echo "AND on BOTH author halves -- winning one half while losing the other"
echo "does not clear the bar."

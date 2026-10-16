#!/bin/bash
#SBATCH --job-name=evcond5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out

# Does risk-conditioned span tagging hold up over 5 folds and the RIGHT average?
#
# WHAT A SINGLE-SPLIT SCREEN SAID (372 val posts): conditioning on GOLD risk
# level gained clearly over the unconditioned base, and conditioning on
# PREDICTED risk level gained by a smaller but still positive margin.
# Precision moved up at unchanged recall, which is the signature you would
# expect: conditioning does not find MORE spans, it stops the tagger
# grabbing text that argues for the wrong rung.
#
# TWO REASONS NOT TO BELIEVE IT YET.
#  1. One split. Single-split arm deltas on this data have flipped sign before;
#     the standing rule is to judge over many splits.
#  2. Wrong average. train_evidence.py calls phrase_f1() at its DEFAULT, which
#     is MICRO. The official scorer is MACRO over posts, and macro hands a free
#     1.0 to every post where gold and prediction are both empty -- exactly the
#     indicator posts, exactly where the risk label acts. A micro win can be a
#     macro loss.
#
# So: 5 folds x 2 arms on the same folds the risk arms use, then
# src/evidence_cond_pool.py scores all 1635 OOF posts as micro, macro, and
# macro-with-the-shipped-suppression-rule, with the cond arm evaluated on both
# GOLD levels (ceiling) and PREDICTED levels (shippable).
#
# WHY THIS RUNS ALONGSIDE THE RISK ARMS. The two lanes compound. The cond@GOLD
# minus cond@PRED gap is exactly what a better risk model hands back, so if
# ladder/fenced land, this arm is worth more than it measures today. Conditioned
# on OOF-predicted levels from artifacts/nligaur/oof_base.csv -- honestly
# out-of-fold, and covering all 1635 posts (champion_oof.csv was missing 18).
#
# PRE-STATED FALSIFIER: cond@PRED must beat base on MACRO+SUPPRESSION at the
# fixed threshold. Micro is reported but does not decide anything.
#
# Usage:
#   sbatch scripts/evidence_risk_cond_5fold.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

EPOCHS="${EPOCHS:-6}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"
RISKSRC="${RISKSRC:-artifacts/nligaur/oof_base.csv}"

[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }
[ -f "$RISKSRC" ] || { echo "ERROR: $RISKSRC missing (run risk_nli_gaur_pool.sh)"; exit 1; }

echo "=== alignment guard ==="
python tests/test_risk_cond_alignment.py || {
    echo "ALIGNMENT TEST FAILED -- not training"; exit 1; }

train_fold () {         # $1 = tag, $2 = fold, $3... = extra flags
    local tag=$1 f=$2; shift 2
    local d="runs/evcond5_${tag}_f${f}"
    if [ -f "${d}/model_final.pt" ]; then
        echo "--- ${tag} fold ${f}: exists, skipping ---"
        return 0
    fi
    echo ""
    echo "================================================================"
    echo "${tag} fold ${f}  $*"
    echo "================================================================"
    python src/train_evidence.py \
        --train "artifacts/folds/train_f${f}.csv" \
        --val   "artifacts/folds/val_f${f}.csv" \
        --model "${MODEL}" --outdir "${d}" --epochs "${EPOCHS}" "$@" \
        || { echo "FAILED ${tag} fold ${f}"; exit 1; }
}

for f in 0 1 2 3 4; do
    train_fold base "$f"
    train_fold cond "$f" --risk-cond
done

echo ""
echo "================================================================"
echo "POOLED 5-FOLD -- micro vs macro vs macro+suppression"
echo "================================================================"
python src/evidence_cond_pool.py --arms base,cond --prefix runs/evcond5 \
    --risk-from "$RISKSRC" --model "$MODEL"

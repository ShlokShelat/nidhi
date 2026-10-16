#!/bin/bash
#SBATCH --job-name=loratest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Score the 378 TEST posts with a 5-fold LoRA risk member.
#
# WHY IT IS NEEDED. A three-member majority vote (encoder + cot72b + LoRA)
# needs all three members present on test. cot72b already has test
# predictions; the LoRA member does not, because risk_lora_5fold.sh scores only
# held-out folds. This job fills that gap.
#
# FOLD AVERAGING, not selection: each adapter saw a different 4/5 of the data,
# so no single one is preferable, and averaging matches how the factor arms
# build their test dumps.
#
# THE SILENT FAILURE IT GUARDS: AutoModelForSequenceClassification adds a
# `score` head that starts RANDOM and gets trained. PEFT with task_type=SEQ_CLS
# should save it with the adapter -- if it did not, loading yields a random
# 4-way head producing confident nonsense indistinguishable from a working
# model. The script snapshots a fresh head, loads the adapter, and REFUSES to
# score if they are bit-identical.
#
# It also prints the predicted class mix against the probe intervals
# (indicator 112-170, ideation 99-142, behavior 69-113, attempt 13-41), so a
# mis-calibrated member cannot enter the vote unexamined.
#
# Usage:
#   sbatch scripts/predict_lora_test.sh
#   sbatch --export=ALL,TAG=loraM7 scripts/predict_lora_test.sh   # another adapter

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TAG="${TAG:-loraL8}"
TEST="${TEST:-data/test.csv}"
# --per-fold by default. Out-of-fold predictions score each post with ONE fold
# model, so a test column built by averaging all five is a different estimator
# from the one any offline verdict was formed on. Writing the single-fold files
# too keeps the matched regime available. Set PERFOLD= to write only the
# average.
PERFOLD="${PERFOLD:---per-fold}"

for f in 0 1 2 3 4; do
    [ -d "runs/risk_${TAG}_f${f}/lora" ] || {
        echo "ERROR: runs/risk_${TAG}_f${f}/lora missing"; exit 1; }
done
[ -f "$TEST" ] || { echo "ERROR: $TEST missing"; exit 1; }

echo "tag ${TAG}   per-fold: ${PERFOLD:-(off)}"
python src/predict_risk_lora_test.py --tag "$TAG" --test "$TEST" ${PERFOLD} \
    || { echo "FAILED"; exit 1; }

echo ""
echo "================================================================"
echo "NEXT (CPU): build the 3-way majority vote"
echo "================================================================"
echo "members on test:"
echo "  encoder risk : submissions/NIDHI_ev17k4.csv (or the current best)"
echo "  cot72b       : artifacts/cot_soft_oof.csv.extra.csv"
echo "  llama        : artifacts/lora_${TAG}_test.csv"
echo ""
echo "NOTE the test-time vote is not the configuration any offline vote was"
echo "evaluated in: on test the encoder member is the full deployed risk chain,"
echo "a stronger and differently-calibrated model than the out-of-fold encoder"
echo "used offline. A stronger member is expected to help, but the substitution"
echo "makes the offline reading an analogy, not a prediction."

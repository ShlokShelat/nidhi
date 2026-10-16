#!/bin/bash
#SBATCH --job-name=qnlidump
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Test-set probabilities for the Qwen factor arm, so `qnli` can actually ship.
#
# `qnli` PASSED its falsifier: added to the 5-arm bag it scores 0.6740 against
# 0.6640 under RATEMATCH -- the rank-only policy, which is the one that matters
# because Qwen's recorded defect was thresholding, not ranking. But it has no
# test dump, and predict_factors_nli.py cannot make one (it loads NLIScorer +
# model.pt; this arm is AutoModelForSequenceClassification + a PEFT adapter).
#
# STEP 1 IS A VERIFICATION, NOT A FORMALITY. It re-scores fold 0's OWN val
# split and compares against the val_probs.npz that fold wrote during training.
# If those disagree, the inference path differs from the training path -- a
# tokenizer, pad-token or hypothesis mismatch -- and the test dump would be a
# silently different model averaged into the bag. That is exactly the class of
# error that makes a result look clean and mean nothing, so the job STOPS
# rather than continuing to step 2.
#
# Qwen has no pad token; the trainer set pad_token = eos_token and wrote
# config.pad_token_id before attaching the adapter. Getting that wrong pools
# hidden states over padding and the deviation check is what catches it.
#
# Usage:
#   sbatch scripts/qnli_test_dump.sh
#   sbatch --export=ALL,SKIP_VERIFY=1 scripts/qnli_test_dump.sh   # not advised

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# the arm was trained on the PUBLISHED definitions -- hypothesis_for reads
# these at call time, so a stale export would change the question asked
unset FACTOR_DEFS_JSON      || true
unset PROTECTIVE_TEMPLATE   || true
unset FACTOR_ASPECTS_JSON   || true
unset FACTOR_SUBSET         || true

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TAG="${TAG:-qnli}"
TEST="${TEST:-data/test.csv}"

[ -f "${MODEL}/config.json" ] || { echo "ERROR: ${MODEL} not found"; exit 1; }
[ -f "${TEST}" ] || { echo "ERROR: ${TEST} not found"; exit 1; }
for f in 0 1 2 3 4; do
    [ -d "runs/factors_${TAG}_f${f}/lora" ] || {
        echo "ERROR: runs/factors_${TAG}_f${f}/lora missing."
        echo "A partial bag is a DIFFERENT model from the one measured."
        exit 1; }
done

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    echo "================================================================"
    echo "STEP 1  verify inference reproduces training (fold 0 own val)"
    echo "================================================================"
    python src/predict_factors_llm_nli.py --model "${MODEL}" --tag "${TAG}" \
        --folds 0 --verify || { echo "VERIFY FAILED"; exit 1; }
    echo ""
    echo "Read the deviation above. Under ~0.01 the path is faithful."
    echo "If it is large, STOP -- do not ship a dump from a different model."
fi

echo ""
echo "================================================================"
echo "STEP 2  score the test set with all 5 fold adapters"
echo "================================================================"
python src/predict_factors_llm_nli.py --model "${MODEL}" --tag "${TAG}" \
    --test "${TEST}" || { echo "FAILED"; exit 1; }

echo ""
echo "================================================================"
echo "DONE -> artifacts/factor_runs/${TAG}_cv_test.npz"
echo "================================================================"
echo "Nothing has been submitted. Next, on CPU:"
echo "  python src/factor_bag_ship.py --arms mnli dmnli cmine prot d4 ${TAG} \\"
echo "      --rule logit --match-counts submissions/NIDHI_canon_on_nlimnli.csv \\"
echo "      --out artifacts/bag6q_matched_factors.csv"
echo "  python src/swap_factor_column.py --base submissions/NIDHI_canon_on_nlimnli.csv \\"
echo "      --factors artifacts/bag6q_matched_factors.csv --out submissions/NIDHI_bag6q.csv"

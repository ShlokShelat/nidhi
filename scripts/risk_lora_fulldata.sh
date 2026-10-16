#!/bin/bash
#SBATCH --job-name=lorafull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Train a LoRA risk member on ALL 1635 posts and score test with that ONE model.
#
# THE ASYMMETRY IT REMOVES. A LoRA fold member is trained on 1308 posts and its
# test predictions AVERAGE five such models, while the champion deploys
# FULL-DATA models trained on all 1635. Comparing the two therefore compares a
# deliberately weakened challenger against a fully trained incumbent.
#
# Full-data training is exactly how the champion is deployed, so this is the
# standard pattern rather than a trick: the fold version's honest OOF remains
# the ESTIMATE, and the full-data model is what ships. That is the same contract
# the champion operates under.
#
# IT ALSO REMOVES THE REGIME MISMATCH. One model at train time and one at test
# time, with no 5-checkpoint average in between. Averaging checkpoints narrows
# the predicted distribution, so a bagged member behaves differently at test
# time from the single model whose OOF was measured.
#
# WHAT YOU CANNOT DO WITH IT. There is no honest validation number for a
# full-data model -- it has seen every post. --no-val-select is therefore
# mandatory (final epoch, no peeking), and the val figure it prints is
# MEANINGLESS because val_split is inside the training data. Judge this member
# by its FOLD sibling's OOF, never by the number this job prints.
#
# Usage:
#   sbatch scripts/risk_lora_fulldata.sh
#   sbatch --export=ALL,MODEL=mistralai/Mistral-7B-Instruct-v0.3,TAG=M7full scripts/risk_lora_fulldata.sh

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TAG="${TAG:-L8full}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-8}"
QUANT="${QUANT:-}"          # set to --load-4bit for a 70B

TRAIN="${TRAIN:-artifacts/full_train.csv}"
[ -f "$TRAIN" ] || { echo "ERROR: $TRAIN missing"; exit 1; }
[ -d "$MODEL" ] || { echo "ERROR: $MODEL missing"; exit 1; }
OUT="runs/risk_${TAG}_full"

echo "model ${MODEL}"
echo "train ${TRAIN}  (ALL posts -- the printed val number is MEANINGLESS)"
echo "out   ${OUT}"

if [ ! -f "${OUT}/run_config.json" ]; then
    python src/train_risk_lora.py \
        --train "$TRAIN" --val artifacts/val_split.csv \
        --model "$MODEL" --outdir "$OUT" \
        --epochs "$EPOCHS" --lr "$LR" --batch "$BATCH" --grad-accum "$ACCUM" \
        --no-val-select ${QUANT} \
        || { echo "FAILED training"; exit 1; }
else
    echo "--- ${OUT} exists, skipping training ---"
fi

echo ""
echo "================================================================"
echo "SCORE TEST with the single full-data model"
echo "================================================================"
python src/predict_risk_lora_test.py \
    --tag "$TAG" --runs "runs/risk_{tag}_full" --folds 1 \
    --test data/test.csv --out "artifacts/lora_${TAG}_test.csv" ${QUANT} \
    || { echo "FAILED scoring"; exit 1; }

echo ""
echo "Marginals are checked against the probe intervals above. This member is"
echo "STRONGER than its fold siblings by construction, so if it also lands"
echo "inside those intervals it is the version that should enter any vote or"
echo "routing rule -- not the 5-fold average."
echo ""
echo "REMEMBER: judge it by the FOLD sibling's OOF (llama8 = 0.8006). The"
echo "number printed during training saw its own validation set."

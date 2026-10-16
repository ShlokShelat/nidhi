#!/bin/bash
#SBATCH --job-name=risk-focal-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2

export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Gamma values to evaluate
GAMMAS=(0.0 1.0 1.5 2.0 2.5 3.0)

echo "=============================================================="
echo "Starting Focal Loss Gamma Sweep"
echo "=============================================================="

for GAMMA in "${GAMMAS[@]}"; do

    TAG=$(echo "$GAMMA" | tr '.' '_')

    echo ""
    echo "=============================================================="
    echo "Training gamma = ${GAMMA}"
    echo "Output directory: runs/risk_focal_g${TAG}"
    echo "=============================================================="

    python src/train_risk_focal.py \
        --train artifacts/train_split.csv \
        --val artifacts/val_split.csv \
        --model microsoft/deberta-v3-large \
        --outdir runs/risk_focal_g${TAG} \
        --max-len 512 \
        --epochs 5 \
        --lr 1e-5 \
        --batch 8 \
        --grad-accum 2 \
        --alpha-ordinal 0.5 \
        --focal-gamma ${GAMMA} \
        --user-context \
        --seed 13

    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo ""
        echo "Training FAILED for gamma=${GAMMA}"
        echo "Stopping sweep."
        exit $STATUS
    fi

    echo ""
    echo "Finished gamma=${GAMMA}"
    echo ""

done

echo "=============================================================="
echo "Gamma sweep completed successfully."
echo "=============================================================="

echo ""
echo "Generated checkpoints:"
ls -d runs/risk_focal_g*

#!/bin/bash
#SBATCH --job-name=risk-focal-rerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

source scripts/env.sh

export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python src/train_risk_focal.py \
    --train artifacts/train_split.csv \
    --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large \
    --outdir runs/risk_focal_rerun \
    --max-len 512 \
    --epochs 5 \
    --lr 1e-5 \
    --batch 8 \
    --grad-accum 2 \
    --alpha-ordinal 0.5 \
    --focal-gamma 2.0 \
    --user-context \
    --seed 13

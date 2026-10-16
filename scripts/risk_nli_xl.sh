#!/bin/bash
#SBATCH --job-name=risk-xl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/train_risk_nli.py \
    --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v2-xlarge \
    --outdir runs/risk_nli_xl \
    --max-len 320 --epochs 4 --batch 2 --grad-accum 8 --lr 7e-6

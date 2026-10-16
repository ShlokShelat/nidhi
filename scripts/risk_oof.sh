#!/bin/bash
#SBATCH --job-name=risk-oof
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
python src/risk_oof.py --train artifacts/train_split.csv \
    --model microsoft/deberta-v3-large --k 3 --epochs 4 \
    --out artifacts/train_split_riskcond.csv

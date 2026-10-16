#!/bin/bash
#SBATCH --job-name=factor-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
python src/train_factors.py --train artifacts/train_aug.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir runs/factors_seed14 --seed 14
python src/train_factors.py --train artifacts/train_aug.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir runs/factors_seed15 --seed 15

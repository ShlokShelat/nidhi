#!/bin/bash
#SBATCH --job-name=errnb-seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2

SEEDS=(42 123 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

python src/train_factors_nli.py \
    --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --external artifacts/error_neighbor_mhi.csv \
    --model microsoft/deberta-v3-large --outdir runs/factors_nli_errneighbor_s${SEED} \
    --neg-ratio 3 --epochs 2 --seed ${SEED}

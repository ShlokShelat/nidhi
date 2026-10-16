#!/bin/bash
#SBATCH --job-name=reproduce-seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2

SEEDS=(42 123 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "Running seed=$SEED (array task $SLURM_ARRAY_TASK_ID)"

python src/train_factors_nli.py \
    --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir runs/factors_nli_reproduce_s${SEED} \
    --neg-ratio 3 --epochs 2 --seed ${SEED}

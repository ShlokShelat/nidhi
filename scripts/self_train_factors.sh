#!/bin/bash
#SBATCH --job-name=selftrain-fac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/self_train_factors.py \
    --raw artifacts/external_raw.csv \
    --nli-run runs/factors_nli,runs/factors_nli_s2 \
    --conf-gate 0.85 --per-factor-cap 60 \
    --out artifacts/external_selftrain.csv

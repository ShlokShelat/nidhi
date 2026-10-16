#!/bin/bash
#SBATCH --job-name=label-ext
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/label_external.py --stage label \
    --raw artifacts/external_raw.csv \
    --qwen Qwen/Qwen2.5-14B-Instruct \
    --conf-gate 0.80 --per-factor-cap 150 \
    --rare-factors "sexual orientation related issues,exposure to others' suicide,substance use,meaning in life,poor school performance,sense of responsibility,physical health/characteristic,cognitive deficits,low socio-economic status,traumatic experience" \
    --aug-out artifacts/external_factor_aug.csv

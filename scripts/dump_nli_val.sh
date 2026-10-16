#!/bin/bash
#SBATCH --job-name=nlival
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out

# One GPU pass to save the NLI risk models' val probabilities. After this,
# rebuilding the champion ensemble and testing Qwen as a seventh member is
# entirely CPU work.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

python src/dump_nli_val_probs.py \
    --val artifacts/val_split.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --max-len 384 \
    --out artifacts/nli_risk_val_probs.csv

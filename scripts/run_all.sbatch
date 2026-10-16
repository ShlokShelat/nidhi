#!/bin/bash
#SBATCH --job-name=risk-cup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

source scripts/env.sh riskcup
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

DATA_DIR=${DATA_DIR:-data}
RUN=${RUN:-v1}

set -e
mkdir -p logs runs artifacts

# 1. audit + split (fast, CPU)
python src/data_audit.py --train "$DATA_DIR/labeled.csv" \
    ${UNLABELED:+--unlabeled "$UNLABELED"} --outdir artifacts

# 2. three models, sequential on one GPU
python src/train_risk.py     --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir "runs/risk_$RUN"
python src/train_evidence.py --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir "runs/evidence_$RUN"
python src/train_factors.py  --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir "runs/factors_$RUN"

# 3. diagnosis
python src/error_analysis.py --val-preds "runs/risk_$RUN/val_predictions.csv" \
    --evidence-preds "runs/evidence_$RUN/val_predictions.csv" \
    | tee "runs/diagnosis_$RUN.txt"

#!/bin/bash
#SBATCH --job-name=fulldata
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
set -e

# build the full-data file (train + val concatenated)
python -c "
import pandas as pd
tr = pd.read_csv('artifacts/train_split.csv'); va = pd.read_csv('artifacts/val_split.csv')
full = pd.concat([tr, va], ignore_index=True)
full.to_csv('artifacts/full_train.csv', index=False)
print('full_train.csv:', len(tr), '+', len(va), '=', len(full))
"

# risk: 3 seeds on full data, 5 epochs (dev runs peaked epoch 4-5)
for S in 13 14 15; do
  python src/train_risk.py --train artifacts/full_train.csv --val artifacts/full_train.csv \
      --model microsoft/deberta-v3-large --outdir runs/risk_full_s$S \
      --epochs 5 --seed $S --no-val
done

# evidence: full data, 4 epochs (dev best was epoch 3-6, 4 is safe), keep min_conf=0.65
python src/train_evidence.py --train artifacts/full_train.csv --val artifacts/full_train.csv \
    --model microsoft/deberta-v3-large --outdir runs/evidence_full \
    --epochs 4 --no-val --fixed-min-conf 0.65

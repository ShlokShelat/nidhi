#!/bin/bash
#SBATCH --job-name=v2-heads
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
set -e

STAMP=$(date +%Y%m%d_%H%M)
mkdir -p submissions/archive

# archive the current submission (the format-fixed v1) before anything overwrites it
[ -f submissions/NIDHI.csv ] && cp submissions/NIDHI.csv "submissions/archive/NIDHI_${STAMP}_pre_v2.csv"

# train the two improved heads (v1 run dirs remain untouched)
python src/train_factors.py --train artifacts/train_split.csv \
    --val artifacts/val_split.csv --model microsoft/deberta-v3-large --outdir runs/factors_v2
python src/train_evidence.py --train artifacts/train_split.csv \
    --val artifacts/val_split.csv --model microsoft/deberta-v3-large --outdir runs/evidence_v2

# generate the v2 submission candidate + archive it with provenance in the name
python src/predict.py --test data/test.csv \
    --risk-run runs/risk_v1 --evidence-run runs/evidence_v2 --factors-run runs/factors_v2 \
    --team NIDHI
cp submissions/NIDHI.csv "submissions/archive/NIDHI_${STAMP}_risk1_ev2_fac2.csv"

echo "DONE. Review the val numbers above BEFORE uploading submissions/NIDHI.csv"

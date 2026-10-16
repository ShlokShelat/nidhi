#!/bin/bash
#SBATCH --job-name=gamma-seed-confirm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

for GAMMA in 2.0 3.0; do
  for SEED in 14 15; do
    GTAG=$(echo $GAMMA | tr '.' '_')
    OUTDIR="runs/risk_focal_rerun_g${GTAG}_s${SEED}"
    echo "=============================================================="
    echo "Training gamma=${GAMMA} seed=${SEED}"
    echo "Output directory: ${OUTDIR}"
    echo "=============================================================="
    python src/train_risk_focal.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model microsoft/deberta-v3-large --outdir ${OUTDIR} \
        --max-len 512 --epochs 5 --lr 1e-5 --batch 8 --grad-accum 2 \
        --alpha-ordinal 0.5 --focal-gamma ${GAMMA} --user-context --seed ${SEED}
    echo ""
    echo "Finished gamma=${GAMMA} seed=${SEED}"
    echo ""
  done
done

echo "=============================================================="
echo "All 4 confirmation runs completed."
echo "=============================================================="

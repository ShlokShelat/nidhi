#!/bin/bash
#SBATCH --job-name=facd6full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Train factors_d6 on ALL 1635 posts and score test with the OOF thresholds.
#
# THE PROCEDURE IS THE ONE factors_oof_cv.py's docstring specifies:
#
#   "the win here is the thresholds and the honest evaluation. To also capture
#    the +29%, train one final model on all 1635 and apply the OOF thresholds"
#
# The deployed factor models -- factors_nli and factors_nli_s2 -- are trained on
# artifacts/train_split.csv, which is 1263 posts. Training on
# artifacts/full_train.csv is 1635, i.e. +29% data.
#
# TWO ZERO-PARAMETER CHANGES STACK HERE:
#
#   1. the d6 hyperparameters (seed 21, lr 2e-5, 3 epochs, max_len 384)
#   2. +29% training data
#
# Neither fits anything on the 372-post val split. That matters because fitting
# 24 thresholds on 372 posts overfits badly enough to invert the sign of a
# factor gain between val and test.
#
# WHERE THE THRESHOLDS COME FROM. Not from this model -- it has no held-out data
# to fit on, since it trains on everything. They come from
# artifacts/factor_thresholds_d6_oof.npy, written by scripts/factors_d6_5fold.sh
# after fitting on 1635 OUT-OF-FOLD posts. That is exactly the incumbent's
# protocol and the whole reason the CV run exists.
#
# RUN THE CV JOB FIRST. Without those thresholds this model cannot be scored
# honestly, and re-fitting them on val would reintroduce the 372-post overfit.
#
# --no-val is required: there is no held-out split, so the trainer cannot tune
# or early-stop. --fixed-min-conf is irrelevant for factors but the flag is
# shared with the evidence trainer.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

TH=artifacts/factor_thresholds_d6_oof.npy
if [ ! -f "${TH}" ]; then
    echo "ERROR: ${TH} not found."
    echo "Run scripts/factors_d6_5fold.sh to completion first -- it fits the"
    echo "24 thresholds on 1635 out-of-fold posts. Fitting 24 thresholds on the"
    echo "372-post val split instead overfits them and inverts the gain."
    exit 1
fi

OUT=artifacts/factor_runs
mkdir -p "${OUT}"
RUN=runs/factors_d6_full

if [ ! -f "${RUN}/model.pt" ]; then
    echo "================================================================"
    echo "TRAIN factors_d6 on all 1635 posts"
    echo "================================================================"
    python src/train_factors_nli.py \
        --train artifacts/full_train.csv \
        --val   artifacts/val_split.csv \
        --model microsoft/deberta-v3-large \
        --outdir "${RUN}" \
        --seed 21 --lr 2e-5 --epochs 3 --max-len 384 \
        || { echo "FAILED"; exit 1; }
    echo ""
    echo "NOTE: --val was passed because the trainer requires it, but val posts"
    echo "are INSIDE full_train.csv, so any val score it prints is in-sample"
    echo "and meaningless. Ignore it. The honest estimate is the CV run's"
    echo "pooled OOF number."
else
    echo "--- ${RUN} exists ---"
fi

for split in val test; do
    src=artifacts/val_split.csv
    [ "${split}" = "test" ] && src=data/test.csv
    dst="${OUT}/d6_full_${split}.npz"
    [ -f "${dst}" ] && { echo "--- ${dst} exists ---"; continue; }
    python src/predict_factors_nli.py \
        --test "${src}" --nli-run "${RUN}" \
        --thresholds "${TH}" \
        --dump-probs "${dst}" \
        --out "${OUT}/d6_full_${split}_labels.csv" \
        || echo "  dump failed ${split}"
done

echo ""
echo "================================================================"
echo "DONE"
echo "================================================================"
echo "Two candidate factor columns now exist:"
echo "  ${OUT}/d6_cv_test.npz     5-fold bag, honestly estimable"
echo "  ${OUT}/d6_full_test.npz   trained on all 1635, +29% data"
echo ""
echo "The CV bag is the one with a measured OOF score. The full-data model"
echo "cannot be validated -- it saw every labelled post -- so prefer it ONLY"
echo "if the CV version already beat the incumbent's 0.6673, which is what"
echo "establishes that the recipe itself is better."
echo ""
echo "Build with src/factor_build_test.py --allow-empty and the OOF thresholds"
echo "above."

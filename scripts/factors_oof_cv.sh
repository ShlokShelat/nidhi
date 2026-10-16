#!/bin/bash
#SBATCH --job-name=fac-oof-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Author-disjoint 5-fold OOF for the NLI factor model.
#
# Why this exists rather than a plain full-data retrain: train_factors_nli.py
# requires --val and uses it BOTH for checkpoint selection and for threshold
# tuning (thresholds.npy). Training on all 1635 posts puts val inside the
# training set, so a full-data run emits a model with no valid thresholds.
# K-fold gives every post an out-of-fold prediction, so thresholds can be
# tuned on 1635 posts / 153 authors instead of 372 / 32 -- which is the
# measured defect (per-label tuning overfit by 0.0684 macro-F1 on val).
#
# Resumable: any fold whose OOF probability dump already exists is skipped,
# so a job that hits the wall clock can simply be resubmitted.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

K=5
SEED=13

echo "================================================================"
echo "STEP 1: build author-disjoint folds (deterministic, no RNG)"
echo "================================================================"
python src/factors_oof_cv.py --make-folds --k $K --seed $SEED || exit 1

echo ""
echo "================================================================"
echo "STEP 2: train each fold and dump its out-of-fold probabilities"
echo "================================================================"
for f in $(seq 0 $((K-1))); do
    if [ -f "artifacts/fac_oof_f${f}.npz" ]; then
        echo "--- fold $f: OOF dump already present, skipping ---"
        continue
    fi
    echo ""
    echo "--- fold $f: training ---"
    python src/train_factors_nli.py \
        --train artifacts/factor_cv/train_f${f}.csv \
        --val   artifacts/factor_cv/val_f${f}.csv \
        --outdir runs/factors_nli_cv_f${f} \
        --seed $SEED || { echo "FOLD $f TRAINING FAILED"; exit 1; }

    echo "--- fold $f: dumping OOF probabilities ---"
    python src/predict_factors_nli.py \
        --test artifacts/factor_cv/val_f${f}.csv \
        --nli-run runs/factors_nli_cv_f${f} \
        --out artifacts/_oof_f${f}.csv \
        --dump-probs artifacts/fac_oof_f${f}.npz \
        || { echo "FOLD $f PREDICT FAILED"; exit 1; }
done

echo ""
echo "================================================================"
echo "STEP 3: pool the OOF predictions and test whether per-label"
echo "        thresholds generalise once fitted on 1635 posts"
echo "================================================================"
python src/factors_oof_cv.py --collect --k $K --seed $SEED \
    --out-thresholds artifacts/factor_thresholds_oof.npy

echo ""
echo "================================================================"
echo "DONE. Read the strategy table in STEP 3."
echo ""
echo "  If per_label's overfit gap collapses (val gave 0.0684), then"
echo "  per-label thresholds were only ever failing for lack of data,"
echo "  and the OOF thresholds are the deployable artefact."
echo ""
echo "  If the gap stays large, global thresholding is confirmed and"
echo "  the threshold route is closed for good."
echo "================================================================"

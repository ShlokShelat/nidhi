#!/bin/bash
#SBATCH --job-name=facinc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Dump the SHIPPED factor model's probabilities on val and test.
#
# The eight diverse factor models (factors_d1..d8) already have their probs in
# artifacts/factor_runs/. The incumbent -- runs/factors_nli, the model that
# produces the deployed S2 factor score of 0.6301 -- does not, so the
# ensemble comparison currently has no baseline to beat.
#
# Also dumps runs/factors_nli_s2 if present: the deployed factor column is
# artifacts/nli_ensemble_factors.csv, which is an ensemble of both NLI seeds,
# so a single-seed baseline would understate the baseline to beat.
#
# This needs a GPU. Running predict_factors_nli.py on the login node scores
# 372 posts x 24 label definitions = ~8900 pairs through DeBERTa-v3-large on
# CPU, which takes the better part of an hour per split.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

OUT=artifacts/factor_runs
mkdir -p "${OUT}"

for run in factors_nli factors_nli_s2; do
    [ -f "runs/${run}/model.pt" ] || { echo "skip ${run}: no checkpoint"; continue; }
    for split in val test; do
        src=artifacts/val_split.csv
        [ "${split}" = "test" ] && src=data/test.csv
        dst="${OUT}/${run}_${split}.npz"
        if [ -f "${dst}" ]; then
            echo "--- ${dst} exists ---"
            continue
        fi
        echo ""
        echo "=== ${run} / ${split} ==="
        python src/predict_factors_nli.py \
            --test "${src}" --nli-run "runs/${run}" \
            --dump-probs "${dst}" \
            --out "${OUT}/${run}_${split}_labels.csv" \
            || echo "FAILED ${run} ${split}"
    done
done

echo ""
echo "probability files now present:"
ls -1 "${OUT}"/*_val.npz | wc -l
ls -1 "${OUT}"/*_test.npz | wc -l
echo ""
echo "Next (CPU, fast): python src/factor_ensemble.py"

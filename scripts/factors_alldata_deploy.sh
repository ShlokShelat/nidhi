#!/bin/bash
#SBATCH --job-name=facall
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# Train the factor model on ALL 1635 labelled posts instead of train_split's
# 1263, then threshold by prevalence rate-matching.
#
# WHY. The deployed recipe trains on 1263 posts and spends val_split's 372 only
# on epoch selection and threshold fitting -- 23% of the labelled data never
# produces a gradient. On the OOF instrument, enlarging a fold's training set by
# the same proportion improves macro-F1 under all three threshold policies
# (deployed, refit and ratematch alike). A change whose sign is the same under
# every decision rule is a change in the probabilities; one whose verdict flips
# with the rule is a change in the thresholds, and does not transfer.
#
# The change is also confined to how much data the model sees: neither the
# estimator nor any component of the pipeline is altered.
#
# WHY THRESHOLDS ARE NOT A PROBLEM HERE, though training on everything leaves no
# held-out set. `ratematch` FITS NOTHING: it sets each label's cutoff so the
# predicted count equals what TRAIN prevalence implies, and prevalence is known
# in advance. With zero fitted parameters there is nothing that can transfer
# badly from a small fitting sample. src/apply_ratematch_factors.py does that
# step.
#
# THE ONE COMPROMISE, DELIBERATE. --val is val_split, which is now INSIDE the
# training set, so epoch selection is in-sample and drifts to the last epoch.
# The fold experiment behaved the same way, so this ships the configuration
# that was measured. The final epoch is slightly worse than the peak, which
# biases the result DOWN rather than up.
#
# The trainer's own thresholds.npy is in-sample and MUST be ignored; the
# ratematch step never reads it.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

unset PROTECTIVE_TEMPLATE FACTOR_DEFS_JSON CONFUSABLE_JSON || true

MODEL="${MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
TAG="${TAG:-mnliall}"
TRAIN="${TRAIN:-artifacts/full_train.csv}"

for f in "${TRAIN}" artifacts/val_split.csv data/test.csv; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done
TRAIN="${TRAIN}" python - <<'PY'
import os
import pandas as pd
n = len(pd.read_csv(os.environ["TRAIN"], keep_default_na=False))
print(f"training on {n} posts (deployed recipe uses 1263)")
if n < 1600:
    raise SystemExit(f"expected ~1635, got {n} -- wrong file?")
PY

echo "model : ${MODEL}"
echo "mirrors scripts/factors_nli_mnli_deploy.sh apart from --train"

R1="runs/factors_nli_${TAG}"
R2="runs/factors_nli_${TAG}_s2"

if [ -f "${R1}/model.pt" ]; then
    echo "--- ${R1} exists, skipping ---"
else
    echo ""
    echo "=== seed 13, 6 epochs -> ${R1} ==="
    python src/train_factors_nli.py \
        --train "${TRAIN}" --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir "${R1}" \
        --neg-ratio 3 --epochs 6 \
        || { echo "FAILED seed 13"; exit 1; }
fi

if [ -f "${R2}/model.pt" ]; then
    echo "--- ${R2} exists, skipping ---"
else
    echo ""
    echo "=== seed 14, 3 epochs -> ${R2} ==="
    python src/train_factors_nli.py \
        --train "${TRAIN}" --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir "${R2}" \
        --neg-ratio 3 --epochs 3 --seed 14 \
        || { echo "FAILED seed 14"; exit 1; }
fi

# Averaged probabilities, exactly as deployment does. The averaged thresholds it
# also writes are in-sample garbage here and are discarded by the next step.
echo ""
echo "=== test probabilities (thresholds from this step are IGNORED) ==="
python src/predict_factors_nli.py \
    --test data/test.csv \
    --nli-run "${R1},${R2}" \
    --dump-probs "artifacts/factor_runs/nli_ensemble_${TAG}_test.npz" \
    --out "artifacts/nli_ensemble_factors_${TAG}_rawthr.csv" \
    || { echo "FAILED prediction"; exit 1; }

echo ""
echo "=== ratematch thresholds from known prevalence (fits nothing) ==="
python src/apply_ratematch_factors.py \
    --probs "artifacts/factor_runs/nli_ensemble_${TAG}_test.npz" \
    --labeled artifacts/full_train.csv \
    --out "artifacts/nli_ensemble_factors_${TAG}.csv" \
    || { echo "FAILED ratematch"; exit 1; }

echo ""
echo "DONE -> artifacts/nli_ensemble_factors_${TAG}.csv"
echo ""
echo "Patch onto the champion. Risk and evidence are untouched, so the risk"
echo "score must come back unchanged and any factor move is this change alone:"
echo ""
echo "  python src/patch_submission_risk.py \\"
echo "      --base submissions/NIDHI_canon_on_nlimnli.csv \\"
echo "      --factors artifacts/nli_ensemble_factors_${TAG}.csv \\"
echo "      --out submissions/NIDHI_factors_${TAG}.csv"

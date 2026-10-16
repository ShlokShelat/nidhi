#!/bin/bash
#SBATCH --job-name=facoof
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# ONE fold of the factor OOF instrument. Submit five of these per arm.
#
#   sbatch --export=ALL,FOLD=0,TAG=mnli,MODEL=MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli scripts/factors_oof_fold.sh
#
# WHY THIS INSTRUMENT EXISTS. On the 372-post val split the standard error of an
# absolute macro-F1 is about 0.0215, which is larger than the effect any single
# factor arm plausibly has -- so a ranking taken there is a ranking of sampling
# noise. Pooling five author-disjoint folds to 1635 posts takes that SE to
# roughly 0.003.
#
# IT REPLICATES THE DEPLOYED RECIPE EXACTLY, so that an arm delta measured here
# is a delta in the thing that actually ships:
#   * same two runs -- seed 13 for 6 epochs, seed 14 for 3 epochs
#   * --val does its real double duty: epoch selection AND threshold fitting
#   * inference through predict_factors_nli.py, which averages the two runs'
#     probabilities and separately averages their two threshold vectors
# Only the data partition changes, and only so that the scored posts are ones
# no part of the fitting ever saw.
#
# THE LEAK IT AVOIDS. Masking a pooled prediction file by `own != f` retains the
# posts the checkpoint trained on, which inflates the score past the annotation
# ceiling and makes every arm look equally good. Here the split is physical:
# make_factor_oof_splits.py asserts the three author groups are disjoint.
#
# THE KNOWN BIAS. TRAIN here is ~1015 posts against deployment's 1263. A uniform
# shortfall largely cancels in a PAIRED delta, but transfer-learning gains
# shrink as data grows, so MNLI-pretrained arms read somewhat high. Upper bound,
# not point estimate.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

FOLD="${FOLD:?set FOLD=0..4}"
TAG="${TAG:?set TAG, e.g. mnli}"
MODEL="${MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
SPLITS="${SPLITS:-artifacts/factor_folds}"
OUT="${OUT:-artifacts/factor_oof}"

# SEED1/SEED2 exist ONLY so a replica arm can be built for --null. Two runs of
# an identical configuration differ by TRAINING nondeterminism, and that gap
# survives author resampling, so pooling more posts cannot shrink it. The only
# way to know how much of an arm delta is seed luck is to run the SAME config
# under a different seed pair and measure it:
#   sbatch --export=ALL,FOLD=$f,TAG=mnlirep,SEED1=15,SEED2=16,MODEL=<same> ...
# Then pass --null mnlirep. Leave these alone for a real arm: 13/14 is what
# deploys.
SEED1="${SEED1:-13}"
SEED2="${SEED2:-14}"

# ALLDATA=1 trains on TRAIN+THRESH (1306 posts) instead of TRAIN (1015).
#
# WHY. Deployment trains the factor model on train_split's 1263 posts while
# val_split's 372 only select an epoch and fit thresholds -- 23% of the labelled
# data never produces a gradient. This arm measures what reclaiming it is worth.
#
# THE ONE CONTAMINATION, AND WHY IT IS SAFE. --val is still THRESH, which is now
# inside the training set, so epoch selection becomes in-sample and will drift
# toward the final epoch. The final epoch is consistently a little WORSE than
# the peak (the 6-epoch runs peak at epoch 2-5 and decline), so this biases the
# measurement DOWN. A gain seen here is a floor, not a ceiling.
#
# THE THRESHOLDS THIS ARM DUMPS ARE IN-SAMPLE AND MUST NOT BE USED. Judge it
# with src/factor_threshold_policies.py, whose cross-fitted `refit` policy is
# equivalent to the deployed one within noise. Compare refit-to-refit, never
# refit against deployed, or the threshold policy confounds the arm.
if [ "${ALLDATA:-0}" = "1" ]; then
    TR="${SPLITS}/f${FOLD}_alldata.csv"
    echo "ALLDATA=1: training on TRAIN+THRESH (thresholds from this run are "
    echo "           in-sample -- refit them cross-fitted before judging)"
else
    TR="${SPLITS}/f${FOLD}_train.csv"
fi
TH="${SPLITS}/f${FOLD}_thresh.csv"
EV="${SPLITS}/f${FOLD}_eval.csv"
for f in "${TR}" "${TH}" "${EV}"; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing -- run "; \
        echo "  python src/make_factor_oof_splits.py --k 5"; exit 1; }
done

# Same opt-in axes as scripts/factors_nli_mnli_deploy.sh, with the same
# semantics. FACTOR_DEFS_JSON is read at BOTH training and inference by
# hypothesis_for(), so it must stay exported across every step below.
# CONFUSABLE_JSON is training-only. PROTECTIVE_TEMPLATE is global at inference,
# so a bag may not mix template and non-template members.
unset PROTECTIVE_TEMPLATE || true
if [ -n "${TEMPLATE:-}" ]; then
    export PROTECTIVE_TEMPLATE=1
    echo "protective template: ON"
fi
for V in FACTOR_DEFS_JSON CONFUSABLE_JSON; do
    eval "v=\${${V}:-}"
    if [ -n "${v}" ]; then
        [ -f "${v}" ] || { echo "ERROR: ${V}=${v} not found"; exit 1; }
        export "${V}=${v}"
        echo "${V}: ${v}"
    else
        unset "${V}" || true
    fi
done

echo "fold ${FOLD}  tag ${TAG}"
echo "model ${MODEL}"
# NOT wc -l: posts contain embedded newlines inside quoted CSV fields, so a line
# count overstates the row count by roughly 50%. Count parsed rows instead.
TR="${TR}" TH="${TH}" EV="${EV}" python - <<'PY'
import os
import pandas as pd
for k, lbl in (("TR", "train "), ("TH", "thresh"), ("EV", "eval  ")):
    n = len(pd.read_csv(os.environ[k], keep_default_na=False))
    note = {"TR": "", "TH": "   (epoch selection + thresholds)",
            "EV": "   (never fitted on)"}[k]
    print(f"{lbl} {n} posts{note}")
PY

R1="runs/foof_${TAG}_f${FOLD}_s${SEED1}"
R2="runs/foof_${TAG}_f${FOLD}_s${SEED2}"
echo "seeds ${SEED1} (6 epochs) / ${SEED2} (3 epochs)"

# ---- 6 epochs -- mirrors scripts/factors_nli.sh -----------------------------
if [ -f "${R1}/model.pt" ]; then
    echo "--- ${R1} exists, skipping ---"
else
    echo ""
    echo "=== seed ${SEED1}, 6 epochs -> ${R1} ==="
    python src/train_factors_nli.py \
        --train "${TR}" --val "${TH}" \
        --model "${MODEL}" --outdir "${R1}" \
        --neg-ratio 3 --epochs 6 --seed "${SEED1}" \
        || { echo "FAILED seed ${SEED1}"; exit 1; }
fi

# ---- 3 epochs -- mirrors scripts/nli_seed2.sh ------------------------------
if [ -f "${R2}/model.pt" ]; then
    echo "--- ${R2} exists, skipping ---"
else
    echo ""
    echo "=== seed ${SEED2}, 3 epochs -> ${R2} ==="
    python src/train_factors_nli.py \
        --train "${TR}" --val "${TH}" \
        --model "${MODEL}" --outdir "${R2}" \
        --neg-ratio 3 --epochs 3 --seed "${SEED2}" \
        || { echo "FAILED seed ${SEED2}"; exit 1; }
fi

# ---- score the held-out fold the deployed way -------------------------------
# No --thresholds override: predict_factors_nli.py averages the two runs' probs
# AND averages their own thresholds, which is byte-for-byte the deployed path.
echo ""
echo "=== scoring the held-out fold ==="
mkdir -p "${OUT}"
python src/predict_factors_nli.py \
    --test "${EV}" \
    --nli-run "${R1},${R2}" \
    --dump-probs "${OUT}/${TAG}_f${FOLD}.npz" \
    --out "${OUT}/${TAG}_f${FOLD}.csv" \
    || { echo "FAILED prediction"; exit 1; }

echo ""
echo "DONE fold ${FOLD} -> ${OUT}/${TAG}_f${FOLD}.npz"
echo "When all five folds are present:"
echo "  python src/factor_oof_instrument.py --arm ${TAG}"

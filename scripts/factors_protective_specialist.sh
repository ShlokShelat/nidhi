#!/bin/bash
#SBATCH --job-name=facprot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# A SPECIALIST for the five protective factors, warm-started from the general
# factor model.
#
# WHY. Held-out F1 splits by polarity: protective 0.504, risk 0.651. Polar
# pairs are worse -- social support 0.533 vs poor social support 0.817,
# psychological capital 0.577 vs hopelessness 0.816, meaning in life 0.231 vs
# hopelessness 0.816. `meaning in life` has precision 0.203 AND recall 0.267,
# so it is a discrimination failure, not calibration; thresholds are already at
# their held-out optimum, and no threshold can repair an ordering.
#
# THE MECHANISM. Every post in this corpus is distressed, so the encoder learns
# a negative prior for the domain. The cross-encoder shares EVERY parameter
# across all 24 factors, so the only thing telling it a question is protective
# is the hypothesis wording -- which has to fight that prior. A specialist never
# switches polarity: its whole training signal is positive content.
#
# PRIOR WORK USES THE SAME SEPARATION. arXiv 2507.10008 uses separate MLPs for
# risk and protective factors, stating they "operate independently rather than
# being the reverse of risk factors". This is that design in a cross-encoder.
#
# WHY WARM-STARTED AND NOT FROM SCRATCH. Five factors is about a quarter of the
# training pairs -- far too thin to learn a representation, ample to specialise
# one. --init-from loads the general model's checkpoint, so the specialist
# begins from an encoder that already reads these posts well.
#
# WHY --full-enum. With 5 factors it is 5 pairs per post instead of 24, so
# exhaustive coverage costs a QUARTER of a normal epoch rather than 2.1x. That
# removes the sampled scheme's coverage imbalance (a post with one gold factor
# saw 3 of 23 negatives; one with six saw all 18) for free. Because full
# enumeration invalidates plain bce's pos_weight=3.0 -- derived from the
# sampling ratio that no longer applies -- the objective moves to
# bce_perlabel, which takes each factor's weight from its actual ratio.
#
# JUDGE IT PER LABEL, not by a composite. The 372-post instrument has a paired
# null band of +/-0.0559, wider than any plausible effect from five of 24
# labels, so a composite figure cannot separate this change from noise. Each of
# the five must move by roughly +0.2 to clear the +/-0.10 swing a single label
# shows between seeds.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${NTHREADS}" MKL_NUM_THREADS="${NTHREADS}"

export FACTOR_SUBSET="${FACTOR_SUBSET:-protective}"
# Aspects and the specialist attack the SAME mechanism from different angles, so
# keep them apart until each is measured alone. Set ASPECTS=1 to combine later.
if [ "${ASPECTS:-0}" = "1" ]; then
    export FACTOR_ASPECTS_JSON=artifacts/aspects.json
    echo "aspects: ON (combined arm)"
else
    unset FACTOR_ASPECTS_JSON || true
    echo "aspects: off (specialist alone)"
fi

TAG="${TAG:-prot}"
EPOCHS="${EPOCHS:-3}"
LOSS="${LOSS:-bce_perlabel}"
BASE1="${BASE1:-runs/factors_nli_mnli}"
BASE2="${BASE2:-runs/factors_nli_mnli_s2}"
MODEL="${MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"

for d in "${BASE1}" "${BASE2}"; do
    [ -f "${d}/model.pt" ] || { echo "ERROR: ${d}/model.pt missing -- the "; \
        echo "general factor pair must exist to warm-start from"; exit 1; }
done
for f in artifacts/train_split.csv artifacts/val_split.csv data/test.csv \
         artifacts/factor_runs/nli_ensemble_mnli_test.npz; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done
echo "subset : ${FACTOR_SUBSET}   epochs ${EPOCHS}   loss ${LOSS}"
echo "warm-start: ${BASE1} / ${BASE2}"

# ---- two seeds, warm-started from the matching champion member --------------
i=0
for SEED in 13 14; do
    i=$((i + 1))
    if [ "${i}" = "1" ]; then INIT="${BASE1}"; else INIT="${BASE2}"; fi
    OUT="runs/factors_nli_${TAG}"
    [ "${i}" = "1" ] || OUT="runs/factors_nli_${TAG}_s2"
    if [ -f "${OUT}/model.pt" ]; then
        echo "--- ${OUT} exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "seed ${SEED}, ${EPOCHS} epochs, init from ${INIT} -> ${OUT}"
    echo "================================================================"
    python src/train_factors_nli.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir "${OUT}" \
        --full-enum --loss "${LOSS}" --epochs "${EPOCHS}" --seed "${SEED}" \
        --init-from "${INIT}" \
        || { echo "FAILED seed ${SEED}"; exit 1; }
done

# ---- score test: only the subset columns are filled, the rest stay 0 --------
echo ""
echo "=== test probabilities (subset columns only) ==="
python src/predict_factors_nli.py \
    --test data/test.csv \
    --nli-run "runs/factors_nli_${TAG},runs/factors_nli_${TAG}_s2" \
    --dump-probs "artifacts/factor_runs/nli_ensemble_${TAG}_test.npz" \
    --out "artifacts/nli_ensemble_factors_${TAG}_raw.csv" \
    || { echo "FAILED prediction"; exit 1; }

# ---- splice: protective from the specialist, the other 19 from the general --
echo ""
echo "=== splice onto the general model's other 19 columns ==="
python src/splice_factor_columns.py \
    --base artifacts/factor_runs/nli_ensemble_mnli_test.npz \
    --specialist "artifacts/factor_runs/nli_ensemble_${TAG}_test.npz" \
    --out "artifacts/nli_ensemble_factors_${TAG}.csv" \
    || { echo "FAILED splice"; exit 1; }

echo ""
echo "DONE -> artifacts/nli_ensemble_factors_${TAG}.csv"
echo ""
echo "Patch onto the base submission. Risk and evidence are untouched, so"
echo "their sub-scores must be unchanged and any factor move is this alone:"
echo ""
echo "  python src/patch_submission_risk.py \\"
echo "      --base submissions/NIDHI_canon_on_nlimnli.csv \\"
echo "      --factors artifacts/nli_ensemble_factors_${TAG}.csv \\"
echo "      --out submissions/NIDHI_factors_${TAG}.csv"

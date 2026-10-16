#!/bin/bash
#SBATCH --job-name=fac2head
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# SHARED TRUNK, SEPARATE MLP HEADS FOR THE 19 RISK FACTORS AND 5 PROTECTIVE.
#
# THE TARGET. Protective factors score 0.5964 against 0.7415 for risk factors
# on the 11-arm OOF. Closing that gap toward parity would be a meaningful
# composite gain. It is the only headroom left anywhere in the system: risk
# is at its annotation ceiling, and the evidence decode is the global maximum
# of an 80-cell grid.
#
# WHY THIS CONFIGURATION AND NOT THE ONE THAT ALREADY FAILED.
# `FACTOR_SUBSET=protective` trained on the 5 labels alone and lost to the
# general arms on 4/5 folds and to the pool on 5/5. The recorded diagnosis is
# that restricting to 5 labels DESTROYS the multi-task benefit of the other 19
# -- so keep the trunk and split only the head.
#
# A LINEAR SPLIT WOULD BE A NO-OP. Linear(h,19) concatenated with Linear(h,5)
# IS Linear(h,24): same parameters, same gradients. Each head therefore gets
# its own NONLINEAR projection (Linear(h,h/2) -> GELU -> Dropout -> Linear),
# which is what "separate MLPs" actually requires. Worth stating because the
# obvious reading of the idea is empty.
#
# HONEST PRIOR: BELOW 10%. Every `mlc` arm scores ~0.45 val macro against the
# NLI pool's 0.677. As a 12th arm this must clear the fixed per-arm dilution
# tax AND beat the pre-specified set -- a bar only `friendfac` has ever
# cleared. It is run because the GPU is otherwise idle and nothing else has
# comparable upside, not because it is likely.
#
# NOTHING IS SUBMITTED HERE.
# ===========================================================================

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"
unset FACTOR_SUBSET || true

TAG="${TAG:-2head}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"
EPOCHS="${EPOCHS:-8}"
mkdir -p logs

echo "node $(hostname)  $(date)"
for f in artifacts/train_split.csv artifacts/val_split.csv; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done

# ---------------------------------------------------------------------------
# SMOKE GATE. One epoch. If the head split does not print, or val macro is
# nonsense, stop before spending five folds.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "SMOKE: 1 epoch, split heads"
echo "################################################################"
rm -rf "runs/factors_${TAG}_smoke"
python src/train_factors_multilabel.py \
    --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model "${MODEL}" --outdir "runs/factors_${TAG}_smoke" \
    --split-heads --epochs 1 \
    || { echo "SMOKE FAILED -- plumbing is broken, not the idea"; exit 1; }

grep -q "SPLIT HEADS: 19 risk-factor labels + 5 protective" \
    "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null \
  || echo "  (could not confirm the SPLIT HEADS line from the log; check it above)"

echo ""
echo "################################################################"
echo "FOLDS: 5 x split-head arm"
echo "################################################################"
for F in 0 1 2 3 4; do
    OUT="runs/factors_${TAG}_cv_f${F}"
    if [ -f "${OUT}/model.pt" ]; then
        echo "--- ${OUT} exists, skipping ---"
        continue
    fi
    echo ""
    echo "=== fold ${F} -> ${OUT} ==="
    python src/train_factors_multilabel.py \
        --train "artifacts/folds/train_f${F}.csv" \
        --val "artifacts/folds/val_f${F}.csv" \
        --model "${MODEL}" --outdir "${OUT}" \
        --split-heads --epochs "${EPOCHS}" --seed $((21 + F)) \
        || { echo "FAILED fold ${F}"; exit 1; }
done

cat <<'NEXT'

################################################################
DONE. Nothing submitted.
################################################################
Next, to judge it as a 12th arm -- it must beat BOTH bars or it does not ship:

  1. dump OOF + test:
       python src/predict_factors_multilabel.py --run runs/factors_2head_cv_f0,...
     (the predictor auto-detects the split head from the checkpoint)
  2. add it to the pool and measure against the SHIPPED 11 with
     author-disjoint win rates -- NOT against BAG8, and NOT against a
     legacy model. It must clear the fixed per-arm dilution tax.
  3. bar: must beat what the shipped pre-specified set achieves, at the
     required win rate against held-out authors. Only `friendfac` has ever
     cleared it.

Read the PER-GROUP val macro in the fold logs before any of that: if
protective has not moved off ~0.60 the architecture did nothing and the arm
is not worth pooling.
NEXT

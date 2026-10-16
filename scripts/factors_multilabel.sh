#!/bin/bash
#SBATCH --job-name=mlcfac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# MULTI-LABEL CLASSIFIER FACTOR ARM -- isolating the FORMULATION axis of the
# friendfac member's contribution.
#
# BACKGROUND. friendfac, part of the deployed factor pool, differs from the
# other pool members on TWO axes at once: (a) backbone (mental-roberta-base
# 125M) and (b) FORMULATION (one forward pass emitting all 24 factor logits
# from a multi-label head, vs the pool's 24 separate entailment questions).
# Axis (a) is covered elsewhere; this job trains the friend's FORMULATION on
# the pool's own backbone, so the two axes become separable -- and if
# formulation carries part of the gain, a retrainable, shippable copy of it
# exists in this repo.
#
# WHY IT COULD PAY: decorrelation. NOT "label dependencies" -- the gold-label
# structure audit showed perfect co-label knowledge reaches only 0.317 macro,
# 87% of it cardinality. friendfac demonstrated empirically that this
# formulation decorrelates from pairwise NLI scoring, and decorrelation is
# what an ensemble pool pays for.
#
# MATCHED CONSTRUCTION, the property shared by the prior changes that
# generalized best from offline measurement: OOF = the held-out fold scored by
# that fold's model(s); test = the bag of the SAME five fold models; every
# dump at --max-len 320 like the rest of the pool. The only new variable is
# the formulation.
#
# TRAINING IS ~24x CHEAPER PER EPOCH than an NLI arm (one forward per post),
# so a fold takes minutes, not hours. Default is ONE seed (21) so the arm's
# OOF/test construction matches every pool member's (1 fold model -> OOF,
# 5-model bag -> test). SEEDS="13 21 42" is available and keeps the matched
# shape (seed-avg per fold -> OOF, all-seeds bag -> test), but run it only
# AFTER the single-seed arm has been judged, or seed count and formulation
# are conflated.
#
# NOTHING SHIPS FROM HERE. Afterwards factor_member_forward.py decides.
# FALSIFIERS:
#   * the arm must beat the fixed per-arm dilution tax that bag8 charges for
#     every added member;
#   * it competes as an ADD on top of the shipped set,
#     BAG8+ltail+friendfac+cbnsynapi -- one that does not survive INTO that
#     set is dead, however it scores standalone;
#   * judge by WIN RATE and per-split sd, never the CI of the mean.
# ===========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

# sbatch defaults to --export=ALL; a dirty login shell would otherwise leak
# hypothesis-config vars. They do nothing in this trainer (no hypotheses),
# but FACTOR_SUBSET would hard-fail it and the others would pollute the
# recorded provenance.
unset FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON PROTECTIVE_TEMPLATE \
      CONFUSABLE_JSON FACTOR_SUBSET 2>/dev/null || true

set -u
TAG="${TAG:-mlc}"
SEEDS="${SEEDS:-21}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"
LOSS="${LOSS:-bce}"          # bce first -- test the formulation cleanly;
                             # LOSS=cbntr is the known-good follow-up knob
EPOCHS="${EPOCHS:-8}"
LR="${LR:-1e-5}"
# EXTRA is passed straight through to train_factors_multilabel.py, e.g.
#   "EXTRA=--lambda-risk 0.3".  Quote the WHOLE assignment when submitting --
#   sbatch --export splits on commas and the shell would split on the space.
# THIS WAS MISSING AND COST A RUN: the arg list below was hardcoded, so an
# EXTRA passed on the command line was silently dropped and the job looked
# like a successful experiment while actually re-running the default config.
# A flag that vanishes without a trace is worse than one that errors.
EXTRA="${EXTRA:-}"
OUT=artifacts/factor_runs
mkdir -p "$OUT" logs

echo "node $(hostname)  $(date)"
echo "TAG=${TAG}  SEEDS=${SEEDS}  MODEL=${MODEL}  LOSS=${LOSS}  EPOCHS=${EPOCHS}  LR=${LR}"
echo "EXTRA=${EXTRA:-(none)}"
df -h . | tail -1

[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }
[ -f data/test.csv ] || { echo "ERROR: data/test.csv missing"; exit 1; }

# TAG-collision guard: every dir this script writes carries the
# "multilabel_arm" marker in run_config.json (written by the trainer itself).
# A directory under this TAG without the marker belongs to another experiment
# -- abort rather than silently pool two different models into one OOF.
for d in runs/factors_${TAG}_s*_cv_f*; do
    [ -d "$d" ] || continue
    if ! grep -q '"multilabel_arm"' "${d}/run_config.json" 2>/dev/null; then
        echo "ERROR: ${d} exists and was not written by train_factors_multilabel.py."
        echo "Another experiment owns TAG=${TAG}. Pick a different TAG."
        exit 1
    fi
done

FAIL=0
for f in 0 1 2 3 4; do
    echo ""
    echo "################################################################"
    echo "FOLD ${f}"
    echo "################################################################"
    R=""
    for s in ${SEEDS}; do
        DST="runs/factors_${TAG}_s${s}_cv_f${f}"
        if [ ! -f "${DST}/model.pt" ]; then
            echo ""
            echo "=== ${TAG} fold ${f} seed ${s}: training ==="
            python src/train_factors_multilabel.py \
                --train "artifacts/folds/train_f${f}.csv" \
                --val   "artifacts/folds/val_f${f}.csv" \
                --model "${MODEL}" \
                --outdir "${DST}" \
                --loss "${LOSS}" \
                --seed "${s}" --lr "${LR}" --epochs "${EPOCHS}" --max-len 320 \
                ${EXTRA} \
                || { echo "TRAIN FAILED ${TAG} fold ${f} seed ${s}"; FAIL=1; continue; }
        else
            echo "--- ${TAG} fold ${f} seed ${s}: exists ---"
        fi
        [ -f "${DST}/model.pt" ] && R="${R}${R:+,}${DST}"
    done

    if [ -z "${R}" ]; then
        echo "fold ${f}: no trained model, skipping OOF dump"
        FAIL=1
        df -h . | tail -1
        continue
    fi

    # OOF = the held-out fold scored by this fold's model(s), --max-len 320
    # matching every other arm's dump.
    O="${OUT}/${TAG}_cv_oof_f${f}.npz"
    if [ ! -f "${O}" ]; then
        python src/predict_factors_multilabel.py \
            --test "artifacts/folds/val_f${f}.csv" \
            --run "${R}" --max-len 320 \
            --dump-probs "${O}" \
            --out "${OUT}/${TAG}_cv_oof_f${f}_labels.csv" \
            || { echo "OOF dump failed ${TAG} fold ${f}"; FAIL=1; }
    else
        echo "--- ${TAG} fold ${f}: OOF dump exists ---"
    fi
    df -h . | tail -1
done

# test = the 5-fold bag of the SAME models the OOF came from.
DST="${OUT}/${TAG}_cv_test.npz"
if [ ! -f "${DST}" ]; then
    R=""; OK=1
    for f in 0 1 2 3 4; do
        for s in ${SEEDS}; do
            d="runs/factors_${TAG}_s${s}_cv_f${f}"
            if [ ! -f "${d}/model.pt" ]; then
                echo "  ${d} missing -- REFUSING a partial test bag (a 4-fold"
                echo "  bag against a 5-fold OOF is exactly the construction"
                echo "  mismatch that loses on this board)."
                OK=0
            fi
            R="${R}${R:+,}${d}"
        done
    done
    if [ ${OK} -eq 1 ]; then
        echo ""
        echo "=== ${TAG} test dump (5-fold bagged, --max-len 320) ==="
        python src/predict_factors_multilabel.py --test data/test.csv \
            --run "${R}" --max-len 320 \
            --dump-probs "${DST}" \
            --out "${OUT}/${TAG}_cv_test_labels.csv" \
            || { echo "test dump failed ${TAG}"; FAIL=1; }
    else
        FAIL=1
    fi
else
    echo "--- ${TAG}: test dump exists ---"
fi
df -h . | tail -1

echo ""
echo "################################################################"
echo "DONE (FAIL=${FAIL}).  Next, all CPU:"
echo "################################################################"
cat <<'NEXT'
  # 1. does the formulation arm earn a slot? Judge as an ADD on top of the
  #    CURRENT set, and remember every added arm pays a fixed dilution tax:
  python src/factor_member_forward.py 2>&1 | tail -30
  #    the set it must improve is BAG8+ltail+friendfac+cbnsynapi.
  #    Judge by WIN RATE and per-split sd (sd ~ mean/z(win)), never the CI
  #    of the mean -- a tight CI on the mean does not guarantee an arm
  #    generalizes to held-out authors.

  # 2. axis attribution: compare this arm's decorrelation against friendfac's.
  #    If mlc adds on top of friendfac, formulation carries signal this
  #    backbone can express; if mlc and friendfac are interchangeable, the
  #    formulation was most of the friend's edge; if mlc adds nothing while
  #    friendfac does, the backbone (axis a) was the edge.

  # 3. only if the add survives at >=90% win with a real per-split sd:
  #    rebuild the factors column with the winning set via
  #    factor_threshold_stabilise.py (--use tfit, NOT tboth), prove the
  #    rebuild reproduces the incumbent column with the CURRENT pool first,
  #    and S1 must return EXACTLY the incumbent's -- only S2 may move.
NEXT

#!/bin/bash
#SBATCH --job-name=ltail
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# ---------------------------------------------------------------------------
# TWO-STAGE LONG-TAIL TRAINING FOR FACTORS, PLUS TEST DUMPS FOR THE ARMS THAT
# WIN ON RARE LABELS.
#
# THE DIAGNOSIS (measured, not assumed)
#
#   Factors are scored by MACRO-F1 over 24 labels -- each worth 1/24 however
#   rare. Training samples `--neg-ratio 3`, so pairs per label ~ 4 x positives:
#
#       meaning in life    45 positives ->   180 pairs
#       hopelessness      745 positives ->  2980 pairs      16.6x more
#       macro-F1 weights them               1 : 1
#
#   `factor_loss_metric_gap.py` confirmed the consequence is in the ORDERING,
#   not the cut: high-support labels have AP 0.91-0.92 while `meaning in life`
#   has AP 0.23. Thresholds cannot fix ordering, and two threshold lanes have
#   already failed.
#
# THE LITERATURE
#
#   Kang et al., ICLR 2020, "Decoupling Representation and Classifier for
#   Long-Tailed Recognition": train the representation on the IMBALANCED data,
#   then retrain with class-balanced sampling. Joint re-balancing from scratch
#   damages representation learning.
#
#   LDAM-DRW: apply re-weighting LATE (deferred), not from step zero.
#
#   Both say the same thing, and it explains two results already observed
#   here:
#     * loss reweighting from scratch backfires here too (the global weight
#       functioned as regularisation) -- exactly the failure the papers
#       describe.
#     * the `fenum` arm (--full-enum, balanced from scratch) is FIRST on the
#       rarest label (substance use 0.772 vs 0.719) and LAST on the common ones
#       (hopelessness 0.846 vs 0.914). Balanced-from-scratch buys the tail and
#       sells the head.
#
#   Two-stage should buy the tail WITHOUT selling the head. That is the whole
#   hypothesis, and it is falsifiable per label.
#
# WHAT THIS RUNS
#
#   PART A  5 folds of a two-stage arm `ltail`
#             stage 1: the known-good d6 recipe (seed 21, lr 2e-5, 3 epochs,
#                      max-len 384) -- trained here rather than initialised
#                      from an existing arm, because factor runs have no
#                      run_config.json and their base model cannot be verified
#             stage 2: --init-from stage 1, --full-enum (every label gets all
#                      1635 posts), --loss bce_perlabel (per-label pos_weight
#                      from the real neg:pos ratio), lr 5e-6, 1 epoch
#           then OOF dumps per fold and a 5-fold-bagged test dump.
#
#   PART B  test dumps for the four arms that beat every BAG8 member on a rare
#           label but have no `_cv_test.npz` and were therefore excluded from
#           every audit: paircm (meaning in life 0.241, low SES 0.875),
#           dv2xl (cognitive deficits 0.634), mnli3, pairprot.
#           Inference only -- the checkpoints exist.
#
# NOTHING HERE IS SHIPPED WITHOUT MEASUREMENT. Afterwards:
#     python src/factor_member_forward.py        # ltail + the 4 new arms
#     python src/factor_arm_ap_protective.py --arms <...> ltail --all-labels
# and the falsifier is per-label: `ltail` must beat `fenum` on common labels
# AND beat BAG8 on rare ones. If it only does one, the two-stage did not work.
# ---------------------------------------------------------------------------

source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

set -u
OUT=artifacts/factor_runs
mkdir -p "$OUT"

if [ ! -f artifacts/folds/train_f0.csv ]; then
    echo "ERROR: artifacts/folds/ missing"; exit 1
fi

TAG=ltail
FAIL=0

# ---------------------------------------------------------------------------
# PART B RUNS FIRST: it is cheap (inference only, ~40 min) and certain, while
# Part A is hours and speculative. If the job is killed the certain value is
# still kept.
#
# ENVIRONMENT-MISMATCH GUARD. These arms have no run_config.json, so the
# hypothesis text they were TRAINED with is unrecorded -- FACTOR_DEFS_JSON,
# FACTOR_ASPECTS_JSON and PROTECTIVE_TEMPLATE are read from the environment at
# both train and inference time. Dumping test predictions under default env
# when the arm was trained under a different one is exactly the bug that made
# the three diverse evidence taggers look useless (trained conditioned, dumped
# unconditioned).
#
# So for each arm: re-dump fold-0 OOF under DEFAULT env and compare it to the
# stored .npz. Identical -> defaults were used, the test dump is valid.
# Different -> the stored OOF came from another env and the arm is unusable
# until that is known. Refuse rather than ship a silent mismatch.
# ---------------------------------------------------------------------------
echo "################################################################"
echo "PART B -- test dumps for rare-label winners (env-verified first)"
echo "################################################################"
for a in paircm dv2xl mnli3 pairprot; do
    DST="${OUT}/${a}_cv_test.npz"
    if [ -f "${DST}" ]; then echo "--- ${a}: test dump exists ---"; continue; fi
    d0="runs/factors_${a}_cv_f0"
    if [ ! -d "$d0" ]; then echo "  ${a}: no ${d0}, skip"; continue; fi

    echo ""
    echo "=== ${a}: verifying env by re-dumping fold-0 OOF under defaults ==="
    TMP="${OUT}/_envcheck_${a}.npz"
    python src/predict_factors_nli.py \
        --test "artifacts/folds/val_f0.csv" --nli-run "${d0}" \
        --dump-probs "${TMP}" --out "/dev/null" \
        || { echo "  ${a}: env re-dump failed, skip"; continue; }
    python - "$a" "$TMP" "${OUT}/${a}_cv_oof_f0.npz" <<'PY'
import sys, numpy as np
arm, tmp, ref = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    A = np.load(tmp, allow_pickle=True)["probs"].astype(float)
    B = np.load(ref, allow_pickle=True)["probs"].astype(float)
except Exception as e:
    print(f"  {arm}: cannot compare ({e}) -> TREAT AS MISMATCH"); sys.exit(3)
if A.shape != B.shape:
    print(f"  {arm}: shape {A.shape} vs stored {B.shape} -> MISMATCH"); sys.exit(3)
d = float(np.abs(A - B).max())
print(f"  {arm}: max |re-dump - stored| = {d:.6f}")
sys.exit(0 if d < 1e-4 else 3)
PY
    if [ $? -eq 0 ]; then
        echo "  ${a}: env VERIFIED (defaults reproduce the stored OOF)"
        R=""
        MISS=0
        for f in 0 1 2 3 4; do
            dd="runs/factors_${a}_cv_f${f}"
            [ -d "$dd" ] || { echo "  ${a}: missing ${dd}"; MISS=1; break; }
            R="${R}${R:+,}${dd}"
        done
        if [ $MISS -eq 0 ]; then
            python src/predict_factors_nli.py --test data/test.csv \
                --nli-run "${R}" --dump-probs "${DST}" \
                --out "${OUT}/${a}_cv_test_labels.csv" \
                || echo "  ${a}: test dump failed (continuing)"
        fi
    else
        echo "  ${a}: ENV MISMATCH -- the stored OOF was produced under a"
        echo "        different hypothesis configuration. NOT dumping test;"
        echo "        shipping it would repeat the diverse-tagger bug."
    fi
    rm -f "${TMP}"
done

echo ""
echo "################################################################"
echo "PART A -- two-stage long-tail arm '${TAG}'"
echo "################################################################"
for f in 0 1 2 3 4; do
    S1="runs/factors_${TAG}s1_cv_f${f}"
    S2="runs/factors_${TAG}_cv_f${f}"

    if [ ! -f "${S1}/model.pt" ]; then
        echo ""
        echo "=== ${TAG} fold ${f} STAGE 1 (standard, imbalanced) ==="
        python src/train_factors_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --model microsoft/deberta-v3-large \
            --outdir "${S1}" \
            --seed 21 --lr 2e-5 --epochs 3 --max-len 384 \
            || { echo "STAGE1 FAILED fold ${f}"; FAIL=1; continue; }
    else
        echo "--- ${TAG} fold ${f} stage 1: exists ---"
    fi

    if [ ! -f "${S2}/model.pt" ]; then
        echo ""
        echo "=== ${TAG} fold ${f} STAGE 2 (balanced, deferred, low LR) ==="
        # --full-enum gives EVERY label all 1635 posts instead of 4x its
        # positives; --loss bce_perlabel supplies the per-label pos_weight that
        # full enumeration then needs (the docstring recommends exactly this
        # pairing). lr 5e-6 and 1 epoch keep the representation intact -- the
        # decoupling result is that the REPRESENTATION must not be retrained
        # under balancing, only the decision surface.
        python src/train_factors_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --model microsoft/deberta-v3-large \
            --outdir "${S2}" --init-from "${S1}" \
            --full-enum --loss bce_perlabel \
            --seed 21 --lr 5e-6 --epochs 1 --max-len 384 \
            || { echo "STAGE2 FAILED fold ${f}"; FAIL=1; continue; }
    else
        echo "--- ${TAG} fold ${f} stage 2: exists ---"
    fi

    DST="${OUT}/${TAG}_cv_oof_f${f}.npz"
    if [ ! -f "${DST}" ]; then
        python src/predict_factors_nli.py \
            --test "artifacts/folds/val_f${f}.csv" \
            --nli-run "${S2}" --dump-probs "${DST}" \
            --out "${OUT}/${TAG}_cv_oof_f${f}_labels.csv" \
            || { echo "OOF dump failed fold ${f}"; FAIL=1; }
    fi
done

DST="${OUT}/${TAG}_cv_test.npz"
if [ ! -f "${DST}" ]; then
    R=""
    for f in 0 1 2 3 4; do R="${R}${R:+,}runs/factors_${TAG}_cv_f${f}"; done
    echo ""
    echo "=== ${TAG} test dump (5-fold bagged, matching how BAG8 test is built) ==="
    python src/predict_factors_nli.py --test data/test.csv --nli-run "${R}" \
        --dump-probs "${DST}" --out "${OUT}/${TAG}_cv_test_labels.csv" \
        || { echo "test dump failed ${TAG}"; FAIL=1; }
fi

echo ""
echo "################################################################"
echo "DONE (FAIL=${FAIL}).  Next, all CPU:"
echo "################################################################"
echo "  # 1. did the two-stage buy the tail WITHOUT selling the head?"
echo "  python src/factor_arm_ap_protective.py --arms mnli dmnli cmine prot d4 qnli mnli7 lnli fenum ltail --all-labels"
echo "  #    falsifier: ltail must beat fenum on hopelessness/mental-health"
echo "  #    AND beat BAG8's best on substance-use / meaning-in-life."
echo "  #    If it only does one, the two-stage did not work."
echo ""
echo "  # 2. does it help the pool? (ltail + the 4 new arms are now shippable)"
echo "  python src/factor_member_forward.py"
echo "  python src/factor_member_forward.py --rounds 3"
echo ""
echo "  # 3. only if something clears >=90% win with a real LOSS rate:"
echo "  #    build the test factor column and patch it, then S1 must return"
echo "  #    EXACTLY 0.8061 -- only S2 may move."

#!/bin/bash
#SBATCH --job-name=redumpm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# IS THE FIRING-RATIO SPREAD CONSTRUCTION DRIFT, OR A REAL PREVALENCE SHIFT?
#
# `factor_matched_dump_probe.py` fits each arm's thresholds on its own OOF and
# applies them to its test dump. The resulting firing ratios are not uniform
# across arms: some land at ~1.00, others at 0.91-0.99.
#
# TWO EXPLANATIONS FIT THAT SPREAD AND THE RATIO CANNOT SEPARATE THEM:
#   (a) construction drift -- a test dump was not built the way its OOF was, so
#       OOF-fitted thresholds sit in the wrong place; or
#   (b) a genuine prevalence difference between the labelled pool and the test
#       set, to which arms respond DIFFERENTLY depending on how much
#       probability mass sits near their thresholds. CB-NTR arms are trained to
#       reshape exactly that tail, so their ratios reading ~1.00 is equally
#       consistent with (b).
#
# Where two dumps of the SAME model on the SAME test set exist, the comparison
# is internally controlled and settles itself: the dumps must agree, and if they
# do not, one is mis-built. Where only ONE dump exists, the ratio alone carries
# no information about which explanation holds.
#
# THIS JOB SUPPLIES THE MISSING CONTROL. It re-dumps the single-dump arms
# through the verified inference path and compares against what is on disk:
#
#   byte-identical    -> the existing dumps are correctly constructed, the
#                        spread is (b), and there is no pool-wide fix.
#   they differ       -> (a) is real and the drift is pool-wide.
#
# Output goes to {arm}_matched_test.npz, a path resolve_test does not reach, so
# running this cannot alter any existing prediction.
# ===========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

set -u
ARMS="${ARMS:-cmine prot d4}"
DIR=artifacts/factor_runs
TEST="${TEST:-data/test.csv}"
FAIL=0

echo "node $(hostname)  $(date)"
echo "arms: ${ARMS}"
df -h . | tail -1

for arm in ${ARMS}; do
    echo ""
    echo "################################################################"
    echo "ARM ${arm}"
    echo "################################################################"

    RUNS=""; NR=0
    for f in 0 1 2 3 4; do
        d="runs/factors_${arm}_cv_f${f}"
        if [ -f "${d}/model.pt" ]; then
            RUNS="${RUNS:+${RUNS},}${d}"; NR=$((NR + 1))
        else
            echo "  MISSING ${d}/model.pt"
        fi
    done
    if [ "${NR}" -ne 5 ]; then
        echo "  ${NR}/5 checkpoints -- REFUSING (a partial bag is a different model)"
        FAIL=$((FAIL + 1)); continue
    fi

    # PROTECTIVE_TEMPLATE is read at INFERENCE and changes the hypothesis
    # string. `prot` was TRAINED with it ON; setting it wrongly asks the model
    # a different question than it was trained on and yields plausible garbage.
    unset FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON CONFUSABLE_JSON FACTOR_SUBSET || true
    case "${arm}" in
        prot|pairprot|mnlip|dmnlip)
            export PROTECTIVE_TEMPLATE=1
            echo "  PROTECTIVE_TEMPLATE=1 (trained with it ON)" ;;
        *)
            unset PROTECTIVE_TEMPLATE || true
            echo "  PROTECTIVE_TEMPLATE unset (trained with it OFF)" ;;
    esac

    # ---- GATE: does the inference path reproduce this arm's STORED OOF? -----
    # Fold 0 only, scored on its own held-out fold. If this does not come back
    # at ~0, the inference environment differs from training and the test dump
    # below would be meaningless.
    echo "  [gate] re-scoring fold 0 against its stored OOF"
    python src/predict_factors_nli.py \
        --test "artifacts/folds/val_f0.csv" \
        --nli-run "runs/factors_${arm}_cv_f0" \
        --dump-probs "${DIR}/_envcheck_${arm}.npz" \
        --out /dev/null \
        || { echo "  gate dump FAILED"; FAIL=$((FAIL + 1)); continue; }

    ARM="${arm}" DIR="${DIR}" python - <<'PY'
import os, sys
import numpy as np
a, d = os.environ["ARM"], os.environ["DIR"]
new = np.load(f"{d}/_envcheck_{a}.npz", allow_pickle=True)
old = np.load(f"{d}/{a}_cv_oof_f0.npz", allow_pickle=True)
ni = [str(x) for x in new["row_id"]]
oi = [str(x) for x in old["row_id"]]
P, Q = new["probs"].astype(float), old["probs"].astype(float)
if ni != oi:
    o = {x: i for i, x in enumerate(ni)}
    if sorted(ni) != sorted(oi):
        print(f"  GATE FAIL {a}: different row sets"); sys.exit(1)
    P = P[[o[x] for x in oi]]
md = float(np.abs(P - Q).max())
print(f"  gate {a}: max |re-dump - stored OOF| = {md:.6f}")
sys.exit(0 if md < 0.01 else 1)
PY
    if [ $? -ne 0 ]; then
        echo "  GATE FAILED -- inference path does not reproduce training."
        echo "  NOT dumping test for ${arm}."
        FAIL=$((FAIL + 1)); continue
    fi

    # ---- the matched test dump ---------------------------------------------
    echo "  [dump] 5-fold bagged test -> ${DIR}/${arm}_matched_test.npz"
    python src/predict_factors_nli.py \
        --test "${TEST}" \
        --nli-run "${RUNS}" \
        --dump-probs "${DIR}/${arm}_matched_test.npz" \
        --out "${DIR}/${arm}_matched_test_labels.csv" \
        || { echo "  test dump FAILED"; FAIL=$((FAIL + 1)); continue; }

    # ---- THE ACTUAL QUESTION ------------------------------------------------
    ARM="${arm}" DIR="${DIR}" python - <<'PY'
import os
import numpy as np
a, d = os.environ["ARM"], os.environ["DIR"]
new = np.load(f"{d}/{a}_matched_test.npz", allow_pickle=True)
oldp = f"{d}/{a}_cv_test.npz"
if not os.path.exists(oldp):
    print(f"  {a}: no existing {a}_cv_test.npz to compare against")
    raise SystemExit(0)
old = np.load(oldp, allow_pickle=True)
ni = [str(x) for x in new["row_id"]]
oi = [str(x) for x in old["row_id"]]
P, Q = new["probs"].astype(float), old["probs"].astype(float)
if ni != oi:
    o = {x: i for i, x in enumerate(ni)}
    P = P[[o[x] for x in oi]]
md = float(np.abs(P - Q).max())
mn = float(np.abs(P - Q).mean())
print(f"  VERDICT {a}: max |matched - existing| = {md:.6f}   "
      f"mean {mn:.6f}")
if md < 1e-6:
    print(f"    -> IDENTICAL. The existing dump was already correctly built.")
    print(f"       The firing ratio for {a} reflects a real prevalence")
    print(f"       difference, NOT construction drift. No fix available here.")
else:
    print(f"    -> THEY DIFFER. Construction drift is real for {a} and the")
    print(f"       problem is pool-wide, not confined to mnli/dmnli.")
PY
    df -h . | tail -1
done

unset PROTECTIVE_TEMPLATE || true
rm -f "${DIR}"/_envcheck_*.npz

echo ""
echo "################################################################"
echo "DONE (FAIL=${FAIL})."
echo "################################################################"
cat <<'NEXT'

Re-run the probe with the new dumps visible:
  python src/factor_matched_dump_probe.py --arms cmine prot d4

READ IT LIKE THIS:
  * every VERDICT says IDENTICAL -> construction is fine pool-wide. The
    ratio spread is prevalence sensitivity, and the only dumps that need
    replacing are the ones with a direct internal control (two dumps of one
    model on one test set).
  * any VERDICT says THEY DIFFER -> drift extends beyond the internally
    controlled arms. Replace the whole matched set at once rather than one
    arm at a time: per-arm increments are individually too small to
    attribute, so a patch cannot be verified.

Either way this is a TEST-SIDE-ONLY change: the OOF files are untouched, so
factor_member_forward cannot see it and no offline metric moves. Change one
thing at a time and hold S1 fixed at 0.8061 while doing it.
NEXT

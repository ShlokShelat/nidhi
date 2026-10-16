#!/bin/bash
#SBATCH --job-name=facdump
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Per-arm TEST probability dumps, so a multi-arm factor bag can be assembled
# and re-assembled on CPU without touching a GPU again.
#
# WHY. `src/factor_inventory.py` resolved only 4 of 6 arms: `cmine` and
# `incmnli` have no test dump, and both appear in most of the strong subsets.
# This is pure INFERENCE -- the fold checkpoints already exist. 378 posts x 24
# factors x 5 models per arm, a few minutes each.
#
# WHY PER-ARM AND NOT ONE BIG AVERAGE. predict_factors_nli.py averages every
# run dir it is given. Averaging all 5xN checkpoints at once equals a mean over
# arms only when every arm contributes the same number of checkpoints, and it
# forecloses logit- or median-averaging entirely. Separate dumps keep every
# combination rule available on CPU afterwards.
#
# ===================== THE ENV VAR THAT SILENTLY CORRUPTS ====================
# PROTECTIVE_TEMPLATE is read at INFERENCE and changes the hypothesis string.
# The `prot` arm was TRAINED with it on; every MNLI arm was trained with it off.
# Setting it wrongly asks the model a different question than it was trained on
# and produces plausible-looking garbage -- this exact class of mismatch is why
# `pairprot` was rejected. Each arm below therefore sets its own environment
# explicitly rather than inheriting whatever is exported.
#
# CONFUSABLE_JSON affects hard-negative SAMPLING during training only, so it is
# irrelevant at inference and stays unset.
#
# Usage:
#   sbatch scripts/factor_arm_test_dumps.sh                 # cmine + incmnli
#   sbatch --export=ALL,ARMS="cmine incmnli d6" scripts/factor_arm_test_dumps.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ARMS="${ARMS:-cmine incmnli}"
TEST="${TEST:-data/test.csv}"
OUT=artifacts/factor_runs
mkdir -p "${OUT}" logs

# CAPTURED ONCE, BEFORE THE LOOP. The per-arm env reset below unsets
# FACTOR_ASPECTS_JSON so one arm cannot leak its hypotheses into the next --
# but that would also destroy any value the CALLER passed, so
# `--export=ALL,FACTOR_ASPECTS_JSON=...` would be silently discarded and the
# `asp*` case would fall back to artifacts/aspects.json regardless. That
# produces plausible-looking output with hypotheses DIFFERENT from the ones an
# arm was trained on -- the exact train/serve skew this script's header warns
# about, with no error raised. Same pattern as CONFUSABLE_JSON_TMPL in
# factors_encoder.sh.
ASPECTS_TMPL="${FACTOR_ASPECTS_JSON:-}"

[ -f "${TEST}" ] || { echo "ERROR: ${TEST} not found"; exit 1; }

echo "arms : ${ARMS}"
echo "test : ${TEST}"
echo ""

FAIL=0
for arm in ${ARMS}; do
    dst="${OUT}/${arm}_cv_test.npz"
    if [ -f "${dst}" ]; then
        echo "=== ${arm}: ${dst} exists, skipping ==="
        continue
    fi

    # ---------------------------------------------------- checkpoint discovery
    RUNS=""
    NR=0
    for f in 0 1 2 3 4; do
        d="runs/factors_${arm}_cv_f${f}"
        if [ -f "${d}/model.pt" ]; then
            RUNS="${RUNS:+${RUNS},}${d}"
            NR=$((NR + 1))
        else
            echo "  MISSING ${d}/model.pt"
        fi
    done
    if [ "${NR}" -ne 5 ]; then
        echo "=== ${arm}: found ${NR}/5 checkpoints, REFUSING ==="
        echo "    A partial bag is a DIFFERENT model from the one measured."
        FAIL=$((FAIL + 1))
        continue
    fi

    # ------------------------------------------------- per-arm inference env
    unset FACTOR_DEFS_JSON  || true
    unset CONFUSABLE_JSON   || true
    unset FACTOR_ASPECTS_JSON || true
    unset FACTOR_SUBSET     || true
    case "${arm}" in
        prot|pairprot|mnlip|dmnlip)
            export PROTECTIVE_TEMPLATE=1
            echo "=== ${arm}: PROTECTIVE_TEMPLATE=1 (trained with it ON) ===" ;;
        asp*|*aspect*)
            # Aspects are applied in score_all, which MAX-POOLS over a factor's
            # several hypotheses. That is an INFERENCE-TIME expansion, so the
            # variable must be set here or the model is asked one abstract
            # question when its OOF probabilities came from several concrete
            # ones -- the same silent train/serve skew as PROTECTIVE_TEMPLATE,
            # and it would make the test dump incomparable with the OOF dump
            # the bag weights were measured on.
            export FACTOR_ASPECTS_JSON="${ASPECTS_TMPL:-artifacts/aspects.json}"
            [ -f "${FACTOR_ASPECTS_JSON}" ] || {
                echo "ERROR: ${FACTOR_ASPECTS_JSON} missing"; FAIL=$((FAIL+1)); continue; }
            unset PROTECTIVE_TEMPLATE || true
            echo "=== ${arm}: FACTOR_ASPECTS_JSON=${FACTOR_ASPECTS_JSON} (max-pooled at inference) ===" ;;
        *)
            unset PROTECTIVE_TEMPLATE || true
            echo "=== ${arm}: PROTECTIVE_TEMPLATE unset (trained with it OFF) ===" ;;
    esac

    echo "    ${NR} checkpoints -> ${dst}"
    python src/predict_factors_nli.py \
        --test "${TEST}" \
        --nli-run "${RUNS}" \
        --dump-probs "${dst}" \
        --out "${OUT}/${arm}_cv_test_labels.csv" \
        || { echo "    FAILED on ${arm}"; FAIL=$((FAIL + 1)); continue; }

    # ---------------------------------------------------------- sanity check
    ARM="${arm}" python - <<'PY'
import os
import numpy as np
arm = os.environ["ARM"]
t = np.load(f"artifacts/factor_runs/{arm}_cv_test.npz", allow_pickle=True)
o = f"artifacts/factor_runs/{arm}_cv_oof_f0.npz"
st = float(t["probs"].std())
msg = f"    test {t['probs'].shape} sd={st:.4f}"
if os.path.exists(o):
    so = float(np.load(o, allow_pickle=True)["probs"].std())
    msg += f"   oof sd={so:.4f}   ratio={st/so:.3f}"
    if st < 0.80 * so:
        msg += "   <-- STRONGLY NARROWER, use a quantile cutoff only"
print(msg)
PY
done

echo ""
echo "================================================================"
if [ "${FAIL}" -gt 0 ]; then
    echo "DONE with ${FAIL} failure(s) -- do NOT bag a partial arm set."
else
    echo "DONE. All requested arms dumped."
fi
echo "Next, on CPU (note: incmnli has NO cv checkpoints and cannot be built,"
echo "and it is the non-MNLI incumbent that dilutes the pair -- leave it out):"
echo "  python src/factor_bag_ship.py --arms mnli dmnli cmine prot d4 --rule logit \\"
echo "      --match-counts submissions/NIDHI_canon_on_nlimnli.csv"
echo "Nothing has been submitted. This writes a FACTOR COLUMN only."
echo "================================================================"

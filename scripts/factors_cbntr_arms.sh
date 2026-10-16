#!/bin/bash
#SBATCH --job-name=cbnarms
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# ADDITIONAL CB-NTR FACTOR ARMS, BUILT AS A CHEAP STAGE 2.
#
# WHY A SECOND STAGE RATHER THAN A FRESH RUN. The decoupling result for
# long-tailed recognition says the representation should be learned on the
# imbalanced distribution and only the classifier re-balanced afterwards. Every
# existing pool member already holds five fold checkpoints trained exactly that
# way, so a 1-epoch CB-NTR pass warm-started from each one is a complete stage 2
# at a fifth of the cost of retraining, and yields a NEW arm per base.
#
# WHY SEVERAL BASES RATHER THAN A KAPPA/BETA SWEEP. The value of an added arm to
# a bagged pool is decorrelation, and decorrelation scales with the number of
# distinct bases, not with the fineness of a hyperparameter grid on one base.
#
# BASES ARE CHOSEN, NOT SWEPT. lnli, mnli7 and qnli are the strongest
# non-template members and each is best-arm on a different set of labels, so
# they are the three least redundant bases available. The template-trained arms
# (prot, pairprot, mnlip, dmnlip, cminep, mnlifep) are DELIBERATELY EXCLUDED:
# they need PROTECTIVE_TEMPLATE=1, which is read at inference as well as
# training, and a wrong setting silently asks the model a question it never saw.
#
# THE MODEL NAME COMES FROM THE BASE, NOT FROM A FLAG. Each member was trained
# on its own encoder, and --init-from checks the state dict against the
# architecture. `runs/<dir>/tok` records the base model; this reads it rather
# than guessing, because a mismatch there is a hard failure five folds deep.
#
# NOTHING SHIPS FROM HERE. Each arm gets OOF plus a 5-fold-bagged test dump,
# both built the same way, and `factor_member_forward.py` decides. Building the
# two artefacts identically is what makes the offline delta comparable to the
# deployed one.
# ===========================================================================

source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

# sbatch defaults to --export=ALL, so a dirty login shell would otherwise
# train and score on hypotheses these arms never saw, silently.
unset FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON PROTECTIVE_TEMPLATE \
      CONFUSABLE_JSON FACTOR_SUBSET 2>/dev/null || true

set -u
BASES="${BASES:-lnli mnli7 qnli}"
LOSSARGS="--loss cbntr --ntr-kappa 0.05 --ntr-lambda 2.0 --cb-beta 0.99 --db-gamma 2.0"
OUT=artifacts/factor_runs
mkdir -p "$OUT" logs

echo "node $(hostname)  $(date)"
echo "bases: ${BASES}"
echo "loss : ${LOSSARGS}"
df -h . | tail -1

[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }

FAIL=0
for BASE in ${BASES}; do
    TAG="cbn${BASE}"
    echo ""
    echo "################################################################"
    echo "ARM ${TAG}  (CB-NTR stage 2 from ${BASE})"
    echo "################################################################"

    # TAG-collision guard: a dir written here carries this marker.
    SKIP=0
    for d in "runs/factors_${TAG}_cv_f"*; do
        [ -d "$d" ] || continue
        grep -q '"cbntr_stage2"' "${d}/run_config.json" 2>/dev/null || {
            echo "ERROR: ${d} exists and was not written by this script."
            echo "Another experiment owns TAG=${TAG}. Skipping this arm."
            SKIP=1; break; }
    done
    [ $SKIP -eq 0 ] || { FAIL=1; continue; }

    for f in 0 1 2 3 4; do
        INIT="runs/factors_${BASE}_cv_f${f}"
        DST="runs/factors_${TAG}_cv_f${f}"
        if [ ! -f "${INIT}/model.pt" ] || [ ! -f "${INIT}/tok" ]; then
            echo "  ${BASE} fold ${f}: missing model.pt or tok, skipping arm"
            FAIL=1; SKIP=1; break
        fi
        MODEL=$(cat "${INIT}/tok")

        if [ ! -f "${DST}/model.pt" ]; then
            echo ""
            echo "=== ${TAG} fold ${f}: from ${INIT}  (${MODEL}) ==="
            python src/train_factors_nli.py \
                --train "artifacts/folds/train_f${f}.csv" \
                --val   "artifacts/folds/val_f${f}.csv" \
                --model "${MODEL}" \
                --outdir "${DST}" --init-from "${INIT}" \
                --full-enum ${LOSSARGS} \
                --seed 21 --lr 5e-6 --epochs 1 --max-len 384 \
                || { echo "FAILED ${TAG} fold ${f}"; FAIL=1; continue; }
            MODEL="${MODEL}" BASE="${BASE}" python - "${DST}" <<'PY'
import json, os, sys
d = sys.argv[1]
if os.path.isdir(d):
    json.dump({"cbntr_stage2": True, "init_from_arm": os.environ.get("BASE"),
               "model": os.environ.get("MODEL"), "max_len": 384,
               "dump_max_len": 320, "seed": 21, "lr": 5e-6, "epochs": 1,
               "loss": "cbntr", "cb_beta": 0.99, "ntr_kappa": 0.05,
               "ntr_lambda": 2.0, "db_gamma": 2.0,
               "FACTOR_DEFS_JSON": os.environ.get("FACTOR_DEFS_JSON", ""),
               "FACTOR_ASPECTS_JSON": os.environ.get("FACTOR_ASPECTS_JSON", ""),
               "PROTECTIVE_TEMPLATE": os.environ.get("PROTECTIVE_TEMPLATE", "")},
              open(os.path.join(d, "run_config.json"), "w"), indent=2)
PY
        else
            echo "--- ${TAG} fold ${f}: exists ---"
        fi

        # --max-len 320 matches every other arm's dump, so the only new
        # variable is the loss.
        O="${OUT}/${TAG}_cv_oof_f${f}.npz"
        if [ ! -f "${O}" ]; then
            python src/predict_factors_nli.py \
                --test "artifacts/folds/val_f${f}.csv" --nli-run "${DST}" \
                --max-len 320 --dump-probs "${O}" \
                --out "${OUT}/${TAG}_cv_oof_f${f}_labels.csv" \
                || { echo "OOF dump failed ${TAG} fold ${f}"; FAIL=1; }
        fi
    done
    [ $SKIP -eq 0 ] || continue

    DST="${OUT}/${TAG}_cv_test.npz"
    if [ ! -f "${DST}" ]; then
        R=""; OK=1
        for f in 0 1 2 3 4; do
            d="runs/factors_${TAG}_cv_f${f}"
            [ -f "${d}/model.pt" ] || { echo "  ${d} missing, no test dump"; OK=0; break; }
            R="${R}${R:+,}${d}"
        done
        if [ $OK -eq 1 ]; then
            echo ""
            echo "=== ${TAG} test dump (5-fold bagged) ==="
            python src/predict_factors_nli.py --test data/test.csv \
                --nli-run "${R}" --max-len 320 --dump-probs "${DST}" \
                --out "${OUT}/${TAG}_cv_test_labels.csv" \
                || { echo "test dump failed ${TAG}"; FAIL=1; }
        fi
    fi
    df -h . | tail -1
done

echo ""
echo "################################################################"
echo "DONE (FAIL=${FAIL}).  Next, CPU:"
echo "################################################################"
cat <<'NEXT'
  python src/factor_member_forward.py 2>&1 | tail -30
  # Judge each new arm as an ADD on top of the pool that is actually deployed.
  # Measuring against an older pool double-counts the contribution of members
  # already in it.

  # Then rebuild against the deployed configuration:
  #   --use tfit                                    the deployed decode
  #   --shipped artifacts/factors_ltail9_test.csv   the CURRENT factor column
  # and prove the rebuild reproduces the incumbent column cell for cell with
  # the current pool before trusting any delta. Without that reproduction a
  # measured delta mixes the new arm with an unintended pipeline change.
NEXT

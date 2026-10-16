#!/bin/bash
#SBATCH --job-name=prot5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# A FIVE-FOLD PROTECTIVE-FACTOR SPECIALIST.
#
# WHY FIVE FOLDS AND NOT THE SINGLE-SPLIT VARIANT IN
# scripts/factors_protective_specialist.sh. That script trains on the single
# train/val split, so the specialist has no out-of-fold predictions -- and
# without OOF its thresholds cannot be fitted on held-out data, which means it
# can never enter the thresholded pipeline at all. Five author-disjoint folds
# fix exactly that: the arm gets `{TAG}_cv_oof_f*.npz` plus a bagged
# `{TAG}_cv_test.npz`, the two artefacts a first-class pool member needs.
#
# THE TARGET, AND ITS ARITHMETIC. Held-out F1 is far lower on the five
# protective labels than on the 19 risk labels, and the polar pairs are the
# tell: each protective label scores well below its risk-side opposite, even
# though the two are decided from the same sentences. The worst of them is
# low in precision AND recall at once, which is a DISCRIMINATION failure, not
# a calibration one -- thresholds are already at their held-out optimum.
#
# Because the score is a 24-label macro, lifting these five labels' F1 moves
# the macro average and, through the 0.3 factor weight, the composite --
# even a modest per-label gain here is amplified by the averaging. That is
# the largest single identified deficit in the component.
#
# THE MECHANISM. Every post in this corpus is distressed, so the encoder
# learns a negative prior for the domain. The cross-encoder shares EVERY
# parameter across all 24 factors, so the only thing telling it a question is
# protective is the hypothesis wording -- which has to fight that prior. A
# specialist never switches polarity: its entire training signal is positive
# content. The organizers do the same thing in arXiv 2507.10008, using separate
# MLPs for risk and protective factors because they "operate independently
# rather than being the reverse of risk factors".
#
# WHY --full-enum IS CHEAP HERE. With 5 factors it is 5 pairs per post instead
# of 24, so exhaustive coverage costs a QUARTER of a normal epoch rather than
# 2.1x. That removes the sampled scheme's coverage imbalance for free. Full
# enumeration invalidates plain bce's pos_weight=3.0 (derived from a sampling
# ratio that no longer applies), so the objective is bce_perlabel.
#
# THIS IS THE CONTROL ARM FOR A HYPOTHESIS REWRITE. It runs with the CURRENT
# hypotheses, deliberately. The constructs here -- hope, optimism, resilience,
# self-efficacy, meaning in life -- are famously correlated in the psychometric
# literature, and their discriminating features are specific (hope = agency
# PLUS pathways; optimism = outcome expectancy WITHOUT a route; self-efficacy =
# task-specific capability; resilience = recovery AFTER a setback). If the
# hypotheses do not encode those distinctions, no amount of capacity separates
# them. A second arm with rewritten hypotheses, passed via FACTOR_DEFS_JSON /
# FACTOR_ASPECTS_JSON, is then a clean single-variable comparison against this.
# Running the control first is the point, not an accident of ordering.
#
# HOW TO JUDGE IT -- on the FIVE, never on the macro:
#   python src/factor_arm_ap_protective.py --arms mnli dmnli cmine prot d4 \
#       qnli mnli7 lnli prot5 --all-labels
# and treat `sexual orientation` (8 positives), `exposure to others' suicide`
# (14) and `poor school performance` (16) as BLIND -- under author-disjoint
# folds they land 0-6 positives per fold, so one post moves a label's F1 by
# 0.2-0.3. None of those three is protective, so the read here is clean.
# ===========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

# Clear the inherited hypothesis environment first -- sbatch defaults to
# --export=ALL, so a dirty login shell would otherwise train and score on
# hypotheses the arm never saw, silently. FACTOR_SUBSET is then set on purpose.
unset FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON PROTECTIVE_TEMPLATE \
      CONFUSABLE_JSON FACTOR_SUBSET 2>/dev/null || true

set -u
export FACTOR_SUBSET=protective

# HYPOTHESIS ARMS. Unset (the default) is the CONTROL: the published
# definitions, no aspects -- which is what TAG=prot5 trained on. Point DEFS or
# ASPECTS at a JSON file to make one clean single-variable change against it.
#
#   DEFS=artifacts/defs_protective_v3.json      -> rewritten definitions
#   ASPECTS=artifacts/aspects_protective_v3.json -> pooled aspect hypotheses
#
# PRECEDENCE MATTERS AND IS EASY TO GET WRONG: hypotheses_for() returns the
# ASPECTS if a factor has any, and only otherwise the templated definition. So
# for any factor present in the aspects file the definition is DEAD TEXT.
# Setting both and then crediting the definitions would be reporting on a
# string the model never saw. Refuse it.
if [ -n "${DEFS:-}" ] && [ -n "${ASPECTS:-}" ]; then
    echo "ERROR: DEFS and ASPECTS both set. For the five protective factors"
    echo "the aspects override the definitions entirely, so the DEFS arm would"
    echo "be unmeasurable. Run them as separate arms."; exit 1
fi
if [ -n "${DEFS:-}" ]; then
    [ -f "${DEFS}" ] || { echo "ERROR: ${DEFS} missing"; exit 1; }
    export FACTOR_DEFS_JSON="${DEFS}"
    echo "hypotheses: DEFS arm -> ${DEFS}"
elif [ -n "${ASPECTS:-}" ]; then
    [ -f "${ASPECTS}" ] || { echo "ERROR: ${ASPECTS} missing"; exit 1; }
    export FACTOR_ASPECTS_JSON="${ASPECTS}"
    echo "hypotheses: ASPECTS arm -> ${ASPECTS}"
else
    echo "hypotheses: CONTROL (published definitions, no aspects)"
fi

# Print what the model will actually be asked, for all five. A hypothesis arm
# whose strings are not in the log cannot be audited afterwards, and the
# failure this guards against is measuring one artefact and shipping another.
python - <<'PY'
import os, sys
sys.path.insert(0, "src")
from train_factors_nli import PROTECTIVE_FACTORS, hypotheses_for
n = []
for f in PROTECTIVE_FACTORS:
    hs = hypotheses_for(f)
    print(f"  {f}  ({len(hs)} hypothesis/es)")
    for h in hs:
        n.append(len(h.split()))
        print(f"      {len(h.split()):>3}w  {h}")
print(f"  word count across all: min {min(n)} max {max(n)} "
      f"-- keep these close; unbalanced length biases the entailment score")
PY

TAG="${TAG:-prot5}"
BASE="${BASE:-mnli}"          # fold-matched warm start, one per fold
MODEL="${MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-13}"
OUT=artifacts/factor_runs
mkdir -p "$OUT" logs

echo "node $(hostname)  $(date)"
echo "tag ${TAG}  subset ${FACTOR_SUBSET}  base ${BASE}  epochs ${EPOCHS}"
df -h . | tail -1

[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }

# TAG-COLLISION GUARD. Reusing a TAG another run owns fails silently: the
# skip-if-exists guards below would quietly adopt that run's checkpoints and
# report them as this arm. A dir written by THIS script carries run_config.json
# with a "specialist" key; anything else stops the job.
for d in "runs/factors_${TAG}_cv_f"*; do
    [ -d "$d" ] || continue
    grep -q '"specialist"' "${d}/run_config.json" 2>/dev/null || {
        echo "ERROR: ${d} exists and was not written by this script."
        echo "Another experiment owns TAG=${TAG}. Rename or clear it."; exit 1; }
done

FAIL=0
for f in 0 1 2 3 4; do
    INIT="runs/factors_${BASE}_cv_f${f}"
    DST="runs/factors_${TAG}_cv_f${f}"

    if [ ! -f "${DST}/model.pt" ]; then
        [ -f "${INIT}/model.pt" ] || {
            echo "ERROR: ${INIT}/model.pt missing -- the specialist warm-starts"
            echo "from the FOLD-MATCHED 24-factor member, so fold ${f} cannot"
            echo "run. Five factors is far too thin to learn a representation."
            FAIL=1; continue; }
        echo ""
        echo "=== fold ${f}: specialist from ${INIT} ==="
        python src/train_factors_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --model "${MODEL}" --outdir "${DST}" --init-from "${INIT}" \
            --full-enum --loss bce_perlabel \
            --epochs "${EPOCHS}" --seed "${SEED}" --max-len 384 \
            || { echo "FAILED fold ${f}"; FAIL=1; continue; }
        python - "${DST}" "${BASE}" <<'PY'
import json, os, sys
d, base = sys.argv[1], sys.argv[2]
if os.path.isdir(d):
    json.dump({"specialist": "protective", "init_from_arm": base,
               "loss": "bce_perlabel", "full_enum": True,
               "model": os.environ.get("MODEL", ""), "max_len": 384,
               "dump_max_len": 320,
               "FACTOR_SUBSET": os.environ.get("FACTOR_SUBSET", ""),
               "FACTOR_DEFS_JSON": os.environ.get("FACTOR_DEFS_JSON", ""),
               "FACTOR_ASPECTS_JSON": os.environ.get("FACTOR_ASPECTS_JSON", ""),
               "PROTECTIVE_TEMPLATE": os.environ.get("PROTECTIVE_TEMPLATE", "")},
              open(os.path.join(d, "run_config.json"), "w"), indent=2)
PY
    else
        echo "--- fold ${f}: exists ---"
    fi

    # --max-len 320 matches every other arm's dump. The 19 non-protective
    # columns will be meaningless (the model never trained on them under
    # FACTOR_SUBSET) -- that is expected and is why this arm is judged and
    # pooled PER LABEL, on the five, never on the 24-label macro.
    O="${OUT}/${TAG}_cv_oof_f${f}.npz"
    if [ ! -f "${O}" ]; then
        python src/predict_factors_nli.py \
            --test "artifacts/folds/val_f${f}.csv" --nli-run "${DST}" \
            --max-len 320 --dump-probs "${O}" \
            --out "${OUT}/${TAG}_cv_oof_f${f}_labels.csv" \
            || { echo "OOF dump failed fold ${f}"; FAIL=1; }
    fi
    df -h . | tail -1
done

# Bagged test dump, built the same way the deployed pool's test column is, so
# the arm is usable the moment it proves itself. An arm with OOF but no matching
# test dump cannot be deployed however well it scores.
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
            || { echo "test dump failed"; FAIL=1; }
    fi
fi

echo ""
echo "################################################################"
echo "DONE (FAIL=${FAIL}).  Next, all CPU:"
echo "################################################################"
cat <<'NEXT'
  # 1. THE READ THAT MATTERS -- per label, on the five protective ones only.
  python src/factor_arm_ap_protective.py \
      --arms mnli dmnli cmine prot d4 qnli mnli7 lnli prot5 --all-labels
  # Bar: the five must each move by roughly +0.2 F1 to clear the +-0.10
  # single-label swing that a change of seed alone produces at this many
  # positives. `meaning in life` is the lowest-AP label in the taxonomy and is
  # the one to watch.

  # 2. does using it ONLY for those five beat the general pool everywhere?
  python src/factor_perlabel_pool.py --shippable-only
  # `all29` has one bit of freedom and cannot overfit the per-label search. A
  # per-label win that `all29` does not share is the search fitting the sample.

  # 3. only then rebuild the factor column with thresholds REFIT on the new
  #    pool, and verify the risk score is bit-identical -- only the factor
  #    score may move, or the change was not single-variable.
NEXT

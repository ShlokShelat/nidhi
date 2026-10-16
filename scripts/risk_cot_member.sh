#!/bin/bash
#SBATCH --job-name=cotmemb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# Score the CoT reasoner over the full OOF set and the test set so it can enter
# the risk blend as a MEMBER, not as a replacement for the encoder.
#
# WHY NOT A REPLACEMENT. A rescue/breakage rate measured on an error-enriched
# sample does not transfer to the deployment distribution. With rescue 0.473 and
# breakage 0.127 on a sample that is half encoder-errors by construction, the
# real distribution gives:
#     encoder wrong  343 x 0.473 = +162 fixed
#     encoder right 1292 x 0.127 = -164 broken
# Net ~= zero. Any override policy has to clear that arithmetic, not the rescue
# rate on its own.
#
# WHY IT IS STILL WORTH A JOB. Backing the same rates out to a representative
# sample gives the reasoner 0.21*0.473 + 0.79*0.873 ~= 0.79 accuracy; the encoder
# is also ~0.79. TWO MODELS AT EQUAL ACCURACY THAT DISAGREE ON HALF OF EACH
# OTHER'S ERRORS is the ensemble setup that actually pays, and it is rare among
# the existing risk members, whose confident errors concentrate on the same posts
# (38 of 39 with neither head right).
#
# TWO DESIGN CHOICES:
#  1. --soft. Reads the model's log-prob over each of the four level strings
#     after its own reasoning, instead of parsing generated text. A blend needs a
#     distribution rather than a hard label, and text parsing fails open: ~10% of
#     greedy generations are unparseable and fall back to `indicator`, which both
#     biases the arm and destroys the information a blend would use.
#  2. ONE MODEL LOAD for both OOF and test. Loading the 4-bit 72B costs ~1.5h
#     (37 shards), more than the scoring itself, so a separate test job would
#     roughly double wall time. --extra-csv scores data/test.csv in the same
#     process.
#
# THE DIAGNOSTIC THAT DECIDES THE NEXT STEP is printed by --soft: mean confidence
# when RIGHT vs when WRONG. A gap means confidence can gate the blend, so the
# encoder is overruled only where the reasoner is sure. No gap means this model
# is confidently wrong in the same way the encoder is, and no blend weight can
# separate them.
#
# Outputs:
#   artifacts/cot_soft_oof.csv            1635 OOF posts, gold + probs
#   artifacts/cot_soft_oof.csv.extra.csv  378 test posts, probs
# Neither is a submission. The blend weight must then be fitted on OOF and
# judged over 50 author splits with arm_robustness, never on one split.
#
# Usage:
#   sbatch scripts/risk_cot_member.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BIG="${BIG:-Qwen/Qwen2.5-72B-Instruct}"
OOF="${OOF:-artifacts/nligaur/oof_base.csv}"
TEST="${TEST:-data/test.csv}"
BATCH="${BATCH:-8}"
# OUT must differ per model or a second reasoner silently overwrites the first.
# The default below is the Qwen reasoner's path (plus its .extra.csv).
OUT="${OUT:-artifacts/cot_soft_oof.csv}"
# Extra flags forwarded verbatim to risk_cot_probe.py. Without this, an
# --export=ALL,EXTRA="--samples 5" is silently DROPPED and the job re-runs
# greedy decoding under a new filename -- 8 hours for a duplicate.
#   EXTRA="--samples 5"            self-consistency (implies sampling)
#   EXTRA="--samples 5 --temperature 0.8"
EXTRA="${EXTRA:-}"

[ -d "$BIG" ]  || { echo "ERROR: $BIG missing"; exit 1; }
[ -f "$OOF" ]  || { echo "ERROR: $OOF missing"; exit 1; }
[ -f "$TEST" ] || { echo "ERROR: $TEST missing"; exit 1; }
python -c "import bitsandbytes" || { echo "ERROR: needs bitsandbytes"; exit 1; }

# --n 0 = every OOF post. The blend weight has to be fitted on the full 1635,
# and an error-enriched subsample would fit the wrong weight.
echo "model: ${BIG}"
echo "out  : ${OUT}  (+ ${OUT}.extra.csv for test)"
echo "extra: ${EXTRA:-(none)}"
if [ -f "$OUT" ]; then
    echo "ERROR: ${OUT} already exists -- set OUT= to a new path so a second"
    echo "reasoner does not overwrite the first."
    exit 1
fi

python src/risk_cot_probe.py \
    --model "$BIG" --load-4bit --batch "$BATCH" \
    --oof "$OOF" --n 0 --soft \
    --extra-csv "$TEST" \
    --out "$OUT" ${EXTRA} \
    || { echo "FAILED"; exit 1; }

echo ""
echo "================================================================"
echo "NEXT (CPU): fit and judge the blend"
echo "================================================================"
echo "The two CSVs are members, not submissions. Judge the blend over 50"
echo "author splits (src/arm_robustness.py), never one -- split-to-split"
echo "variance on this data is larger than the effect being measured, so a"
echo "single split can show either sign."
echo ""
echo "Read the CONFIDENCE GAP first. No gap means no gate, and without a gate"
echo "the net arithmetic above stays at zero however good the rescue rate is."

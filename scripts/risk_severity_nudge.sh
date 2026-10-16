#!/bin/bash
#SBATCH --job-name=sevnudge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# DIAGNOSTIC SWEEP OF THE EVIDENCE-SEVERITY NUDGE ON THE RISK COLUMN.
#
# `champion_risk_inject.py --severity <csv> --w <x>` tilts the risk logits
# toward higher severity in proportion to the evidence taggers' own opinion,
# in LOG space before the argmax, so rules and route_attempt still apply on
# top. `--w` defaults to 0.0, i.e. the nudge is off unless asked for.
#
# WHY THE SIGNAL CARRIES INFORMATION. The taggers hold a severity meter of
# their own: forced to `indicator` they emit 0.54 spans/post, forced to
# `ideation`, 1.29, on identical text. That opinion is formed without the risk
# model, so it is not a restatement of what the risk model already believes.
# It is far too weak to DECIDE a post -- used as an override it degrades the
# column -- so it is applied only as a small additive tilt, which can move
# posts the risk model is already close to indifferent about and cannot
# overturn a confident one.
#
# STAGE 1 EXISTS BECAUSE THE SEVERITY DUMP IS OOF-ONLY.
# artifacts/risk_probe.csv holds 1635 out-of-fold rows and ZERO test rows, and
# risk_tagger_probe.py is fold-based with no test path, so the test-side dump
# has to be built before the nudge can be applied to the test column.
#
# REGIME NOTE. The out-of-fold dump scores each row with a single fold model;
# the test dump below is a 5-fold bag. The two are not the same instrument, so
# judge a weight by the labels it moves and by agreement with an independent
# reader, not by any offline delta.
#
# NOTHING IS SUBMITTED HERE.
# ===========================================================================

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

CHAMP="${CHAMP:-submissions/NIDHI_catonly2.csv}"
PROBE="${PROBE:-artifacts/risk_probe_test.csv}"
RUNS="${RUNS:-runs/evcond5_cond_f{f}}"
mkdir -p logs

echo "node $(hostname)  $(date)"
echo "champion ${CHAMP}"
[ -f "${CHAMP}" ] || { echo "ERROR: ${CHAMP} missing"; exit 1; }

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STAGE 1: build the test-side severity dump"
echo "################################################################"
if [ -f "${PROBE}" ]; then
    echo "  ${PROBE} exists, reusing"
else
    python src/risk_tagger_probe_test.py --runs "${RUNS}" \
        --test data/test.csv --out "${PROBE}" \
        || { echo "PROBE FAILED"; exit 1; }
fi

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STAGE 2: sweep the nudge weight (diagnose only)"
echo "################################################################"
for W in 0.05 0.10 0.15 0.20 0.30 0.50; do
    echo ""
    echo "--------- w=${W} ---------"
    python src/champion_risk_inject.py \
        --champion "${CHAMP}" --severity "${PROBE}" --w "${W}" 2>&1 \
      | grep -E 'GATE|labels change|crossing|row_id|  P0|REFUS' | head -14
done

cat <<'NEXT'

################################################################
DONE. Nothing submitted.
################################################################
Pick the largest w that changes a handful of labels with ZERO indicator
crossings -- a crossing empties or fills an evidence cell and makes it a
two-variable change, which the inject script refuses to write anyway.

Then build it:
  python src/champion_risk_inject.py --champion <champ> \
      --severity artifacts/risk_probe_test.csv --w <best> \
      --out submissions/NIDHI_sev<W>.csv

Verify that evidence and factors differ on 0 rows and that the empty-evidence
count is unchanged, so the risk column is the only thing that moved.

Read the LLM column as one more opinion on the moved posts, not as a veto: it
is a single unvalidated reader, and a handful of posts cannot establish that it
is right where the nudge is wrong.
NEXT

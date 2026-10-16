#!/bin/bash
#SBATCH --job-name=evcondep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# Deploy risk-conditioned tagging to the champion's eight evidence taggers.
#
# WHY. A single-tagger screening run -- 5 folds, 1635 OOF posts, official
# MACRO average with the shipped suppression rule, each arm at its OWN best
# threshold -- showed conditioning on the PREDICTED risk level as a clear
# gain over the unconditioned base, and conditioning on GOLD risk level as a
# much larger but unreachable ceiling. Precision is what moves, not recall:
# conditioning does not find more spans, it stops the tagger grabbing text
# that argues for the wrong risk level.
#
# THAT WAS ONE TAGGER PER FOLD. The champion runs EIGHT with cluster voting, so
# the screening delta is not the shipped delta -- it could go either way. Voting
# already removes some of the errors conditioning removes (both suppress spans
# the ensemble disagrees about), so the gains may not add. This job measures the
# real thing.
#
# THREE STAGES:
#   1. retrain all 8 with --risk-cond, REPLAYING each run's own hyperparameters
#      from its run_config.json so the arm differs in exactly one variable
#   2. dump test+val candidates conditioned on PREDICTED levels -- the champion
#      submission for test, out-of-fold predictions for val
#   3. cluster-vote and write a submission, alongside the champion for diffing
#
# THE CHAMPION IS NEVER OVERWRITTEN. New runs go to runs/<name>_rc, candidates to
# artifacts/evidence_runs_rc, and the submission to a new file.
#
# Usage:
#   sbatch scripts/evidence_cond_deploy.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CHAMP="${CHAMP:-submissions/NIDHI_cluster.csv}"
VALRISK="${VALRISK:-artifacts/nligaur/oof_base.csv}"
OUTDIR="${OUTDIR:-artifacts/evidence_runs_rc}"
SUB="${SUB:-submissions/NIDHI_evcond.csv}"

[ -f "$CHAMP" ]   || { echo "ERROR: champion $CHAMP missing"; exit 1; }
[ -f "$VALRISK" ] || { echo "ERROR: $VALRISK missing"; exit 1; }

echo "=== alignment guard ==="
python tests/test_risk_cond_alignment.py || {
    echo "ALIGNMENT TEST FAILED -- not training"; exit 1; }

echo ""
echo "================================================================"
echo "1/3  retrain 8 taggers with --risk-cond (replaying their configs)"
echo "================================================================"
python src/retrain_evidence_cond.py --dry-run || exit 1
python src/retrain_evidence_cond.py || { echo "retrain FAILED"; exit 1; }

echo ""
echo "================================================================"
echo "2/3  dump candidates conditioned on PREDICTED risk levels"
echo "================================================================"
RC=""
for d in runs/evidence_*_rc; do
    n=$(basename "$d")
    if [ -f "$d/model_final.pt" ] || [ -f "$d/model.pt" ]; then
        RC="${RC}${RC:+,}${n}"
    fi
done
echo "conditioned runs: ${RC}"
[ -n "$RC" ] || { echo "ERROR: no _rc runs found"; exit 1; }

python src/evidence_dump_all.py --runs "$RC" \
    --test data/test.csv --val artifacts/val_split.csv \
    --outdir "$OUTDIR" \
    --risk-cond --test-risk "$CHAMP" --val-risk "$VALRISK" \
    || { echo "dump FAILED"; exit 1; }

echo ""
echo "================================================================"
echo "3/3  cluster-vote -> submission"
echo "================================================================"
# --cluster with the modal representative is the shipped configuration
# (a macro gain over exact vote>=2 on OOF, positive on both author halves).
# --risk-from points at the CHAMPION: the suppression rule keys off its risk
# column, and that column is also what the taggers were conditioned on, so the
# two stay consistent.
python src/evidence_vote_test.py \
    --runs "$RC" --dir "$OUTDIR" --k 2 --cluster --rep modal \
    --risk-from "$CHAMP" --test data/test.csv \
    --out artifacts/evcond_evidence.csv \
    || { echo "vote FAILED"; exit 1; }

python - <<PY || { echo "merge FAILED"; exit 1; }
import pandas as pd
champ = pd.read_csv("${CHAMP}", keep_default_na=False)
ev = pd.read_csv("artifacts/evcond_evidence.csv", keep_default_na=False)
champ["row_id"] = champ["row_id"].astype(str)
ev["row_id"] = ev["row_id"].astype(str)
m = dict(zip(ev["row_id"], ev["evidence"]))
absent = [r for r in champ["row_id"] if r not in m]
if absent:
    raise SystemExit(f"evidence missing for {len(absent)} rows")
out = champ.copy()
out["evidence"] = [m[r] for r in champ["row_id"]]
n_diff = int((out["evidence"] != champ["evidence"]).sum())
n_emp_c = int((champ["evidence"].astype(str).str.strip() == "").sum())
n_emp_o = int((out["evidence"].astype(str).str.strip() == "").sum())
out.to_csv("${SUB}", index=False)
print(f"wrote ${SUB}")
print(f"  {n_diff} of {len(out)} posts differ from champion")
print(f"  empty-evidence posts: champion {n_emp_c} -> new {n_emp_o}")
print("  (empty count is driven by the suppression rule and the risk column "
      "is unchanged, so a LARGE move here means something is wrong)")
PY

echo ""
echo "================================================================"
echo "DONE -- ${SUB}"
echo "================================================================"
echo "Risk and factor columns are the champion's, untouched. ONLY evidence"
echo "changed, so this is a single-variable submission."
echo ""
echo "Before uploading, diff it against the champion and sanity-check the span"
echo "count per post. The offline screening used ONE tagger per fold; the"
echo "ensemble may deliver less because cluster voting already removes some"
echo "of the same errors."

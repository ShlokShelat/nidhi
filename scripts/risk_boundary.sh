#!/bin/bash
#SBATCH --job-name=riskbnd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# Five folds of the plan-or-self-harm boundary model, then the test column.
#
# TARGET, measured on the deployed chain's 1590 out-of-fold posts:
#   behavior -> ideation is 102 errors, 30% of all 343.
#   behavior recall 0.665, ideation precision 0.684.
#   those two classes carry 0.146 of the 0.216 total weighted-F1 loss (68%);
#   `attempt` carries 0.023.
# Each such error is charged twice -- once to behavior's recall and once to
# ideation's precision -- so fixing even a fraction of them yields a
# meaningful composite gain.
#
# WHY A MODEL AND NOT A RULE. The class marginals are already close to correct,
# so moving the operating point toward behavior only trades ideation precision
# for behavior recall at no net gain. What fails on those 102 posts is the
# RANKING, which is what a dedicated model can change and a threshold cannot.
# A lexical promotion rule is no better placed: on this boundary it would need
# above 50% precision merely to break even against the errors it introduces.
#
# AFTER IT TRAINS -- judge it, do not ship it blind:
#   1. Does it beat the chain ON THE BOUNDARY? Restrict the 1590 OOF posts to
#      gold in {ideation, behavior} and compare AP for "is behavior" against
#      the chain's own P(behavior)+P(attempt) from artifacts/champion_oof.csv.
#      If it does not beat that, it has learned nothing new and the lane closes.
#   2. Only then, as a MEMBER: blend it into the chain and judge over 50 author
#      partitions against an estimator-matched baseline. Do NOT read
#      champion_oof.csv as a member-addition instrument: it is a single fixed
#      dump built from one estimator, so comparing a blend against it mixes the
#      member's effect with the change of estimator and reports the sum.
#
# PRE-STATED FALSIFIER: boundary AP must exceed the chain's by >= +0.03 on the
# ideation/behavior subset. Below that, a member blend cannot recover 25 of 102
# errors and the lane closes.
#
# Usage:
#   sbatch scripts/risk_boundary.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

MODEL="${MODEL:-microsoft/deberta-v3-large}"
EPOCHS="${EPOCHS:-4}"

[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }

RUNS=""
for f in 0 1 2 3 4; do
    d="runs/risk_bnd_f${f}"
    if [ -f "${d}/model.pt" ]; then
        echo "--- fold ${f}: exists, skipping ---"
    else
        echo ""
        echo "================================================================"
        echo "boundary fold ${f}   ${MODEL}"
        echo "================================================================"
        python src/train_risk_boundary.py --fold "${f}" --outdir "${d}" \
            --model "${MODEL}" --epochs "${EPOCHS}" \
            || { echo "FAILED fold ${f}"; exit 1; }
    fi
    RUNS="${RUNS:+${RUNS},}${d}"
done

echo ""
echo "================================================================"
echo "POOLED OOF -- does it beat the chain on the boundary?"
echo "================================================================"
python - <<'PY'
import glob
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "src")
from sklearn.metrics import average_precision_score as APS
from scoring import normalize_label

fr = [pd.read_csv(p) for p in sorted(glob.glob("runs/risk_bnd_f*/oof_boundary.csv"))]
if len(fr) < 5:
    print(f"only {len(fr)} folds present -- pooled read is incomplete")
b = pd.concat(fr, ignore_index=True)
b["row_id"] = b["row_id"].astype(str)
b = b.drop_duplicates("row_id").set_index("row_id")

ch = pd.read_csv("artifacts/champion_oof.csv")
ch["row_id"] = ch["row_id"].astype(str)
ch = ch.drop_duplicates("row_id").set_index("row_id")

rows = sorted(set(b.index) & set(ch.index))
gold = np.array([normalize_label(x) for x in ch.loc[rows, "gold"]])
# the decision that costs us: among posts whose gold is ideation or behavior,
# which are behavior?
m = np.isin(gold, ["ideation", "behavior"])
y = (gold[m] == "behavior").astype(int)
bp = b.loc[rows, "p_plan_or_act"].to_numpy(float)[m]
cp = (ch.loc[rows, "prob_behavior"].to_numpy(float)
      + ch.loc[rows, "prob_attempt"].to_numpy(float))[m]

print(f"{int(m.sum())} posts with gold in {{ideation, behavior}}, "
      f"{int(y.sum())} behavior")
print(f"  chain    P(behavior)+P(attempt)   AP {APS(y, cp):.4f}")
print(f"  boundary P(plan or act)           AP {APS(y, bp):.4f}")
print(f"  delta {APS(y, bp) - APS(y, cp):+.4f}")
print(f"  correlation of the two scores: {np.corrcoef(bp, cp)[0,1]:.3f}")
print("\nA LOW correlation with a comparable AP is still useful -- that is a")
print("decorrelated member. A high correlation with no AP gain means it")
print("learned the same thing and the lane closes.")
print("\nFALSIFIER: >= +0.03 AP to proceed to a member blend.")
PY

echo ""
echo "================================================================"
echo "TEST COLUMN"
echo "================================================================"
python src/train_risk_boundary.py --predict --runs "${RUNS}" \
    --test data/test.csv --out artifacts/risk_boundary_test.csv \
    || { echo "FAILED predict"; exit 1; }

echo ""
echo "DONE. Nothing has been submitted."

#!/bin/bash
#SBATCH --job-name=nligaur
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# Compare two phrasings of the risk-level definitions fed to the NLI risk
# classifier: the terse Table III one-liners, and definitions expanded with the
# operational criteria of the paper those one-liners derive from.
#
# THE GROUNDING. Both organizer papers cite Gaur et al. WWW 2019 as the approved
# annotation criteria (Li et al. 2022, HKIE Trans 29(4): "the adopted labels and
# their corresponding annotation criteria have been approved by domain experts
# working in psychology (Gaur et al., 2019)"). Gaur operationalises `behavior`
# as self-harm OR active planning OR possessing/musing over a means of death OR
# preparation -- far broader than the Table III one-liner ("explicit suicidal
# expressions and self-harm or a suicidal plan").
#
# WHY THE TERSE WORDING IS RISKY. "Plan" is the load-bearing word, and it means
# something NARROWER in ordinary usage and in C-SSRS -- where a named method
# (level 3) and even a specific plan with intent (level 5) are still IDEATION,
# and self-harm without suicidal intent is a separate category. A pretrained NLI
# encoder carries those priors, so a terse definition pushes it to under-call
# `behavior` and route those posts to `ideation`.
#
# BOTH ARMS ARE TRAINED HERE so the comparison is like-for-like on identical
# folds, rather than against a number produced under a different protocol.
#
# PRE-STATED FALSIFIER. On the pooled 5-fold OOF, the gaur arm must
#   (a) raise BEHAVIOR RECALL by >= +0.03, and
#   (b) not lose weighted-F1 overall.
# Recall is the target because the failure mode is under-calling behavior; wF1
# is the guard against simply trading ideation precision away. Miss either and
# the definitions lane closes for risk.
#
# Usage:
#   sbatch scripts/risk_nli_gaur.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

EPOCHS="${EPOCHS:-4}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"
[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }

run_arm () {          # $1 = tag, $2 = RISK_DEFS value ("" or "gaur")
    local tag=$1 defs=$2
    for f in 0 1 2 3 4; do
        local d="runs/risknli_${tag}_f${f}"
        if [ -f "${d}/model.pt" ]; then
            echo "--- ${tag} fold ${f}: exists, skipping ---"
            continue
        fi
        echo ""
        echo "================================================================"
        echo "${tag} fold ${f}   RISK_DEFS='${defs}'"
        echo "================================================================"
        if [ -n "${defs}" ]; then export RISK_DEFS="${defs}"; else unset RISK_DEFS; fi
        python src/train_risk_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --model "${MODEL}" --outdir "${d}" --epochs "${EPOCHS}" \
            || { echo "FAILED ${tag} fold ${f}"; exit 1; }
        unset RISK_DEFS
    done
}

run_arm base ""
run_arm gaur "gaur"

echo ""
echo "================================================================"
echo "POOLED 5-FOLD OOF -- does gaur lift BEHAVIOR RECALL?"
echo "================================================================"
python - <<'PY'
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1


def pool(tag):
    fr = []
    for f in range(5):
        p = f"runs/risknli_{tag}_f{f}/val_predictions.csv"
        if not os.path.exists(p):
            return None
        fr.append(pd.read_csv(p))
    d = pd.concat(fr, ignore_index=True)
    d["row_id"] = d["row_id"].astype(str)
    return d.drop_duplicates("row_id").set_index("row_id")


lab = pd.read_csv("artifacts/full_train.csv", keep_default_na=False)
lab["row_id"] = lab["row_id"].astype(str)
lab = lab.set_index("row_id")

arms = {t: pool(t) for t in ("base", "gaur")}
if any(v is None for v in arms.values()):
    raise SystemExit("a fold is missing val_predictions.csv -- cannot compare")
rows = sorted(set(arms["base"].index) & set(arms["gaur"].index) & set(lab.index))
y = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
print(f"{len(rows)} pooled OOF posts")

pcol = "pred_risk_level"
res = {}
for t, d in arms.items():
    p = np.array([normalize_label(x) for x in d.loc[rows, pcol]])
    res[t] = p
    print(f"\n--- {t} ---   weighted-F1 {risk_weighted_f1(list(y), list(p)):.4f}")
    print(f"{'class':<12}{'gold':>6}{'pred':>6}{'prec':>8}{'rec':>8}{'F1':>8}")
    for c in RISK_LEVELS:
        g, q = (y == c), (p == c)
        tp = int((g & q).sum())
        pr = tp / max(int(q.sum()), 1)
        rc = tp / max(int(g.sum()), 1)
        f1 = 0.0 if tp == 0 else 2 * pr * rc / (pr + rc)
        print(f"{c:<12}{int(g.sum()):>6}{int(q.sum()):>6}{pr:>8.3f}"
              f"{rc:>8.3f}{f1:>8.3f}")
    print(f"  behavior->ideation errors: "
          f"{int(((y == 'behavior') & (p == 'ideation')).sum())}")

b, g_ = res["base"], res["gaur"]
gm = y == "behavior"
rb = float((b[gm] == "behavior").mean())
rg = float((g_[gm] == "behavior").mean())
wb = risk_weighted_f1(list(y), list(b))
wg = risk_weighted_f1(list(y), list(g_))
print("\n" + "=" * 64)
print(f"behavior recall  base {rb:.4f} -> gaur {rg:.4f}   delta {rg - rb:+.4f}")
print(f"weighted-F1      base {wb:.4f} -> gaur {wg:.4f}   delta {wg - wb:+.4f}")
print(f"behavior->ideation  {int(((y=='behavior')&(b=='ideation')).sum())} -> "
      f"{int(((y=='behavior')&(g_=='ideation')).sum())}")
ok = (rg - rb) >= 0.03 and (wg - wb) >= 0.0
print("\nPASSES -- retrain the 2 deployed seeds with RISK_DEFS=gaur and reblend."
      if ok else
      "FAILS the pre-stated bar (behavior recall >= +0.03 AND wF1 not worse).\n"
      "The definitions lane closes for risk.")
print("=" * 64)
PY

#!/bin/bash
#SBATCH --job-name=indbin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# A DEDICATED INDICATOR DETECTOR. THE HIGHEST-WEIGHT DECISION IN THE TASK IS
# ONE CLASS OF A 4-WAY SOFTMAX.
#
# THE ARITHMETIC. Risk is support-weighted F1, and the class supports are
# very uneven:
#
#     indicator  37.5% of the weight
#     ideation   31.5%
#     behavior   24.0%
#     attempt     7.0%
#
# Gold evidence is EMPTY iff gold risk is indicator, so the same decision also
# sets the evidence score on 135 of 378 test posts. Because indicator affects
# both the risk sub-score and the evidence sub-score at once, an improvement
# in indicator F1 carries an order of magnitude more composite leverage than
# the same improvement on `attempt`, which carries only 7% of the risk
# weight alone.
#
# WHY A SEPARATE MODEL SHOULD BEAT ONE SOFTMAX HEAD. The 4-class model spends
# capacity separating ideation / behavior / attempt, which differ by the
# presence of a PLAN or a COMPLETED ACT -- a subtle distinction and the source
# of most of its errors. Indicator vs the rest is a different and cleaner
# question: does the post explicitly reference suicide at all? A binary head
# sees 593 vs 1042 examples for one boundary instead of four.
#
# A lexical baseline shows the boundary is real but not closed: an EXPLICIT
# regex alone lands well below the model, and it fails on open-vocabulary
# phrasing that never names suicide directly. A trained binary model gets the
# semantics AND all of its capacity on that one line.
#
# HONEST VALIDATION. Five author-disjoint folds, each model scoring only its
# held-out fold, compared against `champion_oof.csv`'s own indicator calls --
# fold model vs fold model, so the comparison is fair.
#
# THE BAR IS DELIBERATELY HIGH. Deployment scores the test set with an ENSEMBLE
# while OOF scores each post with ONE model, and the ensemble reads several
# points higher on every component. A small OOF win does not survive that gap.
# Require >= +0.03 indicator F1 before considering a submission.
#
# Usage:
#   sbatch scripts/indicator_binary.sh
#   sbatch --export=ALL,EPOCHS=4 scripts/indicator_binary.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

EPOCHS="${EPOCHS:-4}"
TAG="${TAG:-indbin}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"
mkdir -p artifacts/indbin

for f in 0 1 2 3 4; do
    [ -f "artifacts/folds/train_f${f}.csv" ] || {
        echo "ERROR: artifacts/folds/train_f${f}.csv missing"; exit 1; }
done

# binary label files: indicator vs everything else, written as a 2-class
# risk_level so train_risk.py needs no modification
python - <<'PY'
import os
import pandas as pd
import sys
sys.path.insert(0, "src")
from scoring import normalize_label
os.makedirs("artifacts/indbin", exist_ok=True)
for f in range(5):
    for part in ("train", "val"):
        d = pd.read_csv(f"artifacts/folds/{part}_f{f}.csv",
                        keep_default_na=False)
        d["risk_level"] = ["indicator" if normalize_label(x) == "indicator"
                           else "ideation" for x in d["risk_level"]]
        d.to_csv(f"artifacts/indbin/{part}_f{f}.csv", index=False)
    print(f"fold {f}: wrote binary label files")
PY

for f in 0 1 2 3 4; do
    dir="runs/${TAG}_f${f}"
    if [ -f "${dir}/model_final.pt" ]; then
        echo "--- ${TAG} fold ${f}: exists ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "${TAG} fold ${f}  --  indicator vs rest, ${EPOCHS} epochs"
    echo "================================================================"
    python src/train_risk.py \
        --train "artifacts/indbin/train_f${f}.csv" \
        --val   "artifacts/indbin/val_f${f}.csv" \
        --model "${MODEL}" --outdir "${dir}" \
        --epochs "${EPOCHS}" --num-workers 0 --seed 13 \
        || { echo "FAILED ${TAG} fold ${f}"; exit 1; }
done

echo ""
echo "================================================================"
echo "DOES A DEDICATED HEAD BEAT THE 4-CLASS MODEL ON ITS OWN BOUNDARY?"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from scoring import normalize_label

TAG = os.environ["TAG"]
lab = pd.read_csv("artifacts/full_train.csv", keep_default_na=False)
lab["row_id"] = lab["row_id"].astype(str); lab = lab.set_index("row_id")

P, rid = [], []
for f in range(5):
    p = f"runs/{TAG}_f{f}/val_predictions.csv"
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    d = pd.read_csv(p, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    col = next(c for c in d.columns if "indicator" in c.lower()
               and c.startswith("prob"))
    P.extend(d[col].astype(float)); rid.extend(d["row_id"])
P = np.asarray(P)

gold = np.array([normalize_label(lab.loc[r, "risk_level"]) == "indicator"
                 for r in rid])
ch = pd.read_csv("artifacts/champion_oof.csv", keep_default_na=False)
ch["row_id"] = ch["row_id"].astype(str); ch = ch.set_index("row_id")
have = [i for i, r in enumerate(rid) if r in ch.index]
cm = np.array([normalize_label(ch.loc[rid[i], "pred_risk_level"]) ==
               "indicator" for i in have])
cp = ch.loc[[rid[i] for i in have], "prob_indicator"].astype(float).to_numpy()

def f1(pred, tgt):
    tp = int((pred & tgt).sum()); fp = int((pred & ~tgt).sum())
    fn = int((~pred & tgt).sum())
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    return 2*p*r/max(p+r,1e-9), p, r

g2 = gold[have]
print(f"{len(have)} posts, {int(g2.sum())} gold indicator\n")
f_c, p_c, r_c = f1(cm, g2)
print(f"  4-class champion_oof   F1 {f_c:.4f}  P {p_c:.3f}  R {r_c:.3f}")
best = (0, 0)
for t in np.arange(0.20, 0.85, 0.025):
    f_b, p_b, r_b = f1(P[have] >= t, g2)
    if f_b > best[0]:
        best = (f_b, t)
f_b, p_b, r_b = f1(P[have] >= best[1], g2)
print(f"  BINARY head  t={best[1]:.2f} F1 {f_b:.4f}  P {p_b:.3f}  R {r_b:.3f}"
      f"   delta {f_b-f_c:+.4f}")
mix = 0.5*(P[have] + cp)
bm = max(((f1(mix >= t, g2)[0], t) for t in np.arange(0.2, 0.85, 0.025)))
print(f"  MEAN of both t={bm[1]:.2f} F1 {bm[0]:.4f}"
      f"   delta {bm[0]-f_c:+.4f}")
print("\n  BAR: >= +0.03. Deployment scores test with an ENSEMBLE while this")
print("  is ONE model per post, and the ensemble reads several points higher,")
print("  so a small win here does not survive the gap.")
print("  Worth ~0.4*0.375*dF1 on risk PLUS ~0.3*dF1 on evidence.")
PY

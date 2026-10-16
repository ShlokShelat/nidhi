#!/bin/bash
#SBATCH --job-name=riskrat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# Risk with RATIONALE SUPERVISION, trained as a 5-fold author-disjoint bag.
#
# WHY THIS IS THE ONE JUSTIFIED RISK CHANGE LEFT. Everything else is measured
# shut:
#   decoding      38 of 39 confident ideation<->behavior errors have NEITHER
#                 head right, median margin 0.640
#   regex         8.8% base rate on eligible posts; best candidate 20% precision
#   factor labels worse than guessing the majority on exactly those errors
#   within-user   model errors correlate 0.32-0.33 within author
#
# What is NOT shut, verified independently on those 39 errors:
#   classify gold-ideation vs gold-behavior from ...
#     the annotator's GOLD SPAN text      0.692
#     majority baseline                   0.538
#     the FULL POST text                  0.308   <- worse than chance
# The rationale carries the signal; the surrounding post text actively misleads.
# Predicted spans only reach 0.590, so this must be a TRAINING-time head, not a
# test-time feature -- and a training head needs no spans at inference.
#
#   loss = (1-alpha)*CE + alpha*CORN + lambda*spanBCE,  lambda FIXED at 0.5
#
# THE FOLD BAG FIXES TWO MORE THINGS. The shipped seeds train on ~1263 posts and
# burn 372 on epoch selection, and all five shipped checkpoints choose their
# epoch on that same 372-post split (risk_v1_s15 ships at equal weight with a
# degraded best-val of 0.7129). Author-disjoint folds reclaim the burned data and
# select each epoch on held-out authors.
#
# Mask quality verified before writing this: 96.0% of gold spans locate in the
# post (96-98% for ideation/behavior/attempt); indicator posts get near-empty
# masks, which is correct supervision rather than missing supervision.
#
# HONEST SIZING. A modest member-level gain is expected, with an unreachable
# ceiling if every confident error flipped; through the blend weight that
# translates to a small composite gain. The val evidence is n=39 at p~0.03 --
# real but marginal. Prior risk submissions have shown offline gains fail to
# transfer, so judge ONLY by replicating the deployed recipe on the pooled
# OOF, never by the per-fold numbers this prints.
#
# Nothing here touches src/train_risk.py or any shipped run directory.
#
# Roughly 2-3h per fold. Run all five, or split across two jobs:
#   sbatch --export=ALL,"FOLDS=0 1 2" scripts/risk_rationale_5fold.sh
#   sbatch --export=ALL,"FOLDS=3 4"   scripts/risk_rationale_5fold.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

FOLDS="${FOLDS:-0 1 2 3 4}"
TAG="${TAG:-rat}"
LAMBDA="${LAMBDA:-0.5}"
EPOCHS="${EPOCHS:-5}"

for f in ${FOLDS}; do
    for p in "artifacts/folds/train_f${f}.csv" "artifacts/folds/val_f${f}.csv"; do
        [ -f "${p}" ] || { echo "ERROR: ${p} missing"; exit 1; }
    done
done

echo "folds  : ${FOLDS}"
echo "lambda : ${LAMBDA}   (fixed a priori, never swept)"
echo "epochs : ${EPOCHS}"

declare -A INC=( [0]=0 [1]=0 [2]=0 [3]=0 [4]=0 )

for f in ${FOLDS}; do
    OUT="runs/risk_${TAG}_f${f}"
    if [ -f "${OUT}/model.pt" ]; then
        echo "--- fold ${f}: ${OUT} exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "fold ${f} -> ${OUT}"
    echo "================================================================"
    python src/train_risk_rationale.py \
        --fold "${f}" --outdir "${OUT}" \
        --lambda-span "${LAMBDA}" --epochs "${EPOCHS}" \
        || { echo "FAILED fold ${f}"; exit 1; }
done

echo ""
echo "================================================================"
echo "POOLED OUT-OF-FOLD, rationale bag vs the shipped 3-seed DeBERTa"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import glob, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from data_audit import load
from scoring import normalize_label
from sklearn.metrics import f1_score

TAG = os.environ["TAG"]
LEV = ["indicator", "ideation", "behavior", "attempt"]
lab = load("artifacts/full_train.csv").set_index("row_id")

rid, P = [], []
for f in range(5):
    p = f"runs/risk_{TAG}_f{f}/val_probs.npz"
    if not os.path.exists(p):
        print(f"  missing {p} -- pooled figure will be partial")
        continue
    d = np.load(p, allow_pickle=True)
    rid.extend(list(d["row_id"]))
    P.append(d["probs"])
if not P:
    raise SystemExit("no folds finished")
P = np.concatenate(P)
y = np.array([LEV.index(normalize_label(lab.loc[r, "risk_level"])) for r in rid])
pred = P.argmax(1)
print(f"{len(rid)} out-of-fold posts")
print(f"  pooled weighted-F1 = {f1_score(y, pred, average='weighted'):.4f}")
print("\nper class")
for k, l in enumerate(LEV):
    print(f"  {l:<12}{f1_score((y==k).astype(int), (pred==k).astype(int), zero_division=0):>8.3f}")

for f in range(5):
    c = f"runs/risk_{TAG}_f{f}/run_config.json"
    if os.path.exists(c):
        print(f"  fold {f} best val wF1 "
              f"{json.load(open(c)).get('best_val_weighted_f1'):.4f}")

print("\nREFERENCE POINTS")
print("  3-seed DeBERTa mix on the 372-post val split : 0.7498")
print("  categorical head alone on that split         : 0.7579")
print("  1-fold DeBERTa on the 1590-post instrument   : 0.7311")
print("  full champion blend + rules there            : 0.7841")
print("\nThese are DIFFERENT instruments; do not compare across them. The")
print("decisive test is to swap this bag into src/champion_oof.py in place of")
print("the deberta member, replicate the deployed recipe, and compare against")
print("artifacts/champion_oof.csv over 50 author partitions.")
PY

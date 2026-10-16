#!/bin/bash
#SBATCH --job-name=facdefs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Train the factor model on definitions DERIVED FROM THE ANNOTATIONS.
#
# WHY. Reading the errors on `meaning in life` showed the model applies the
# published definition correctly and is punished for it:
#
#   published: "a sense of purpose, goals, or reasons for living"
#   model YES / gold NO:  "he was my point to keep going"
#                         "She was the reason I started to try in life"
#                         "I don't see the purpose"
#   gold YES / model 0.01: "Breathe Breathe everything's fine ... just breathe"
#
# The published definition is not the annotators' operational rule. Nine
# previous factor experiments changed the MODEL or the DATA; every one failed
# because the labels encode conventions only the annotations carry. This
# changes the QUESTION, and derives it from those same annotations.
#
# LEAKAGE CONTROL, and it is the whole ballgame. Each fold uses definitions
# written from THAT FOLD'S TRAINING PORTION ONLY. Fold f's definitions never
# saw fold f's posts. A definition derived from the evaluation posts would show
# a large gain that is pure circularity, so the per-fold file is not optional.
#
# BAR: 0.6673, the incumbent on the same 1635 out-of-fold posts.
#
# Read the per-label table, not just the macro. The hypothesis predicts gains
# concentrated where the published definition and annotator behaviour diverge
# most -- the protective five and `exposure to others' suicide` -- and roughly
# nothing on labels like `hopelessness` where the definition already matches
# (AP 0.890).

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# DEFS and TAG are overridable so the all-derived and hybrid arms run as a
# clean A/B. TAG MUST differ between them: the fold loop skips any fold whose
# model.pt already exists, so a shared tag would silently reuse the other arm's
# models and report its numbers as new.
#
#   all derived : sbatch --export=ALL scripts/factors_derived_defs.sh
#   hybrid      : sbatch --export=ALL,DEFS=artifacts/hybrid_defs,TAG=hyb \
#                     scripts/factors_derived_defs.sh
DEFS="${DEFS:-artifacts/derived_defs}"
TAG="${TAG:-defs}"
for f in 0 1 2 3 4; do
    if [ ! -f "${DEFS}/defs_f${f}.json" ]; then
        echo "ERROR: ${DEFS}/defs_f${f}.json missing."
        echo "Run src/derive_definitions.py (and, for the hybrid,"
        echo "src/build_hybrid_defs.py) on the login node first."
        exit 1
    fi
done
echo "definitions: ${DEFS}"
echo "tag        : ${TAG}"

OUT=artifacts/factor_runs
mkdir -p "${OUT}"

for f in 0 1 2 3 4; do
    # per-fold definitions: fold f's file was built from fold f's TRAINING data
    export FACTOR_DEFS_JSON="${DEFS}/defs_f${f}.json"
    dir="runs/factors_${TAG}_cv_f${f}"
    if [ ! -f "${dir}/model.pt" ]; then
        echo ""
        echo "================================================================"
        echo "${TAG} fold ${f}   defs=${FACTOR_DEFS_JSON}"
        echo "================================================================"
        python src/train_factors_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --model microsoft/deberta-v3-large \
            --outdir "${dir}" \
            || { echo "FAILED fold ${f}"; exit 1; }
    else
        echo "--- fold ${f}: exists ---"
    fi
    dst="${OUT}/${TAG}_cv_oof_f${f}.npz"
    # SAME env var at inference -- the hypothesis must match what it trained on
    [ -f "${dst}" ] || python src/predict_factors_nli.py \
        --test "artifacts/folds/val_f${f}.csv" --nli-run "${dir}" \
        --dump-probs "${dst}" --out "${OUT}/${TAG}_cv_oof_f${f}_labels.csv" \
        || echo "  dump failed fold ${f}"
done

# TEST predictions, bagging all five fold models -- without this a winning arm
# cannot be turned into a submission without a second GPU run.
#
# The definitions differ per fold, so the bag cannot be produced in one call
# the way the other factor scripts do it. Each fold is scored separately under
# ITS OWN definitions, then the five probability matrices are averaged. Scoring
# fold 3's model under fold 0's definitions would ask it a question it was
# never trained on.
TESTDST="${OUT}/${TAG}_cv_test.npz"
if [ ! -f "${TESTDST}" ]; then
    for f in 0 1 2 3 4; do
        export FACTOR_DEFS_JSON="${DEFS}/defs_f${f}.json"
        part="${OUT}/${TAG}_test_part_f${f}.npz"
        [ -f "${part}" ] || python src/predict_factors_nli.py \
            --test data/test.csv --nli-run "runs/factors_${TAG}_cv_f${f}" \
            --dump-probs "${part}" \
            --out "${OUT}/${TAG}_test_part_f${f}_labels.csv" \
            || echo "  test dump failed fold ${f}"
    done
    unset FACTOR_DEFS_JSON
    TAG="${TAG}" OUT="${OUT}" python - <<'PY'
import os
import numpy as np
tag, out = os.environ["TAG"], os.environ["OUT"]
parts = [np.load(f"{out}/{tag}_test_part_f{f}.npz", allow_pickle=True)
         for f in range(5)]
rid0 = list(parts[0]["row_id"])
for p in parts[1:]:
    assert list(p["row_id"]) == rid0, "test row order differs between folds"
probs = np.mean([p["probs"] for p in parts], axis=0)
np.savez(f"{out}/{tag}_cv_test.npz", probs=probs,
         thresholds=np.mean([p["thresholds"] for p in parts], axis=0),
         row_id=np.array(rid0, dtype=object))
print(f"bagged 5 folds -> {out}/{tag}_cv_test.npz  {probs.shape}")
PY
fi
unset FACTOR_DEFS_JSON

echo ""
echo "================================================================"
echo "POOLED OOF vs THE INCUMBENT"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import glob, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

TAG = os.environ.get("TAG", "defs")

PROT = {"social support", "coping strategy", "psychological capital",
        "sense of responsibility", "meaning in life"}
lab = pd.read_csv("artifacts/full_train.csv").set_index("row_id")
idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def gold(rids):
    Y = np.zeros((len(rids), 24), dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def per_label(P, Y):
    out = []
    for j in range(24):
        tp = int((P[:, j] & Y[:, j]).sum())
        out.append(0.0 if tp == 0 else 2 * tp /
                   (2 * tp + int((P[:, j] & (1 - Y[:, j])).sum())
                    + int(((1 - P[:, j]) & Y[:, j]).sum())))
    return out


ir, IP, ITH = [], [], []
for f in range(5):
    d = np.load(f"artifacts/fac_oof_f{f}.npz", allow_pickle=True)
    ir.extend(list(d["row_id"]))
    IP.append(d["probs"])
    ITH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
IY = gold(ir)
inc = per_label((np.concatenate(IP) >= np.concatenate(ITH)).astype(int), IY)

paths = sorted(glob.glob(f"artifacts/factor_runs/{TAG}_cv_oof_f*.npz"))
if len(paths) < 5:
    raise SystemExit(f"only {len(paths)}/5 folds")
sr, SP = [], []
for p in paths:
    d = np.load(p, allow_pickle=True)
    sr.extend(list(d["row_id"]))
    SP.append(d["probs"])
SP = np.concatenate(SP)
SY = gold(sr)
th = np.full(24, 0.5)
for j in range(24):
    best, bt = -1.0, 0.5
    for t in GRID:
        p = (SP[:, j] >= t).astype(int)
        tp = int((p & SY[:, j]).sum())
        if tp == 0:
            continue
        f1 = 2 * tp / (2 * tp + int((p & (1 - SY[:, j])).sum())
                       + int(((1 - p) & SY[:, j]).sum()))
        if f1 > best:
            best, bt = f1, t
    th[j] = bt
new = per_label((SP >= th).astype(int), SY)

print(f"{'factor':<44}{'incumb':>8}{'derived':>9}{'delta':>8}")
print("-" * 70)
for j, t in enumerate(FACTOR_TAXONOMY):
    print(f"{t:<44}{inc[j]:>8.3f}{new[j]:>9.3f}{new[j]-inc[j]:>+8.3f}"
          f"{'  prot' if t in PROT else ''}")
pj = [idx[t] for t in FACTOR_TAXONOMY if t in PROT]
rj = [idx[t] for t in FACTOR_TAXONOMY if t not in PROT]
print(f"\n{'MACRO (all 24)':<44}{np.mean(inc):>8.4f}{np.mean(new):>9.4f}"
      f"{np.mean(new)-np.mean(inc):>+8.4f}")
print(f"{'  protective (5)':<44}{np.mean([inc[j] for j in pj]):>8.4f}"
      f"{np.mean([new[j] for j in pj]):>9.4f}"
      f"{np.mean([new[j]-inc[j] for j in pj]):>+8.4f}")
print(f"{'  risk (19)':<44}{np.mean([inc[j] for j in rj]):>8.4f}"
      f"{np.mean([new[j] for j in rj]):>9.4f}"
      f"{np.mean([new[j]-inc[j] for j in rj]):>+8.4f}")
np.save(f"artifacts/factor_thresholds_{TAG}_oof.npy", th)
print(f"\nthresholds -> artifacts/factor_thresholds_{TAG}_oof.npy")
print("BAR: 0.6673. Expect gains where the published definition and the")
print("annotators diverge, and nothing where it already matches.")
PY

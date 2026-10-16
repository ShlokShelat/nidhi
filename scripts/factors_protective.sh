#!/bin/bash
#SBATCH --job-name=facprot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Ask the RIGHT QUESTION about the five protective factors.
#
# THE OBSERVATION. Four of the five worst labels on 1635 out-of-fold posts are
# the protective ones:
#
#     meaning in life          0.324      social support     0.654
#     sense of responsibility  0.560      coping strategy    0.743
#     psychological capital    0.598
#                                         protective average 0.576
#                                         19 risk factors    ~0.70
#
# They are 5 of 24 labels, so 21% of a macro-averaged score, and they sit 0.12
# below the rest as a group.
#
# THE LIKELY CAUSE. The NLI hypothesis is
#
#     "This post shows meaning in life: a sense of purpose, goals, or reasons
#      for living."
#
# That is the right question for a RISK factor -- the post shows hopelessness.
# It is the wrong question for a PROTECTIVE one, where what matters is whether
# the PERSON HAS the thing, not whether the post displays it. PROTECTIVE_TEMPLATE=1
# switches those five to
#
#     "The person in this post has meaning in life: ..."
#
# WHY THIS IS WORTH A RUN WHEN SEVEN OTHER FACTOR IDEAS FAILED. Every one of
# those tried to add something from outside the annotated data -- keyword rules
# (26% precision), synthetic posts (starved labels got worse), a fine-tuned LLM
# (-0.16), a zero-shot LLM (-0.3 on the labels it should have owned). All failed
# the same way: the labels encode these annotators' conventions, which only the
# data carries.
#
# This adds nothing from outside. It changes how the question is PHRASED to a
# model already trained on that data, on five labels where the current phrasing
# is arguably malformed. Zero new parameters, one variable.
#
# BAR: 0.6673, the incumbent on these same 1635 OOF posts. Read the five
# protective rows first -- if the mechanism is right, the gain concentrates
# there and the other nineteen barely move.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# Applies to training AND inference. Both build the hypothesis through
# train_factors_nli.hypothesis_for, so the env var keeps them consistent -- a
# CLI flag on only one path would train on one question and ask another.
export PROTECTIVE_TEMPLATE=1

if [ ! -f artifacts/folds/train_f0.csv ]; then
    echo "ERROR: artifacts/folds/ missing (scripts/nli_risk_5fold.sh writes it)."
    exit 1
fi

OUT=artifacts/factor_runs
mkdir -p "${OUT}"
TAG=prot

for f in 0 1 2 3 4; do
    dir="runs/factors_${TAG}_cv_f${f}"
    if [ ! -f "${dir}/model.pt" ]; then
        echo ""
        echo "================================================================"
        echo "${TAG} fold ${f}   (incumbent config, protective template)"
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
    [ -f "${dst}" ] || python src/predict_factors_nli.py \
        --test "artifacts/folds/val_f${f}.csv" --nli-run "${dir}" \
        --dump-probs "${dst}" --out "${OUT}/${TAG}_cv_oof_f${f}_labels.csv" \
        || echo "  dump failed fold ${f}"
done

dst="${OUT}/${TAG}_cv_test.npz"
if [ ! -f "${dst}" ]; then
    runs=""
    for f in 0 1 2 3 4; do runs="${runs}${runs:+,}runs/factors_${TAG}_cv_f${f}"; done
    python src/predict_factors_nli.py --test data/test.csv --nli-run "${runs}" \
        --dump-probs "${dst}" --out "${OUT}/${TAG}_cv_test_labels.csv" \
        || echo "  test dump failed"
fi

echo ""
echo "================================================================"
echo "POOLED OOF vs THE INCUMBENT -- protective labels marked"
echo "================================================================"
python - <<'PY'
import glob, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

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

paths = sorted(glob.glob("artifacts/factor_runs/prot_cv_oof_f*.npz"))
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

print(f"{'factor':<44}{'incumb':>8}{'prot':>8}{'delta':>8}")
print("-" * 68)
for j, t in enumerate(FACTOR_TAXONOMY):
    star = "  <- protective" if t in PROT else ""
    print(f"{t:<44}{inc[j]:>8.3f}{new[j]:>8.3f}{new[j]-inc[j]:>+8.3f}{star}")
pj = [idx[t] for t in FACTOR_TAXONOMY if t in PROT]
rj = [idx[t] for t in FACTOR_TAXONOMY if t not in PROT]
print(f"\n{'MACRO (all 24)':<44}{np.mean(inc):>8.4f}{np.mean(new):>8.4f}"
      f"{np.mean(new)-np.mean(inc):>+8.4f}")
print(f"{'  protective (5)':<44}{np.mean([inc[j] for j in pj]):>8.4f}"
      f"{np.mean([new[j] for j in pj]):>8.4f}"
      f"{np.mean([new[j]-inc[j] for j in pj]):>+8.4f}")
print(f"{'  risk (19)':<44}{np.mean([inc[j] for j in rj]):>8.4f}"
      f"{np.mean([new[j] for j in rj]):>8.4f}"
      f"{np.mean([new[j]-inc[j] for j in rj]):>+8.4f}")
np.save("artifacts/factor_thresholds_prot_oof.npy", th)
print("\nThe protective row is the test. If it rises and the risk row holds,")
print("the phrasing was the problem. If both move together, something else")
print("changed and the mechanism is not what I claimed.")
PY

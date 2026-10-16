#!/bin/bash
#SBATCH --job-name=facsyn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# The INCUMBENT recipe, retrained with synthetic data for the starved labels.
# One variable: the training set.
#
# WHY NOT d6. Measured properly -- five author-disjoint folds, thresholds fitted
# on 1635 out-of-fold posts, exactly the incumbent's protocol -- both d6 and
# d4 score below the incumbent. d6's apparent gain on the 372-post val split
# was an artefact of that split, the same artefact that made the top-3
# ensemble look like a large gain on val while performing far worse on
# held-out data. Hyperparameters are not the problem, so this run keeps the
# incumbent's configuration untouched (the trainer defaults:
# max_len 320, epochs 3, lr 1e-5, batch 16, grad_accum 2, neg_ratio 3,
# hard_frac 0.5, loss bce) and changes ONLY the data.
#
# WHAT THE DATA FIXES. S2 is macro-F1, so all 24 labels weigh 1/24 each.
# Thirteen of them have 8-118 real examples:
#
#     sexual orientation issues      8      cognitive deficits      33
#     exposure to others' suicide   14      meaning in life         45
#     poor school performance       16      low socio-economic      54
#     substance use                 33      sense of responsibility 58
#
# The earlier synthetic effort (gpt5_generated_all_targeted.csv, 1720 posts)
# generated for the NINE labels that already had 200-745 examples and zero for
# these thirteen. artifacts/syn_rare_factors.csv inverts that.
#
# THE DISCIPLINE THAT MAKES THIS MEASURABLE. Synthetic posts enter TRAINING
# ONLY, through --external. Every fold still validates on real held-out posts,
# and the 24 thresholds are still fitted on the pooled 1635 REAL out-of-fold
# predictions. The moment synthetic data touches a threshold or an evaluation
# set, the comparison against 0.6673 stops meaning anything.
#
# READ THE PER-LABEL TABLE, NOT JUST THE MEAN. If this works it will show up as
# large gains on the thirteen starved labels and roughly nothing elsewhere.
# A uniform shift would mean something else changed.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# SYN and TAG are overridable so the raw and style-matched arms can run as a
# clean A/B. TAG must differ between them: the fold loop skips any fold whose
# model.pt already exists, so reusing a tag would silently reuse the other
# arm's models and report its numbers as if they were new.
#
#   raw    : sbatch --export=ALL scripts/factors_syn_5fold.sh
#   styled : sbatch --export=ALL,SYN=artifacts/syn_rare_factors_styled.csv,TAG=synsty \
#                scripts/factors_syn_5fold.sh
SYN="${SYN:-artifacts/syn_rare_factors.csv}"
TAG="${TAG:-syn}"
if [ ! -f "${SYN}" ]; then
    echo "ERROR: ${SYN} not found -- run scripts/gen_rare_factors.sh first."
    exit 1
fi
echo "synthetic file: ${SYN}"
echo "tag           : ${TAG}"
if [ ! -f artifacts/folds/train_f0.csv ]; then
    echo "ERROR: artifacts/folds/ missing (scripts/nli_risk_5fold.sh writes it)."
    exit 1
fi
echo "synthetic posts: $(( $(wc -l < ${SYN}) - 1 ))"

OUT=artifacts/factor_runs
mkdir -p "${OUT}"

for f in 0 1 2 3 4; do
    dir="runs/factors_${TAG}_cv_f${f}"
    if [ ! -f "${dir}/model.pt" ]; then
        echo ""
        echo "================================================================"
        echo "syn fold ${f}  (incumbent config + synthetic external)"
        echo "================================================================"
        python src/train_factors_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --model microsoft/deberta-v3-large \
            --outdir "${dir}" \
            --external "${SYN}" \
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
echo "POOLED OOF vs THE INCUMBENT -- per label"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import collections, glob, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

lab = pd.read_csv("artifacts/full_train.csv").set_index("row_id")
idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
real_cnt = collections.Counter(
    f for c in lab["factors"]
    for f in set(parse_factors(c)) & set(FACTOR_TAXONOMY))


def gold_for(rids):
    Y = np.zeros((len(rids), len(FACTOR_TAXONOMY)), dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def fit_and_score(PR, Y):
    th = np.full(PR.shape[1], 0.5)
    for j in range(PR.shape[1]):
        best, bt = -1.0, 0.5
        for t in GRID:
            p = (PR[:, j] >= t).astype(int)
            tp = int((p & Y[:, j]).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((p & (1 - Y[:, j])).sum())
                           + int(((1 - p) & Y[:, j]).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt
    P = (PR >= th).astype(int)
    f1s = []
    for j in range(P.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp /
                   (2 * tp + int((P[:, j] & (1 - Y[:, j])).sum())
                    + int(((1 - P[:, j]) & Y[:, j]).sum())))
    return np.mean(f1s), f1s, th


# incumbent, deployed thresholds, same 1635 posts
ir, IP, ITH = [], [], []
for f in range(5):
    d = np.load(f"artifacts/fac_oof_f{f}.npz", allow_pickle=True)
    ir.extend(list(d["row_id"]))
    IP.append(d["probs"])
    ITH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
IP, ITH = np.concatenate(IP), np.concatenate(ITH)
IY = gold_for(ir)
IB = (IP >= ITH).astype(int)
inc = []
for j in range(IB.shape[1]):
    tp = int((IB[:, j] & IY[:, j]).sum())
    inc.append(0.0 if tp == 0 else 2 * tp /
               (2 * tp + int((IB[:, j] & (1 - IY[:, j])).sum())
                + int(((1 - IB[:, j]) & IY[:, j]).sum())))

TAG = os.environ.get("TAG", "syn")
paths = sorted(glob.glob(f"artifacts/factor_runs/{TAG}_cv_oof_f*.npz"))
if len(paths) < 5:
    raise SystemExit(f"only {len(paths)}/5 folds")
sr, SP = [], []
for p in paths:
    d = np.load(p, allow_pickle=True)
    sr.extend(list(d["row_id"]))
    SP.append(d["probs"])
SP = np.concatenate(SP)
SY = gold_for(sr)
m, per, th = fit_and_score(SP, SY)

print(f"{'factor':<44}{'real':>6}{'incumb':>8}{'syn':>8}{'delta':>8}")
print("-" * 74)
for j, t in enumerate(FACTOR_TAXONOMY):
    star = "  *" if real_cnt.get(t, 0) < 130 else ""
    print(f"{t:<44}{real_cnt.get(t,0):>6}{inc[j]:>8.3f}{per[j]:>8.3f}"
          f"{per[j]-inc[j]:>+8.3f}{star}")
starved = [j for j, t in enumerate(FACTOR_TAXONOMY)
           if real_cnt.get(t, 0) < 130]
print(f"\n{'MACRO (all 24)':<44}{'':>6}{np.mean(inc):>8.4f}{m:>8.4f}"
      f"{m-np.mean(inc):>+8.4f}")
print(f"{'starved labels only (*)':<44}{'':>6}"
      f"{np.mean([inc[j] for j in starved]):>8.4f}"
      f"{np.mean([per[j] for j in starved]):>8.4f}"
      f"{np.mean([per[j]-inc[j] for j in starved]):>+8.4f}")
np.save(f"artifacts/factor_thresholds_{TAG}_oof.npy", th)
print(f"\nthresholds -> artifacts/factor_thresholds_{TAG}_oof.npy")
print("\nSHIP ONLY IF the MACRO row beats the incumbent baseline computed")
print("above -- both d6 and d4 looked like wins on val but lost under this")
print("fold-based protocol.")
PY

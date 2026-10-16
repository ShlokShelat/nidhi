#!/bin/bash
#SBATCH --job-name=facqnli
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# Qwen as the factor cross-encoder, WITH the definitions.
#
# WHY DEFINITION CONDITIONING IS NOT OPTIONAL. Several factor labels have single
# digit support in the training data, so the only thing that lets a model handle
# them is the definition text in the hypothesis, not the examples. An LLM given
# 24 anonymous sigmoid outputs and no definitions is solving a different, much
# harder problem and says nothing about the encoder's capacity.
#
# This imports make_pairs and hypothesis_for from train_factors_nli, so the
# positives, the hard negatives from the CONFUSABLE map, and the hypothesis
# strings are identical to the DeBERTa arm. Only the encoder differs.
#
# TWO FOLDS, NOT FIVE. Fold 3 is the reference model's hardest (0.6023 tuned) and
# fold 0 its easiest (0.6743), so the pair brackets the range. Roughly 4-5h per
# fold against ~40 min for DeBERTa -- if AP does not move on the two folds that
# bound the range, five folds will not change the conclusion.
#
# READ MEAN AP, NOT AUC. F1 tracks average precision, while AUC is uninformative
# at this class imbalance: `meaning in life` reads AUC 0.817 against AP 0.173,
# because AUC rewards ranking the vast negative majority correctly.
#
# TEMPER EXPECTATIONS. The worst labels are bounded by ANNOTATION AGREEMENT, not
# model capacity -- `meaning in life` has an oracle F1 of 0.235 for any reader.
# Gains, if any, come from mid-tier labels where the text supports the
# annotation and the current model is merely imprecise.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# published definitions only -- the encoder is the single variable
unset FACTOR_DEFS_JSON || true
unset PROTECTIVE_TEMPLATE || true

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TAG="${TAG:-qnli}"
FOLDS="${FOLDS:-0 3}"
EXTRA="${EXTRA:-}"

if [ ! -f "${MODEL}/config.json" ]; then
    echo "ERROR: ${MODEL}/config.json not found (compute nodes are offline)."
    exit 1
fi
if [ ! -f artifacts/folds/train_f0.csv ]; then
    echo "ERROR: artifacts/folds/ missing."
    exit 1
fi

# reference per-fold tuned macro-F1 of the DeBERTa arm this is compared against
declare -A INC=( [0]=0.6743 [1]=0.6828 [2]=0.7145 [3]=0.6023 [4]=0.6103 )

for f in ${FOLDS}; do
    OUT="runs/factors_${TAG}_f${f}"
    if [ -f "${OUT}/val_probs.npz" ]; then
        echo "--- fold ${f}: done, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "fold ${f}   ${MODEL}   (reference tuned macro-F1 = ${INC[$f]})"
    echo "================================================================"
    python src/train_factors_llm_nli.py \
        --train "artifacts/folds/train_f${f}.csv" \
        --val   "artifacts/folds/val_f${f}.csv" \
        --model "${MODEL}" --outdir "${OUT}" \
        --max-len 288 --epochs 2 --batch 8 --grad-accum 2 ${EXTRA} \
        || { echo "FAILED fold ${f}"; exit 1; }
done

echo ""
echo "================================================================"
echo "AP PER LABEL vs THE BASELINE OOF, on the same folds"
echo "================================================================"
TAG="${TAG}" FOLDS="${FOLDS}" python - <<'PY'
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

TAG = os.environ["TAG"]
folds = [int(x) for x in os.environ["FOLDS"].split()]
lab = pd.read_csv("artifacts/full_train.csv").set_index("row_id")
idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}


def gold(rids):
    Y = np.zeros((len(rids), 24), dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


qr, QP, ir, IP = [], [], [], []
for f in folds:
    p = f"runs/factors_{TAG}_f{f}/val_probs.npz"
    if not os.path.exists(p):
        print(f"missing {p}")
        continue
    d = np.load(p, allow_pickle=True)
    qr.extend(list(d["row_id"]))
    QP.append(d["probs"])
    di = np.load(f"artifacts/fac_oof_f{f}.npz", allow_pickle=True)
    ir.extend(list(di["row_id"]))
    IP.append(di["probs"])
if not QP:
    raise SystemExit("no Qwen folds finished")
QP, IP = np.concatenate(QP), np.concatenate(IP)

# align the baseline to the Qwen row order before comparing
o = {r: i for i, r in enumerate(ir)}
missing = [r for r in qr if r not in o]
if missing:
    raise SystemExit(f"{len(missing)} rows absent from the baseline OOF")
IP = IP[[o[r] for r in qr]]
Y = gold(qr)

print(f"{len(qr)} posts from folds {folds}\n")
print(f"{'factor':<44}{'n':>5}{'AP inc':>9}{'AP qwen':>9}{'delta':>9}")
print("-" * 78)
di, dq = [], []
for j, t in enumerate(FACTOR_TAXONOMY):
    if Y[:, j].sum() == 0:
        continue
    a = average_precision_score(Y[:, j], IP[:, j])
    b = average_precision_score(Y[:, j], QP[:, j])
    di.append(a)
    dq.append(b)
    print(f"{t:<44}{int(Y[:,j].sum()):>5}{a:>9.3f}{b:>9.3f}{b-a:>+9.3f}")
print(f"\n{'MEAN AP':<44}{'':>5}{np.mean(di):>9.4f}{np.mean(dq):>9.4f}"
      f"{np.mean(dq)-np.mean(di):>+9.4f}")
won = sum(1 for a, b in zip(di, dq) if b > a + 0.02)
print(f"\nQwen ranks better (AP +0.02 or more) on {won}/{len(di)} labels")
print("\nMean AP up          -> extend to five folds")
print("Up on labels DeBERTa handles badly, even at a lower mean -> ENSEMBLE")
print("Flat or down        -> the reader is not the constraint, and the")
print("                       annotation ceiling from factor_error_read is")
print("                       the real answer")
PY

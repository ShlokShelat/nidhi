#!/bin/bash
#SBATCH --job-name=champqwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Everything needed on GPU to add Qwen to the champion risk ensemble.
#
# Already on disk, no work required:
#   artifacts/ensembled_risk.csv   0.5 x mean(3 risk_v1 seeds) + 0.5 x friend,
#                                  test probs in p_Indicator..p_Attempt
#
# Missing, produced here:
#   artifacts/nli_risk_test_probs.csv    the 0.30 NLI term on test
#   artifacts/qwen_ord_test_probs.csv    the ordinal Qwen arm, 4-adapter bag
#
# After this, assembling the submission is pure CPU.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

QWEN="${QWEN:-Qwen/Qwen2.5-14B-Instruct}"

echo "================================================================"
echo "STEP 1  NLI risk probabilities on TEST"
echo "================================================================"
python src/dump_nli_probs.py \
    --data data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --max-len 384 \
    --out artifacts/nli_risk_test_probs.csv || exit 1

echo ""
echo "================================================================"
echo "STEP 2  re-dump NLI on VAL through the same code path"
echo "================================================================"
# Regenerated deliberately: the val probs were written by the older val-only
# script. If these two disagree, the test probs above cannot be trusted either.
python src/dump_nli_probs.py \
    --data artifacts/val_split.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --max-len 384 \
    --out artifacts/nli_risk_val_probs2.csv || exit 1
python - <<'PY'
import pandas as pd, numpy as np
a = pd.read_csv('artifacts/nli_risk_val_probs.csv').set_index('row_id')
b = pd.read_csv('artifacts/nli_risk_val_probs2.csv').set_index('row_id')
c = [f'prob_{x}' for x in ['indicator','ideation','behavior','attempt']]
d = float(np.abs(a.loc[b.index, c].to_numpy() - b[c].to_numpy()).max())
print(f"max |old - new| on val NLI probs: {d:.2e}")
print("MATCH" if d < 1e-6 else "*** MISMATCH -- do not trust the test probs ***")
PY

echo ""
echo "================================================================"
echo "STEP 3  ordinal Qwen bag over TEST (folds 1-4)"
echo "================================================================"
python v2_rebuild/src/predict_risk_llm2.py \
    --test data/test.csv \
    --model "${QWEN}" \
    --runs 'v2_rebuild/runs/risk_llm2_ord_f{f}' \
    --folds "1 2 3 4" \
    --out artifacts/qwen_ord_test_probs.csv \
    --batch 8 --max-len 512 || exit 1

echo ""
echo "================================================================"
echo "DONE. Next (CPU):"
echo "  python src/assemble_champion_qwen.py --w 0.30"
echo "================================================================"

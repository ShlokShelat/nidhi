#!/bin/bash
#SBATCH --job-name=lora5f
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# Rebuild the LoRA risk member 5-fold, so its gain can be measured honestly.
#
# WHAT WAS FOUND, AND WHY IT IS NOT YET TRUSTWORTHY. runs/risk_lora7b (a
# LoRA'd Qwen2.5-7B, 3 epochs) scores 0.7694 on val_split against a risk-NLI
# model's 0.7683 on the same posts -- equal accuracy, 48.8% error overlap, and
# a real CONFIDENCE GAP: mean confidence 0.892 when right vs 0.703 when wrong.
# The encoder has no such gap; its errors are confidently wrong. A zero-parameter
# `maxconf` rule (believe whichever model is surer) improved weighted F1.
#
# THREE REASONS THAT NUMBER CANNOT BE BELIEVED, all fixed here:
#  1. EPOCH-SELECTION BIAS. train_risk_lora.py saved its BEST-on-val epoch, so
#     the probabilities were tuned on the split they were then scored against.
#     --no-val-select saves the final epoch instead.
#  2. 32 AUTHORS. Every CI on val_split includes zero and always will. Pooling
#     5 folds gives 1635 posts / 153 authors.
#  3. BASELINE AMBIGUITY, the worst of the three. Against oof_base.csv the
#     gain has a CI spanning zero; against champion_oof.csv it has a CI
#     excluding zero -- because champion_oof scores WORSE on these posts than
#     a single fresh risk-NLI model. The significance appeared only when the
#     baseline got weaker, which is precisely the pattern that makes an
#     offline delta untrustworthy as a predictor of the held-out result.
#
# The fix for (3) is structural: these folds are artifacts/folds/, the SAME
# folds risk_nli_gaur.sh used to build artifacts/nligaur/oof_base.csv. So the
# comparison becomes paired by construction -- same posts, same splits, both
# honestly out-of-fold, one unambiguous baseline.
#
# PRE-STATED FALSIFIER: pooled over 1635 posts / 153 authors, `maxconf` must
# beat oof_base with a 95% paired-author CI excluding zero.
# Below that, the LoRA member is not worth a submission slot however good the
# confidence gap looks.
#
# NOT INCLUDED HERE: the CoT arm. Its traces come from the cotmemb job, which
# generates reasoning for all 1635 posts WITHOUT seeing gold; only traces
# where the 72B model independently reached the RIGHT answer are then kept for
# training. Distilling gold-conditioned traces would teach confabulation on the
# posts whose labels rest on annotator context. Run that as a second arm on
# these same folds once cotmemb lands.
#
# Usage:
#   sbatch scripts/risk_lora_5fold.sh

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="${TAG:-lora7b}"
EPOCHS="${EPOCHS:-3}"
BASE="${BASE:-artifacts/nligaur/oof_base.csv}"

[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }
[ -f "$BASE" ] || { echo "ERROR: $BASE missing"; exit 1; }
if [ ! -d "$MODEL" ]; then
    # the original run used Qwen/Qwen2.5-7B-Instruct
    ALT="Qwen/Qwen2.5-7B-Instruct"
    [ -d "$ALT" ] && MODEL="$ALT" || { echo "ERROR: no Qwen2.5-7B dir"; exit 1; }
fi
echo "model: ${MODEL}"

for f in 0 1 2 3 4; do
    d="runs/risk_${TAG}_f${f}"
    if [ -f "${d}/val_predictions.csv" ]; then
        echo "--- ${d}: exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "${TAG} fold ${f}   (final-epoch probs, no val selection)"
    echo "================================================================"
    python src/train_risk_lora.py \
        --train "artifacts/folds/train_f${f}.csv" \
        --val   "artifacts/folds/val_f${f}.csv" \
        --model "${MODEL}" --outdir "${d}" \
        --epochs "${EPOCHS}" --no-val-select \
        || { echo "FAILED fold ${f}"; exit 1; }
done

echo ""
echo "================================================================"
echo "POOL the 5 folds into one honest OOF file"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import os
import pandas as pd
tag = os.environ["TAG"]
fr = []
for f in range(5):
    p = f"runs/risk_{tag}_f{f}/val_predictions.csv"
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    d = pd.read_csv(p)
    d["fold"] = f
    fr.append(d)
d = pd.concat(fr, ignore_index=True)
d["row_id"] = d["row_id"].astype(str)
n0 = len(d)
d = d.drop_duplicates("row_id")
out = f"artifacts/lora_{tag}_oof.csv"
d.to_csv(out, index=False)
print(f"{n0} rows -> {len(d)} unique posts   wrote {out}")
PY

echo ""
echo "================================================================"
echo "JUDGE -- paired against the SAME folds' encoder OOF"
echo "================================================================"
python src/risk_lora_member.py \
    --lora-csv "artifacts/lora_${TAG}_oof.csv" \
    --enc "$BASE" --val artifacts/full_train.csv \
    || { echo "eval FAILED"; exit 1; }

echo ""
echo "BAR: maxconf must beat oof_base with the 95% CI excluding zero."
echo "153 authors now, so the CI can finally resolve an effect that size."

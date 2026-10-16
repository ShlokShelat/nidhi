#!/bin/bash
#SBATCH --job-name=facl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out

# Fine-tune a decoder LLM on the 24 FACTOR labels. One fold at a time.
#
# WHY A TRAINED MODEL RATHER THAN ANOTHER DECODE. Every recombination of the
# existing NLI arms -- prompting a larger model, per-label arm selection, extra
# pool members, author pooling, further decode refinements -- reuses the same
# underlying ranking. The decode operates on that ranking and cannot reorder it.
# What is left to improve is RANKING quality, which needs a different model.
#
# WHAT IT MUST FIX. Four of the five worst average precisions in the taxonomy
# are protective labels, and the errors are not calibration errors: the posts
# ranked highest for one protective label usually carry a DIFFERENT protective
# label in gold. The model locates protective content and cannot tell which of
# five overlapping boxes the annotator used. Discriminating between correlated
# constructs is what supervised training fixes and prompting does not.
#
# THE BAR. src/factor_fold_baseline.py scores the deployed bag on the same fold.
# Judge the PROTECTIVE column, not the macro: a model that matches the bag
# overall but not on those five labels has reproduced the bag rather than
# improved it. Fold difficulty is not the same for factors as for risk, so
# compare against the SAME fold's baseline, never against the pooled
# out-of-fold macro.
#
# LADDER. The 8B runs first: it validates the 24-way head, the weighted loss and
# the decode cheaply before anything expensive. An 8B that merely matches the
# bar does not settle the question -- at this corpus size an 8B tends to tie an
# encoder ensemble while a 70B pulls ahead, so escalate rather than conclude.
#
# FOLD VARIANCE. Per-fold scores on 1635 author-disjoint posts swing widely, so
# one fold is a smoke test, not evidence. Run a second fold before committing
# the remaining three.
#
# Usage:
#   sbatch scripts/factor_lora_pilot.sh                       # 8B, fold 3
#   sbatch --export=ALL,FOLD=0 scripts/factor_lora_pilot.sh   # second fold
#   sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-70B-Instruct,TAG=fq70,QUANT=--load-4bit,BATCH=1,ACCUM=16 scripts/factor_lora_pilot.sh
#   sbatch --export=ALL,EXTRA=--faint-weight\ 2.0,TAG=facw2 scripts/factor_lora_pilot.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TAG="${TAG:-facL8}"
FOLD="${FOLD:-3}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-8}"
MAXLEN="${MAXLEN:-512}"
QUANT="${QUANT:-}"
EXTRA="${EXTRA:-}"

[ -d "$MODEL" ] || { echo "ERROR: $MODEL missing"; exit 1; }
[ -f "artifacts/folds/train_f${FOLD}.csv" ] || {
    echo "ERROR: artifacts/folds/train_f${FOLD}.csv missing"; exit 1; }

OUT="runs/fac_${TAG}_f${FOLD}"
if [ -f "${OUT}/val_predictions.csv" ]; then
    echo "--- ${OUT} already trained, nothing to do ---"
    exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ""
echo "model ${MODEL}   tag ${TAG}   fold ${FOLD}"
echo "bar: src/factor_fold_baseline.py on this fold -- judge the PROTECTIVE column"
echo ""

python src/train_factors_lora.py \
    --train "artifacts/folds/train_f${FOLD}.csv" \
    --val   "artifacts/folds/val_f${FOLD}.csv" \
    --model "$MODEL" --outdir "$OUT" \
    --epochs "$EPOCHS" --lr "$LR" --batch "$BATCH" --grad-accum "$ACCUM" \
    --max-len "$MAXLEN" --no-val-select ${QUANT} ${EXTRA} \
    || { echo "FAILED"; exit 1; }

cat <<'EOF'

================================================================
HOW TO READ IT  (bar = src/factor_fold_baseline.py, same fold)
================================================================
protective clearly ABOVE the bar   run a second fold to check it is not a
                                   one-fold outlier, THEN the rest.
protective at or below the bar     a decoder LLM does not separate those five
                                   either, and the lane is closed.
macro up but protective flat       the bag was rebuilt, not improved. The trap
                                   that looks like success. Do not escalate.

An 8B that merely MATCHES the bar does not settle it -- at this corpus size an
8B tends to tie an encoder ensemble while a 70B pulls ahead. Escalate with:
  sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-70B-Instruct,TAG=fq70,QUANT=--load-4bit,BATCH=1,ACCUM=16 scripts/factor_lora_pilot.sh
EOF

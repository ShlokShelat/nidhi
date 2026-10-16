#!/bin/bash
#SBATCH --job-name=qlora70
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# PILOT: QLoRA a 70B risk classifier on ONE fold.
#
# WHY SCALE IS THE AXIS TO TRY. A LoRA'd 8B, three epochs, untuned, already
# matches the tuned DeBERTa encoder chain on honest 5-fold OOF:
#
#     LoRA Llama-3.1-8B   0.8006   (honest 5-fold OOF)
#     encoder chain       0.7985
#
# The 70B/72B checkpoints on disk have only ever been PROMPTED, never
# fine-tuned, so parameter scale is the one axis of the recipe left untouched.
#
# A SINGLE MODEL ALSO AVOIDS THE MEASUREMENT TRAPS OF COMBINATION RULES:
#   1. OOF necessarily uses FOLD models (1308 posts) while deployment uses
#      FULL-DATA models (1635), so OOF understates the deployed champion and
#      any "add a member" rule tuned on OOF exploits that gap rather than a
#      real improvement.
#   2. A LoRA member's OOF is one fold model per post, while its test
#      predictions average five -- a different event from the one measured.
# A single stronger model has neither problem: it is not a vote, not a blend,
# not a routing rule, so what is measured is what deploys.
#
# ONE FOLD, NOT FIVE. Five folds is ~25 GPU-hours; one is ~5. Fold 3 is chosen
# because every member scores highest there (encoder-comparable runs hit
# 0.84-0.85), so a 70B that cannot beat the 8B on the easiest fold will not beat
# it anywhere. Judge against the SAME fold's 8B number, printed below.
#
# QLoRA specifics: NF4 4-bit puts a 70B at ~40GB on one 94GB H100, leaving room
# for gradient checkpointing and the optimiser. prepare_model_for_kbit_training
# is required -- without it the adapter trains on a frozen quantised graph and
# the loss barely moves. batch 1 with grad-accum 16 keeps the effective batch
# where the 8B runs had it.
#
# PRE-STATED BAR: fold-3 val weighted-F1 >= 0.87. The 8B scored 0.8496 there and
# the deployed champion is ~0.8203 overall. Below 0.87 the extra 9x parameters
# are not buying enough to justify 25 hours for the remaining folds.
#
# Usage:
#   sbatch scripts/risk_qlora70_pilot.sh
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-72B-Instruct,TAG=q72 scripts/risk_qlora70_pilot.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
TAG="${TAG:-ql70}"
FOLD="${FOLD:-3}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-16}"
MAXLEN="${MAXLEN:-512}"

[ -d "$MODEL" ] || { echo "ERROR: $MODEL missing"; exit 1; }
[ -f "artifacts/folds/train_f${FOLD}.csv" ] || {
    echo "ERROR: artifacts/folds/train_f${FOLD}.csv missing"; exit 1; }
python -c "import bitsandbytes, peft; print('bitsandbytes', bitsandbytes.__version__, 'peft', peft.__version__)" \
    || { echo "ERROR: needs bitsandbytes + peft"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ""
echo "model ${MODEL}   fold ${FOLD}   4-bit QLoRA"
echo "BAR: fold-${FOLD} val weighted-F1 >= 0.87"
echo "  same fold, LoRA Llama-3.1-8B  = 0.8496"
echo "  same fold, LoRA Mistral-7B    = 0.8341"
echo "  deployed champion overall     ~0.8203"
echo ""

python src/train_risk_lora.py \
    --train "artifacts/folds/train_f${FOLD}.csv" \
    --val   "artifacts/folds/val_f${FOLD}.csv" \
    --model "$MODEL" --outdir "runs/risk_${TAG}_f${FOLD}" \
    --epochs "$EPOCHS" --lr "$LR" --batch "$BATCH" --grad-accum "$ACCUM" \
    --max-len "$MAXLEN" --load-4bit --no-val-select \
    || { echo "FAILED"; exit 1; }

echo ""
echo "================================================================"
echo "IF IT CLEARS 0.87"
echo "================================================================"
echo "Train the other four folds, pool, and judge with"
echo "  src/risk_lora_member.py --lora-csv artifacts/lora_${TAG}_oof.csv"
echo "A single model at that level would REPLACE the champion's risk"
echo "column outright -- no vote, no blend, no routing, and therefore"
echo "none of the OOF-vs-deployment mismatch a combination rule carries."
echo ""
echo "IF IT DOES NOT: scale is not the lever, and the remaining risk"
echo "work is the narrow class-restricted overrides."

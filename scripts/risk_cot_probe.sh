#!/bin/bash
#SBATCH --job-name=cotprobe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# Screening probe, run BEFORE building any CoT training set: can a strong model
# reason its way to the risk label without being told the answer?
#
# WHY THE PRIOR QUESTION MATTERS. Distilling reasoning traces means generating
# traces for the training posts. A trace generated FROM the gold label is a
# rationalisation -- the model justifies whatever label it is handed, and where
# the label rests on annotator context rather than post text (a multi-post
# countdown series; near-duplicate posts labelled from an author's history) it
# will invent a reason that is not there. Training on invented reasons teaches
# invention. So the probe hides the gold label: if reasoning cannot crack the
# posts the encoder misses, teaching the encoder to reason will not crack them.
#
# WHY QWEN2.5-72B. The 2024 competition winner was few-shot CoT Qwen2-72B plus
# LoRA generative models, 1st overall. This probe tests the reasoning half of
# that recipe.
#
# WHY THE MECHANISM IS PLAUSIBLE. On the confident ideation/behavior errors,
# classifying from the ANNOTATOR'S SPAN scores 0.692 while classifying from the
# FULL POST scores 0.308 -- worse than chance. One clause decides the label and
# the surrounding despair vocabulary drowns it. Isolating that clause is exactly
# what step-by-step reasoning does.
#
# WHAT DECIDES IT -- not accuracy. A zero-shot LLM will not beat the tuned
# encoder chain outright. COMPLEMENTARITY decides it: of the posts the encoder
# gets WRONG, how many does reasoning get RIGHT? The members of the encoder
# chain fail on largely the SAME posts, so a new member only pays if it breaks
# that pattern.
#
# PRE-STATED BAR: rescue >= 0.35 on encoder-wrong posts. Chance among the three
# remaining classes is ~0.33 and the encoder's own second choice is right ~45%
# of the time, so below 0.35 this adds nothing that rescoring the encoder's own
# probabilities would not give for free.
#
# THREE ARMS, same fixed sample:
#   7B smoke     pipeline check on 40 posts -- catches prompt/parse bugs before
#                the expensive model loads
#   72B + CoT    the real question
#   72B no-CoT   ablation: same definitions, same few-shot, reasoning REMOVED.
#                If these two match, the definitions are doing the work and the
#                reasoning is decoration.
#
# EVERYTHING LOCAL. No dataset post leaves the cluster.
#
# Usage:
#   sbatch scripts/risk_cot_probe.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BIG="${BIG:-Qwen/Qwen2.5-72B-Instruct}"
SMALL="${SMALL:-Qwen/Qwen2.5-7B-Instruct}"
N="${N:-300}"
BATCH="${BATCH:-8}"
OOF="${OOF:-artifacts/nligaur/oof_base.csv}"

# ONE GPU. A 72B in bf16 needs ~145GB and two cards are hard to co-schedule
# here; NF4 puts it at ~40GB, which fits an 80GB card with room for the KV
# cache at batch 8. Quality cost of NF4 on a 72B is small and this is a
# screening question, not a shipped model.
QUANT="${QUANT:---load-4bit}"

# --focus-errors samples HALF from posts the encoder gets wrong. Rescue rate is
# what decides this lane, so generations are spent there rather than on posts
# the encoder already handles. 150 encoder-wrong posts estimate rescue to about
# +/-0.08, which is enough to clear or miss a 0.35 bar.
FOCUS="${FOCUS:---focus-errors}"

[ -f "$OOF" ] || { echo "ERROR: $OOF missing"; exit 1; }
for m in "$BIG" "$SMALL"; do
    [ -d "$m" ] || { echo "ERROR: $m missing"; exit 1; }
done
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The smoke test runs UNQUANTISED, so it would not catch a missing
# bitsandbytes -- check now rather than after the 7B arm has already passed.
if [ -n "$QUANT" ]; then
    python -c "import bitsandbytes, torch; print('bitsandbytes', bitsandbytes.__version__)" \
        || { echo "ERROR: --load-4bit needs bitsandbytes. Either"; \
             echo "  pip install bitsandbytes"; \
             echo "  or rerun with QUANT= BIG=Qwen/Qwen2.5-7B-Instruct"; \
             exit 1; }
fi

echo ""
echo "================================================================"
echo "1/3  SMOKE TEST -- Qwen2.5-7B, 40 posts, catches prompt/parse bugs"
echo "================================================================"
python src/risk_cot_probe.py --model "$SMALL" --oof "$OOF" --n 40 $FOCUS \
    --batch "$BATCH" --out artifacts/cot_probe_7b.csv \
    || { echo "SMOKE TEST FAILED -- not loading the 72B"; exit 1; }

echo ""
echo "================================================================"
echo "2/3  Qwen2.5-72B WITH reasoning -- ${N} posts"
echo "================================================================"
python src/risk_cot_probe.py --model "$BIG" --oof "$OOF" --n "$N" $FOCUS \
    $QUANT --batch "$BATCH" --out artifacts/cot_probe_72b.csv \
    || { echo "FAILED: 72B CoT arm"; exit 1; }

echo ""
echo "================================================================"
echo "3/3  ABLATION -- same prompt, reasoning REMOVED, same ${N} posts"
echo "================================================================"
python src/risk_cot_probe.py --model "$BIG" --oof "$OOF" --n "$N" $FOCUS \
    $QUANT --batch "$BATCH" --no-cot --max-new 8 \
    --out artifacts/cot_probe_72b_nocot.csv \
    || { echo "FAILED: 72B ablation"; exit 1; }

echo ""
echo "================================================================"
echo "READ IN THIS ORDER"
echo "================================================================"
echo "1. the COMPLEMENTARITY block of arm 2 -- rescue >= 0.35 or the lane closes"
echo "2. arm 2 vs arm 3 -- if they match, reasoning is decoration and the"
echo "   definitions plus few-shot examples were doing all the work"
echo "3. artifacts/cot_probe_72b.csv -- read the traces on RESCUED posts. Sound"
echo "   reasoning is worth distilling; lucky guessing dressed as reasoning is"
echo "   the same confabulation failure arriving by another road."

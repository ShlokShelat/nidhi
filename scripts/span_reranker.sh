#!/bin/bash
#SBATCH --job-name=span-rerank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Supervised cross-encoder reranker over BIO-tagger span candidates.
#
# The target, measured by headroom_analysis.py on val: a perfect selector
# over the candidate pool reaches a ceiling far above the best single tagger
# -- the pool already holds 75.7% of gold spans, so selection is the deficit.
#
# Two selectors already failed and reveal why: a hand-feature cross-fitted
# selector underperforms because its features never read the text, and a
# frontier LLM used as a zero-shot grader ranks candidates WORSE than simply
# counting tagger agreement, because gold spans follow an ANNOTATION
# CONVENTION the LLM has no access to and the taggers were trained on.
#
# So the selector has to LEARN the convention. This trains on train_split
# only, so val stays a clean measurement for both the taggers and the
# reranker.
#
# Evaluation is author-disjoint cross-fitted, and the bar is:
#     beat vote>=2 (0.6984) on FULL val AND on BOTH halves.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "================================================================"
echo "STEP 1: inspect the generated training pairs (CPU, seconds)"
echo "================================================================"
python src/train_span_reranker.py --dump-pairs artifacts/reranker_pairs.csv || exit 1

echo ""
echo "================================================================"
echo "STEP 2a: deberta-v3-large, 2 epochs"
echo "================================================================"
# Attempt 1 used 3 epochs and collapsed to loss 0.025 -- memorisation. The
# provenance shortcut that caused it is now removed, but 400M parameters on
# ~7k pairs still overfits easily, so the epoch count comes down too.
python src/train_span_reranker.py \
    --train artifacts/train_split.csv \
    --val   artifacts/val_split.csv \
    --outdir runs/span_reranker_large \
    --model microsoft/deberta-v3-large \
    --epochs 2 --batch 16 --max-len 320 --seed 13 \
    --cache-dir artifacts/evidence_per_run --tag val_split

echo ""
echo "================================================================"
echo "STEP 2b: deberta-v3-base, 3 epochs  (4.6x fewer parameters)"
echo "================================================================"
# If capacity is the problem rather than the shortcut, the smaller encoder
# should hold up where the larger one does not. Both variants are cheap, so
# run them together rather than serialising two submissions.
python src/train_span_reranker.py \
    --train artifacts/train_split.csv \
    --val   artifacts/val_split.csv \
    --outdir runs/span_reranker_base \
    --model microsoft/deberta-v3-base \
    --epochs 3 --batch 16 --max-len 320 --seed 13 \
    --cache-dir artifacts/evidence_per_run --tag val_split

echo ""
echo "================================================================"
echo "DONE."
echo ""
echo "  Bar: cross-fitted F1 must beat vote>=2 (0.7037) on full val AND"
echo "  on BOTH author halves, for EITHER variant."
echo ""
echo "  Watch the per-epoch mean loss. Attempt 1 went 0.266 -> 0.061 ->"
echo "  0.025: that was the provenance shortcut being memorised. If the"
echo "  loss now plateaus near 0.15-0.30 the harder negatives are doing"
echo "  their job; if it collapses below 0.05 again, capacity is still"
echo "  the problem and the base variant is the one to read."
echo ""
echo "  This is attempt 4 at evidence SELECTION (hand features, LLM"
echo "  binary, LLM graded, supervised v1 all missed). If both variants"
echo "  miss too, the selection route closes and the remaining evidence"
echo "  lever is a better TAGGER, not a better selector."
echo "================================================================"

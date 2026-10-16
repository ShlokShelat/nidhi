#!/bin/bash
#SBATCH --job-name=evllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x_%j.out

# Measure whether an LLM can PROPOSE gold spans that the encoder taggers never
# put in the candidate pool.
#
# THIS IS GENERATION, NOT SELECTION. Better ranking cannot lift evidence past
# the pool it ranks:
#
#     a PERFECT selector over the tagger pool reaches only 0.8617,
#     because 24% of gold spans appear in NO candidate at ANY confidence.
#
# No ranking reaches a span nobody proposed, so the only remaining lever is
# adding proposals.
#
# THE BAR IS NOT "BEATS THE TAGGERS". A proposer only has to be DIFFERENTLY
# wrong. The decisive number is: of the gold spans the pool misses, what share
# does the LLM recover? Bar >= 0.20; below that the ensemble gain cannot
# meaningfully improve macro-F1 even with perfect selection afterwards.
#
# Two arms:
#   GOLD levels  the ceiling -- is the information reachable at all?
#   PRED levels  shippable -- does it survive the pipeline's risk error rate?
#
# Every generated quote is substring-verified against the post with the same
# gate the taggers pass. A quote not literally in the post scores zero under the
# official matcher, so it is dropped rather than counted as a proposal.
#
# Runs on posts WITH gold spans only: empty-gold posts are a suppression
# question, not a span-generation one.
#
# Usage:
#   sbatch scripts/evidence_llm_propose.sh

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# THREE FAMILIES, DELIBERATELY: dense Qwen, dense Llama, sparse MoE Mixtral.
# For a PROPOSER the goal is coverage, not quality -- a model only helps if it
# is DIFFERENTLY wrong, so picking the single "best" model optimises the wrong
# quantity. Mixtral is included for that reason and not because it is stronger:
# it is older and smaller in active parameters, but a MoE routes differently and
# so may quote differently. In 4-bit it is ~25GB and only two experts fire per
# token, making it the cheapest of the three to run.
#
# Relevant external evidence (ACL-Verbatim, 2026): on verbatim span extraction a
# ModernBERT extractor beat EVERY LLM tested, because LLMs "extract
# plausible-sounding but off-topic text" while a small model abstains. Expect
# poor precision here and judge ONLY on pool-recall gain.
#
# DeepSeek-R1-Distill is deliberately NOT used: it is a REASONING model, and
# reasoning models paraphrase and elaborate, which is exactly what fails the
# substring gate. DeepSeek V4 Flash is API-only and would send dataset posts off
# this cluster; everything here stays local.
MODELS="${MODELS:-Qwen/Qwen2.5-72B-Instruct meta-llama/Llama-3.1-70B-Instruct mistralai/Mixtral-8x7B-Instruct-v0.1}"
CANDS="${CANDS:-artifacts/evidence_runs}"
PREDRISK="${PREDRISK:-artifacts/nligaur/oof_base.csv}"
N="${N:-250}"
QUANT="${QUANT:---load-4bit}"

for m in $MODELS; do
    [ -d "$m" ] || { echo "ERROR: $m missing"; exit 1; }
done
ls "$CANDS"/*_val_candidates.csv >/dev/null 2>&1 || {
    echo "ERROR: no *_val_candidates.csv in $CANDS -- run scripts/evidence_dump.sh"
    exit 1; }
if [ -n "$QUANT" ]; then
    python -c "import bitsandbytes" || { echo "ERROR: needs bitsandbytes"; exit 1; }
fi

for m in $MODELS; do
    tag=$(basename "$m" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9.-')
    echo ""
    echo "################################################################"
    echo "# ${tag}"
    echo "################################################################"

    echo ""
    echo "--- GOLD levels (the ceiling) ---"
    python src/evidence_llm_propose.py --model "$m" $QUANT --n "$N" \
        --cand-dir "$CANDS" --out "artifacts/evllm_${tag}_gold.csv" \
        || { echo "FAILED gold arm for ${tag}"; exit 1; }

    echo ""
    echo "--- PREDICTED levels (shippable) ---"
    python src/evidence_llm_propose.py --model "$m" $QUANT --n "$N" \
        --cand-dir "$CANDS" --risk-from "$PREDRISK" \
        --out "artifacts/evllm_${tag}_pred.csv" \
        || { echo "FAILED pred arm for ${tag}"; exit 1; }
done

echo ""
echo "================================================================"
echo "HOW TO READ THIS"
echo "================================================================"
echo "1. POOL RECALL block per arm -- bar is 0.20 of MISSED spans recovered."
echo "   Ignore F1 and precision; a proposer only has to be differently wrong."
echo "2. GOLD clears the bar but PRED does not -> the proposer works and the"
echo "   risk labels are the limit, which risk conditioning addresses."
echo "3. Compare the MODELS' recovered spans. If they recover DIFFERENT"
echo "   spans, use both -- union of proposers is the point. If they recover"
echo "   the same ones, one is enough."
echo "4. If none clears the bar, generation is not the binding constraint"
echo "   either."

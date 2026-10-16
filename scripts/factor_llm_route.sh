#!/bin/bash
#SBATCH --job-name=facllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out

# Factor ROUTING with a reasoning LLM: collect per-label LLM opinions so that
# specific DISAGREEMENT CELLS can be routed to whichever model owns them.
#
# THE MECHANISM. Using an LLM's scores globally requires it to out-rank the
# encoder bag everywhere, which average precision says it does not. Routing asks
# a strictly narrower question --
#
#     on posts where the LLM says factor k is PRESENT and the bag says ABSENT,
#     which one is right?
#
# -- and only that cell has to hold. A model with worse global AP can still be
# the more accurate reader inside one disagreement cell, because global AP
# averages over the whole label distribution while a cell is conditioned on the
# exact configuration where the two models differ. With 24 factor labels there
# are 24 candidate cells to test, against 4 for risk.
#
# WHY FACTORS. Factors are 0.30 of the composite, and the encoder-arm and
# combination-rule levers there operate on global ranking, which is the thing
# routing does not depend on. That makes it the component where a cell-level
# mechanism is most likely to add something the existing arms cannot.
#
# WHAT THIS DOES: one pass per post over 1635 OOF + 378 test, asking which of the
# 24 factors are present, with SHORT CONCRETE glosses rather than the published
# academic definitions. Verbatim academic definitions rank far worse than
# plain-English ones (`meaning in life` quoted from George & Park as "involves
# cognitive component, motivational component and affective component" scores
# 0.231, against 0.816 for `hopelessness` stated plainly), because the abstract
# wording shares little surface with how the construct appears in a post.
#
# One pass, not 24 per post: 2k generations instead of 48k. Routing needs a few
# precise labels, not calibrated per-label scores. A label with a strong cell can
# be rescored on its own afterwards.
#
# WHAT DECIDES IT: per-label PRECISION on the LLM's YES calls. A label where the
# LLM is precise is a routing candidate; one where it sprays is not. Expect
# over-listing (gold averages 2.92 of 24) -- that is fine, imprecise labels
# simply do not get routed.
#
# EVERYTHING LOCAL. No dataset post leaves the cluster.
#
# Usage:
#   sbatch scripts/factor_llm_route.sh

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BIG="${BIG:-Qwen/Qwen2.5-72B-Instruct}"
BATCH="${BATCH:-8}"

[ -d "$BIG" ] || { echo "ERROR: $BIG missing"; exit 1; }
[ -f artifacts/full_train.csv ] || { echo "ERROR: full_train.csv missing"; exit 1; }
[ -f data/test.csv ] || { echo "ERROR: data/test.csv missing"; exit 1; }
python -c "import bitsandbytes" || { echo "ERROR: needs bitsandbytes"; exit 1; }

python src/factor_llm_opinions.py \
    --model "$BIG" --load-4bit --batch "$BATCH" \
    --oof artifacts/full_train.csv \
    --extra-csv data/test.csv \
    --out artifacts/factor_llm_oof.csv \
    || { echo "FAILED"; exit 1; }

echo ""
echo "================================================================"
echo "READ THE PER-LABEL PRECISION TABLE"
echo "================================================================"
echo "A label with HIGH precision on the LLM's YES calls is a routing"
echo "candidate. Low precision means it sprays and will not be routed."
echo ""
echo "Then run the cell analysis on CPU:"
echo "  python src/factor_route_eval.py"
echo ""
echo "Bar, matching how route_attempt was judged: the rule must win on"
echo "BOTH available factor baselines with a paired-author CI clear of"
echo "zero, and it must fit ZERO parameters -- 0-param rules have"
echo "outperformed fitted ones here before."

#!/bin/bash
#SBATCH --job-name=evdiv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# ---------------------------------------------------------------------------
# ONE DIVERSE EVIDENCE TAGGER PER JOB.
#
# WHY DIVERSITY AND NOT MORE SEEDS. `evidence_pool_analysis.py` categorised all
# 50 unmatched gold spans on val:
#
#     ABSENT           21  42%   no candidate anywhere near it
#     PARTIAL          12  24%   a candidate overlaps, neither contains
#     in pool, dropped 10  20%   the VOTE discarded it -- selection, CLOSED
#     TRUNCATED         7  14%   past token 512, unreachable
#
# 56% (ABSENT + TRUNCATED) needs a different or longer model; a 9th DeBERTa-v3
# seed proposes near-identical candidates to the eight we have and cannot fill
# ABSENT. Everything decode-side is already closed against a verified 0.8824
# baseline (the decode is already optimal), and min_conf is not starving the
# pool (oracle flat at 0.9493 from 0.05 to 0.50).
#
# WHAT LOADS OFFLINE (transformers 4.46.3, HF_HUB_OFFLINE=1):
#   ModernBERT-large      FAILS -- needs >=4.48. The 8192 window is out.
#   mental-roberta-large  GATED. mental-roberta-base not fully cached.
#   electra-large-discriminator  OK 334M -- replaced-token-detection
#                         pretraining, the most DIFFERENT objective available,
#                         so its candidates should overlap ours least
#   roberta-large-mnli    OK 355M -- different architecture and tokenizer
#   deberta-v2-xlarge     OK 885M -- same family, bigger
#   deberta-v3-large @1024 OK -- verified forward pass, logits (1,1008,3).
#                         Relative position embeddings extend past the 512
#                         max_position_embeddings, so the TRUNCATED bucket is
#                         reachable without ModernBERT.
#
# Trains on train_split and validates on val_split -- the same protocol as the
# eight deployed taggers, so the result is honestly measurable BEFORE shipping.
# Nothing here touches full data (full-data runs leak onto val).
#
# Usage (one per backbone, so a partial result is still usable):
#   sbatch --job-name=ev1024 --export=ALL,TAG=s31_rc,MODEL=microsoft/deberta-v3-large,MAXLEN=1024,BATCH=2,ACCUM=8 scripts/evidence_diverse.sh
#   sbatch --job-name=evelec --export=ALL,TAG=s32_rc,MODEL=google/electra-large-discriminator,MAXLEN=512,BATCH=8,ACCUM=2 scripts/evidence_diverse.sh
#   sbatch --job-name=evrob  --export=ALL,TAG=s33_rc,MODEL=FacebookAI/roberta-large-mnli,MAXLEN=512,BATCH=8,ACCUM=2 scripts/evidence_diverse.sh
# ---------------------------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

TAG="${TAG:?set TAG, e.g. s31_rc}"
MODEL="${MODEL:?set MODEL}"
MAXLEN="${MAXLEN:-512}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-2}"
SEED="${SEED:-31}"
EPOCHS="${EPOCHS:-6}"

TR=artifacts/train_split.csv
VA=artifacts/val_split.csv
RISK=artifacts/nligaur/oof_base.csv     # PREDICTED risk for val conditioning
OUT="runs/evidence_${TAG}"

echo "=============================================================="
echo "tagger ${TAG}"
echo "  model   ${MODEL}"
echo "  max_len ${MAXLEN}   batch ${BATCH} x accum ${ACCUM}   seed ${SEED}"
echo "=============================================================="
set -e

python src/train_evidence.py \
    --train "$TR" --val "$VA" \
    --model "$MODEL" --outdir "$OUT" \
    --risk-cond --val-risk-from "$RISK" \
    --max-len "$MAXLEN" --batch "$BATCH" --grad-accum "$ACCUM" \
    --seed "$SEED" --epochs "$EPOCHS"

echo ""
echo "=============================================================="
echo "dump val + test candidates for evidence_${TAG}"
echo "=============================================================="
python src/evidence_dump_all.py --runs "evidence_${TAG}" \
    --test data/test.csv --val "$VA" --outdir artifacts/evidence_runs

echo ""
echo "DONE ${TAG}. Score it INTO the vote once all the new runs exist:"
echo ""
echo '  RC8=evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_v1_rc,evidence_v2_rc'
echo '  for R in "$RC8" "$RC8,evidence_'"${TAG}"'"; do'
echo '    for K in 2 3; do'
echo '      python src/evidence_vote_test.py --runs "$R" --dir artifacts/evidence_runs \'
echo '        --split val --test artifacts/val_split.csv --k $K --cluster --rep modal \'
echo '        --risk-from artifacts/nligaur/oof_base.csv \'
echo '        --out artifacts/tmp_vote_k${K}_$(echo $R | md5sum | cut -c1-6).csv'
echo '    done; done'
echo '  python src/score_evidence_variants.py --files "artifacts/tmp_vote_*.csv"'
echo ""
echo "k is a FRACTION of the pool -- at 9 runs, k=2 is looser than at 8."
echo "Sweep k or the added-member comparison is not like-for-like."

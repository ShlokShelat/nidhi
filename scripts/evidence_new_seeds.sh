#!/bin/bash
#SBATCH --job-name=evseeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# ---------------------------------------------------------------------------
# NEW EVIDENCE TAGGERS.
#
# Decode-side selection has been extensively explored against a verified
# 0.8824 baseline (five selection rules, a learned re-ranker, a learned count
# budget, four cluster representatives, conditioning hedges) without
# exceeding it. Per-candidate AUC says the vote rule already extracts what
# the features hold.
#
# What is NOT closed is the candidate POOL itself. The oracles say the right
# spans are often there (best-subset 0.9493 vs shipped 0.8824), and 13 gold
# spans were never proposed by ANY tagger. More/better taggers change the pool,
# which is the one thing a decode cannot.
#
# THIS IS HONESTLY MEASURABLE. Train on train_split, validate on val_split --
# exactly how the eight current taggers were validated. Only FULL-DATA training
# cannot be measured (it trains on val), so nothing here uses full data.
#
# THREE CONFIGS, each targeting something the diagnosis actually found:
#
#   s21  cond-dropout 0.3      Training always shows the GOLD risk prefix;
#                              deployment shows the PREDICTED one, wrong on
#                              ~17% of posts. As AUGMENTATION this also shows
#                              each post under several conditionings, which is
#                              the defensible half of the argument -- the
#                              robustness half was weakened by the finding
#                              that a WRONG label (0.5789) still beats NO
#                              label (0.5166).
#
#   s22  max-len 1024          7 of 358 val gold spans (2.0%) sit past token
#                              512 and are unreachable at any training budget.
#                              This is the only measured HARD ceiling in the
#                              component.
#
#   s23  per-post loss         The scorer macro-averages over POSTS; the default
#                              loss averages over TOKENS, so long posts pull
#                              ~5x the gradient. Same defect class already
#                              addressed once in the factors component.
#
# Expect modest returns: an 8-tagger vote has diminishing returns from a 9th.
# The honest case for running it is that it is the ONLY remaining lever, it is
# measurable before shipping, and the GPU hours are otherwise idle.
# ---------------------------------------------------------------------------

source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

TR=artifacts/train_split.csv
VA=artifacts/val_split.csv
RISK=artifacts/nligaur/oof_base.csv     # PREDICTED risk for val conditioning

set -e

echo "=============================================================="
echo "s21  cond-dropout 0.3"
echo "=============================================================="
python src/train_evidence.py --train "$TR" --val "$VA" \
    --outdir runs/evidence_s21_rc --risk-cond --cond-dropout 0.3 \
    --val-risk-from "$RISK" --seed 21 --epochs 6

echo "=============================================================="
echo "s22  max-len 1024 (rescues the 2% of gold spans past token 512)"
echo "=============================================================="
python src/train_evidence.py --train "$TR" --val "$VA" \
    --outdir runs/evidence_s22_rc --risk-cond --max-len 1024 \
    --val-risk-from "$RISK" --seed 22 --epochs 6 --batch 4 --grad-accum 4

echo "=============================================================="
echo "s23  per-post loss + cond-dropout"
echo "=============================================================="
python src/train_evidence.py --train "$TR" --val "$VA" \
    --outdir runs/evidence_s23_rc --risk-cond --per-post --cond-dropout 0.3 \
    --val-risk-from "$RISK" --seed 23 --epochs 6

echo "=============================================================="
echo "dump val + test candidates for the three new runs"
echo "=============================================================="
python src/evidence_dump_all.py \
    --runs evidence_s21_rc,evidence_s22_rc,evidence_s23_rc \
    --test data/test.csv --val "$VA" --outdir artifacts/evidence_runs

echo ""
echo "DONE. Next -- score the 8-tagger vote against the 11-tagger vote on VAL:"
echo ""
echo '  RC8=evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_v1_rc,evidence_v2_rc'
echo '  RC11=$RC8,evidence_s21_rc,evidence_s22_rc,evidence_s23_rc'
echo '  for R in "$RC8" "$RC11"; do'
echo '    python src/evidence_vote_test.py --runs "$R" --dir artifacts/evidence_runs \'
echo '      --split val --test artifacts/val_split.csv --k 2 --cluster --rep modal \'
echo '      --risk-from artifacts/nligaur/oof_base.csv --out artifacts/tmp_vote_$(echo $R | tr , _ | cut -c1-40).csv'
echo '  done'
echo '  python src/score_evidence_variants.py --files "artifacts/tmp_vote_*.csv"'
echo ""
echo "k is a FRACTION of the pool: at 11 runs, k=2 is a looser vote than at 8."
echo "Sweep k=2,3 on the 11-run pool or the comparison is not like-for-like."

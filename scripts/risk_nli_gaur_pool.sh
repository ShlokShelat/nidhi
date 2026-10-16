#!/bin/bash
#SBATCH --job-name=nlipool
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Recover the comparison that risk_nli_gaur.sh could not run.
#
# That script's final block read runs/risknli_<tag>_f<k>/val_predictions.csv --
# a file train_risk_nli.py has never written; it saves model.pt only. Ten
# trainings succeeded and the comparison died on a missing file.
#
# The per-fold val numbers already decide the SHIPPING question: gaur lost on 4
# of 5 folds, mean weighted-F1 moved from 0.7983 to 0.7778, so the pre-stated
# falsifier's guard condition ("not worse on wF1") has failed. This job is for
# the DIAGNOSTIC question the falsifier's first condition asks: did behavior
# recall actually move? If it rose noticeably while wF1 fell, the mechanism is
# real and only the trade-off is bad, which argues for the severity-ladder
# variant. If it did not move, definition-widening does not reach the boundary
# at all and the lane closes.
#
# It also prints the INDICATOR row for both arms, which is now the more valuable
# boundary: gold evidence is empty exactly on indicator posts, so indicator
# errors cost the risk term AND the evidence term.
#
# Inference only -- no training. ~10 model loads over the 5 held-out folds.
#
# Usage:
#   sbatch scripts/risk_nli_gaur_pool.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

for t in base gaur; do
    for f in 0 1 2 3 4; do
        [ -f "runs/risknli_${t}_f${f}/model.pt" ] || {
            echo "ERROR: runs/risknli_${t}_f${f}/model.pt missing"; exit 1; }
    done
done

python src/risk_nli_pool_oof.py --arms base,gaur --outdir artifacts/nligaur

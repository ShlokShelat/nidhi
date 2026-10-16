#!/bin/bash
#SBATCH --job-name=risknlimnli
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# The two DEPLOYABLE risk-NLI seeds, with an MNLI-pretrained encoder.
#
# WHY. src/train_risk_nli.py scores (post, "This post shows suicide risk level
# <level>: <definition>") -- an entailment task -- while the existing members
# runs/risk_nli and runs/risk_nli_s2 both start from microsoft/deberta-v3-large,
# a checkpoint that has never seen an entailment example. Starting from an
# encoder already fine-tuned on MNLI matches the pretraining objective to the
# task format, which is the same correction that pays on the factor side across
# two unrelated base architectures.
#
# Risk is ARGMAX over 4 classes, so there is no threshold fitted on held-out
# data to mis-transfer to the test set; the encoder swap is the only variable.
#
# The 5-fold checkpoints from scripts/nli_risk_mnli_5fold.sh cannot be shipped:
# the deployed pipeline (scripts/rebuild_and_apply_rules.sh,
# src/blend_risk_saved.py --nli-run) needs models trained on
# artifacts/train_split.csv. This produces them, matching scripts/risk_nli.sh
# and risk_nli_s2.sh exactly except for --model.
#
# Seeds 13 and 14, mirroring the pair being replaced. Both run in one job
# because the Slurm submission cap is 5 and two seeds are one logical unit.
#
# The encoder must be swapped in BOTH seeds. Mixing an MNLI seed with a plain
# one averages two different models, so a difference in the blended output
# could not be attributed to the encoder.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
EPOCHS="${EPOCHS:-4}"

for f in artifacts/train_split.csv artifacts/val_split.csv; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done

echo "model : ${MODEL}"
echo "epochs: ${EPOCHS}"
echo "replacing: runs/risk_nli (seed 13) and runs/risk_nli_s2 (seed 14)"
echo "the originals are NOT touched; new dirs are runs/risk_nli_mnli{,_s2}"

# seed 13 -- mirrors scripts/risk_nli.sh
if [ -f runs/risk_nli_mnli/model.pt ]; then
    echo "--- runs/risk_nli_mnli exists, skipping ---"
else
    echo ""
    echo "================================================================"
    echo "seed 13  -> runs/risk_nli_mnli"
    echo "================================================================"
    python src/train_risk_nli.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir runs/risk_nli_mnli --epochs "${EPOCHS}" \
        || { echo "FAILED seed 13"; exit 1; }
fi

# seed 14 -- mirrors scripts/risk_nli_s2.sh
if [ -f runs/risk_nli_mnli_s2/model.pt ]; then
    echo "--- runs/risk_nli_mnli_s2 exists, skipping ---"
else
    echo ""
    echo "================================================================"
    echo "seed 14  -> runs/risk_nli_mnli_s2"
    echo "================================================================"
    python src/train_risk_nli.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir runs/risk_nli_mnli_s2 --epochs "${EPOCHS}" \
        --seed 14 \
        || { echo "FAILED seed 14"; exit 1; }
fi

echo ""
echo "================================================================"
echo "val weighted-F1, new vs the seeds being replaced"
echo "================================================================"
python - <<'PY'
import json, os
pairs = [("runs/risk_nli", "runs/risk_nli_mnli", 13),
         ("runs/risk_nli_s2", "runs/risk_nli_mnli_s2", 14)]
for old, new, seed in pairs:
    def get(d):
        p = os.path.join(d, "run_config.json")
        if not os.path.exists(p):
            return None
        return json.load(open(p)).get("best_val_weighted_f1")
    o, n = get(old), get(new)
    if o is None or n is None:
        print(f"  seed {seed}: missing run_config ({old}={o}, {new}={n})")
        continue
    print(f"  seed {seed}:  {o:.4f} -> {n:.4f}   {n - o:+.4f}")
print()
print("These are 372-post val figures and are NOISY: at that sample size a")
print("single post moves weighted F1 by more than the effect being measured.")
print("The 1635-post out-of-fold measurement is the one that decides.")
print("Use these only to confirm nothing is broken.")
PY

echo ""
echo "================================================================"
echo "NEXT -- rebuild the risk column with these two, changing NOTHING else"
echo "================================================================"
cat <<'EOF'
  python src/blend_risk_saved.py \
      --nli-run runs/risk_nli_mnli,runs/risk_nli_mnli_s2 \
      --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
      --ens-test artifacts/ensembled_risk.csv \
      --out artifacts/risk_blend_mnli.csv

  # then apply the SAME rules, then patch onto the champion:
  python src/patch_submission_risk.py \
      --base submissions/NIDHI_evidence_vote8.csv \
      --risk <rules-applied csv> \
      --out submissions/NIDHI_risk_mnli.csv

Ship the ENCODER SWAP FIRST with w_nli left at 0.30. Dropping the other
blend members is a SECOND, separate submission: bundling two changes leaves
a negative result uninterpretable, because neither change can be attributed
the outcome.
EOF

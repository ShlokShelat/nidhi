#!/bin/bash
#SBATCH --job-name=evcond
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# Does telling the tagger WHICH RISK LEVEL it is justifying help it pick the
# right span?
#
# THE GAP. src/train_evidence.py encoded the post ALONE -- one sequence, no risk
# signal. The risk label entered the evidence pipeline at exactly one point, as
# a delete-everything switch in evidence_vote_test.py:223. So the tagger could
# only hunt for generically risky-sounding text.
#
# WHY THAT IS WRONG. Gold spans are level-SPECIFIC justifications: the clause
# chosen as evidence is the one that establishes THAT level, not the most
# emotionally salient sentence in the post.
#
# The sharpest form of this is a single author posting repeatedly, where
# consecutive posts differ only in an incrementing countdown at the end. The
# body of such a post can be ordinary rumination with no suicidal content at
# all, and it is gold `behavior` ONLY because of that trailing countdown -- a
# stated timeline, which is a behavior criterion. A tagger with no level to
# argue for has no reason to prefer that clause over the emotional body text,
# so it cannot find it.
#
# WHY THIS IS THE RIGHT KIND OF ATTEMPT. The evidence memo closes the encoder
# lane with "any future evidence work has to change the OBJECTIVE or the
# DECODER, not the backbone". This changes the objective. Five span-SELECTION
# attempts have failed; this is not selection, it is a different training signal
# for generation.
#
# THREE NUMBERS, and the gap between the last two is the whole diagnosis:
#   base            no conditioning
#   cond @ GOLD     oracle ceiling -- is the idea sound at all?
#   cond @ PRED     shippable -- does it survive the risk model's 21% error rate?
# If oracle is large and pred is flat, the idea works and the RISK model is the
# bottleneck (which is the indicator-boundary lane). If oracle is flat too, the
# idea is wrong and this closes.
#
# PRE-STATED FALSIFIER: cond @ PRED must beat base val phrase-F1 by a margin
# clearly outside the noise already measured on this metric.
#
# Usage:
#   sbatch scripts/evidence_risk_cond.sh

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

TRAIN="${TRAIN:-artifacts/train_split.csv}"
VAL="${VAL:-artifacts/val_split.csv}"
RISKSRC="${RISKSRC:-artifacts/champion_oof.csv}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"
EPOCHS="${EPOCHS:-6}"

for f in "$TRAIN" "$VAL" "$RISKSRC"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

echo "=== alignment guard (paired offsets restart at 0; a silent corruption) ==="
python tests/test_risk_cond_alignment.py || {
    echo "ALIGNMENT TEST FAILED -- not training"; exit 1; }

train_arm () {          # $1 = outdir, $2... = extra flags
    local d=$1; shift
    if [ -f "${d}/model_final.pt" ]; then
        echo "--- ${d}: exists, skipping ---"
        return 0
    fi
    echo ""
    echo "================================================================"
    echo "training ${d}  $*"
    echo "================================================================"
    python src/train_evidence.py \
        --train "$TRAIN" --val "$VAL" --model "$MODEL" \
        --outdir "${d}" --epochs "$EPOCHS" "$@" \
        || { echo "FAILED ${d}"; exit 1; }
}

train_arm runs/evcond_base
train_arm runs/evcond_on --risk-cond

echo ""
echo "================================================================"
echo "EVAL -- base vs cond@GOLD vs cond@PRED on the same val split"
echo "================================================================"
VAL="$VAL" RISKSRC="$RISKSRC" MODEL="$MODEL" python - <<'PY'
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, "src")
from scoring import parse_evidence, phrase_f1
from train_evidence import TaggerModel, collect_candidates, spans_at
from transformers import AutoTokenizer

VAL = os.environ["VAL"]
RISKSRC = os.environ["RISKSRC"]
MODEL = os.environ["MODEL"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

va = pd.read_csv(VAL)
posts = va["post"].astype(str).tolist()
gold = [parse_evidence(c) for c in va["evidence"]]
gold_lvl = [str(x) for x in va["risk_level"]]

rk = pd.read_csv(RISKSRC, keep_default_na=False)
col = "risk_level" if "risk_level" in rk.columns else "pred_risk_level"
m = {str(a): b for a, b in zip(rk["row_id"], rk[col])}
pred_lvl, n_fallback = [], 0
for r, g in zip(va["row_id"].astype(str), gold_lvl):
    if r in m:
        pred_lvl.append(m[r])
    else:
        pred_lvl.append(g)
        n_fallback += 1
if n_fallback:
    print(f"NOTE: {n_fallback} of {len(va)} val rows absent from {RISKSRC}; "
          f"gold level used for those -- the PRED number is that much optimistic")

tok = AutoTokenizer.from_pretrained(MODEL)


def score(outdir, risks):
    mp = os.path.join(outdir, "model_final.pt")
    cfg = os.path.join(outdir, "run_config.json")
    mc = json.load(open(cfg)).get("evidence_min_conf", 0.0) \
        if os.path.exists(cfg) else 0.0
    mdl = TaggerModel(MODEL)
    mdl.load_state_dict(torch.load(mp, map_location="cpu"))
    mdl.to(dev)
    c = collect_candidates(mdl, tok, posts, 512, dev, risks=risks)
    best = max(((phrase_f1(gold, spans_at(c, t)).f1, t)
                for t in [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]))
    at_stored = phrase_f1(gold, spans_at(c, mc)).f1
    del mdl
    torch.cuda.empty_cache()
    return at_stored, best[0], best[1], mc


rows = [("base", "runs/evcond_base", None),
        ("cond @ GOLD  (oracle)", "runs/evcond_on", gold_lvl),
        ("cond @ PRED  (shippable)", "runs/evcond_on", pred_lvl)]
res = {}
print(f"\n{'arm':<26}{'F1@stored':>11}{'F1@best':>10}{'conf':>7}")
print("-" * 54)
for name, d, risks in rows:
    if not os.path.exists(os.path.join(d, "model_final.pt")):
        print(f"{name:<26}   (missing {d})")
        continue
    a, b, t, mc = score(d, risks)
    res[name] = b
    print(f"{name:<26}{a:>11.4f}{b:>10.4f}{t:>7.2f}")

print("\n" + "=" * 54)
if "base" in res and "cond @ PRED  (shippable)" in res:
    d_pred = res["cond @ PRED  (shippable)"] - res["base"]
    d_or = res.get("cond @ GOLD  (oracle)", float("nan")) - res["base"]
    print(f"cond@PRED  vs base   {d_pred:+.4f}   "
          f"(composite {0.3 * d_pred:+.4f})")
    print(f"cond@GOLD  vs base   {d_or:+.4f}   <- ceiling if risk were perfect")
    if d_pred >= 0.010:
        print("\nPASSES. Retrain the champion's taggers with --risk-cond and "
              "re-vote.")
    elif d_or >= 0.010:
        print("\nFAILS as shipped but the ORACLE is real: the idea works and "
              "the\nrisk model is the bottleneck. That argues for the "
              "indicator-boundary\nlane, not for more evidence work.")
    else:
        print("\nFAILS at the ceiling too. Label conditioning does not help; "
              "close it.")
print("=" * 54)
PY

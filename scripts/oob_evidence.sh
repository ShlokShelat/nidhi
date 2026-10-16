#!/bin/bash
#SBATCH --job-name=oobev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x_%j.out

# AN EVIDENCE INSTRUMENT THAT MATCHES DEPLOYMENT.
#
# Deployment scores a TEST post with an ENSEMBLE of 5 fold models. The ordinary
# OOF instrument scores each post with the ONE model that held it out. Arm
# comparisons, decode sweeps and thresholds fitted that way therefore rank
# SINGLE models and are then applied to an ENSEMBLE, which is a different
# estimator: ensembling raises every component by more than the effects being
# ranked, so the ordering measured on single models need not survive.
#
# This trains B bagged models (each on a random 80% of AUTHORS) and scores
# every post with the ~5 models that excluded its author. That is a clean
# 5-model ensemble per post -- the same shape deployment uses -- so an A/B
# measured here is a measurement of the system that actually ships.
#
# COST. B=25 at ~35 min is roughly 15h in one job, the same order as
# evidence_5fold.sh.
#
# RISK-CONDITIONED BY DEFAULT, because the deployed evidence column is
# risk-conditioned. Training conditions on GOLD risk, matching how the deployed
# taggers were trained; inference conditions on PREDICTED risk, because that is
# all that exists at test time. Set RC=0 for the unconditioned family.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

BAGS="${BAGS:-25}"
RC="${RC:-1}"
TAG="${TAG:-oobrc}"
EXTRA="${EXTRA:---max-len 512 --epochs 6 --lr 1e-5 --batch 8 --grad-accum 2}"
RISKS="${RISKS:-artifacts/nligaur/oof_base.csv}"

if [ ! -f artifacts/oob/train_b0.csv ]; then
    echo "ERROR: run  python src/oob_bags.py --bags ${BAGS}  first."
    exit 1
fi

COND=""
[ "${RC}" = "1" ] && COND="--risk-cond"

for b in $(seq 0 $((BAGS-1))); do
    dir="runs/oob_${TAG}_b${b}"
    if [ -f "${dir}/model_final.pt" ]; then
        echo "--- ${TAG} bag ${b}: exists ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "${TAG} bag ${b} / ${BAGS}   ${COND} ${EXTRA}"
    echo "================================================================"
    python src/train_evidence.py \
        --train "artifacts/oob/train_b${b}.csv" \
        --val   "artifacts/oob/val_b${b}.csv" \
        --model microsoft/deberta-v3-large \
        --outdir "${dir}" --num-workers 0 ${COND} ${EXTRA} \
        || { echo "FAILED ${TAG} bag ${b}"; exit 1; }
done

echo ""
echo "================================================================"
echo "DUMP OUT-OF-BAG CANDIDATES -- each post scored by the models that"
echo "EXCLUDED its author, which is the honest ensemble"
echo "================================================================"
TAG="${TAG}" BAGS="${BAGS}" RC="${RC}" RISKS="${RISKS}" python - <<'PY'
import json, os, sys
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from scoring import normalize_label
from train_evidence import RISK_PREFIX, TaggerModel, _decode_candidates

TAG = os.environ["TAG"]; B = int(os.environ["BAGS"])
RC = os.environ["RC"] == "1"; RISKS = os.environ["RISKS"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
rk = {}
if RC:
    r = pd.read_csv(RISKS, keep_default_na=False)
    r["row_id"] = r["row_id"].astype(str)
    col = next(c for c in ("pred_risk_level", "risk_level") if c in r.columns)
    rk = dict(zip(r["row_id"], r[col]))

rows = []
for b in range(B):
    d = f"runs/oob_{TAG}_b{b}"
    cfg = json.load(open(f"{d}/run_config.json"))
    ml = int(cfg.get("max_len", 512))
    va = pd.read_csv(f"artifacts/oob/val_b{b}.csv", keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    tok = AutoTokenizer.from_pretrained(cfg.get("model",
                                                "microsoft/deberta-v3-large"))
    m = TaggerModel(cfg.get("model", "microsoft/deberta-v3-large"))
    m.load_state_dict(torch.load(f"{d}/model_final.pt", map_location="cpu"))
    m.to(dev).eval()
    n = 0
    with torch.no_grad():
        for rid, post in zip(va["row_id"], va["post"].astype(str)):
            if RC:
                enc = tok(RISK_PREFIX.format(normalize_label(rk[rid])), post,
                          truncation="only_second", max_length=ml,
                          return_offsets_mapping=True, return_tensors="pt")
                seq = enc.sequence_ids(0)
            else:
                enc = tok(post, truncation=True, max_length=ml,
                          return_offsets_mapping=True, return_tensors="pt")
                seq = None
            off = enc.pop("offset_mapping").squeeze(0).tolist()
            enc = {k: v.to(dev) for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")}
            pr = torch.softmax(m(**enc)["logits"].squeeze(0).float(),
                               -1).cpu().numpy()
            if seq is not None:
                keep = [t for t in range(len(off)) if seq[t] == 1]
                pr = pr[keep]; off = [off[t] for t in keep]
            for s, c in _decode_candidates(post, pr, off):
                rows.append((rid, s, float(c), cfg.get("evidence_min_conf",
                                                       0.0), b))
                n += 1
    del m; torch.cuda.empty_cache()
    print(f"  bag {b}: {len(va)} posts, {n} candidates", flush=True)

out = pd.DataFrame(rows, columns=["row_id", "span", "conf", "min_conf", "bag"])
dst = f"artifacts/evidence_oof/{TAG}_oof_candidates.csv"
out.to_csv(dst, index=False)
per = out.groupby("row_id")["bag"].nunique()
print(f"\nwrote {dst}  {len(out)} candidates over {per.size} posts")
print(f"clean models contributing per post: median {int(per.median())}  "
      f"mean {per.mean():.1f}")
print("The `bag` column IS the voter identity -- vote across bags exactly as")
print("deployment votes across folds.")
PY

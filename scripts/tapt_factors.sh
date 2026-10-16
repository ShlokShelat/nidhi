#!/bin/bash
#SBATCH --job-name=tapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# TASK-ADAPTIVE PRETRAINING, then a factor arm on the adapted backbone.
#
# WHY THIS AND NOT ANOTHER LOSS OR ANOTHER HYPOTHESIS. Gururangan et al.,
# ACL 2020 (arXiv:2004.10964) is the strongest published result at this sample
# size: on ACL-ARC -- 1,688 labeled examples, imbalanced, essentially the 1,635
# here -- RoBERTa goes 63.0 -> 75.4 macro-F1 with domain-adaptive pretraining,
# and TAPT alone (MLM on the task's own unlabeled text) reaches 67.4, +4.4.
#
# It also acts on a different part of the model from the other factor arms: a
# hypothesis rewrite changes the QUESTION asked, CB-NTR changes the LOSS, this
# changes the REPRESENTATION. A successful backbone is reusable by every later
# arm.
#
# NO EXTERNAL DATA. MLM runs on the text of `artifacts/train_split.csv` +
# `artifacts/val_split.csv` only -- posts already held, labels unused. The
# competition requires external LABELLED data to be declared; this uses none,
# so there is nothing to declare. data/test.csv is deliberately EXCLUDED:
# transductive MLM on the eval set would have to be re-run by the organizers on
# their own unseen authors, which the cold-run reproduction does not cover.
#
# THE HONEST PRIOR. Gururangan's numbers are single-label classification with a
# linear head; this is a 24-label NLI cross-encoder. The mechanism (better
# in-domain representations) transfers, the magnitude need not. Judge the arm
# the way every other arm is judged -- as an ADD on the deployed set, on
# author-disjoint win rate, against the dilution cost of a further member.
# ===========================================================================

source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

# these change what hypothesis the factor trainer asks; a leaked value from the
# login shell would silently make this a different experiment
unset FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON PROTECTIVE_TEMPLATE \
      CONFUSABLE_JSON FACTOR_SUBSET 2>/dev/null || true

set -u
BASE="${BASE:-microsoft/deberta-v3-large}"
OUT="${OUT:-runs/tapt_dv3}"
TAG="${TAG:-tapt}"
EPOCHS_MLM="${EPOCHS_MLM:-10}"
MLM_LR="${MLM_LR:-5e-5}"
MLM_BS="${MLM_BS:-8}"
MAXLEN="${MAXLEN:-320}"

echo "node $(hostname)  $(date)"
echo "base ${BASE} -> ${OUT}   mlm epochs ${EPOCHS_MLM} lr ${MLM_LR}"
df -h . | tail -1

for f in artifacts/train_split.csv artifacts/val_split.csv artifacts/folds/train_f0.csv; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done

# ---------------------------------------------------------------------------
if [ -f "${OUT}/config.json" ]; then
    echo "--- ${OUT} exists, skipping MLM ---"
else
python - <<PY
import os, math
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

BASE, OUT = "${BASE}", "${OUT}"
EPOCHS, LR, BS, ML = ${EPOCHS_MLM}, ${MLM_LR}, ${MLM_BS}, ${MAXLEN}

# TASK text only -- both labelled splits, labels unused, test EXCLUDED.
texts = []
for p in ("artifacts/train_split.csv", "artifacts/val_split.csv"):
    d = pd.read_csv(p, keep_default_na=False)
    texts += [str(x) for x in d["post"] if str(x).strip()]
print(f"MLM corpus: {len(texts)} posts (train_split + val_split, no test)")

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForMaskedLM.from_pretrained(BASE)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev).train()

class DS(Dataset):
    def __init__(self, t): self.t = t
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = tok(self.t[i], truncation=True, max_length=ML, padding="max_length")
        return {k: torch.tensor(v) for k, v in e.items()
                if k in ("input_ids", "attention_mask")}

coll = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=0.15)
dl = DataLoader(DS(texts), batch_size=BS, shuffle=True, collate_fn=coll)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
tot = len(dl) * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(0.1 * tot), tot)
use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
print(f"{len(dl)} steps/epoch x {EPOCHS} epochs, bf16={use_bf16}")

for ep in range(EPOCHS):
    run = []
    for i, b in enumerate(dl):
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast(device_type=dev, dtype=torch.bfloat16,
                            enabled=use_bf16):
            loss = model(**b).loss
        if not torch.isfinite(loss):
            raise SystemExit(f"NON-FINITE MLM LOSS at epoch {ep+1} step {i}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad()
        run.append(loss.item())
        if (i + 1) % 50 == 0:
            print(f"  epoch {ep+1} step {i+1} loss {np.mean(run[-50:]):.4f}")
    m = float(np.mean(run))
    print(f"== epoch {ep+1}: MLM loss {m:.4f}  perplexity {math.exp(min(m,20)):.2f}")

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"wrote {OUT}")
PY
    [ $? -eq 0 ] || { echo "MLM FAILED"; exit 1; }
fi

df -h . | tail -1

# ---------------------------------------------------------------------------
# The factor arm on the adapted backbone. AutoModel.from_pretrained will load
# the encoder and drop the LM head with a warning -- that is expected and
# correct; the LM head was only the pretraining objective.
echo ""
echo "################################################################"
echo "FACTOR ARM on the TAPT backbone"
echo "################################################################"
MODEL="${OUT}" TAG="${TAG}" LR="${LR:-1e-5}" bash scripts/factors_encoder.sh

cat <<'NEXT'

JUDGE IT LIKE ANY OTHER ARM -- nothing about TAPT earns it a pass:
  python src/factor_member_forward.py --extra ltail,friendfac,cbnsynapi,tapt | tail -18
Bar: an add must clear the fixed per-arm dilution tax and reach >=90% win on
HELD-OUT authors.

Also read the arm's SOLO per-label table against the incumbent. TAPT should
help most where in-domain vocabulary matters -- the tail labels. If the mean
moves but the tail does not, the representation did not do what the paper
says it does, and the result is a seed effect (run-to-run variance at fixed
seed is +-0.025, so treat anything under 0.03 as noise).
NEXT

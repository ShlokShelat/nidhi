"""
Bisect the NaN. Runs in ~30s. Usage (any node, GPU preferred):
  python src/debug_nan.py
  python src/debug_nan.py microsoft/deberta-v3-base artifacts/train_split.csv

Stages:
  1) tokenizer sanity: are token ids within the vocab?
  2) bare pretrained encoder forward: NaN in hidden states?
  3) DualHeadModel forward: NaN in loss?
  4) backward: NaN in gradients, and of which parameters?

Interpretation:
  fails at 1 or 2 -> the transformers installation is at fault (a pre-release
                     v5 mishandling DeBERTa does this); fix = pinned stable env.
  fails at 3 or 4 -> bug in the model code here; report which parameters.
"""

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no hub calls; cache only (override: HF_HUB_OFFLINE=0)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from scoring import RISK_LEVELS, normalize_label
from train_risk import DualHeadModel

model_name = sys.argv[1] if len(sys.argv) > 1 else "microsoft/deberta-v3-base"
train_csv = sys.argv[2] if len(sys.argv) > 2 else "artifacts/train_split.csv"

print(f"torch {torch.__version__} | transformers {transformers.__version__} "
      f"| cuda={torch.cuda.is_available()}")

tok = AutoTokenizer.from_pretrained(model_name)
df = pd.read_csv(train_csv).head(8)
texts = df["post"].astype(str).tolist()

# ---- 1) tokenizer ----
enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
max_id = enc["input_ids"].max().item()
vocab = getattr(tok, "vocab_size", None)
emb_note = ""
print(f"1) max input_id = {max_id} | tokenizer.vocab_size = {vocab}")
sample = tok.convert_ids_to_tokens(enc["input_ids"][0][:20].tolist())
print(f"   first tokens of post 0: {sample}")

device = "cuda" if torch.cuda.is_available() else "cpu"
ids = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

# ---- 2) bare encoder ----
base = AutoModel.from_pretrained(model_name).to(device).eval()
emb_size = base.get_input_embeddings().weight.shape[0]
print(f"   model embedding rows = {emb_size} "
      f"{'<-- IDS OUT OF RANGE!' if max_id >= emb_size else '(ids in range)'}")
with torch.no_grad():
    h = base(**ids).last_hidden_state
print(f"2) bare encoder hidden states: nan={torch.isnan(h).any().item()} "
      f"inf={torch.isinf(h).any().item()} max|h|={h.abs().max().item():.2f}")
del base

# ---- 3) DualHeadModel forward ----
label_ids = torch.tensor(
    [ {l: i for i, l in enumerate(RISK_LEVELS)}[normalize_label(x)]
      for x in df["risk_level"] ]).to(device)
m = DualHeadModel(model_name).to(device)
m.train()
out = m(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
        labels=label_ids)
loss = out["loss"]
print(f"3) DualHeadModel: loss={loss.item():.4f} "
      f"logits_nan={torch.isnan(out['logits']).any().item()}")

# ---- 4) backward ----
loss.backward()
gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)
bad = [n for n, p in m.named_parameters()
       if p.grad is not None and torch.isnan(p.grad).any()]
print(f"4) grad_norm={gn.item():.4f} | params with nan grads: {len(bad)}")
for n in bad[:10]:
    print(f"   nan grad: {n}")

if max_id >= emb_size or torch.isnan(h).any():
    print("\nVERDICT: broken at stage 1/2 -> transformers install, not model code.")
elif torch.isnan(loss) or bad:
    print("\nVERDICT: broken at stage 3/4 -> model code; paste this output.")
else:
    print("\nVERDICT: all healthy on this batch -> NaN is data-dependent or "
          "Trainer-specific; paste this output.")

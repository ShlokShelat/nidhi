"""
Replay the poison batch through the EXACT training path and localize the
first non-finite tensor.

Usage (GPU node):
  python src/batch_repro.py runs/risk_v0_manual/poison_batch.csv

Tests, for bf16-autocast and fp32 separately:
  encoder hidden -> pooled -> cat/ord logits -> CE -> CORN -> total loss
  -> backward -> gradient norms
"""

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no hub calls; cache only (override: HF_HUB_OFFLINE=0)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scoring import RISK_LEVELS, normalize_label
from train_risk import DualHeadModel, PostDataset

csv_path = sys.argv[1] if len(sys.argv) > 1 else "runs/risk_v0_manual/poison_batch.csv"
model_name = sys.argv[2] if len(sys.argv) > 2 else "microsoft/deberta-v3-base"

df = pd.read_csv(csv_path)
print("labels in batch:", [normalize_label(x) for x in df["risk_level"]])

tok = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
ds = PostDataset(df, tok, max_len=512, use_user_context=True)
batch = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False)))
batch.pop("idx", None)
batch = {k: v.to(device) for k, v in batch.items()}
labels = batch["labels"]


def chk(name, t):
    bad = bool(torch.isnan(t).any() or torch.isinf(t).any())
    mx = t.abs().max().item() if t.numel() else float("nan")
    print(f"   {name:12s} finite={not bad}  max|x|={mx:.4g}")
    return bad


for prec_name, use_bf16 in [("bf16", device == "cuda"), ("fp32", False)]:
    if prec_name == "bf16" and device != "cuda":
        continue
    print(f"\n===== {prec_name} =====")
    torch.manual_seed(13)
    model = DualHeadModel(model_name).to(device)
    model.train()

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        out_enc = model.encoder(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
        h = out_enc.last_hidden_state
        bad = chk("hidden", h)
        pooled = model.dropout(h[:, 0]).to(model.cat_head.weight.dtype)
        bad |= chk("pooled", pooled)
        cat_logits = model.cat_head(pooled)
        ord_logits = model.ord_head(pooled)
        bad |= chk("cat_logits", cat_logits)
        bad |= chk("ord_logits", ord_logits)
        ce = nn.CrossEntropyLoss()(cat_logits, labels)
        corn = model._corn_loss(ord_logits, labels)
        print(f"   CE={ce.item():.6f}  CORN={corn.item():.6f}")
        loss = 0.5 * ce + 0.5 * corn
        print(f"   loss={loss.item():.6f}")

    if torch.isfinite(loss):
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        nan_params = [n for n, p in model.named_parameters()
                      if p.grad is not None and torch.isnan(p.grad).any()]
        print(f"   backward: grad_norm={gn.item():.4f}  "
              f"nan-grad params={len(nan_params)}")
        for n in nan_params[:6]:
            print(f"      {n}")
    else:
        print("   loss non-finite; skipping backward")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

print("\nIf bf16 fails where fp32 is clean: the offending op is a bf16 "
      "overflow, and --precision fp32 is cheap at this scale. If BOTH are "
      "clean here: the failure needs optimizer state/history -> next suspect "
      "is the earlier batches shifting the weights before this one arrives.")

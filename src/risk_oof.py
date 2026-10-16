"""
User-level k-fold OOF risk prediction.

Needed for risk-conditioned factors: if the factor model were conditioned on
risk predictions from a risk model that TRAINED on those same posts, the
factor model would see unrealistically confident/accurate risk labels
during training but only realistic (noisier) ones at test time — a
leakage-shaped mismatch that would make the conditioning look better in
training than it actually helps.

This trains a lightweight 3-fold risk model (fewer folds/epochs than the
full CV for speed — this is a supporting signal, not the final risk model)
and produces honest OOF risk predictions for every train post, using the
SAME user-level fold logic as train_factors_cv.py (imported, not
reimplemented) so the two CV harnesses can't disagree about fold
membership.

Usage:
  python src/risk_oof.py --train artifacts/train_split.csv \
      --model microsoft/deberta-v3-large --k 3 --epochs 4 \
      --out artifacts/train_split_riskcond.csv
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS
from train_risk import PostDataset, DualHeadModel, build_text
from train_factors_cv import user_folds  # reuse the SAME fold assignment logic


def train_fold_risk(tr_df, model_name, max_len, epochs, lr, batch,
                    grad_accum, device, use_bf16, seed):
    tok = AutoTokenizer.from_pretrained(model_name)
    tr_ds = PostDataset(tr_df, tok, max_len, True)  # use_user_context=True
    loader = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0)
    model = DualHeadModel(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = (len(loader) // grad_accum + 1) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    torch.manual_seed(seed)
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        for b, batch_d in enumerate(loader):
            batch_d.pop("idx", None)
            batch_d = {k: v.to(device) for k, v in batch_d.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch_d)["loss"]
            (loss / grad_accum).backward()
            if (b + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
    return model, tok


@torch.no_grad()
def predict_fold(model, tok, df, max_len, device, use_bf16):
    model.eval()
    d = df.copy()
    d["user_post_count"] = d.groupby("anon_user_id")["post"].transform("size")
    d["post_index_in_user"] = d.groupby("anon_user_id").cumcount() + 1
    preds = []
    for _, row in d.iterrows():
        enc = tok(build_text(row, True), truncation=True, max_length=max_len,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(**enc)["logits"]
        preds.append(RISK_LEVELS[logits.argmax(-1).item()])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    df = pd.read_csv(args.train).reset_index(drop=True)
    folds = user_folds(df, args.k, args.seed)
    pred_risk = [None] * len(df)

    for f in range(args.k):
        held = folds[f]
        train_mask = np.ones(len(df), dtype=bool); train_mask[held] = False
        tr_df = df[train_mask].copy()
        print(f"\n=== fold {f+1}/{args.k}: train {len(tr_df)} (held {len(held)}) ===")
        model, tok = train_fold_risk(tr_df, args.model, args.max_len, args.epochs,
                                     args.lr, args.batch, args.grad_accum,
                                     device, use_bf16, args.seed + f)
        held_df = df.iloc[held].reset_index(drop=True)
        preds = predict_fold(model, tok, held_df, args.max_len, device, use_bf16)
        for idx, p in zip(held, preds):
            pred_risk[idx] = p
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = df.copy()
    out["pred_risk_level"] = pred_risk
    out.to_csv(args.out, index=False)
    n_match = sum(a.lower() == b.lower() for a, b in zip(out["risk_level"], out["pred_risk_level"]))
    print(f"\nOOF risk-cond accuracy on train: {n_match}/{len(out)} = {n_match/len(out):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""
compare_head_combination.py

Diagnostic, no retraining. The risk models train with alpha_ordinal=0.5
(categorical CE + CORN ordinal, averaged 50/50 at inference in
DualHeadModel._combine()). This checks whether that 50/50 average is the
best combination, or whether the categorical head is diluting a stronger
ordinal signal (or vice versa) — a question that can be settled from
stored logits, without spending a training sweep on it.

Loads the EXISTING risk_v1 checkpoint, runs the encoder once, and
scores val three ways from the SAME forward pass:
  - categorical-only  (argmax of cat_head logits)
  - ordinal-only      (argmax of CORN-derived cumulative class probs)
  - combined (current) (argmax of 0.5*cat_p + 0.5*ord_p — what's deployed)

Usage:
  python src/compare_head_combination.py \
      --val artifacts/val_split.csv \
      --checkpoint runs/risk_v1
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, risk_weighted_f1
from train_risk import DualHeadModel, PostDataset

ORDER = {l: i for i, l in enumerate(RISK_LEVELS)}


@torch.no_grad()
def get_raw_logits(model, loader, device, use_bf16):
    model.eval()
    all_cat, all_ord, all_labels = [], [], []
    for batch in loader:
        batch.pop("idx")
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model.encoder(**batch)
            pooled = model.dropout(out.last_hidden_state[:, 0])
            pooled = pooled.to(model.cat_head.weight.dtype)
            cat_logits = model.cat_head(pooled)
            ord_logits = model.ord_head(pooled)
        all_cat.append(cat_logits.float().cpu())
        all_ord.append(ord_logits.float().cpu())
        all_labels.append(labels)
    return torch.cat(all_cat), torch.cat(all_ord), torch.cat(all_labels)


def ordinal_class_probs(ord_logits):
    cond = torch.sigmoid(ord_logits)
    cum = torch.cumprod(cond, dim=-1)
    ones = torch.ones_like(cum[:, :1])
    upper = torch.cat([ones, cum], dim=-1)
    lower = torch.cat([cum, torch.zeros_like(cum[:, :1])], dim=-1)
    return (upper - lower).clamp_min(1e-8)


def confusion_breakdown(gold, pred):
    adjacent, far, correct = 0, 0, 0
    for g, p in zip(gold, pred):
        if g == p:
            correct += 1
            continue
        dist = abs(ORDER[g] - ORDER[p])
        if dist == 1:
            adjacent += 1
        else:
            far += 1
    return correct, adjacent, far


def report(name, gold, pred):
    f1 = risk_weighted_f1(gold, pred)
    correct, adjacent, far = confusion_breakdown(gold, pred)
    total = len(gold)
    print(f"\n{name}")
    print(f"  weighted-F1: {f1:.4f}")
    print(f"  correct: {correct}/{total}   adjacent-errors: {adjacent}   far-errors: {far}")
    print(f"  prediction distribution: {dict(Counter(pred))}")
    return f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--checkpoint", required=True, help="e.g. runs/risk_v1")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    with open(os.path.join(args.checkpoint, "run_config.json")) as f:
        cfg = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")
    print(f"checkpoint config: model={cfg['model']} alpha_ordinal={cfg['alpha_ordinal']} "
         f"user_context={cfg['user_context']}")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    va_df = pd.read_csv(args.val)

    from torch.utils.data import DataLoader
    va_loader = DataLoader(
        PostDataset(va_df, tok, cfg["max_len"], cfg["user_context"]),
        batch_size=args.batch, shuffle=False, num_workers=0)

    model = DualHeadModel(cfg["model"], alpha_ordinal=cfg["alpha_ordinal"]).to(device)
    state = torch.load(os.path.join(args.checkpoint, "model_final.pt"), map_location=device)
    model.load_state_dict(state, strict=False)

    cat_logits, ord_logits, labels = get_raw_logits(model, va_loader, device, use_bf16)
    gold = [RISK_LEVELS[i] for i in labels.tolist()]

    ord_p = ordinal_class_probs(ord_logits)
    cat_p = torch.softmax(cat_logits, dim=-1)

    # deployed baseline (alpha=0.5) computed first so every other alpha can
    # report how many predictions it actually changes vs what's live now
    deployed_p = 0.5 * cat_p + 0.5 * ord_p
    deployed_pred = [RISK_LEVELS[i] for i in deployed_p.argmax(-1).tolist()]

    print("\n" + "=" * 90)
    print(f"ALPHA SWEEP on {args.checkpoint} (val, n={len(gold)})")
    print(f"  combined_p = alpha * cat_p + (1-alpha) * ord_p")
    print(f"  alpha=1.0 -> pure categorical | alpha=0.0 -> pure ordinal | alpha=0.5 -> CURRENT deployed")
    print("=" * 90)

    alphas = [round(a, 2) for a in np.arange(0.0, 1.001, 0.05)]
    results = []
    print(f"\n{'alpha':>7}{'weighted-F1':>14}{'correct':>10}{'adjacent':>10}{'far':>8}{'chg_vs_deploy':>15}")
    for alpha in alphas:
        combined_p = alpha * cat_p + (1 - alpha) * ord_p
        pred = [RISK_LEVELS[i] for i in combined_p.argmax(-1).tolist()]
        f1 = risk_weighted_f1(gold, pred)
        correct, adjacent, far = confusion_breakdown(gold, pred)
        changed = sum(p1 != p2 for p1, p2 in zip(pred, deployed_pred))
        tag = "  <- deployed" if alpha == 0.5 else ""
        print(f"{alpha:>7.2f}{f1:>14.4f}{correct:>10}{adjacent:>10}{far:>8}{changed:>15}{tag}")
        results.append((alpha, f1, correct, adjacent, far, changed))

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    best_alpha, best_f1, _, _, _, best_changed = max(results, key=lambda r: r[1])
    deployed_f1 = next(r[1] for r in results if r[0] == 0.5)
    print(f"  best alpha: {best_alpha:.2f} (weighted-F1 {best_f1:.4f}, "
         f"{best_changed} predictions differ from deployed alpha=0.5)")
    print(f"  deployed alpha=0.5: weighted-F1 {deployed_f1:.4f}")
    print(f"  delta vs deployed: {best_f1 - deployed_f1:+.4f}")
    if best_alpha != 0.5 and best_f1 - deployed_f1 > 0.005:
        direction = "MORE ordinal (lower alpha)" if best_alpha < 0.5 else "MORE categorical (higher alpha)"
        if best_changed < 10:
            print(f"\n  -> gain comes from only {best_changed} changed predictions — "
                 f"a handful of flips on a few hundred posts is inside the "
                 f"noise floor. Treat with skepticism even though it looks positive.")
        else:
            print(f"\n  -> real signal that leaning {direction} may help "
                 f"({best_changed} predictions changed, above the noise floor). "
                 f"Worth testing this same sweep on risk_v1_s14 and risk_v1_s15 "
                 f"to see if all 3 seeds agree before acting on it.")
    else:
        print(f"\n  -> alpha=0.5 (current) is already near-optimal, or differences "
             f"are within noise. Head-combination is likely not the lever here.")


if __name__ == "__main__":
    main()

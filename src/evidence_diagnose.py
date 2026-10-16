"""
Evidence error diagnosis — where exactly is phrase-F1 lost?

Evidence carries 0.3 of the composite (same as factors) and the val
phrase-F1 is 0.6902 with the threshold already optimal, so further gains
must come from changing WHAT is emitted, not the cutoff. This script
localizes the loss.

Scoring reminder: phrase-F1 is computed PER POST then averaged.
  * gold empty + pred empty        -> 1.0 (free point for correct "none")
  * gold empty + pred non-empty    -> 0.0 (pure precision loss)
  * gold non-empty + pred empty    -> 0.0 (pure recall loss)
  * both non-empty                 -> containment matching, 3x length cap,
                                      one-to-one greedy
So the two catastrophic buckets are the mismatched-empty cases: each costs
a full 1.0 on that post. If many of the 147 empty-prediction posts have
gold spans, that alone caps the ceiling.

Usage:
  python src/evidence_diagnose.py --val artifacts/val_split.csv \
      --run runs/evidence_v2
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

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import normalize_span, parse_evidence, phrase_f1
from train_evidence import TaggerModel, collect_candidates, spans_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(args.run, "run_config.json")))
    stored = cfg.get("evidence_min_conf", 0.65)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = TaggerModel(cfg["model"])
    model.load_state_dict(
        torch.load(os.path.join(args.run, "model_final.pt"), map_location=device),
        strict=False)
    model.to(device).eval()

    df = load(args.val)
    posts = df["post"].astype(str).tolist()
    gold = [parse_evidence(c) for c in df["evidence"]]
    cands = collect_candidates(model, tok, posts, args.max_len, device)

    print(f"val posts: {len(posts)}, stored min_conf={stored}\n")

    # ---- bucket analysis at the operating threshold ----
    pred = spans_at(cands, stored)
    both_empty = sum(1 for g, p in zip(gold, pred) if not g and not p)
    gold_empty_pred_not = sum(1 for g, p in zip(gold, pred) if not g and p)
    pred_empty_gold_not = sum(1 for g, p in zip(gold, pred) if g and not p)
    both_full = sum(1 for g, p in zip(gold, pred) if g and p)
    print("=== per-post buckets at operating threshold ===")
    print(f"  both empty (free 1.0):            {both_empty}")
    print(f"  gold empty, predicted non-empty:  {gold_empty_pred_not}  <- precision loss")
    print(f"  gold present, predicted none:     {pred_empty_gold_not}  <- recall loss")
    print(f"  both non-empty (scored):          {both_full}")
    lost = gold_empty_pred_not + pred_empty_gold_not
    print(f"  => {lost} posts score a hard 0.0 purely from empty/non-empty mismatch")
    print(f"     (max recoverable F1 from fixing those: +{lost/len(posts):.4f})\n")

    # score on only the both-non-empty subset, to isolate matching quality
    idx = [i for i in range(len(posts)) if gold[i] and pred[i]]
    if idx:
        sub = phrase_f1([gold[i] for i in idx], [pred[i] for i in idx]).f1
        print(f"  phrase-F1 restricted to both-non-empty posts: {sub:.4f}")
        print("  (if this is high, span MATCHING is fine and the loss is"
              " concentrated in the empty/non-empty decision)\n")

    # ---- span-count comparison ----
    g_tot = sum(len(g) for g in gold)
    p_tot = sum(len(p) for p in pred)
    print("=== span counts ===")
    print(f"  gold spans: {g_tot} ({g_tot/len(posts):.2f}/post)")
    print(f"  pred spans: {p_tot} ({p_tot/len(posts):.2f}/post)")
    multi_gold = sum(1 for g in gold if len(g) > 1)
    print(f"  posts with >1 gold span: {multi_gold}")
    under = sum(1 for g, p in zip(gold, pred) if g and p and len(p) < len(g))
    print(f"  posts with FEWER predicted spans than gold: {under}\n")

    # ---- threshold curve with P/R breakdown ----
    print("=== threshold sweep (F1 / how many posts get a hard 0) ===")
    for t in np.arange(0.30, 0.86, 0.05):
        p_t = spans_at(cands, t)
        f1 = phrase_f1(gold, p_t).f1
        zeros = sum(1 for g, p in zip(gold, p_t)
                    if (not g and p) or (g and not p))
        tot = sum(len(p) for p in p_t)
        print(f"  min_conf={t:.2f}: F1={f1:.4f}  hard-zeros={zeros:3d}  spans={tot}")


if __name__ == "__main__":
    main()

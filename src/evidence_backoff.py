"""
Evidence: fix the empty/non-empty decision (the real loss center).

Diagnosis showed span MATCHING is fine (0.7485 on posts where both
gold and prediction are non-empty), but 41/372 posts score a hard 0.0
purely because one side is empty and the other isn't:
  * 22 posts: gold has spans, none emitted    -> recall catastrophe
  * 19 posts: gold empty, something emitted   -> precision catastrophe

This tests a BACKOFF strategy: keep the tuned threshold for normal span
selection, but when NO candidate clears it, fall back to emitting the
single best candidate if it clears a lower floor. That converts some of
the 22 zero-scoring posts into positive scores. The risk is turning some
of the 137 correctly-empty posts (free 1.0 each) into zeros, so the floor
is swept and the net effect measured.

It ALSO reimplements phrase-F1 exactly as the official task describes it
(per-post F1, then averaged; 1.0 when both empty; 0.0 when exactly one is
empty; containment matching with 3x length cap and one-to-one greedy
assignment) and reports both the repo's scorer and this one side by side.
The repo's scorer appears to micro-average, which would understate how much
the hard-zero posts actually cost.

Usage:
  python src/evidence_backoff.py --val artifacts/val_split.csv \
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
from scoring import parse_evidence, phrase_f1
from train_evidence import TaggerModel, collect_candidates, spans_at


def _norm(s):
    return " ".join(str(s).lower().split())


def _ntok(s):
    return len(_norm(s).split())


def official_post_f1(gold, pred):
    """Per-post phrase F1 exactly as the task describes."""
    g = [_norm(x) for x in gold if _norm(x)]
    p = [_norm(x) for x in pred if _norm(x)]
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    used, matched = set(), 0
    for pi in p:
        for gi, gv in enumerate(g):
            if gi in used:
                continue
            if (gv in pi or pi in gv) and _ntok(pi) <= 3 * _ntok(gv):
                used.add(gi)
                matched += 1
                break
    if matched == 0:
        return 0.0
    prec, rec = matched / len(p), matched / len(g)
    return 2 * prec * rec / (prec + rec)


def official_f1(gold_lists, pred_lists):
    return float(np.mean([official_post_f1(g, p)
                          for g, p in zip(gold_lists, pred_lists)]))


def apply_backoff(cands, t_hi, t_lo, max_extra=1):
    """Spans at t_hi; if a post has none, take up to max_extra best
    candidates that clear t_lo."""
    out = []
    for post_c in cands:
        keep = [s for s, c in post_c if c >= t_hi]
        if not keep and t_lo is not None:
            fallback = sorted([(c, s) for s, c in post_c if c >= t_lo],
                              reverse=True)[:max_extra]
            keep = [s for _, s in fallback]
        out.append(keep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(args.run, "run_config.json")))
    t_hi = cfg.get("evidence_min_conf", 0.65)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = TaggerModel(cfg["model"])
    model.load_state_dict(
        torch.load(os.path.join(args.run, "model_final.pt"), map_location=device),
        strict=False)
    model.to(device).eval()

    df = load(args.val)
    gold = [parse_evidence(c) for c in df["evidence"]]
    cands = collect_candidates(model, tok, df["post"].astype(str).tolist(),
                               args.max_len, device)

    base = spans_at(cands, t_hi)
    print("=== scorer comparison on current output ===")
    print(f"  repo phrase_f1 (likely micro): {phrase_f1(gold, base).f1:.4f}")
    print(f"  official per-post macro:       {official_f1(gold, base):.4f}")
    print("  (if these differ materially, optimize against the official one)\n")

    baseline = official_f1(gold, base)

    # min_conf was tuned against the MICRO scorer; re-sweep t_hi against the
    # OFFICIAL per-post macro metric, jointly with the backoff floor and how
    # many fallback spans to emit.
    print("=== joint sweep: t_hi x t_lo x max_extra (OFFICIAL macro) ===")
    print(f"  {'t_hi':>5}  {'t_lo':>5}  {'extra':>5}  {'official':>9}  "
          f"{'repo':>7}  {'zeros':>5}  {'spans':>6}")
    best = (baseline, t_hi, None, 1)
    for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for tl in [None, 0.40, 0.30, 0.20, 0.10, 0.0]:
            for mx in [1, 2]:
                if tl is None and mx != 1:
                    continue
                pred = apply_backoff(cands, th, tl, mx)
                off = official_f1(gold, pred)
                if off < baseline - 0.02:
                    continue  # keep the table readable
                rep = phrase_f1(gold, pred).f1
                zeros = sum(1 for g, p in zip(gold, pred)
                            if (not g and p) or (g and not p))
                spans = sum(len(p) for p in pred)
                lab = "none" if tl is None else f"{tl:.2f}"
                star = " *" if off > best[0] else ""
                print(f"  {th:>5.2f}  {lab:>5}  {mx:>5}  {off:>9.4f}  "
                      f"{rep:>7.4f}  {zeros:>5}  {spans:>6}{star}")
                if off > best[0]:
                    best = (off, th, tl, mx)

    print(f"\nBASELINE (t_hi={t_hi}, no backoff): official {baseline:.4f}")
    print(f"BEST: official {best[0]:.4f} at t_hi={best[1]} t_lo={best[2]} "
          f"max_extra={best[3]}  (delta {best[0]-baseline:+.4f})")
    print(f"\nEstimated composite gain: {0.3*(best[0]-baseline):+.4f}")


if __name__ == "__main__":
    main()

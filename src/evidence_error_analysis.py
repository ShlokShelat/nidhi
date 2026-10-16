"""
evidence_error_analysis.py

Applies the symbolic-discovery methodology to the EVIDENCE component, which
carries the same 0.3 composite weight as factors and has never been analysed
this way.

The official phrase metric scores a predicted span as matching a gold span
when one CONTAINS the other (after normalisation) and the longer is at most
3x the shorter, with one-to-one greedy assignment. That means there are three
very different ways to lose a point, and they need completely different fixes:

  NEAR-MISS (overlap, fails the rule)  -> fixable by post-processing: the
      model found the right region but the boundary is wrong, usually too
      long (blows the 3x cap) or too short (fails containment).
  WRONG REGION (no overlap at all)     -> a model error, not a boundary error.
  NOTHING EMITTED / SPURIOUS           -> a decision error about whether the
      post has evidence at all.

Only the first class is reachable by a symbolic layer. This script measures
how big each class is before any effort is spent.

Usage:
  python src/evidence_error_analysis.py
  python src/evidence_error_analysis.py --show 25
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, normalize_span, token_len, _span_pair_matches


def overlaps(a, b):
    """Loose token-level overlap, independent of the official match rule."""
    ta, tb = set(normalize_span(a).split()), set(normalize_span(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--pred", default="runs/evidence_v2/val_predictions.csv")
    ap.add_argument("--pred-col", default="pred_evidence")
    ap.add_argument("--overlap-min", type=float, default=0.5)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    val = pd.read_csv(args.val)[["row_id", "post", "evidence"]]
    pr = pd.read_csv(args.pred)[["row_id", args.pred_col]]
    df = val.merge(pr, on="row_id")

    stats = {"matched": 0, "near_too_long": 0, "near_too_short": 0,
             "near_other": 0, "wrong_region": 0, "missed_no_pred": 0}
    spurious = 0
    near_examples, ratios = [], []

    for _, r in df.iterrows():
        gold = parse_evidence(r["evidence"])
        pred = parse_evidence(r[args.pred_col])

        used = set()
        for g in gold:
            hit = None
            for i, p in enumerate(pred):
                if i in used:
                    continue
                if _span_pair_matches(p, g):
                    hit = i
                    break
            if hit is not None:
                used.add(hit)
                stats["matched"] += 1
                continue
            if not pred:
                stats["missed_no_pred"] += 1
                continue
            # best overlapping unused prediction
            best, bo = None, 0.0
            for i, p in enumerate(pred):
                if i in used:
                    continue
                o = overlaps(p, g)
                if o > bo:
                    best, bo = i, o
            if best is None or bo < args.overlap_min:
                stats["wrong_region"] += 1
                continue
            p = pred[best]
            lp, lg = token_len(p), token_len(g)
            ratio = max(lp, lg) / max(min(lp, lg), 1)
            ratios.append(ratio)
            if lp > lg and ratio > 3:
                stats["near_too_long"] += 1
                kind = f"pred {lp}tok vs gold {lg}tok (ratio {ratio:.1f}, over 3x cap)"
            elif lp < lg and ratio > 3:
                stats["near_too_short"] += 1
                kind = f"pred {lp}tok vs gold {lg}tok (ratio {ratio:.1f}, over 3x cap)"
            else:
                stats["near_other"] += 1
                kind = f"pred {lp}tok vs gold {lg}tok (ratio {ratio:.1f}, containment failed)"
            near_examples.append((r["row_id"], kind, p, g, bo))

        spurious += len(pred) - len(used)

    n_gold = sum(stats.values())
    print("=" * 92)
    print("EVIDENCE ERROR DECOMPOSITION  (per GOLD span)")
    print("=" * 92)
    for k, v in stats.items():
        print(f"  {k:<18} {v:>5}  ({v / n_gold:.1%})")
    print(f"  {'TOTAL gold spans':<18} {n_gold:>5}")
    print(f"\n  spurious predicted spans (matched nothing): {spurious}")

    fixable = stats["near_too_long"] + stats["near_too_short"] + stats["near_other"]
    print(f"\n  BOUNDARY-FIXABLE (right region, wrong extent): {fixable} "
          f"({fixable / n_gold:.1%} of gold spans)")
    print(f"  MODEL-LEVEL (wrong region / nothing emitted) : "
          f"{stats['wrong_region'] + stats['missed_no_pred']}")

    if ratios:
        import statistics
        print(f"\n  length ratio on near-misses: median {statistics.median(ratios):.2f}, "
              f"max {max(ratios):.1f}")
        too_long = sum(1 for x in ratios if x > 3)
        print(f"  near-misses failing ONLY the 3x cap: {too_long}")

    print(f"\n{'=' * 92}\nNEAR-MISS EXAMPLES (the boundary-fixable class)\n{'=' * 92}")
    for rid, kind, p, g, o in near_examples[:args.show]:
        print(f"\n  {rid}  {kind}  token-overlap {o:.2f}")
        print(f"     PRED: {p[:190]}")
        print(f"     GOLD: {g[:190]}")


if __name__ == "__main__":
    main()

"""
evidence_diagnosis.py

Diagnose the evidence component. It has never been looked at.

WHY IT IS NOW THE ONLY LIVE COMPONENT.

  risk     ~0.82   LLM replacement, override and blending all measured, all lose
  factors   0.63   consistency, thresholds, an LLM and ensembling all closed
  evidence ~0.75   produced the only win of the campaign and has never been
                   diagnosed

WHAT THE METRIC REWARDS, because it is asymmetric and that asymmetry is
exploitable. From scoring.py:

    a predicted span matches a gold span if either CONTAINS the other, AND
    token_len(pred) <= 3 * token_len(gold)

Only the PREDICTION is length-capped. A short span inside a long gold span
always matches; a long span around a short gold span fails once it exceeds 3x.
So short predictions are strictly safer, and the shipped spans already average
3.5 words against gold's 4.6 -- already on the safe side and losing nothing
there.

THE HYPOTHESIS THIS TESTS. Micro-F1 with precision approx equal to recall has
a specific property: adding one more prediction is net positive whenever its
probability of matching exceeds roughly F1/2. At F1 = 0.75 that is about 0.38 --
so emitting a second span is worth it even if it is wrong 60% of the time.

Gold carries 1.80 spans per non-indicator post; the shipped decode emits 1.69. If the tagger has
a plausible runner-up on the posts where it currently emits one span, taking it
should raise F1 even though most runners-up are wrong.

That is a claim about the METRIC's shape, not a tuned threshold -- the same
category as the consistency fix that worked, and a different category from
the seven threshold changes that all failed.

This script measures whether the headroom is really there before anything is
built:
  - precision / recall / F1, and which side is limiting
  - how many posts miss entirely vs predict-but-mismatch
  - the ceiling from a perfect second span, which bounds the whole idea
  - per-risk-level breakdown, since indicator posts should have no spans at all

Usage:
  python src/evidence_diagnosis.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span, parse_evidence,
                     phrase_f1, token_len, _span_pair_matches)


def match_counts(gold, pred):
    """Maximum one-to-one matches between one post's gold and predicted spans,
    using the official pairing rule. Greedy is enough at these sizes."""
    used = set()
    n = 0
    for p in pred:
        for i, g in enumerate(gold):
            if i in used:
                continue
            if _span_pair_matches(p, g):
                used.add(i)
                n += 1
                break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--pred", default="runs/evidence_v2/val_predictions.csv")
    ap.add_argument("--pred-col", default=None)
    args = ap.parse_args()

    val = pd.read_csv(args.val)
    pr = pd.read_csv(args.pred)
    col = args.pred_col or next(
        (c for c in ("pred_evidence", "evidence_pred", "evidence", "pred")
         if c in pr.columns), None)
    if col is None:
        raise SystemExit(f"no prediction column; have {list(pr.columns)}")
    print(f"prediction column: {col!r}   ({len(pr)} rows)")

    pmap = dict(zip(pr["row_id"], pr[col]))
    rows = [r for r in val["row_id"] if r in pmap]
    v = val.set_index("row_id").loc[rows]
    G = [parse_evidence(x) for x in v["evidence"]]
    P = [parse_evidence(pmap[r]) for r in rows]
    risk = [normalize_label(x) for x in v["risk_level"]]
    print(f"{len(rows)} posts")

    res = phrase_f1(G, P)
    print(f"\nphrase-F1 {res.f1:.4f}   P {res.precision:.4f}   "
          f"R {res.recall:.4f}")
    print(f"  gold spans {res.n_gold}   predicted {res.n_pred}   "
          f"matched {res.n_matched}")
    limiting = "RECALL" if res.recall < res.precision else "PRECISION"
    print(f"  limiting side: {limiting}")

    ng = np.array([len(g) for g in G])
    npd = np.array([len(p) for p in P])
    nm = np.array([match_counts(G[i], P[i]) for i in range(len(rows))])

    print(f"\nspans per post   gold {ng.mean():.2f}   ours {npd.mean():.2f}")
    print(f"{'risk':<12}{'n':>6}{'gold/post':>11}{'ours/post':>11}"
          f"{'matched/post':>14}")
    for r in ("indicator", "ideation", "behavior", "attempt"):
        m = np.array([x == r for x in risk])
        if not m.any():
            continue
        print(f"{r:<12}{int(m.sum()):>6}{ng[m].mean():>11.2f}"
              f"{npd[m].mean():>11.2f}{nm[m].mean():>14.2f}")

    print(f"\nPOST-LEVEL BREAKDOWN")
    print(f"  gold>0, predicted 0    : {int(((ng > 0) & (npd == 0)).sum()):>4}"
          f"   (pure misses -- {int(ng[(ng > 0) & (npd == 0)].sum())} gold spans lost)")
    print(f"  gold=0, predicted >0   : {int(((ng == 0) & (npd > 0)).sum()):>4}"
          f"   (pure false positives)")
    print(f"  both>0, nothing matched: {int(((ng > 0) & (npd > 0) & (nm == 0)).sum()):>4}"
          f"   (a span was found but the wrong one)")
    print(f"  fully matched          : {int(((ng > 0) & (nm == ng)).sum()):>4}")
    print(f"  partially matched      : {int(((nm > 0) & (nm < ng)).sum()):>4}")

    # Ceiling on the second-span idea: posts where gold still has an unmatched
    # span AND fewer spans are currently emitted than gold has.
    room = (ng - nm > 0) & (npd < ng)
    extra = int((ng - nm)[room].sum())
    print(f"\nSECOND-SPAN HEADROOM")
    print(f"  posts with an unmatched gold span and room to add: {int(room.sum())}")
    print(f"  gold spans recoverable in principle              : {extra}")
    if extra:
        # If `extra` predictions were added and ALL matched -- the ceiling.
        p2 = (res.n_matched + extra) / (res.n_pred + extra)
        r2 = (res.n_matched + extra) / res.n_gold
        print(f"  CEILING if every added span matched: F1 "
              f"{2 * p2 * r2 / (p2 + r2):.4f} (now {res.f1:.4f})")
        # Break-even: adding k spans of which m match keeps F1 flat when
        # m/k is about F1/2. Show the realistic case.
        for hit in (0.3, 0.4, 0.5):
            m2 = res.n_matched + extra * hit
            pp = m2 / (res.n_pred + extra)
            rr = m2 / res.n_gold
            f = 2 * pp * rr / (pp + rr)
            print(f"  if {int(100 * hit)}% of added spans match: F1 {f:.4f} "
                  f"({f - res.f1:+.4f})   -> {0.3 * (f - res.f1):+.4f} composite")
        print(f"  break-even hit-rate is about F1/2 = {res.f1 / 2:.2f}")

    gl = [token_len(s) for g in G for s in g]
    pl = [token_len(s) for p in P for s in p]
    print(f"\nspan length (tokens)   gold mean {np.mean(gl):.1f}   "
          f"pred mean {np.mean(pl):.1f}")
    toolong = sum(1 for i in range(len(rows)) for p in P[i]
                  for g in G[i]
                  if (normalize_span(p) in normalize_span(g)
                      or normalize_span(g) in normalize_span(p))
                  and token_len(p) > 3 * token_len(g))
    print(f"  containment matches LOST to the 3x cap: {toolong}"
          f"   (0 means length is not costing anything)")

    print("\n" + "=" * 70)
    if limiting == "RECALL" and extra > 20:
        print("Recall-limited with real headroom. Emitting a second span is")
        print("worth building IF the tagger has runner-up candidates -- check")
        print("the break-even hit-rate above against what a runner-up would")
        print("plausibly achieve.")
    elif limiting == "PRECISION":
        print("Precision-limited. Adding spans would make it WORSE; the gain")
        print("has to come from better spans, not more of them.")
    else:
        print("Little headroom from extra spans. Evidence is close to what")
        print("this tagger can give.")
    print("=" * 70)


if __name__ == "__main__":
    main()

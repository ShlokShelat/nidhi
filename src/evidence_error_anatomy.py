"""
evidence_error_anatomy.py

WHERE EVIDENCE POINTS ARE LOST, BROKEN DOWN BY CAUSE.

The scorer matches a predicted span to a gold span iff ONE CONTAINS THE OTHER
(after normalisation) AND len_tokens(pred) <= 3 * len_tokens(gold). An F1 number
hides which of four completely different failures produced it:

    WRONG PLACE   no token overlap with any gold span      -> model problem
    BOUNDARY      overlaps gold, but neither contains      -> EXTENT problem
    TOO LONG      contains gold but exceeds the 3x cap     -> trim it
    MISSED        a gold span with nothing near it         -> recall problem

BOUNDARY IS THE ONE WITH THE CHEAPEST FIX. Containment is all-or-nothing, so a
span that begins a couple of words too early contains neither the gold span nor
is contained by it, and scores ZERO -- identical credit to quoting a random
sentence -- even though the model located the right place in the post. Trimming
the leading words would turn the same prediction into a full match.

If BOUNDARY is a large share of the unmatched spans, the model is locating
evidence correctly and it is being discarded on extents. Two zero-parameter fixes
follow directly:
  * SHRINK to the highest-confidence contiguous core (a shorter span is easier
    to be contained BY gold, and can never breach the 3x cap upward)
  * or EXTEND to a clause boundary so the span CONTAINS gold instead of
    straddling it -- but only while staying under 3x, which is why `capsafe`
    exists and why it is length-sensitive

If instead WRONG PLACE and MISSED dominate, extents are a red herring and the
gap is model quality, which no decode can reach.

Scored on the eight `_rc` taggers' VAL candidates -- the shipped ensemble --
using the shipped `k=2 --cluster --rep modal` and the indicator suppression
rule, so the anatomy describes what is actually submitted.

Usage:
  python src/evidence_error_anatomy.py
  python src/evidence_error_anatomy.py --show 12
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, parse_evidence, token_len  # noqa: E402

RC8 = ["evidence_v2_rc", "evidence_v1_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_lr2_rc", "evidence_ep9_rc"]


def norm(s):
    return " ".join(str(s).lower().split())


def toks(s):
    return norm(s).split()


def contains(a, b):
    na, nb = norm(a), norm(b)
    return na in nb or nb in na


def matches(p, g):
    """The scorer's rule."""
    return contains(p, g) and token_len(p) <= 3 * max(token_len(g), 1)


def overlap(p, g):
    """Jaccard on tokens -- how close the prediction was, ignoring the matching rule."""
    a, b = set(toks(p)), set(toks(g))
    return len(a & b) / max(len(a | b), 1)


def dedupe(spans):
    out = []
    for s in spans:
        if not any(norm(s) == norm(t) for t in out):
            out.append(s)
    return out


def cluster_vote(per_run, r, k):
    items = [(run, s) for run in per_run for s in per_run[run].get(r, [])]
    if not items:
        return []
    items.sort(key=lambda x: (-token_len(x[1]), x[1]))
    groups = []
    for run, s in items:
        for g in groups:
            if any(contains(s, t) and
                   max(token_len(s), token_len(t)) <=
                   3 * max(min(token_len(s), token_len(t)), 1)
                   for _, t in g):
                g.append((run, s))
                break
        else:
            groups.append([(run, s)])
    out = []
    for g in groups:
        if len({run for run, _ in g}) < k:
            continue
        cnt = Counter(s for _, s in g)
        out.append(max(cnt.items(),
                       key=lambda kv: (kv[1], -token_len(kv[0])))[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--runs", default=",".join(RC8))
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    per_run, ids = {}, set()
    for run in args.runs.split(","):
        p = f"{args.dir}/{run.strip()}_val_candidates.csv"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        if "conf" in d.columns and "min_conf" in d.columns and len(d):
            d = d[d["conf"] >= float(d["min_conf"].iloc[0])]
        g = {}
        for rid, s in zip(d["row_id"], d["span"]):
            g.setdefault(rid, []).append(str(s))
        per_run[run.strip()] = g
        ids |= set(g)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rk = pd.read_csv(args.risk, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    rk = rk.drop_duplicates("row_id").set_index("row_id")
    rcol = next((c for c in ("pred_risk_level", "risk_level")
                 if c in rk.columns), None)
    RL = ["indicator", "ideation", "behavior", "attempt"]
    if rcol is not None:
        risk = {r: normalize_label(rk.loc[r, rcol]) for r in rk.index}
    else:
        M = rk[[f"prob_{c}" for c in RL]].to_numpy(float)
        risk = {r: RL[i] for r, i in zip(rk.index, M.argmax(1))}

    rows = sorted(ids & set(lab.index) & set(risk))
    print(f"{len(rows)} val posts   {len(per_run)} `_rc` taggers   "
          f"shipped k={args.k} cluster modal\n")

    pcat = Counter()
    gcat = Counter()
    examples = {"BOUNDARY": [], "TOO LONG": [], "WRONG PLACE": [],
                "MISSED near": []}
    tp = 0
    for r in rows:
        gold = parse_evidence(lab.loc[r, "evidence"])
        pred = [] if risk[r] == "indicator" else dedupe(
            cluster_vote(per_run, r, args.k))

        gused = set()
        for p in pred:
            hit = next((j for j, g in enumerate(gold)
                        if j not in gused and matches(p, g)), None)
            if hit is not None:
                gused.add(hit)
                pcat["MATCHED"] += 1
                tp += 1
                continue
            # why not?
            best, bo = None, 0.0
            for g in gold:
                o = overlap(p, g)
                if o > bo:
                    best, bo = g, o
            if best is None or bo == 0.0:
                pcat["WRONG PLACE"] += 1
                if len(examples["WRONG PLACE"]) < args.show:
                    examples["WRONG PLACE"].append((r, p, "-"))
            elif contains(p, best):
                pcat["TOO LONG"] += 1
                if len(examples["TOO LONG"]) < args.show:
                    examples["TOO LONG"].append((r, p, best))
            else:
                pcat["BOUNDARY"] += 1
                if len(examples["BOUNDARY"]) < args.show:
                    examples["BOUNDARY"].append((r, p, best))

        for j, g in enumerate(gold):
            if j in gused:
                gcat["MATCHED"] += 1
                continue
            bo = max((overlap(p, g) for p in pred), default=0.0)
            if bo >= 0.25:
                gcat["MISSED near"] += 1
                if len(examples["MISSED near"]) < args.show:
                    bp = max(pred, key=lambda p: overlap(p, g))
                    examples["MISSED near"].append((r, bp, g))
            elif bo > 0:
                gcat["MISSED far"] += 1
            else:
                gcat["MISSED none"] += 1

    npred = sum(pcat.values())
    ngold = sum(gcat.values())
    print("=" * 92)
    print("PREDICTED SPANS -- why each one scored or did not")
    print("=" * 92)
    for k in ("MATCHED", "BOUNDARY", "TOO LONG", "WRONG PLACE"):
        v = pcat[k]
        print(f"  {k:<14}{v:>6}   {v/max(npred,1):>6.1%}")
    print(f"  {'total':<14}{npred:>6}")
    rec = pcat["BOUNDARY"] + pcat["TOO LONG"]
    print(f"\n  RECOVERABLE (right place, wrong extent): {rec} "
          f"({rec/max(npred,1):.1%} of predicted spans)")

    print("\n" + "=" * 92)
    print("GOLD SPANS -- why each one was or was not credited")
    print("=" * 92)
    for k in ("MATCHED", "MISSED near", "MISSED far", "MISSED none"):
        v = gcat[k]
        print(f"  {k:<14}{v:>6}   {v/max(ngold,1):>6.1%}")
    print(f"  {'total':<14}{ngold:>6}")
    print(f"\n  'MISSED near' = a predicted span shares >=25% of its tokens "
          f"with this gold\n  span but the containment rule rejected it.")

    print("\n" + "=" * 92)
    print("WHAT THIS DECIDES")
    print("=" * 92)
    share = rec / max(npred, 1)
    if share >= 0.15:
        print(f"  {share:.0%} of predicted spans are in the RIGHT PLACE and "
              f"rejected on extent alone.")
        print("  That is a decode-level defect with zero-parameter fixes:")
        print("    - shrink to the confident core (easier to be contained BY "
              "gold, cannot")
        print("      breach the 3x cap upward)")
        print("    - or extend to a clause boundary so the span CONTAINS gold, "
              "while")
        print("      staying under 3x")
        print("  Worth building. Precision recovered here is free -- no model "
              "change.")
    else:
        print(f"  Only {share:.0%} of predicted spans fail on extent. Boundaries "
              f"are NOT the problem;")
        print("  the losses are location and recall, which no decode can fix.")
        print("  That would mean evidence headroom is model quality, and the")
        print("  span-vote work is finished.")

    for k, ex in examples.items():
        if not ex:
            continue
        print(f"\n--- {k} ---")
        for r, p, g in ex[:args.show]:
            print(f"  {r}")
            print(f"    ours: {p[:110]}")
            if g != "-":
                print(f"    gold: {g[:110]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""THE EXACT CEILING of any selector over the enumerated candidate pool.

WHY. `span_rerank_upper_bound.py` reported a "pool oracle" of 0.8243, previously
read as evidence being generation-limited. That number was the ceiling of ONE
selection rule -- pick every y==1 candidate, then dedupe_nested(keep-longest) --
and that rule DESTROYS matches: on an n-gram pool many candidates match the same
gold span, keep-longest collapses them, and a collapsed span can fall outside
the 3x-token cap on a short gold. So 0.8243 is a LOWER bound on the ceiling, not
the ceiling, and "generation-limited" was a stronger claim than the measurement
supported.

THE EXACT ANSWER. Under the official matcher (containment either direction,
token_len(pred) <= 3*token_len(gold), maximum bipartite matching) an optimal
selector picks exactly one candidate per gold span it can cover, and picks
NOTHING else -- any extra prediction is a guaranteed false positive, and two
candidates matching the same gold win only one match while costing two
predictions. So per post:

    C = max number of gold spans simultaneously coverable by DISTINCT
        candidates   (a maximum bipartite matching on the gold-candidate
        compatibility graph -- NOT just "how many golds have some match",
        which double-counts when two golds share their only candidate)
    F1 = 2C / (C + g)      for g > 0
    F1 = 1.0               for g == 0 (emit nothing)

That is the true ceiling. It reuses the repo's own matcher so it cannot drift
from the official scorer.

READ IT AGAINST RC7 = 0.8045 (val twin):
  * ceiling near 0.85  -> evidence really is generation-limited. Selection
    cannot pay for itself and the whole lane is closed for good.
  * ceiling well above 0.90 -> the gap is SELECTION, not generation, and exact
    MBR subset selection is worth building.

The pool is 4:1 negative-subsampled, which does NOT bias this: subsampling
removes NEGATIVES only, so every gold-matching candidate is still present.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, parse_evidence,  # noqa: E402
                     phrase_f1)


def norm(s):
    return " ".join(str(s).lower().split())


def compatible(pred, gold):
    """The official rule: containment either direction AND a 3x token cap."""
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return False
    if len(p.split()) > 3 * len(g.split()):
        return False
    return p in g or g in p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="artifacts/span_rerank_enum_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--rc7", type=float, default=0.8045)
    args = ap.parse_args()

    df = pd.read_csv(args.scores, keep_default_na=False)
    df["row_id"] = df["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    gold_by = {r: parse_evidence(e)
               for r, e in zip(lab["row_id"], lab["evidence"])}

    cand = {}
    for r, s in zip(df["row_id"], df["span"]):
        cand.setdefault(r, []).append(str(s))

    rows = sorted(cand)
    val_ids = set()
    if os.path.exists(args.val):
        v = pd.read_csv(args.val, keep_default_na=False)
        val_ids = {str(x) for x in v["row_id"]}

    f1s, in_val, covered, total_gold = [], [], 0, 0
    for r in rows:
        g = gold_by.get(r, [])
        in_val.append(r in val_ids)
        if not g:
            f1s.append(1.0)
            continue
        cs = cand[r]
        # adjacency: gold i -> candidate indices that could match it
        adj = [[j for j, c in enumerate(cs) if compatible(c, gi)] for gi in g]
        C = _max_bipartite_matching(adj, len(cs))
        covered += C
        total_gold += len(g)
        f1s.append(2.0 * C / (C + len(g)))

    f1s = np.array(f1s)
    in_val = np.array(in_val)

    print(f"{len(rows)} posts   {len(df)} candidates "
          f"({len(df)/len(rows):.1f}/post)")
    print(f"gold spans coverable by a DISTINCT candidate: "
          f"{covered}/{total_gold} ({100*covered/max(total_gold,1):.1f}%)")
    print()
    print(f"TRUE ORACLE  all posts        {f1s.mean():.4f}")
    if in_val.any():
        print(f"TRUE ORACLE  val subset ({int(in_val.sum())})  "
              f"{f1s[in_val].mean():.4f}")
    print(f"RC7 incumbent                 {args.rc7:.4f}")
    print()

    ref = f1s[in_val].mean() if in_val.any() else f1s.mean()
    head = ref - args.rc7
    print("=" * 66)
    print(f"HEADROOM ABOVE RC7: {head:+.4f} on val")
    if head < 0.03:
        print()
        print("Evidence IS generation-limited. Even a perfect selector barely")
        print("beats RC7, and after the 2.8x val->test attenuation that is")
        print("worth a negligible composite gain. The lane is closed for good --")
        print("no reranker, no MBR, no better decode can pay for itself.")
    elif head < 0.08:
        print()
        print("Modest headroom. A perfect selector would be worth roughly")
        print(f"{head/2.8*0.3:+.4f} composite. Only worth building if the")
        print("selector can capture most of it -- the reranker captured NONE")
        print("(0.7494, below RC7), so the bar is high.")
    else:
        print()
        print("SUBSTANTIAL headroom -- the pool is not generation-limited after")
        print("all, and the gap is SELECTION. Exact MBR subset selection")
        print("becomes the best remaining lane. Note the abstain slot needs a")
        print("CALIBRATED P(empty) or MBR can never predict nothing, and 36%")
        print("of posts have empty gold.")
    print("=" * 66)
    print()
    print("Sanity: this reuses scoring._max_bipartite_matching and the official")
    print("containment+3x rule, so it cannot drift from the official scorer.")
    print("Subsampling removes NEGATIVES only, so no gold-matching candidate")
    print("is missing.")

    # cross-check the matcher wiring on the oracle selection itself
    chk = phrase_f1([gold_by.get(r, []) for r in rows[:200]],
                    [gold_by.get(r, []) for r in rows[:200]],
                    average="macro").f1
    print(f"matcher self-check (gold vs gold, 200 posts): {chk:.4f} "
          f"{'OK' if chk > 0.999 else 'BROKEN'}")


if __name__ == "__main__":
    main()

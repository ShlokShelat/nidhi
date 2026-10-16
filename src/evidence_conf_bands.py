"""
evidence_conf_bands.py

Test whether the deployed confidence cut discards candidate spans the metric
would reward.

THE BREAK-EVEN. Phrase-F1 is micro-averaged, and where precision and recall
are close, adding one more prediction is net positive whenever its chance of
matching a gold span exceeds roughly F1/2. Any candidate whose confidence sits
between that break-even and the deployed `min_conf` is being discarded even
though emitting it should pay. This script prints the break-even alongside the
per-band hit rates so the two can be compared directly.

GUARDING THE SWEEP. Sweeping min_conf is itself a parameter choice, and a
parameter chosen on the same split it is scored on will look better than it
is. Two controls:
  - the sweep is cross-fitted across author-disjoint halves (choose on one,
    apply blind to the other), and must win on both
  - the per-band hit rates are printed, so the decision can be read off the
    break-even identity (keep a band iff its hit rate > F1/2) rather than off
    whichever cut happens to top the sweep

Where the band hit rates and the sweep disagree, the hit rates govern: they
are a property of the metric, not of this val split.

Usage:
  python src/evidence_conf_bands.py --run evidence_v2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, parse_evidence, phrase_f1,
                     _span_pair_matches)


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="evidence_v2")
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--shipped", type=float, default=0.65)
    args = ap.parse_args()

    val = pd.read_csv(args.val)
    cand = pd.read_csv(os.path.join(args.dir, f"{args.run}_val_candidates.csv"))
    rows = list(val["row_id"])
    gold = {r: parse_evidence(x) for r, x in zip(val["row_id"], val["evidence"])}
    risk = {r: normalize_label(x)
            for r, x in zip(val["row_id"], val["risk_level"])}
    half = dict(zip(rows, author_halves(list(val["anon_user_id"]))))

    by_post = {}
    for r, s, c in zip(cand["row_id"], cand["span"], cand["conf"]):
        by_post.setdefault(r, []).append((str(s), float(c)))
    print(f"{len(cand)} candidates over {len(by_post)} posts")

    # A candidate "hits" if it matches ANY gold span in its own post. This
    # slightly overstates value because matching is one-to-one, but it is the
    # right per-candidate quantity for a band hit-rate.
    hit = {}
    for r, lst in by_post.items():
        g = gold.get(r, [])
        for s, c in lst:
            hit[(r, s)] = any(_span_pair_matches(s, x) for x in g)

    print(f"\nHIT RATE BY CONFIDENCE BAND")
    print(f"{'band':<14}{'n':>7}{'hits':>7}{'hit rate':>10}")
    edges = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        sel = [(r, s) for r, lst in by_post.items() for s, c in lst
               if lo <= c < hi]
        if not sel:
            continue
        h = sum(hit[k] for k in sel)
        print(f"{lo:.2f}-{hi:<8.2f}{len(sel):>7}{h:>7}{h / len(sel):>10.2f}")

    def build(mc):
        out = []
        for r in rows:
            s = [x for x, c in by_post.get(r, []) if c >= mc]
            if risk[r] == "indicator":       # the deployed consistency rule
                s = []
            elif not s and by_post.get(r):
                s = [min(by_post[r], key=lambda t: len(t[0].split()))[0]]
            out.append(s)
        return out

    G = [gold[r] for r in rows]
    hv = np.array([half[r] for r in rows])

    def sc(P, m=None):
        if m is None:
            return phrase_f1(G, P)
        return phrase_f1([g for g, k in zip(G, m) if k],
                         [p for p, k in zip(P, m) if k])

    grid = np.round(np.arange(0.20, 0.91, 0.05), 2)
    base = sc(build(args.shipped))
    bA = sc(build(args.shipped), hv == "A")
    bB = sc(build(args.shipped), hv == "B")
    print(f"\ndeployed min_conf={args.shipped}: F1 {base.f1:.4f} "
          f"(P {base.precision:.4f} R {base.recall:.4f})")
    print(f"  break-even hit rate = F1/2 = {base.f1 / 2:.2f}"
          f"   -> keep any band above this")

    print(f"\n{'min_conf':>9}{'F1':>9}{'P':>8}{'R':>8}{'pred':>7}"
          f"{'delta':>9}{'halfA':>9}{'halfB':>9}")
    cache = {}
    for mc in grid:
        P = build(mc)
        cache[mc] = P
        r = sc(P)
        print(f"{mc:>9.2f}{r.f1:>9.4f}{r.precision:>8.4f}{r.recall:>8.4f}"
              f"{r.n_pred:>7}{r.f1 - base.f1:>+9.4f}"
              f"{sc(P, hv == 'A').f1 - bA.f1:>+9.4f}"
              f"{sc(P, hv == 'B').f1 - bB.f1:>+9.4f}")

    print("\nCROSS-FITTED (cut chosen on the OTHER half)")
    tot = []
    for tgt, src in (("A", "B"), ("B", "A")):
        mt, ms = hv == tgt, hv == src
        bt = sc(build(args.shipped), mt).f1
        best = max(grid, key=lambda m: sc(cache[m], ms).f1)
        got = sc(cache[best], mt).f1
        tot.append(got - bt)
        print(f"  tune on {src}, apply to {tgt}: min_conf={best:.2f}  "
              f"{bt:.4f} -> {got:.4f}  ({got - bt:+.4f})")
    m = float(np.mean(tot))
    print(f"  mean cross-fitted delta {m:+.4f}"
          f"   ({0.3 * m:+.4f} composite)")

    print("\n" + "=" * 70)
    if m > 0 and all(x > 0 for x in tot):
        print("Lowering the cut pays, cross-fitted and on both halves.")
        print("Cross-check it against the band table: the chosen cut should sit")
        print("where the hit rate crosses the break-even, not somewhere else.")
    else:
        print("The deployed cut is already at or past the useful point. The")
        print("decoder filters internally, so the band below the cut is small")
        print("and not worth harvesting.")
    print("=" * 70)


if __name__ == "__main__":
    main()

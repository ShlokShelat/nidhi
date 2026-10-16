"""
evidence_pool_ceiling.py

Two falsifiers that decide whether the evidence POOL has anything left.

A perfect selector can never exceed the gold-span recall of the candidate pool
it selects from, so the pool bounds the whole component. Both tests run on the
full OOF set rather than the smaller val split, because a roster or threshold
conclusion drawn from a few hundred posts is not separable from sampling noise.

FALSIFIER 1 -- ROSTER SATURATION.
Union gold-span recall as a function of roster size, averaged over every subset
of that size. Same-family taggers propose overlapping spans, so the marginal
recall of the k-th member decays geometrically and the union converges well
below 1.0.
  KILL if the last marginal recall is <= 2%. Then more same-family members
  convert nothing and the roster lane is dead.

FALSIFIER 2 -- SEMANTIC BLINDNESS vs THRESHOLDING.
Of the gold spans NOT in the pool, how many have a SUB-THRESHOLD candidate
overlapping them? `evidence_vote_oof.py` filters each tagger at its shipped
min_conf; these dumps store the full confidence band, so dropping the filter
separates the two causes: a tagger that saw the text and was merely unsure is
recoverable by a threshold change, while one that produced no trace at any
confidence is not reachable by any threshold, vote rule or roster change.
  KILL if fewer than 30% of missed golds have an overlapping sub-threshold
  candidate. Above 30%, a low-threshold pool-widening arm is worth one
  measurement.

Both are CPU, seconds. Neither fits any parameter.

Usage:
  python src/evidence_pool_ceiling.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_vote_oof import TAGS  # noqa: E402
from scoring import _span_pair_matches, normalize_span  # noqa: E402
from scoring import parse_evidence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--dir", default="artifacts/evidence_oof")
    ap.add_argument("--tags", default=",".join(TAGS))
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    tags = [t.strip() for t in args.tags.split(",")]
    # TWO views of every tagger: at its shipped cutoff, and unfiltered.
    shipped, full = {}, {}
    for t in tags:
        p = os.path.join(args.dir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            print(f"  skip {t}: no {p}")
            continue
        d = pd.read_csv(p)
        d["row_id"] = d["row_id"].astype(str)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        for name, dd in (("shipped", d[d["conf"] >= mc]), ("full", d)):
            tgt = shipped if name == "shipped" else full
            m = {}
            for r, s in zip(dd["row_id"], dd["span"]):
                m.setdefault(r, []).append(str(s))
            tgt[t] = m
        print(f"  {t}: min_conf={mc}  {len(d[d['conf'] >= mc])} shipped / "
              f"{len(d)} total spans")
    tags = [t for t in tags if t in shipped]
    if len(tags) < 2:
        raise SystemExit("need >= 2 taggers")

    rows = sorted(set(lab.index) & set().union(
        *[set(v) for v in shipped.values()]))
    G = {r: parse_evidence(lab.loc[r, "evidence"]) for r in rows}
    n_gold = sum(len(v) for v in G.values())
    print(f"\n{len(rows)} OOF posts, {n_gold} gold spans")

    def recall(subset, view):
        hit = 0
        for r in rows:
            pool = [s for t in subset for s in view[t].get(r, [])]
            for g in G[r]:
                if any(_span_pair_matches(s, g) for s in pool):
                    hit += 1
        return hit / max(n_gold, 1)

    print("\n" + "=" * 64)
    print("FALSIFIER 1  roster saturation (mean over ALL subsets of each size)")
    print("=" * 64)
    print(f"{'k':>3}{'subsets':>9}{'union recall':>15}{'marginal':>11}")
    print("-" * 40)
    prev = None
    for k in range(1, len(tags) + 1):
        vals = [recall(c, shipped)
                for c in itertools.combinations(tags, k)]
        m = float(np.mean(vals))
        mar = "" if prev is None else f"{100 * (m - prev):+.2f}%"
        print(f"{k:>3}{len(vals):>9}{m:>15.4f}{mar:>11}")
        prev = m
    print("\nKILL if the last marginal is <= 2%: more same-family members")
    print("convert nothing and the roster lane is dead.")

    print("\n" + "=" * 64)
    print("FALSIFIER 2  blindness vs thresholding")
    print("=" * 64)
    missed, sub, zero = [], 0, 0
    for r in rows:
        pool_s = [s for t in tags for s in shipped[t].get(r, [])]
        pool_f = [s for t in tags for s in full[t].get(r, [])]
        for g in G[r]:
            if any(_span_pair_matches(s, g) for s in pool_s):
                continue
            missed.append((r, g))
            gn = set(normalize_span(g).split())
            if any(_span_pair_matches(s, g) for s in pool_f):
                sub += 1
            elif any(gn & set(normalize_span(s).split()) for s in pool_f):
                sub += 1
            else:
                zero += 1
    n = len(missed)
    print(f"gold spans NOT matched by the shipped pool: {n} "
          f"({100 * n / max(n_gold,1):.1f}% of gold)")
    if n:
        print(f"  recoverable below threshold (match or token overlap): "
              f"{sub} ({100 * sub / n:.1f}%)")
        print(f"  ZERO trace at any confidence (semantic blindness): "
              f"{zero} ({100 * zero / n:.1f}%)")
        print(f"\n  examples of the blind misses:")
        shown = 0
        for r, g in missed:
            pool_f = [s for t in tags for s in full[t].get(r, [])]
            gn = set(normalize_span(g).split())
            if not any(gn & set(normalize_span(s).split()) for s in pool_f):
                print(f"    {r}: {g!r}")
                shown += 1
                if shown >= 10:
                    break
    print("\nKILL if recoverable < 30%: the taggers never see this text, so no")
    print("threshold, vote rule or roster change reaches it -- and evidence")
    print("generation is closed along with selection.")


if __name__ == "__main__":
    main()

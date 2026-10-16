"""
duplicate_consistency.py

Measures whether the corpus contains near-duplicate posts and whether the
model's predictions on them are self-consistent.

Consecutive row_ids sometimes carry near-identical text and near-identical
gold. Where two posts are near-duplicates, a model that assigns them DIFFERENT
labels is contradicting itself on the same evidence, so at least one of those
calls is wrong. Forcing agreement needs no training and no new model, and it
is transductive: it applies to the test set exactly as it does here.

This asks nothing new of any model. It asks only whether the predictions
already on disk are internally consistent.

WHAT IS MEASURED

  1. how many near-duplicate pairs exist, by normalised-token Jaccard, at
     several thresholds -- and how many are within the same author (a user
     reposting) versus across authors (the corpus containing the same text
     twice)
  2. GOLD agreement within a duplicate pair, per component. If gold itself
     disagrees on near-identical posts the labels are noisy and forcing
     consistency would fight the annotator, so this bounds the whole idea
     BEFORE any rule is written
  3. MODEL agreement within a pair. The gap between gold agreement and model
     agreement is the headroom
  4. the effect of enforcing consistency: within each duplicate cluster take
     the highest-confidence member's label and propagate it, cross-fitted by
     author, on risk and factors

If gold agreement is no higher than the model's, there is nothing to recover.
If gold agrees far more often than the model does, the difference is the
model's own contradictions and they are correctable.

Usage:
  python src/duplicate_consistency.py
  python src/duplicate_consistency.py --thresh 0.8
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, parse_evidence, parse_factors  # noqa

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def toks(s):
    return set(re.findall(r"[a-z0-9']+", str(s).lower()))


def jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def clusters(sim_pairs, n):
    """Union-find over the near-duplicate pairs."""
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i, j in sim_pairs:
        a, b = find(i), find(j)
        if a != b:
            par[a] = b
    g = defaultdict(list)
    for i in range(n):
        g[find(i)].append(i)
    return [v for v in g.values() if len(v) > 1]


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--test", default="data/test.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--thresh", type=float, default=0.75)
    args = a_.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    T = [toks(p) for p in lab["post"]]
    n = len(lab)
    auth = lab["anon_user_id"].astype(str).tolist()
    print(f"labelled set: {n} posts, {lab['anon_user_id'].nunique()} authors")

    print("\n" + "=" * 74)
    print("1. HOW MANY NEAR-DUPLICATES?")
    print("=" * 74)
    # only compare posts of similar length -- a full O(n^2) Jaccard on 1635
    # posts is fine, but the length filter removes obvious non-matches fast
    order = sorted(range(n), key=lambda i: len(T[i]))
    found = {t: [] for t in (0.6, 0.75, 0.9, 0.98)}
    for a in range(n):
        i = order[a]
        for b in range(a + 1, n):
            j = order[b]
            if len(T[j]) > 2 * max(len(T[i]), 1):
                break
            s = jac(T[i], T[j])
            for t in found:
                if s >= t:
                    found[t].append((i, j, s))
    hdr = "posts in a cluster"
    print(f"  {'jaccard':>9}{'pairs':>8}{'same author':>13}{hdr:>22}")
    for t in sorted(found):
        pr = found[t]
        same = sum(1 for i, j, _ in pr if auth[i] == auth[j])
        cl = clusters([(i, j) for i, j, _ in pr], n)
        print(f"  {t:>9.2f}{len(pr):>8}{same:>13}"
              f"{sum(len(c) for c in cl):>22}")

    pairs = [(i, j, s) for i, j, s in found[min(found, key=lambda t:
                                                abs(t - args.thresh))]]
    if not pairs:
        print("\n  no near-duplicates at this threshold -- nothing to enforce")
        return

    print("\n" + "=" * 74)
    print(f"2. DOES GOLD AGREE WITHIN A PAIR?  (jaccard >= {args.thresh})")
    print("=" * 74)
    gr = [normalize_label(v) for v in lab["risk_level"]]
    gf = [frozenset(x.strip().lower() for x in parse_factors(v))
          for v in lab["factors"]]
    ge = [frozenset(x.strip().lower() for x in parse_evidence(v))
          for v in lab["evidence"]]
    rs = np.mean([gr[i] == gr[j] for i, j, _ in pairs])
    fs = np.mean([gf[i] == gf[j] for i, j, _ in pairs])
    fj = np.mean([jac(set(gf[i]), set(gf[j])) if (gf[i] or gf[j]) else 1.0
                  for i, j, _ in pairs])
    es = np.mean([bool(ge[i]) == bool(ge[j]) for i, j, _ in pairs])
    print(f"  {len(pairs)} pairs")
    print(f"  gold RISK identical            {rs:>7.1%}")
    print(f"  gold FACTOR SET identical      {fs:>7.1%}")
    print(f"  gold factor Jaccard (mean)     {fj:>7.3f}")
    print(f"  gold evidence empty/non-empty  {es:>7.1%}")
    base_r = np.mean([gr[i] == gr[np.random.randint(n)]
                      for i, j, _ in pairs])
    print(f"\n  chance risk agreement (random pairs) ~{base_r:.1%}")
    print("  If gold agreement is near chance, near-duplicate posts get")
    print("  genuinely different labels and forcing consistency would fight")
    print("  the annotator. If it is very high, disagreements are model errors.")

    print("\n" + "=" * 74)
    print("3. DOES THE MODEL AGREE?  (val posts, where predictions exist)")
    print("=" * 74)
    if not os.path.exists(args.probs):
        print(f"  {args.probs} missing -- skipping")
        return
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pr_map = {r: normalize_label(v)
              for r, v in zip(pv["row_id"], pv["pred_risk_level"])}
    conf = {r: float(c) for r, c in
            zip(pv["row_id"], pv[[f"prob_{c}" for c in LEVELS]].max(1))}
    rid = lab["row_id"].tolist()
    vp = [(i, j, s) for i, j, s in pairs
          if rid[i] in pr_map and rid[j] in pr_map]
    print(f"  {len(vp)} pairs where both posts have predictions")
    if not vp:
        print("  none -- the duplicates do not fall in the val split")
        return
    ours = np.mean([pr_map[rid[i]] == pr_map[rid[j]] for i, j, _ in vp])
    goldv = np.mean([gr[i] == gr[j] for i, j, _ in vp])
    print(f"  gold agrees on   {goldv:>7.1%}")
    print(f"  model agrees on  {ours:>7.1%}")
    print(f"  gap              {goldv-ours:>+7.1%}   <- the headroom")

    cl = clusters([(i, j) for i, j, _ in vp], n)
    fixed = wrong = 0
    for c in cl:
        m = [i for i in c if rid[i] in pr_map]
        if len(m) < 2:
            continue
        lead = max(m, key=lambda i: conf[rid[i]])
        for i in m:
            if i == lead:
                continue
            if pr_map[rid[i]] != pr_map[rid[lead]]:
                if pr_map[rid[lead]] == gr[i]:
                    fixed += 1
                elif pr_map[rid[i]] == gr[i]:
                    wrong += 1
    print(f"\n  propagating the most-confident label within each cluster:")
    print(f"    posts corrected {fixed}   posts broken {wrong}   "
          f"clusters {len(cl)}")
    if fixed + wrong:
        print(f"    fix/break {fixed/max(wrong,1):.2f}")
    print("\n  Read fix/break, not the raw counts: a rule that touches more")
    print("  posts will show more of both, and only the ratio says whether")
    print("  propagating the label is net positive.")

    print("\n" + "=" * 74)
    print("4. HOW MANY DUPLICATES ARE IN THE TEST SET?")
    print("=" * 74)
    te = pd.read_csv(args.test, keep_default_na=False)
    TT = [toks(p) for p in te["post"]]
    m = len(te)
    o2 = sorted(range(m), key=lambda i: len(TT[i]))
    tp = []
    for a in range(m):
        i = o2[a]
        for b in range(a + 1, m):
            j = o2[b]
            if len(TT[j]) > 2 * max(len(TT[i]), 1):
                break
            if jac(TT[i], TT[j]) >= args.thresh:
                tp.append((i, j))
    tc = clusters(tp, m)
    print(f"  {len(tp)} near-duplicate pairs among {m} test posts")
    print(f"  {len(tc)} clusters covering {sum(len(c) for c in tc)} posts")
    if tc:
        print("  example clusters (row_ids):")
        for c in tc[:6]:
            print("    " + " ".join(str(te['row_id'].iloc[i]) for i in c))
    print("\n  This bounds the effect on test: a consistency rule can only")
    print("  touch posts inside a cluster.")


if __name__ == "__main__":
    main()

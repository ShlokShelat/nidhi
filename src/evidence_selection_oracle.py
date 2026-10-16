"""
evidence_selection_oracle.py

THE CANDIDATES ARE RIGHT. THE SELECTOR PICKS THE WRONG ONES.

`evidence_tagger_diagnosis.py` split every gold span three ways:

    matched by the shipped decode          308   91.1%
    PROPOSED by a tagger, NOT selected      17    5.0%
    never matched by any candidate          13    3.8%   (and >=1 of those is
                                                          a 3x-cap failure, not
                                                          a generation failure)

and found the partial-credit posts PRECISION-bound: **37 surplus predicted
spans against 25 missed gold spans**, with 17 of the 25 already sitting in the
candidate pool.

So the pool contains the right clauses and the decode emits the wrong subset.
That also explains why `k=3` underperforms: a global vote threshold removes
surplus and correct spans together. What is needed is a selector that SWAPS,
not one that CUTS.

WHAT THIS MEASURES, in order of usefulness:

  1. THE SELECTION ORACLE. For each post, choose the subset of ITS OWN
     candidates that maximises that post's F1. Exhaustive for small pools,
     greedy above that. This is the ceiling for any selection rule, reachable
     with zero model changes -- and unlike the risk oracles from earlier, the
     features needed to approach it (vote count, confidence, length, position)
     are all available at inference.

  2. THE COUNT ORACLE. Keep the shipped ranking but emit the RIGHT NUMBER of
     spans. Separates "the ranking is wrong" from "the stopping point is
     wrong" -- these need different fixes, and the count half is a
     one-parameter model.

  3. A CROSS-FITTED SELECTOR. Rank candidates by a small logistic on features
     available at inference, cut at a learned budget, coefficients from FIT
     authors and scored on HELD-OUT authors. This is the shippable version and
     the only number that counts.

The scorer is macro over posts; deltas are quoted that way and multiplied by
0.3 for composite. Only the evidence column moves, so risk and factors stay
bit-identical.

Usage:
  python src/evidence_selection_oracle.py
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def f1_of(gold, pred):
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    adj = [[j for j, g in enumerate(gold) if _span_pair_matches(p, g)]
           for p in pred]
    m = _max_bipartite_matching(adj, len(gold))
    pr, rc = m / len(pred), m / len(gold)
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def best_subset(cands, gold, cap=14):
    """Highest-F1 subset of `cands`. Exhaustive when small, greedy otherwise."""
    if not gold:
        return []
    if len(cands) <= cap:
        best, bs = 0.0, []
        for k in range(1, len(cands) + 1):
            for c in combinations(cands, k):
                v = f1_of(gold, list(c))
                if v > best:
                    best, bs = v, list(c)
            if best == 1.0:
                break
        return bs
    cur, best = [], 0.0
    pool = list(cands)
    while pool:
        gains = [(f1_of(gold, cur + [s]), s) for s in pool]
        v, s = max(gains, key=lambda x: x[0])
        if v <= best:
            break
        best, cur = v, cur + [s]
        pool.remove(s)
    return cur


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--votes", type=int, default=2)
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    S = json.load(open(args.cache))
    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    va = va[va["row_id"].isin(S)].reset_index(drop=True)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    posts = list(va["post"].astype(str))
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pred_risk = {r: normalize_label(v)
                 for r, v in zip(pv["row_id"], pv["pred_risk_level"])}

    def dedupe_nested(spans):
        """Longest-first, drop any span contained in one already kept. This is
        the LAST step of the shipped decode and must be applied identically to
        every candidate set, or the comparison is not like-for-like."""
        out = []
        for s in sorted(spans, key=lambda x: -token_len(x)):
            n = normalize_span(s)
            if n and not any(n in normalize_span(k) for k in out):
                out.append(s)
        return out

    # THE BASELINE BUG THIS REPLACES. The first version counted votes over ALL
    # candidates, deduped nested spans on the full pool, and only THEN applied
    # the vote/confidence filter. Deduping before filtering lets a long
    # low-vote span swallow a short high-vote one, after which the survivor is
    # filtered out and both are lost -- it read 1.30 spans/post and MACRO
    # 0.8153, against the true decode's 1.61 and 0.8824. Every gain measured
    # against it was partly recovering that bug. Order is now: confidence
    # filter -> vote count -> dedupe, matching evidence_failure_anatomy.py.
    POOL, FEAT, SHIPPED = [], [], []
    for i, r in enumerate(rows):
        lv = pred_risk.get(r, "indicator")
        if lv == "indicator":
            POOL.append([])
            FEAT.append(np.zeros((0, 5)))
            SHIPPED.append([])
            continue
        agg = {}
        for s, cf in S[r][lv]:
            if cf < args.min_conf:
                continue
            n = normalize_span(s)
            if not n:
                continue
            e = agg.setdefault(n, {"n": 0, "mx": 0.0, "sum": 0.0, "s": s})
            e["n"] += 1
            e["mx"] = max(e["mx"], cf)
            e["sum"] += cf
        items = list(agg.values())
        # the POOL keeps nested spans -- the selector is allowed to prefer a
        # short variant over the long one containing it, which is exactly the
        # choice the fixed decode order takes away from it
        cands = [e["s"] for e in items]
        pos = [posts[i].lower().find(normalize_span(e["s"])[:20])
               for e in items]
        F = np.array([[e["n"], e["mx"], e["sum"] / max(e["n"], 1),
                       token_len(e["s"]),
                       (p / max(len(posts[i]), 1)) if p >= 0 else 0.5]
                      for e, p in zip(items, pos)], dtype=float) \
            if items else np.zeros((0, 5))
        POOL.append(cands)
        FEAT.append(F)
        SHIPPED.append(dedupe_nested([e["s"] for e in items
                                      if e["n"] >= args.votes]))

    def macro(pred):
        return float(np.mean([f1_of(g, p) for g, p in zip(gold, pred)]))

    base = macro(SHIPPED)
    nz = [i for i in range(len(rows)) if gold[i]]
    print(f"{len(rows)} val posts   shipped decode MACRO {base:.4f}")
    # SANITY GATE. evidence_failure_anatomy.py measures the same decode at
    # 0.8824 / 1.61 spans-per-post. If this baseline drifts from that, the
    # decode has been mis-reimplemented and every delta below is measured
    # against the wrong reference -- which is exactly what happened on the
    # first version of this script.
    if abs(base - 0.8824) > 0.01:
        print(f"  !! WARNING: baseline {base:.4f} differs from the known "
              f"0.8824 by more than 0.01.")
        print("     The decode is not being reproduced; deltas below are "
              "measured against the WRONG baseline. Fix before believing "
              "anything.")
    print(f"  candidates/post (non-empty gold) "
          f"{np.mean([len(POOL[i]) for i in nz]):.2f}"
          f"   shipped {np.mean([len(SHIPPED[i]) for i in nz]):.2f}"
          f"   gold {np.mean([len(gold[i]) for i in nz]):.2f}\n")

    print("=" * 76)
    print("1. SELECTION ORACLE -- best subset of OUR OWN candidates")
    print("=" * 76)
    ORC = [best_subset(POOL[i], gold[i]) if gold[i] else []
           for i in range(len(rows))]
    o = macro(ORC)
    print(f"  shipped   {base:.4f}")
    print(f"  ORACLE    {o:.4f}   ({o-base:+.4f} evidence = "
          f"{0.3*(o-base):+.4f} composite)")
    print(f"  oracle spans/post {np.mean([len(ORC[i]) for i in nz]):.2f}")
    print("\n  Reachable with ZERO model changes if a selector could rank the")
    print("  pool correctly. Unlike the risk oracles, every feature it needs")
    print("  (vote count, confidence, length, position) exists at inference.")

    print("\n" + "=" * 76)
    print("2. COUNT ORACLE -- shipped ranking, right NUMBER of spans")
    print("=" * 76)
    CNT = []
    for i in range(len(rows)):
        if not gold[i]:
            CNT.append([])
            continue
        order = sorted(range(len(POOL[i])),
                       key=lambda j: (-FEAT[i][j, 0], -FEAT[i][j, 1]))
        CNT.append(dedupe_nested([POOL[i][j] for j in order[:len(gold[i])]]))
    c = macro(CNT)
    print(f"  shipped {base:.4f}   right-count {c:.4f}   ({c-base:+.4f})")
    print("  If this captures most of the oracle, the RANKING is fine and only")
    print("  the stopping rule is wrong -- a one-parameter fix. If it captures")
    print("  little, the ranking itself is the problem.")

    print("\n" + "=" * 76)
    print("3. CROSS-FITTED SELECTOR -- logistic ranker + learned budget")
    print("=" * 76)
    from sklearn.linear_model import LogisticRegression
    Y = []
    for i in range(len(rows)):
        Y.append(np.array([1 if any(_span_pair_matches(s, g) for g in gold[i])
                           else 0 for s in POOL[i]]))
    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    res = {k: [] for k in ("logistic", "logistic+budget")}
    for _ in range(args.splits):
        ordr = list(ua)
        rng.shuffle(ordr)
        k = max(int(len(ordr) * 0.25), 2)
        ho = set(ordr[:k])
        hm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) < 40 or len(hi) < 15:
            continue
        X = np.vstack([FEAT[i] for i in fi if len(POOL[i])])
        y = np.concatenate([Y[i] for i in fi if len(POOL[i])])
        if len(np.unique(y)) < 2:
            continue
        m = LogisticRegression(C=1.0, max_iter=400).fit(X, y)
        # score once per post, reuse -- and dedupe_nested the selection so the
        # selector's output gets the SAME post-processing as the baseline
        SC = {i: (m.predict_proba(FEAT[i])[:, 1] if len(POOL[i])
                  else np.zeros(0)) for i in list(fi) + list(hi)}

        def pick(i, t, cap=None):
            if not len(POOL[i]):
                return []
            sc = SC[i]
            idx = [j for j in np.argsort(-sc) if sc[j] >= t]
            if cap is not None:
                idx = idx[:cap]
            return dedupe_nested([POOL[i][j] for j in idx])

        best_t, bv = 0.5, -1
        for t in np.arange(0.15, 0.85, 0.05):
            v = np.mean([f1_of(gold[i], pick(i, t)) for i in fi])
            if v > bv:
                bv, best_t = v, t
        ref = [SHIPPED[i] for i in hi]
        gh = [gold[i] for i in hi]
        rb = np.mean([f1_of(g, p) for g, p in zip(gh, ref)])
        res["logistic"].append(
            np.mean([f1_of(gold[i], pick(i, best_t)) for i in hi]) - rb)
        res["logistic+budget"].append(
            np.mean([f1_of(gold[i], pick(i, best_t, max(len(SHIPPED[i]), 1)))
                     for i in hi]) - rb)
    print(f"  {'selector':<20}{'d evidence':>12}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'sd':>9}")
    for k, v in res.items():
        d = np.array(v)
        if not len(d):
            continue
        print(f"  {k:<20}{d.mean():>+12.4f}{0.3*d.mean():>+9.4f}"
              f"{np.mean(d > 0):>5.0%}{np.mean(d < 0):>6.0%}"
              f"{0.3*d.std():>9.4f}")
    print("\n  5 features and one threshold, cross-fitted by author. If this")
    print("  clears 90% win it is a shippable gain, and it touches ONLY the")
    print("  evidence column.")


if __name__ == "__main__":
    main()

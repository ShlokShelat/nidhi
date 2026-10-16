"""
evidence_failure_anatomy.py

WHY DOES A PREDICTED SPAN FAIL? THE SCORER'S RULE MAKES THIS GEOMETRIC.

Evidence scores 0.777 overall, but that number is flattered: ~42% of posts are
predicted `indicator` and ship EMPTY evidence, scoring 1.0 free when gold is
also empty. Solving 0.42*1.0 + 0.58*X = 0.7864 (the twin) gives **X ~ 0.63 on
posts that actually have evidence**. That is the real number and it is poor.

THE MATCH RULE, from scoring._span_pair_matches:

    if p not in g and g not in p:  return False        # substring, either way
    return token_len(p) <= 3 * token_len(g)            # length cap

**If the prediction is a substring of the gold span the cap is automatic**
(len(p) <= len(g) <= 3*len(g)). A single word inside the right clause scores
exactly as well as the whole clause. So a span can only fail three ways:

  PARTIAL   it overlaps a gold span but neither contains the other -- the
            tagger found the right region and drew the boundary elsewhere.
            FIXABLE BY TRIMMING to the shared core.
  TOO_LONG  it contains a gold span but exceeds 3x its tokens.
            FIXABLE BY TRIMMING.
  DISJOINT  it shares nothing with any gold span. Only a better model fixes
            this -- the tagger is looking in the wrong place.

That split decides the entire lane. An earlier analysis concluded "all
evidence headroom is span quality" without separating these, leaving evidence
work without this finer-grained target.

WHAT IS REPORTED
  1. score decomposition -- empty vs non-empty posts, and the per-post F1
     histogram. How many posts score exactly 0, and how many exactly 1?
  2. predicted-span outcomes: MATCHED / PARTIAL / TOO_LONG / DISJOINT
  3. gold-span outcomes: was anything predicted near it at all?
  4. THE TRIM ORACLE -- for every predicted span, try every contiguous word
     sub-window and ask whether ANY of them would match a gold span. This is
     an upper bound on what trimming can buy, computed without changing a
     single model weight.
  5. count analysis -- is the right NUMBER of spans predicted? With bipartite
     matching, an extra span costs precision even when the others are perfect.

Everything runs on the cached conditioned spans, so it is CPU and instant.

Usage:
  python src/evidence_failure_anatomy.py
  python src/evidence_failure_anatomy.py --show 12
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_span_pair_matches, normalize_label,  # noqa: E402
                     normalize_span, parse_evidence, phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def pf1(gold, pred):
    return phrase_f1(gold, pred).f1


def post_f1(gold, pred):
    """One post's phrase-F1 under the scorer's own rule."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    adj = [[j for j, g in enumerate(gold) if _span_pair_matches(p, g)]
           for p in pred]
    from scoring import _max_bipartite_matching
    m = _max_bipartite_matching(adj, len(gold))
    pr, rc = m / len(pred), m / len(gold)
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def words(s):
    return normalize_span(s).split()


def classify(p, golds):
    """MATCHED / TOO_LONG / PARTIAL / DISJOINT against the nearest gold."""
    if any(_span_pair_matches(p, g) for g in golds):
        return "MATCHED", None
    np_ = normalize_span(p)
    best, bj = None, 0.0
    for g in golds:
        ng = normalize_span(g)
        a, b = set(np_.split()), set(ng.split())
        j = len(a & b) / max(len(a | b), 1)
        if j > bj:
            bj, best = j, g
    if best is None or bj == 0:
        return "DISJOINT", None
    if normalize_span(best) in np_:
        return "TOO_LONG", best
    return "PARTIAL", best


def trim_oracle(p, golds, min_words=2):
    """Would ANY contiguous word sub-window of p match a gold span?
    Upper bound on trimming, since it is allowed to peek at gold."""
    w = words(p)
    for L in range(len(w), min_words - 1, -1):
        for s in range(0, len(w) - L + 1):
            sub = " ".join(w[s:s + L])
            if any(_span_pair_matches(sub, g) for g in golds):
                return sub
    return None


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--votes", type=int, default=1,
                    help="how many of the 5 fold taggers must propose a span. "
                         "1 = UNION, which is what the first version of this "
                         "script used and is NOT the champion's decode -- the "
                         "champion votes. A union over-predicts by "
                         "construction, so the over-prediction finding has to "
                         "be re-read at the vote level that matches shipping.")
    a_.add_argument("--show", type=int, default=8)
    args = a_.parse_args()

    S = json.load(open(args.cache))
    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    va = va[va["row_id"].isin(S)].reset_index(drop=True)
    rows = va["row_id"].tolist()
    gold_ev = [parse_evidence(v) for v in va["evidence"]]
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pred_risk = {r: normalize_label(v) for r, v in
                 zip(pv["row_id"], pv["pred_risk_level"])}

    def decode(min_conf, votes):
        """The 5 fold taggers each propose spans; keep those proposed by at
        least `votes` of them. votes=1 is a union."""
        out_all = []
        for r in rows:
            lv = pred_risk.get(r, "indicator")
            if lv == "indicator":
                out_all.append([])
                continue
            c = {}
            for s, cf in S[r][lv]:
                if cf < min_conf:
                    continue
                n = normalize_span(s)
                e = c.setdefault(n, [0, s])
                e[0] += 1
            keep = [v[1] for v in c.values() if v[0] >= votes]
            keep.sort(key=lambda x: -token_len(x))
            out = []
            for s in keep:
                n = normalize_span(s)
                if n and not any(n in normalize_span(k) for k in out):
                    out.append(s)
            out_all.append(out)
        return out_all

    def macro(pred):
        """The scorer's aggregation: per-post F1 averaged over POSTS.
        phrase_f1() pools micro and reads ~0.07 lower here, so a delta
        quoted in micro is not the delta that determines the official
        score."""
        return float(np.mean([post_f1(g, p) for g, p in zip(gold_ev, pred)]))

    print("=" * 78)
    print("0. DECODE SWEEP -- vote level and confidence, against spans/post")
    print("=" * 78)
    ng_all = np.mean([len(g) for g in gold_ev if g])
    print(f"  gold spans/post on non-empty posts: {ng_all:.2f}")
    print(f"  {'votes':>6}{'min_conf':>10}{'spans/post':>12}{'MACRO':>9}"
          f"{'micro':>8}")
    for v in (1, 2, 3):
        for mc in (0.35, 0.50, 0.65):
            d = decode(mc, v)
            nz = [len(p) for p, g in zip(d, gold_ev) if g]
            print(f"  {v:>6}{mc:>10.2f}{np.mean(nz):>12.2f}"
                  f"{macro(d):>9.4f}{pf1(gold_ev, d):>8.4f}")
    print("\n  MACRO is what the scorer pays. If a tighter decode raises MACRO")
    print("  while moving spans/post toward gold, over-prediction is real and")
    print("  costing accuracy; if MACRO is flat, the union was never the problem.")

    P = decode(args.min_conf, args.votes)
    print(f"\n{len(rows)} val posts   votes={args.votes} "
          f"min_conf={args.min_conf}")
    print(f"  MACRO (scorer) {macro(P):.4f}   micro {pf1(gold_ev, P):.4f}\n")
    print("=" * 78)
    print("1. WHERE THE SCORE COMES FROM")
    print("=" * 78)
    per = np.array([post_f1(g, p) for g, p in zip(gold_ev, P)])
    emp = np.array([not g for g in gold_ev])
    pe = np.array([not p for p in P])
    print(f"  gold EMPTY posts       {int(emp.sum()):>4}   mean F1 "
          f"{per[emp].mean():.4f}")
    print(f"  gold NON-EMPTY posts   {int((~emp).sum()):>4}   mean F1 "
          f"{per[~emp].mean():.4f}   <- the real number")
    print(f"  predicted empty on     {int(pe.sum()):>4}   of which gold empty "
          f"{int((pe & emp).sum())}")
    h = Counter(np.round(per, 1))
    print("\n  per-post F1 histogram")
    for k in sorted(h):
        print(f"    {k:>4.1f}  {'#' * int(60 * h[k] / len(per))} {h[k]}")
    z = int(((per == 0) & ~emp).sum())
    print(f"\n  {z} non-empty posts score EXACTLY 0 "
          f"({z / max((~emp).sum(), 1):.0%} of them). Those are where the")
    print("  headroom is: each is a full point of macro score, and the scorer")
    print("  gives no partial credit for being close.")

    print("\n" + "=" * 78)
    print("2. WHY EACH PREDICTED SPAN FAILS")
    print("=" * 78)
    cnt, examples = Counter(), {"PARTIAL": [], "TOO_LONG": [], "DISJOINT": []}
    for i, (g, p) in enumerate(zip(gold_ev, P)):
        if not g:
            continue
        for s in p:
            k, near = classify(s, g)
            cnt[k] += 1
            if k != "MATCHED" and len(examples[k]) < args.show:
                examples[k].append((rows[i], s, near))
    tot = sum(cnt.values())
    for k in ("MATCHED", "PARTIAL", "TOO_LONG", "DISJOINT"):
        print(f"  {k:<10}{cnt[k]:>5}  {cnt[k]/max(tot,1):>6.1%}  "
              f"{'#' * int(40 * cnt[k] / max(tot, 1))}")
    fix = cnt["PARTIAL"] + cnt["TOO_LONG"]
    print(f"\n  PARTIAL + TOO_LONG = {fix} ({fix/max(tot,1):.1%}) are BOUNDARY")
    print("  errors -- the tagger found the right region. DISJOINT is the only")
    print("  class that needs a better model.")

    print("\n" + "=" * 78)
    print("3. THE TRIM ORACLE -- could a sub-window of the predicted span match?")
    print("=" * 78)
    Q, fixed = [], 0
    for g, p in zip(gold_ev, P):
        if not g:
            Q.append(p)
            continue
        out = []
        for s in p:
            if any(_span_pair_matches(s, x) for x in g):
                out.append(s)
                continue
            t = trim_oracle(s, g)
            if t:
                out.append(t)
                fixed += 1
            else:
                out.append(s)
        Q.append(out)
    o, base = macro(Q), macro(P)
    print(f"  spans rescued by trimming  {fixed}")
    print(f"  MACRO evidence  {base:.4f} -> {o:.4f}   ({o-base:+.4f} = "
          f"{0.3*(o-base):+.4f} composite)")
    print(f"  (micro, for reference: {pf1(gold_ev, P):.4f} -> "
          f"{pf1(gold_ev, Q):.4f})")
    print("\n  THIS IS AN ORACLE -- it peeks at gold to choose the window. It")
    print("  bounds what any confidence-based trimming rule could reach. If it")
    print("  is small the boundary story is wrong; if it is large, the next")
    print("  step is picking the window from TAGGER CONFIDENCE instead of gold")
    print("  and measuring how much of this survives.")

    print("\n" + "=" * 78)
    print("4. IS THE RIGHT NUMBER OF SPANS PREDICTED?")
    print("=" * 78)
    ng = np.array([len(g) for g in gold_ev])
    npd = np.array([len(p) for p in P])
    m = ~emp
    print(f"  gold spans/post (non-empty)  {ng[m].mean():.2f}")
    print(f"  pred spans/post (non-empty)  {npd[m].mean():.2f}")
    print(f"  {'gold n':<8}{'posts':>7}{'pred mean n':>12}{'mean F1':>10}")
    for k in range(1, 5):
        s = m & (ng == k)
        if s.sum() < 3:
            continue
        print(f"  {k:<8}{int(s.sum()):>7}{npd[s].mean():>12.2f}"
              f"{per[s].mean():>10.3f}")
    print("\n  Bipartite matching means an extra span costs precision even when")
    print("  every other span is perfect. Over-predicting on 1-gold posts is")
    print("  the cheapest thing to fix if it shows up here.")

    for k in ("PARTIAL", "TOO_LONG", "DISJOINT"):
        if not examples[k]:
            continue
        print("\n" + "=" * 78)
        print(f"EXAMPLES -- {k}")
        print("=" * 78)
        for rid, s, near in examples[k]:
            print(f"  [{rid}]")
            print(f"    ours: {s[:110]!r}")
            print(f"    gold: {(near or '(nothing nearby)')[:110]!r}")


if __name__ == "__main__":
    main()

"""
evidence_span_budget.py

KEEP THE RANKING. LEARN WHERE TO CUT.

`evidence_selection_oracle.py`, against a baseline verified at 0.8824:

    shipped decode                                    0.8824
    ORACLE best subset of the current candidates      0.9493
    COUNT ORACLE -- current ranking, right NUMBER     0.9234
    learned re-ranking (5-feature logistic)           0.8741   below baseline

Two things follow, and they point the same way:

  * **61% of the entire selection headroom is the STOPPING RULE**, not the
    order. The existing rank (vote count, then max confidence) is already
    good -- re-ranking it LOSES more often than it wins.
  * the miscount is PER-POST, not global: the decode emits 1.61 spans against
    gold's 1.66, right on average and wrong post by post.

So the problem is one integer per post: how many spans does this post need?
That is a far better-posed question than "which spans", it leaves the proven
ranking untouched, and it is the only evidence lane with a large verified
oracle behind a correct baseline.

WHY THE VOTE THRESHOLD CANNOT DO THIS. `k` is a GLOBAL knob: k=2 emits 1.61
spans/post, k=3 measurably strips correct spans from posts that needed them
while still over-emitting on posts that needed one.
A per-post budget is exactly the degree of freedom a global threshold lacks.

WHAT IS TESTED, all cross-fitted by author (model from FIT, scored on HELD-OUT):

  shipped        the current decode, unchanged -- the reference
  cap N          fixed budgets, to see whether a constant beats the vote rule
  predicted      a small model over features available at inference:
                 candidate count, votes at each level, confidence spread, post
                 length, predicted risk level, mean candidate length
  oracle count   the ceiling from above, reprinted for scale

FEATURES ARE INFERENCE-TIME ONLY. Nothing here uses gold, and the budget model
never sees the held-out authors it is scored on.

Usage:
  python src/evidence_span_budget.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

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


def dedupe_nested(spans):
    out = []
    for s in sorted(spans, key=lambda x: -token_len(x)):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--votes", type=int, default=2)
    a_.add_argument("--splits", type=int, default=300)
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
    pv = pv.set_index("row_id")
    pred_risk = np.array([normalize_label(v)
                          for v in pv.loc[rows, "pred_risk_level"]])

    RANK, SHIPPED, X = [], [], []
    for i, r in enumerate(rows):
        lv = pred_risk[i]
        if lv == "indicator":
            RANK.append([])
            SHIPPED.append([])
            X.append(np.zeros(11))
            continue
        agg = {}
        for s, cf in S[r][lv]:
            if cf < args.min_conf:
                continue
            n = normalize_span(s)
            if not n:
                continue
            e = agg.setdefault(n, {"n": 0, "mx": 0.0, "s": s})
            e["n"] += 1
            e["mx"] = max(e["mx"], cf)
        items = list(agg.values())
        # THE SHIPPED RANKING, untouched: vote count desc, then max conf desc
        items.sort(key=lambda e: (-e["n"], -e["mx"]))
        RANK.append([e["s"] for e in items])
        SHIPPED.append(dedupe_nested([e["s"] for e in items
                                      if e["n"] >= args.votes]))
        v = np.array([e["n"] for e in items], dtype=float)
        c = np.array([e["mx"] for e in items], dtype=float)
        L = np.array([token_len(e["s"]) for e in items], dtype=float)
        X.append(np.array([
            len(items), (v >= 2).sum(), (v >= 3).sum(), (v >= 4).sum(),
            c.mean() if len(c) else 0.0, c.std() if len(c) else 0.0,
            L.mean() if len(L) else 0.0,
            np.log1p(len(posts[i].split())),
            float(lv == "ideation"), float(lv == "behavior"),
            float(lv == "attempt")]))
    X = np.vstack(X)

    def emit(i, budget):
        if not RANK[i] or budget <= 0:
            return []
        return dedupe_nested(RANK[i][:int(budget)])

    def macro(pred):
        return float(np.mean([f1_of(g, p) for g, p in zip(gold, pred)]))

    base = macro(SHIPPED)
    nz = [i for i in range(len(rows)) if gold[i]]
    print(f"{len(rows)} val posts   shipped MACRO {base:.4f}")
    if abs(base - 0.8824) > 0.01:
        print(f"  !! WARNING baseline {base:.4f} != 0.8824 -- decode not "
              f"reproduced, stop here")
    ngold = np.array([len(g) for g in gold])
    nship = np.array([len(p) for p in SHIPPED])
    print(f"  spans/post  shipped {nship[nz].mean():.2f}   "
          f"gold {ngold[nz].mean():.2f}   "
          f"corr {np.corrcoef(nship[nz], ngold[nz])[0,1]:.3f}")
    print(f"  exact count match on {np.mean(nship[nz] == ngold[nz]):.1%} "
          f"of non-empty posts")
    print("\n  gold count distribution: "
          + "  ".join(f"{k}:{v}" for k, v in
                      sorted(Counter(ngold[nz]).items())))
    print("  ours:                    "
          + "  ".join(f"{k}:{v}" for k, v in
                      sorted(Counter(nship[nz]).items())))

    print("\n" + "=" * 74)
    print("FIXED BUDGETS -- does a constant beat the vote rule?")
    print("=" * 74)
    print(f"  {'rule':<16}{'MACRO':>9}{'d':>9}{'spans/post':>12}")
    for b in (1, 2, 3):
        p = [emit(i, b) if gold[i] or RANK[i] else [] for i in range(len(rows))]
        p = [[] if pred_risk[i] == "indicator" else p[i]
             for i in range(len(rows))]
        print(f"  cap {b:<12}{macro(p):>9.4f}{macro(p)-base:>+9.4f}"
              f"{np.mean([len(p[i]) for i in nz]):>12.2f}")
    orc = [emit(i, len(gold[i])) if gold[i] else [] for i in range(len(rows))]
    print(f"  {'ORACLE count':<16}{macro(orc):>9.4f}{macro(orc)-base:>+9.4f}"
          f"{np.mean([len(orc[i]) for i in nz]):>12.2f}")

    print("\n" + "=" * 74)
    print("LEARNED BUDGET -- cross-fitted by author")
    print("=" * 74)
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    res = {"ridge": [], "gbr": []}
    acc = {"ridge": [], "gbr": []}
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        fi = np.array([i for i in np.flatnonzero(~hm)
                       if pred_risk[i] != "indicator"])
        hi = np.array([i for i in np.flatnonzero(hm)])
        if len(fi) < 40 or len(hi) < 15:
            continue
        y = ngold[fi].astype(float)
        for nm, mdl in (("ridge", Ridge(alpha=1.0)),
                        ("gbr", GradientBoostingRegressor(
                            n_estimators=60, max_depth=2, random_state=0))):
            mdl.fit(X[fi], y)
            pb = np.clip(np.rint(mdl.predict(X[hi])), 1, 4).astype(int)
            pred = []
            for q, i in enumerate(hi):
                pred.append([] if pred_risk[i] == "indicator"
                            else emit(i, pb[q]))
            gh = [gold[i] for i in hi]
            ref = [SHIPPED[i] for i in hi]
            res[nm].append(np.mean([f1_of(g, p) for g, p in zip(gh, pred)])
                           - np.mean([f1_of(g, p) for g, p in zip(gh, ref)]))
            m2 = np.array([i for q, i in enumerate(hi) if gold[i]])
            q2 = np.array([q for q, i in enumerate(hi) if gold[i]])
            if len(m2):
                acc[nm].append(float(np.mean(pb[q2] == ngold[m2])))
    print(f"  {'model':<12}{'d evidence':>12}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'sd':>9}{'count acc':>11}")
    for nm in ("ridge", "gbr"):
        d = np.array(res[nm])
        if not len(d):
            continue
        print(f"  {nm:<12}{d.mean():>+12.4f}{0.3*d.mean():>+9.4f}"
              f"{np.mean(d > 0):>5.0%}{np.mean(d < 0):>6.0%}"
              f"{0.3*d.std():>9.4f}{np.mean(acc[nm]):>11.1%}")
    print("\n  `count acc` is how often the predicted budget equals the gold")
    print("  span count. Compare it against the shipped decode's own exact-")
    print("  count rate printed at the top -- if the model is not beating")
    print("  that, it cannot beat the decode, whatever the F1 says.")
    print("  Only the evidence column moves: d comp = 0.3 * d evidence.")


if __name__ == "__main__":
    main()

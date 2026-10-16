"""
evidence_feature_ceiling.py

IS THERE ANY SIGNAL LEFT IN THE FEATURES, OR IS THE VOTE RULE ALREADY OPTIMAL?

Selection attempts have failed against a verified 0.8824 baseline:

    learned RE-RANKING (5-feature logistic)   below baseline, low win rate
    learned COUNT budget (ridge / gbr)        below baseline, low win rate
                                               count acc 76.8% vs the
                                               decode's 74.1%

while the oracles say the headroom is real:

    best subset of the current candidates     large headroom
    current ranking, oracle COUNT             smaller but real headroom

Losing twice is suggestive but not conclusive -- both models could simply be
underpowered. This settles it directly by asking what the FEATURES can
discriminate, independent of any model or decode:

  per-candidate target: does this span match a gold span under the scorer's
  own rule?

  AUC of each feature alone, of the vote count (which is what the shipped rule
  thresholds), and of a cross-fitted logistic over all of them.

READING IT
  * if the full model's AUC is barely above vote-count-alone, the vote rule
    already extracts what the features contain, and no selector can beat it.
    The oracle is then real but unreachable, like the risk oracles before it,
    and the only way forward is NEW information -- better or more taggers.
  * if the full model's AUC is materially higher yet it still loses on F1, the
    problem is the OBJECTIVE, not the information: a per-span classifier
    optimises the wrong thing, because phrase-F1 is a SET score with bipartite
    matching and is not decomposable into independent per-span decisions. That
    would justify a listwise/expected-F1 selector, which is a different build.

Also reports the best achievable F1 from a PERFECT per-span classifier that
still uses the shipped post-processing -- separating "can the candidates be
ranked well" from "can a good ranking be turned into a good set".

Usage:
  python src/evidence_feature_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
NAMES = ["n_votes", "max_conf", "mean_conf", "tok_len", "rel_pos",
         "conf_rank", "votes_rank", "len_ratio", "n_cands"]


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


def auc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--votes", type=int, default=2)
    a_.add_argument("--splits", type=int, default=100)
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

    POOL, F, Y, PIDX, SHIPPED = [], [], [], [], []
    for i, r in enumerate(rows):
        lv = pred_risk[i]
        if lv == "indicator":
            POOL.append([])
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
        items.sort(key=lambda e: (-e["n"], -e["mx"]))
        POOL.append([e["s"] for e in items])
        SHIPPED.append(dedupe_nested([e["s"] for e in items
                                      if e["n"] >= args.votes]))
        if not items:
            continue
        mx = max(e["mx"] for e in items)
        mv = max(e["n"] for e in items)
        pl = max(len(posts[i].split()), 1)
        for e in items:
            p = posts[i].lower().find(normalize_span(e["s"])[:20])
            # RELATIVE features included deliberately: the decision is
            # per-post, so a confidence of 0.6 means something different when
            # the post's best is 0.65 than when it is 0.95. The earlier
            # logistic used absolute features only.
            F.append([e["n"], e["mx"], e["sum"] / max(e["n"], 1),
                      token_len(e["s"]), (p / max(len(posts[i]), 1))
                      if p >= 0 else 0.5,
                      e["mx"] / max(mx, 1e-9), e["n"] / max(mv, 1),
                      token_len(e["s"]) / pl, len(items)])
            Y.append(1 if any(_span_pair_matches(e["s"], g)
                              for g in gold[i]) else 0)
            PIDX.append(i)
    F = np.array(F, dtype=float)
    Y = np.array(Y)
    PIDX = np.array(PIDX)
    print(f"{len(rows)} posts   {len(Y)} candidates   "
          f"{Y.mean():.1%} of them match a gold span")
    base = float(np.mean([f1_of(g, p) for g, p in zip(gold, SHIPPED)]))
    print(f"  shipped MACRO {base:.4f}"
          + ("" if abs(base - 0.8824) <= 0.01 else "   !! baseline drift"))

    print("\n" + "=" * 70)
    print("1. WHAT EACH FEATURE KNOWS (AUC for 'this candidate is gold')")
    print("=" * 70)
    for j, nm in enumerate(NAMES):
        a = auc(F[:, j], Y)
        print(f"  {nm:<14}{a:>7.3f}{'  <- what the vote rule uses' if nm == 'n_votes' else ''}")

    from sklearn.linear_model import LogisticRegression
    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    aucs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([auth[i] in ho for i in PIDX])
        if hm.sum() < 30 or (~hm).sum() < 100:
            continue
        if len(np.unique(Y[~hm])) < 2 or len(np.unique(Y[hm])) < 2:
            continue
        m = LogisticRegression(C=1.0, max_iter=500).fit(F[~hm], Y[~hm])
        aucs.append(auc(m.predict_proba(F[hm])[:, 1], Y[hm]))
    print(f"\n  cross-fitted logistic (all 9)   {np.mean(aucs):>7.3f}"
          f"  +- {np.std(aucs):.3f}")
    print(f"  n_votes alone                   {auc(F[:, 0], Y):>7.3f}")
    print("\n  If these two are close, the vote rule already extracts what the")
    print("  features contain and no selector can beat it -- the oracle is")
    print("  real but unreachable. If the model is materially higher yet still")
    print("  loses on F1, the OBJECTIVE is wrong (phrase-F1 is a set score,")
    print("  not a sum of independent per-span decisions).")

    print("\n" + "=" * 70)
    print("2. CEILING OF A PERFECT PER-SPAN CLASSIFIER")
    print("=" * 70)
    perf = []
    for i in range(len(rows)):
        if pred_risk[i] == "indicator" or not POOL[i]:
            perf.append([])
            continue
        keep = [s for s in POOL[i]
                if any(_span_pair_matches(s, g) for g in gold[i])]
        perf.append(dedupe_nested(keep))
    pv_ = float(np.mean([f1_of(g, p) for g, p in zip(gold, perf)]))
    print(f"  keep exactly the candidates that DO match   {pv_:.4f}"
          f"  ({pv_-base:+.4f})")
    print("  That is the ceiling for per-span classification with the shipped")
    print("  post-processing. Compare it with the best-subset oracle (0.9493):")
    print("  the difference is what SET-level reasoning buys over per-span,")
    print("  and it is the only part a listwise selector could add.")


if __name__ == "__main__":
    main()

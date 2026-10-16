"""
evidence_cond_robust.py

Score spans by CONDITIONING ROBUSTNESS: how many of the four risk-level
conditionings propose the same span.

The tagger is handed the predicted risk level and asked to justify it. Where
that level is wrong, it can find a justification for the wrong answer, and the
evidence F1 on those posts collapses. Rules that try to repair this by picking
the RIGHT conditioning need to know which conditioning is correct; this one
does not.

A span that survives being told the post is an indicator AND that it is an
attempt is anchored in the text. A span that appears under one conditioning
only may be the model confabulating a justification for whatever label it was
handed. Robustness is therefore a per-span quality signal available at
inference. `n_votes` counts agreement across MODELS; this counts agreement
across QUESTIONS, and the two are independent.

WHAT IS SWEPT, against the deployed single-conditioning decode, cross-fitted by
author:

  robust>=r         keep spans proposed under at least r of 4 conditionings
  weighted>=w       weight each conditioning by the risk model's probability
                    for it, so a confident post is barely affected and an
                    uncertain one draws on several
  predicted AND robust>=r    tighten the deployed set
  predicted OR robust>=r     admit conditioning-robust spans the predicted
                             conditioning missed -- the only rule here that can
                             reach spans the current decode cannot produce

Runs on the cached `evcond_val_spans.json`, so it is CPU-only and fast.

Usage:
  python src/evidence_cond_robust.py
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
    a_.add_argument("--splits", type=int, default=250)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    S = json.load(open(args.cache))
    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    va = va[va["row_id"].isin(S)].reset_index(drop=True)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    gold_risk = np.array([normalize_label(v) for v in va["risk_level"]])
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    P = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    pred_risk = np.array([normalize_label(v)
                          for v in pv.loc[rows, "pred_risk_level"]])

    # per post: span -> (n_conditionings, weighted, in_predicted, text)
    info = []
    for i, r in enumerate(rows):
        d = {}
        for j, lv in enumerate(LEVELS):
            for s, cf in S[r][lv]:
                if cf < args.min_conf:
                    continue
                n = normalize_span(s)
                if not n:
                    continue
                e = d.setdefault(n, {"n": 0, "w": 0.0, "inp": False, "s": s})
                e["n"] += 1
                e["w"] += P[i, j]
                if lv == pred_risk[i]:
                    e["inp"] = True
        info.append(d)

    print(f"{len(rows)} val posts   risk correct on "
          f"{np.mean(pred_risk == gold_risk):.1%}")

    print("\n" + "=" * 74)
    print("1. IS ROBUSTNESS INFORMATIVE? precision by conditioning count")
    print("=" * 74)
    print(f"  {'proposed under':<18}{'spans':>8}{'match a gold span':>20}")
    for k in (1, 2, 3, 4):
        tot = hit = 0
        for i, d in enumerate(info):
            if not gold[i]:
                continue
            for e in d.values():
                if e["n"] == k:
                    tot += 1
                    hit += any(_span_pair_matches(e["s"], g) for g in gold[i])
        if tot:
            print(f"  {f'{k} of 4':<18}{tot:>8}{hit/tot:>20.1%}")
    print("\n  A rising precision means robustness separates real evidence")
    print("  from label-driven confabulation -- and it is INDEPENDENT of the")
    print("  vote count, which measures agreement across MODELS rather than")
    print("  across QUESTIONS.")

    def build(rule, r=2, w=0.5):
        out = []
        for i, rid in enumerate(rows):
            if pred_risk[i] == "indicator":
                out.append([])
                continue
            d = info[i]
            if rule == "predicted":
                k = [e["s"] for e in d.values() if e["inp"]]
            elif rule == "robust":
                k = [e["s"] for e in d.values() if e["n"] >= r]
            elif rule == "weighted":
                k = [e["s"] for e in d.values() if e["w"] >= w]
            elif rule == "and":
                k = [e["s"] for e in d.values() if e["inp"] and e["n"] >= r]
            else:
                k = [e["s"] for e in d.values() if e["inp"] or e["n"] >= r]
            out.append(dedupe_nested(k))
        return out

    def macro(pred, ix=None):
        ix = range(len(rows)) if ix is None else ix
        return float(np.mean([f1_of(gold[i], pred[i]) for i in ix]))

    base = build("predicted")
    nz = [i for i in range(len(rows)) if gold[i]]
    print(f"\n  DEPLOYED (predicted conditioning): macro {macro(base):.4f}   "
          f"spans/post {np.mean([len(base[i]) for i in nz]):.2f}")

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        subs.append((np.flatnonzero(~hm), np.flatnonzero(hm)))

    print("\n" + "=" * 74)
    print("2. CROSS-FITTED -- parameter chosen on FIT, scored on HELD-OUT")
    print("=" * 74)
    print(f"  {'rule':<16}{'d evidence':>12}{'d comp':>9}{'win':>6}{'LOSS':>6}"
          f"{'sd':>9}{'param':>12}")
    cache = {}
    grids = {"robust": [2, 3, 4], "and": [2, 3, 4], "or": [2, 3, 4],
             "weighted": [0.3, 0.5, 0.7, 0.9, 1.2]}
    for rule, grid in grids.items():
        for g in grid:
            cache[(rule, g)] = (build(rule, r=g) if rule != "weighted"
                                else build(rule, w=g))
        ds, ps = [], []
        for fi, hi in subs:
            bg = max(grid, key=lambda g: macro(cache[(rule, g)], fi))
            ds.append(macro(cache[(rule, bg)], hi) - macro(base, hi))
            ps.append(bg)
        d = np.array(ds)
        print(f"  {rule:<16}{d.mean():>+12.4f}{0.3*d.mean():>+9.4f}"
              f"{np.mean(d > 0):>5.0%}{np.mean(d < 0):>6.0%}"
              f"{0.3*d.std():>9.4f}"
              f"{f'{np.mean(ps):.2f}+-{np.std(ps):.2f}':>12}")

    print("\n" + "=" * 74)
    print("3. WHERE IT ACTS -- risk-right vs risk-wrong posts")
    print("=" * 74)
    ok = np.flatnonzero(pred_risk == gold_risk)
    bad = np.flatnonzero(pred_risk != gold_risk)
    best = max(grids, key=lambda rl: max(
        macro(cache[(rl, g)]) for g in grids[rl]))
    bg = max(grids[best], key=lambda g: macro(cache[(best, g)]))
    v = cache[(best, bg)]
    print(f"  best full-sample rule: {best} @ {bg}")
    print(f"  {'set':<16}{'deployed':>10}{best:>10}{'delta':>9}")
    for nm, ix in (("all", None), ("risk RIGHT", ok), ("risk WRONG", bad)):
        b, n = macro(base, ix), macro(v, ix)
        print(f"  {nm:<16}{b:>10.4f}{n:>10.4f}{n-b:>+9.4f}")
    print("\n  The mechanism predicts the gain comes from risk-WRONG posts.")
    print("  A gain concentrated on risk-right posts instead means the rule")
    print("  acts through some other route than the stated one.")


if __name__ == "__main__":
    main()

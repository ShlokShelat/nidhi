"""
headroom_analysis.py

Measures the CEILING of the existing models, so that an architecture proposal
can be justified by an upper bound before it is built.

Two ceilings, both on val, both with the official metrics:

 1. EVIDENCE -- reranking ceiling.
    Pool every span any tagger proposes, then keep exactly the ones that match
    gold. That is what a perfect reranker over the existing candidate pool
    would score. The gap between it and the best single tagger is the prize
    available WITHOUT a better tagger -- only a better selector. The gold
    spans no model proposes bound the part that selection can never reach.

 2. RISK -- oracle model-selection ceiling.
    If an oracle picked the best of the available risk models per post, what
    would weighted-F1 be? The gap over the deployed blend bounds what any
    router, stacker or confidence-gated mixture could ever win. Pairwise
    disagreement is printed alongside it, because routing can only exploit
    diversity that exists.

A ceiling is not a forecast. It is an upper bound that says whether an idea is
worth building at all: if perfect reranking only buys 0.01, do not build a
reranker.

Usage:
  python src/headroom_analysis.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (RISK_LEVELS, normalize_label, parse_evidence, phrase_f1,
                     risk_weighted_f1, _span_pair_matches)

PROBCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]
FRIEND_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]
EV_RUNS = ["evidence_v2", "evidence_v1", "evidence_s14", "evidence_s15"]
RISK_RUNS = ["risk_v1", "risk_v1_s14", "risk_v1_s15"]


def evidence_ceiling(val):
    gold = [parse_evidence(x) for x in val["evidence"]]
    per_run = {}
    for r in EV_RUNS:
        p = f"runs/{r}/val_predictions.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        col = next(c for c in d.columns if "pred" in c.lower())
        m = val[["row_id"]].merge(d[["row_id", col]], on="row_id", how="left")
        per_run[r] = [parse_evidence(x) for x in m[col]]
    if not per_run:
        print("  no evidence runs found")
        return

    print(f"{'model':<20}{'P':>8}{'R':>8}{'F1':>8}")
    for r, spans in per_run.items():
        s = phrase_f1(gold, spans)
        print(f"{r:<20}{s.precision:>8.4f}{s.recall:>8.4f}{s.f1:>8.4f}")

    n_posts = len(gold)
    pool, oracle = [], []
    for i in range(n_posts):
        cands = []
        for spans in per_run.values():
            for s in spans[i]:
                if not any(_span_pair_matches(s, c) for c in cands):
                    cands.append(s)
        pool.append(cands)
        # keep only candidates that match an as-yet-unmatched gold span
        keep, used = [], set()
        for g_i, g in enumerate(gold[i]):
            for c in cands:
                if c in keep:
                    continue
                if _span_pair_matches(c, g) and g_i not in used:
                    keep.append(c)
                    used.add(g_i)
                    break
        oracle.append(keep)

    u = phrase_f1(gold, pool)
    o = phrase_f1(gold, oracle)
    n_cand = sum(len(c) for c in pool)
    n_gold = sum(len(g) for g in gold)
    print(f"\n{'union of all models':<20}{u.precision:>8.4f}{u.recall:>8.4f}{u.f1:>8.4f}"
          f"   ({n_cand} candidate spans)")
    print(f"{'ORACLE rerank':<20}{o.precision:>8.4f}{o.recall:>8.4f}{o.f1:>8.4f}"
          f"   <- perfect selector over that pool")
    best = max(phrase_f1(gold, s).f1 for s in per_run.values())
    print(f"\n  candidate pool contains {o.n_matched}/{n_gold} gold spans "
          f"({o.n_matched / n_gold:.1%} recall ceiling)")
    print(f"  headroom from BETTER SELECTION alone: {o.f1:.4f} - {best:.4f} = "
          f"{o.f1 - best:+.4f} phrase-F1 = {0.3 * (o.f1 - best):+.4f} composite")
    print(f"  gold spans NO model proposes: {n_gold - o.n_matched} "
          f"-- unreachable without a better tagger")


def risk_ceiling(val):
    row_ids = val["row_id"].tolist()
    gold = [normalize_label(x) for x in val["risk_level"]]
    models = {}
    for r in RISK_RUNS:
        d = pd.read_csv(f"runs/{r}/val_predictions.csv").set_index("row_id")
        models[r] = d.loc[row_ids, PROBCOLS].values.astype(float)
    equal = np.mean(list(models.values()), axis=0)

    fr = pd.read_csv("artifacts/friend_risk_probs_val.csv").set_index("row_id")
    friend = np.zeros((len(row_ids), 4))
    n_missing = 0
    for i, rid in enumerate(row_ids):
        if rid in fr.index:
            p = fr.loc[rid, FRIEND_COLS].values.astype(float)
        else:
            p = equal[i]
            n_missing += 1
        friend[i] = p / p.sum()
    models["friend"] = friend
    models["equal_avg_3"] = equal
    models["0.5*equal+0.5*friend"] = 0.5 * equal + 0.5 * friend

    print(f"  ({n_missing} val rows fall back to the 3-model average for "
          f"friend)\n")
    print(f"{'model':<26}{'weighted-F1':>13}")
    preds = {}
    for name, P in models.items():
        pr = [RISK_LEVELS[int(i)] for i in P.argmax(1)]
        preds[name] = pr
        print(f"{name:<26}{risk_weighted_f1(gold, pr):>13.4f}")

    base = [m for m in RISK_RUNS] + ["friend"]
    oracle = []
    for i in range(len(gold)):
        hit = next((preds[m][i] for m in base if preds[m][i] == gold[i]), None)
        oracle.append(hit if hit else preds["0.5*equal+0.5*friend"][i])
    dep = risk_weighted_f1(gold, preds["0.5*equal+0.5*friend"])
    orc = risk_weighted_f1(gold, oracle)
    print(f"\n{'ORACLE per-post selection':<26}{orc:>13.4f}   <- perfect router "
          f"over 4 base models")
    print(f"  headroom from ROUTING alone: {orc - dep:+.4f} weighted-F1 = "
          f"{0.4 * (orc - dep):+.4f} composite")

    n_any = sum(1 for i in range(len(gold))
                if any(preds[m][i] == gold[i] for m in base))
    n_all_wrong = len(gold) - n_any
    print(f"  posts where NO base model is right: {n_all_wrong}/{len(gold)} "
          f"({n_all_wrong / len(gold):.1%}) -- unreachable by routing")

    # pairwise disagreement -- how much diversity is there to exploit?
    print(f"\n  pairwise disagreement between base models:")
    for a, b in itertools.combinations(base, 2):
        dis = sum(1 for i in range(len(gold)) if preds[a][i] != preds[b][i])
        print(f"    {a:<14} vs {b:<14} {dis:>4}/{len(gold)} ({dis/len(gold):.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    args = ap.parse_args()
    val = pd.read_csv(args.val)

    print("=" * 76)
    print("1. EVIDENCE -- what a perfect reranker over existing candidates buys")
    print("=" * 76)
    evidence_ceiling(val)

    print("\n" + "=" * 76)
    print("2. RISK -- what a perfect per-post router over existing models buys")
    print("=" * 76)
    risk_ceiling(val)


if __name__ == "__main__":
    main()

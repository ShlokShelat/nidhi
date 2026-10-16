"""
evidence_match_quality.py

Report whether evidence points are earned by exact spans or by the scorer's
containment slack.

THE RULE (scoring._span_pair_matches), which is NOT exact match:

    match(p, g) = (norm(p) in norm(g) or norm(g) in norm(p))
                  and token_len(p) <= 3 * token_len(g)

then maximum bipartite matching, precision = m/|pred|, recall = m/|gold|,
F1 per post, MACRO over posts. Both empty -> 1.0, one empty -> 0.0.

A prediction can therefore match three different ways, and they are not
equally robust:

    EXACT        norm(p) == norm(g)
    P_IN_G       the prediction is a FRAGMENT of the annotator's clause. The
                 length cap is automatic here (len(p) <= len(g) <= 3*len(g)),
                 so this match can never be lost to the cap.
    G_IN_P       the prediction SWALLOWS the annotator's clause. Survives only
                 while len(p) <= 3*len(g) -- one word longer and it scores 0.

The composition of the matched set matters because the three types carry
different exposure:

  * a high EXACT share means the tagger reproduces annotator boundaries and
    the score is robust
  * a high G_IN_P share near the 3x cap is fragile: the same spans against
    slightly shorter gold clauses would fail the cap
  * a high P_IN_G share means systematic under-spanning, which is SAFE under
    this scorer but would collapse under any stricter one

ALSO REPORTED
  * the macro score the same predictions would earn if the scorer demanded
    EXACT match -- the size of the containment slack
  * the token-length ratio distribution, and how many matches sit within one
    token of the 3x cliff
  * the posts scoring exactly 0, split into "gold empty, spans emitted" and
    "gold non-empty, nothing emitted" -- opposite problems needing opposite
    fixes

Usage:
  python src/evidence_match_quality.py
  python src/evidence_match_quality.py --votes 2
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


def post_f1(gold, pred, matcher=_span_pair_matches):
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    adj = [[j for j, g in enumerate(gold) if matcher(p, g)] for p in pred]
    m = _max_bipartite_matching(adj, len(gold))
    pr, rc = m / len(pred), m / len(gold)
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def exact_only(p, g):
    return normalize_span(p) == normalize_span(g)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--votes", type=int, default=2)
    a_.add_argument("--show", type=int, default=6)
    args = a_.parse_args()

    S = json.load(open(args.cache))
    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    va = va[va["row_id"].isin(S)].reset_index(drop=True)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    posts = list(va["post"].astype(str))
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pred_risk = {r: normalize_label(v)
                 for r, v in zip(pv["row_id"], pv["pred_risk_level"])}

    P = []
    for r in rows:
        lv = pred_risk.get(r, "indicator")
        if lv == "indicator":
            P.append([])
            continue
        c = {}
        for s, cf in S[r][lv]:
            if cf < args.min_conf:
                continue
            e = c.setdefault(normalize_span(s), [0, s])
            e[0] += 1
        keep = sorted([v[1] for v in c.values() if v[0] >= args.votes],
                      key=lambda x: -token_len(x))
        out = []
        for s in keep:
            n = normalize_span(s)
            if n and not any(n in normalize_span(k) for k in out):
                out.append(s)
        P.append(out)

    macro = float(np.mean([post_f1(g, p) for g, p in zip(gold, P)]))
    macro_ex = float(np.mean([post_f1(g, p, exact_only)
                              for g, p in zip(gold, P)]))
    print(f"{len(rows)} val posts   votes>={args.votes}")
    print(f"  MACRO under the real rule (containment + 3x)  {macro:.4f}")
    print(f"  MACRO if EXACT match were required            {macro_ex:.4f}")
    print(f"  the scorer's leniency is worth                "
          f"{macro - macro_ex:+.4f}  ({0.3*(macro-macro_ex):+.4f} composite)")

    print("\n" + "=" * 74)
    print("HOW EACH MATCH IS EARNED")
    print("=" * 74)
    kinds, ratios, tight, examples = Counter(), [], 0, {"EXACT": [],
                                                        "P_IN_G": [],
                                                        "G_IN_P": []}
    for i, (g, p) in enumerate(zip(gold, P)):
        if not g or not p:
            continue
        used = set()
        for s in p:
            for j, gg in enumerate(g):
                if j in used or not _span_pair_matches(s, gg):
                    continue
                used.add(j)
                ns, ng = normalize_span(s), normalize_span(gg)
                if ns == ng:
                    k = "EXACT"
                elif ns in ng:
                    k = "P_IN_G"
                else:
                    k = "G_IN_P"
                kinds[k] += 1
                rr = token_len(s) / max(token_len(gg), 1)
                ratios.append(rr)
                if k == "G_IN_P" and rr > 2.0:
                    tight += 1
                if len(examples[k]) < args.show:
                    examples[k].append((rows[i], s, gg, rr))
                break
    tot = sum(kinds.values())
    for k in ("EXACT", "P_IN_G", "G_IN_P"):
        lbl = {"EXACT": "exact string match",
               "P_IN_G": "pred is INSIDE gold (fragment)",
               "G_IN_P": "pred SWALLOWS gold (needs <=3x)"}[k]
        print(f"  {lbl:<36}{kinds[k]:>5}{kinds[k]/max(tot,1):>8.1%}  "
              f"{'#' * int(34 * kinds[k] / max(tot, 1))}")
    print(f"\n  matched spans {tot}   median token ratio pred/gold "
          f"{np.median(ratios):.2f}")
    print(f"  G_IN_P matches above 2x (within reach of the 3x cliff): {tight}")
    print("\n  P_IN_G can never be lost to the cap. G_IN_P near 3x is fragile:")
    print("  the same span against a shorter gold clause scores zero.")

    print("\n" + "=" * 74)
    print("THE POSTS SCORING EXACTLY ZERO -- opposite problems")
    print("=" * 74)
    per = [post_f1(g, p) for g, p in zip(gold, P)]
    z = [i for i, v in enumerate(per) if v == 0.0]
    spoke = [i for i in z if not gold[i] and P[i]]
    missed = [i for i in z if gold[i] and not P[i]]
    wrongsp = [i for i in z if gold[i] and P[i]]
    print(f"  total zero-scoring posts             {len(z)}")
    print(f"    gold EMPTY, spans emitted          {len(spoke)}   "
          f"(over-firing: risk said non-indicator)")
    print(f"    gold NON-EMPTY, nothing emitted    {len(missed)}   "
          f"(suppression: risk said indicator)")
    print(f"    both non-empty, nothing matched    {len(wrongsp)}   "
          f"(tagger miss)")
    print("\n  The first two are RISK errors surfacing in the evidence score:")
    print("  suppression keys off the predicted risk label, so they are only")
    print("  fixable upstream. Only the third is the tagger's, and it bounds")
    print("  what a better tagger could buy: "
          f"{len(wrongsp)}/{len(rows)} = "
          f"{len(wrongsp)/len(rows):.1%} of posts.")

    for k in ("G_IN_P", "P_IN_G", "EXACT"):
        if not examples[k]:
            continue
        print("\n" + "-" * 74)
        print(f"EXAMPLES -- {k}")
        for rid, s, gg, rr in examples[k]:
            print(f"  [{rid}] ratio {rr:.1f}x")
            print(f"    pred: {s[:100]!r}")
            print(f"    gold: {gg[:100]!r}")


if __name__ == "__main__":
    main()

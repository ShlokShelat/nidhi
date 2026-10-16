#!/usr/bin/env python
"""WHERE IS THE EVIDENCE GAP -- selection, or candidates? CPU, seconds.

THE UNASKED QUESTION. `evidence_true_oracle.py` measures the exact ceiling of
the ENUM pool at 0.9864, which could be read as "evidence is
selection-limited, 0.18 of headroom". But the shipped decode does not run on
the enum pool. It runs on the RC7 TAGGER pool, and 61% of RC7's spans are not
even present in the enum pool -- the two candidate sets are largely disjoint.
So the 0.9864 figure says nothing about how much room the actual selector has.

This computes the exact oracle SEPARATELY for:

    RC7    the seven shipped taggers' candidates (what the decode uses today)
    ENUM   the model-free in-sentence n-gram pool
    UNION  both together

Exact, not a bound: under the official matcher an optimal selector takes one
candidate per gold span it can cover and nothing else, so per post
    F1 = 2C/(C+g)   where C = max bipartite matching of gold to candidates
    F1 = 1.0        when g = 0 (emit nothing)

WHY THE ANSWER DECIDES THE NEXT MOVE.

  oracle(RC7) close to the shipped decode's score
        -> selection inside RC7 is nearly exhausted, matching what two
           rerankers, the cap, and the expected-F1 decoder all found.
           The gap is CANDIDATES. Fixing it means a better or additional
           tagger, not a better rule.
  oracle(RC7) far above the shipped decode's score
        -> selection really is the problem, and MBR/listwise work would be
           justified.
  oracle(UNION) >> oracle(RC7)
        -> the enum pool carries gold that RC7 never proposes, and merging the
           pools raises the ceiling even if neither selector is improved.

An external system scores an implied 0.8213 on evidence against the shipped
system's 0.7872, so a gap of that size is demonstrably reachable by somebody.
This says which door it came through.

Read-only.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     phrase_f1)

RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def compatible(pred, gold):
    return _span_pair_matches(pred, gold) or _span_pair_matches(gold, pred)


def oracle(gold, cands):
    """Exact per-post ceiling, and the coverage that produces it."""
    f1, cov, tot = [], 0, 0
    for g, cs in zip(gold, cands):
        if not g:
            f1.append(1.0)
            continue
        adj = [[j for j, c in enumerate(cs) if compatible(c, gi)] for gi in g]
        C = _max_bipartite_matching(adj, len(cs))
        cov += C
        tot += len(g)
        f1.append(2.0 * C / (C + len(g)))
    return float(np.mean(f1)), cov, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RC7))
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--split", default="val")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--enum", default="artifacts/span_rerank_enum_oof.csv")
    ap.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = [str(x) for x in va["row_id"]]
    gold = [parse_evidence(e) for e in va["evidence"]]
    ng = sum(len(g) for g in gold)
    print(f"{len(rows)} val posts, {ng} gold spans, "
          f"{sum(1 for g in gold if not g)} empty-gold")

    # ---- RC7 candidates ----------------------------------------------------
    per, votes = {}, {}
    for run in [x.strip() for x in args.runs.split(",") if x.strip()]:
        p = os.path.join(args.dir, f"{run}_{args.split}_candidates.csv")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        if "conf" in d.columns:
            d = d[d["conf"] >= mc]
        for r, s in zip(d["row_id"], d["span"]):
            n = normalize_span(str(s))
            if not n:
                continue
            per.setdefault(r, {})
            if n not in per[r]:
                per[r][n] = str(s)
            votes.setdefault(r, {}).setdefault(n, 0)
            votes[r][n] += 1
    rc7 = [list(per.get(r, {}).values()) for r in rows]

    # ---- ENUM candidates ---------------------------------------------------
    en = {}
    if os.path.exists(args.enum):
        e = pd.read_csv(args.enum, keep_default_na=False)
        e["row_id"] = e["row_id"].astype(str)
        for r, s in zip(e["row_id"], e["span"]):
            en.setdefault(r, set()).add(str(s))
    enum = [sorted(en.get(r, set())) for r in rows]

    union = []
    for a, b in zip(rc7, enum):
        seen, u = set(), []
        for s in list(a) + list(b):
            n = normalize_span(s)
            if n and n not in seen:
                seen.add(n)
                u.append(s)
        union.append(u)

    # ---- what is actually shipped, for reference ---------------------------
    rk = pd.read_csv(args.risk, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    rc = "pred_risk_level" if "pred_risk_level" in rk.columns else "risk_level"
    supp = {r: normalize_label(v) == "indicator"
            for r, v in zip(rk["row_id"], rk[rc])}
    shipped = []
    for r in rows:
        if supp.get(r, False):
            shipped.append([])
            continue
        shipped.append([per[r][n] for n, c in votes.get(r, {}).items()
                        if c >= args.k])
    cur = phrase_f1(gold, shipped, average="macro").f1

    print()
    print(f"{'pool':<8}{'cands/post':>12}{'gold covered':>15}{'ORACLE':>10}"
          f"{'headroom':>11}")
    print(f"{'-'*8}{'-'*12:>12}{'-'*15:>15}{'-'*10:>10}{'-'*11:>11}")
    res = {}
    for name, cs in (("RC7", rc7), ("ENUM", enum), ("UNION", union)):
        o, cov, tot = oracle(gold, cs)
        res[name] = o
        print(f"{name:<8}{np.mean([len(c) for c in cs]):>12.1f}"
              f"{cov}/{tot} = {100*cov/max(tot,1):>5.1f}%"
              f"{o:>10.4f}{o-cur:>+11.4f}")
    print(f"\nthe shipped decode (vote>={args.k}, suppressed): {cur:.4f}")

    print("\n" + "=" * 70)
    hr = res["RC7"] - cur
    print(f"SELECTION headroom inside RC7 : {hr:+.4f}")
    print(f"CANDIDATE headroom from ENUM  : {res['UNION'] - res['RC7']:+.4f}")
    print()
    if hr < 0.05:
        print("SELECTION IS NEARLY EXHAUSTED inside RC7. Two rerankers, the")
        print("per-post cap and the expected-F1 decoder all failing was not")
        print("bad luck -- there was almost nothing there to win. The gap is")
        print("CANDIDATES, so the move is a better or additional tagger.")
    else:
        print("SELECTION really is the bottleneck inside RC7 -- every selector")
        print("built so far was simply bad. MBR / listwise ranking is justified.")
    if res["UNION"] - res["RC7"] > 0.05:
        print()
        print("AND the enum pool carries gold RC7 never proposes. Merging the")
        print("pools raises the ceiling without improving any selector -- the")
        print("cheapest form of a 'better tagger' available, since the enum")
        print("candidates are model-free and already on disk.")
    print("=" * 70)
    print("\nDivide any val gain by ~2.8 for val->test before believing it.")
    print("Read the oracle ceiling above as the score a PERFECT selector over")
    print("this candidate pool would reach. The distance between it and the")
    print("shipped score is the headroom a better selector could recover; the")
    print("distance above it is what only a better pool can reach.")


if __name__ == "__main__":
    main()

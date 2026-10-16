#!/usr/bin/env python
"""Give RC7's candidates a reranker score WITHOUT a GPU, by joining to the
enumerated pool that was already scored.

WHY. RC7 (7-tagger vote, 0.8045) and the span reranker (cluster-then-rank,
~0.806) are two independent selectors sitting at the same level, against an
exact oracle of 0.9864 (`evidence_true_oracle.py`). Neither reaches it, and
they are differently wrong -- so FUSING them is the one untried move sized to
the remaining gap.

`apply_span_reranker.py` already implements the fusion rules (`vote>=2`,
`rerank>=t`, `votes OR rerank`, `votes AND rerank`), the author cross-fit, and
the macro scorer. All it lacks is a `rerank` column for RC7's spans, which it
would otherwise compute on a GPU. But RC7 emits in-sentence n-grams and the
enum pool IS every in-sentence 2-8 gram, so most RC7 spans are ALREADY in
`span_rerank_enum_oof.csv` with an author-disjoint OOF score attached. This
script does that join and writes the cache file.

THE JOIN KEY MATTERS. `apply_span_reranker.py:130` merges on the RAW
`["row_id", "span"]` strings, while it DEDUPES on `normalize_span`. So this
script reproduces its aggregation exactly (same normalize_span, same raw-string
retention) and matches to the enum pool on the NORMALIZED form. Emitting a
cache keyed on anything else would merge to all-NaN and silently score nothing.

COVERAGE IS THE RESULT. If most RC7 spans get a score, fusion is testable on
CPU today. If coverage is poor, the pools disagree more than expected and the
honest path is a GPU re-score -- the script says which, and never pretends an
unmatched span was scored.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_span  # noqa: E402

RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RC7))
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--enum", default="artifacts/span_rerank_enum_oof.csv")
    ap.add_argument("--out", default="artifacts/span_rerank_val.csv")
    ap.add_argument("--fill", default="median",
                    choices=["median", "zero", "drop"],
                    help="what to do with RC7 spans absent from the enum pool. "
                         "'median' is neutral-ish, 'zero' makes the reranker "
                         "veto them, 'drop' leaves them NaN so downstream "
                         "rules exclude them. Report coverage before trusting "
                         "any of these.")
    args = ap.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    rows = {str(x) for x in va["row_id"]}

    # ---- rebuild RC7's candidate frame EXACTLY as apply_span_reranker does --
    agg = {}
    used = 0
    for run in [x.strip() for x in args.runs.split(",")]:
        p = os.path.join(args.dir, f"{run}_val_candidates.csv")
        if not os.path.exists(p):
            print(f"  MISSING {p}")
            continue
        used += 1
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        for r, s in zip(d["row_id"], d["span"]):
            n = normalize_span(str(s))
            if not n:
                continue
            e = agg.setdefault((r, n), {"row_id": r, "span": str(s), "n": 0})
            e["n"] += 1
    if used == 0:
        raise SystemExit(f"no *_val_candidates.csv found under {args.dir}")
    cand = pd.DataFrame(list(agg.values()))
    cand = cand[cand.row_id.isin(rows)].reset_index(drop=True)
    print(f"{used} taggers, {len(rows)} val posts, {len(cand)} distinct "
          f"candidates ({len(cand)/max(len(rows),1):.2f}/post)")

    # ---- enum pool lookup, keyed on (row_id, normalized span) --------------
    en = pd.read_csv(args.enum, keep_default_na=False)
    en["row_id"] = en["row_id"].astype(str)
    look = {}
    for r, s, sc in zip(en["row_id"], en["span"], en["rerank"]):
        k = (r, normalize_span(str(s)))
        v = float(sc)
        if k not in look or v > look[k]:
            look[k] = v
    print(f"enum pool: {len(en)} rows -> {len(look)} distinct (row_id, span)")

    scores, hit = [], 0
    for r, s in zip(cand["row_id"], cand["span"]):
        v = look.get((r, normalize_span(str(s))))
        if v is None:
            scores.append(np.nan)
        else:
            scores.append(v)
            hit += 1
    cov = hit / max(len(cand), 1)
    print(f"COVERAGE: {hit}/{len(cand)} RC7 candidates found in the enum pool "
          f"({100*cov:.1f}%)")

    # posts where the enum pool has nothing at all -- these cannot be fused
    enum_rows = {r for r, _ in look}
    nomatch = sorted(rows - enum_rows)
    if nomatch:
        print(f"  {len(nomatch)} val posts have NO enum candidates at all")

    s = pd.Series(scores, dtype=float)
    if args.fill == "median" and s.notna().any():
        s = s.fillna(float(s.median()))
    elif args.fill == "zero":
        s = s.fillna(0.0)
    cand["rerank"] = s

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cand[["row_id", "span", "n", "rerank"]].to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print()
    print("=" * 68)
    if cov >= 0.80:
        print(f"COVERAGE {100*cov:.1f}% -- fusion is testable on CPU now:")
        print()
        print("  python src/apply_span_reranker.py \\")
        print(f"      --cache {args.out} 2>&1 | tail -40")
        print()
        print("Read the `or` and `and` rules against the `vote` baseline. The")
        print("diagnosis predicts OR wins: RC7's misses are single-tagger")
        print("spans the reranker likes. A win must clear ~90% and then be")
        print("divided by ~2.8 for val->test attenuation before it ships.")
    else:
        print(f"COVERAGE ONLY {100*cov:.1f}% -- do NOT trust a fusion result")
        print("built on this. The filled scores would dominate the rules and")
        print("whatever the sweep picked would be an artefact of --fill.")
        print("The honest path is a GPU re-score of RC7's own candidates via")
        print("apply_span_reranker.py with no --cache.")
    print("=" * 68)


if __name__ == "__main__":
    main()

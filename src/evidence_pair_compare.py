"""
evidence_pair_compare.py

PAIRED AUTHOR-SPLIT TEST BETWEEN TWO VOTE OUTPUTS.

Both configurations are supplied as already-built val prediction CSVs, so
neither is chosen on the data judging it and there is no winner's curse to
correct for. Reports WIN RATE and per-split sd, never the CI of the mean: the
CI of the mean shrinks with the number of resamples and can exclude zero at a
win rate barely above chance, which says nothing about a single held-out draw.

Acceptance bar: >= 90% win AND >= +0.0023 macro, which is +0.0007 composite at
the 0.30 evidence weight.

Usage:
  python src/evidence_vote_test.py ... --out artifacts/_A.csv
  python src/evidence_vote_test.py ... --out artifacts/_B.csv
  python src/evidence_pair_compare.py --a artifacts/_A.csv --b artifacts/_B.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1  # noqa: E402


def per_post(val, pred_csv):
    pv = pd.read_csv(pred_csv, keep_default_na=False)
    pv["row_id"] = pv["row_id"].astype(str)
    m = dict(zip(pv["row_id"], pv[pv.columns[-1]]))
    out = []
    for r in val.itertuples():
        g = parse_evidence(r.evidence)
        p = parse_evidence(m.get(str(r.row_id), ""))
        out.append(float(phrase_f1([g], [p], average="macro").f1))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline (the shipped config)")
    ap.add_argument("--b", required=True, help="challenger")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    amap = dict(zip(lab["row_id"], lab[acol].astype(str)))
    authors = np.array([amap.get(r, r) for r in va["row_id"]])

    fa = per_post(va, args.a)
    fb = per_post(va, args.b)
    diff = int((fa != fb).sum())
    print(f"{len(va)} val posts, {len(set(authors))} authors")
    print(f"  A {os.path.basename(args.a):<28} macro {fa.mean():.4f}")
    print(f"  B {os.path.basename(args.b):<28} macro {fb.mean():.4f}")
    print(f"  full-sample d {fb.mean()-fa.mean():+.4f} macro   "
          f"posts differing {diff}")

    rng = np.random.default_rng(args.seed)
    ua = np.array(sorted(set(authors)))
    d = []
    for _ in range(args.splits):
        h = set(rng.choice(ua, size=max(1, len(ua) // 3), replace=False))
        mk = np.array([a in h for a in authors])
        if mk.sum() >= 5:
            d.append(fb[mk].mean() - fa[mk].mean())
    d = np.array(d)
    w = float((d > 0).mean())
    print(f"  held-out    mean d {d.mean():+.4f}   sd {d.std():.4f}"
          f"   WIN {100*w:.0f}%   over {len(d)} splits")
    print(f"  composite estimate {0.30*(fb.mean()-fa.mean()):+.5f}")
    ok = w >= 0.90 and (fb.mean() - fa.mean()) >= 0.0023
    print("\n  " + ("CLEARS BOTH BARS -- build the test column."
                    if ok else
                    "DOES NOT CLEAR (need >=90% win AND >=+0.0023 macro)."))
    if diff and diff <= 12:
        print(f"\n  only {diff} posts differ -- per-post detail:")
        for i, r in enumerate(va.itertuples()):
            if fa[i] != fb[i]:
                print(f"    {r.row_id}  A {fa[i]:.3f} -> B {fb[i]:.3f}")


if __name__ == "__main__":
    main()

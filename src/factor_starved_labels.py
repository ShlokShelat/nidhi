"""
factor_starved_labels.py

Compares decode variants for the handful of factor labels whose score is
decided by one or two predicted cells.

Factor macro-F1 averages over 24 labels, so ONE label is worth 1/24 = 0.0417
of it -- and 0.30 * 0.0417 = 0.0125 composite. Moving a single label's F1 by
0.04 is +0.0005 composite.

The rarest labels (`poor school performance`, `exposure to others' suicide`,
`sexual orientation issues`) each have under 1% gold prevalence and fire on
one or two posts. At one firing the threshold is not trading precision against
recall in any meaningful sense; the label's whole score rides on whether that
single cell is right. If it is wrong the label contributes 0.000 and costs
0.0125 composite.

This is a narrower question than factor_smallsample_thresholds.py, whose
"low-support" set was every label with <= 60 gold -- eight labels moved
jointly, which lets the healthier ones dilute whatever the starved ones do.
The labels here have <= 16 gold and sit in a different regime.

WHAT IT TESTS, all cross-fitted on FIT authors and scored on HELD-OUT authors
so the threshold is never chosen on the data it is judged on:

    tfit      per-label argmax-F1 threshold            (baseline)
    prev      threshold set so the FIRING RATE matches gold prevalence
    half      halfway between tfit and prev, in threshold space
    zero      never predict the label at all

`zero` is the decisive control: a starved label that scores BETTER switched
off is one whose argmax threshold buys nothing, and removing it costs no
parameters.

Judged by WIN RATE and per-split sd over author-disjoint splits, never the CI
of the mean, because the split-to-split spread is what a single held-out draw
is exposed to. Bar: >= 90% win AND a positive macro gain.

Usage (CPU, under a minute):
  python src/factor_starved_labels.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402
from factor_structure import SHIPPED11, load_oof, logit_mean  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, c in enumerate(col):
        for x in set(parse_factors(str(c))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def f1(y, p):
    tp = float((y & p).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (y.sum() + p.sum())


def thr_argmax(p, y):
    best, bt = -1.0, 0.5
    for t in TGRID:
        s = f1(y, (p >= t).astype(np.int8))
        if s > best:
            best, bt = s, t
    return float(bt)


def thr_rate(p, rate):
    """threshold that makes the firing rate equal `rate`"""
    k = max(0, min(len(p), int(round(rate * len(p)))))
    if k == 0:
        return 1.01
    return float(np.sort(p)[::-1][k - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=SHIPPED11)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top", type=int, default=4,
                    help="how many of the rarest labels to treat as starved")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    ids, P = load_oof(args.dir, args.arms, args.folds)
    P = logit_mean(P)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id").loc[[str(i) for i in ids]].reset_index()
    Y = cells(lab["factors"], len(ids))
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    authors = lab[acol].astype(str).to_numpy()
    print(f"{len(ids)} OOF posts, {len(set(authors))} authors, "
          f"{len(args.arms)} arms")

    support = Y.sum(0)
    order = np.argsort(support)
    starved = list(order[:args.top])
    print("\nstarved labels (rarest by gold support):")
    for j in starved:
        print(f"  {FACTOR_TAXONOMY[j]:<38} gold {support[j]:>4}  "
              f"({100*support[j]/len(ids):.2f}%)")

    rng = np.random.default_rng(args.seed)
    ua = np.array(sorted(set(authors)))
    rows = []
    for _ in range(args.splits):
        hold = set(rng.choice(ua, size=max(1, len(ua) // 3), replace=False))
        h = np.array([a in hold for a in authors])
        f = ~h
        if h.sum() < 20 or f.sum() < 20:
            continue
        # every label keeps its argmax-F1 threshold, fitted on FIT authors only
        base = np.array([thr_argmax(P[f, j], Y[f, j]) for j in range(N)])
        scores = {}
        for name in ("tfit", "prev", "half", "zero"):
            t = base.copy()
            pred = np.zeros((h.sum(), N), dtype=np.int8)
            for j in range(N):
                if j in starved and name != "tfit":
                    if name == "zero":
                        t[j] = 1.01
                    elif name == "prev":
                        t[j] = thr_rate(P[f, j], Y[f, j].mean())
                    else:
                        t[j] = 0.5 * (base[j]
                                      + thr_rate(P[f, j], Y[f, j].mean()))
                pred[:, j] = (P[h, j] >= t[j]).astype(np.int8)
            scores[name] = float(np.mean([f1(Y[h, j], pred[:, j])
                                          for j in range(N)]))
        rows.append(scores)

    df = pd.DataFrame(rows)
    b = df["tfit"]
    print(f"\ncross-fitted over {len(df)} author-disjoint splits")
    print(f"  {'variant':<8}{'macro':>9}{'d vs tfit':>12}{'win':>7}{'sd':>9}")
    for c in df.columns:
        d = df[c] - b
        w = float((d > 0).mean())
        flag = ""
        if c != "tfit" and w >= 0.90 and d.mean() >= 0.0017:
            flag = "   <-- CLEARS BOTH BARS"
        print(f"  {c:<8}{df[c].mean():>9.4f}{d.mean():>+12.4f}"
              f"{100*w:>6.0f}%{d.std():>9.4f}{flag}")

    print("\n  Bar: >=90% win AND a positive macro gain.")
    print("  `zero` winning would mean a starved label scores no better at its")
    print("  argmax threshold than it does switched off.")
    print("  Per-label firing at the argmax threshold, on the full OOF:")
    full = np.array([thr_argmax(P[:, j], Y[:, j]) for j in range(N)])
    for j in starved:
        p = (P[:, j] >= full[j]).astype(np.int8)
        print(f"    {FACTOR_TAXONOMY[j]:<38} fires {p.sum():>4}  "
              f"gold {Y[:, j].sum():>4}  F1 {f1(Y[:, j], p):.3f}")


if __name__ == "__main__":
    main()

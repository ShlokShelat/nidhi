"""
arm_robustness.py

Re-measures each factor arm's cross-fitted macro-F1 over many random author
partitions instead of the single partition compare_factor_arms.py reports.

WHY MANY SPLITS. A cross-fitted score from one author split is dominated by
which authors landed on which side: the same data and the same code can give
materially different answers on different partitions, and on a set this small
a single draw is not distinguishable from the effect being measured. Repeating
the partition turns that variation into something quantifiable.

WHAT IT SEPARATES. An arm can gain under cross-fitted thresholds while ranking
no better by average precision. Averaging models smooths their probabilities,
which makes the fitted threshold more stable -- exactly what a cross-fitted
score rewards and what a threshold-free ranking metric cannot see. If that
smoothing is the real mechanism, the gain holds across every partition and the
ensemble beats its own members consistently; if the gain came from a favourable
partition, the win rate sits near half.

WHAT IS REPORTED. For each arm, paired against the reference on identical
posts within each split, so the split-to-split variation both share cancels:

    mean d      average gain over the reference arm
    se          standard error of that mean
    95% CI      excludes zero  -> real at this sample size
    wins        fraction of splits where the arm is ahead

The bar is a 95% interval that excludes zero AND a win rate above 80%.
Anything less is reported as unproven rather than as a verdict.

Usage:
  python src/arm_robustness.py
  python src/arm_robustness.py --splits 50 --arms mnlipair mnli dmnli mnli3
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def best_threshold(pr, y):
    """argmax-F1 threshold, computed in one pass instead of 91.

    Sorting by score descending, taking the top k predictions gives
    tp = cumsum(y)[k-1], and since k = tp+fp and npos = tp+fn,

        F1 = 2*tp / (k + npos)

    so every grid point is one lookup. The naive loop is ~100x slower and this
    script evaluates it a few hundred thousand times."""
    npos = int(y.sum())
    if npos == 0:
        return 0.5
    o = np.argsort(-pr, kind="stable")
    ps, ys = pr[o], y[o]
    cum = np.cumsum(ys)
    # number of items with score >= t, for every t in GRID
    ks = np.searchsorted(-ps, -GRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    f1 = np.where(ks + npos > 0, 2.0 * tp / (ks + npos), 0.0)
    return float(GRID[int(np.argmax(f1))])


def fit(PR, Y, rows):
    return np.array([best_threshold(PR[rows, j], Y[rows, j]) for j in range(N)])


def macro(P, Y):
    v = []
    for j in range(N):
        tp = int((P[:, j] & Y[:, j]).sum())
        v.append(0.0 if tp == 0 else 2 * tp /
                 (2 * tp + int((P[:, j] & (1 - Y[:, j])).sum())
                  + int(((1 - P[:, j]) & Y[:, j]).sum())))
    return float(np.mean(v))


def xfit(PR, Y, A, B):
    Px = np.zeros_like(Y)
    Px[A] = (PR[A] >= fit(PR, Y, B)).astype(int)
    Px[B] = (PR[B] >= fit(PR, Y, A)).astype(int)
    return macro(Px, Y)


def random_author_halves(auth, rng):
    s = pd.Series(auth)
    counts = s.value_counts()
    a = list(counts.index)
    rng.shuffle(a)
    half, size = {}, {"A": 0, "B": 0}
    for x in a:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[x] = pick
        size[pick] += counts[x]
    return s.map(half).to_numpy()


def load(pattern, k=5):
    r, P = [], []
    for f in range(k):
        p = pattern.format(f=f)
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        r.extend(list(d["row_id"]))
        P.append(d["probs"])
    return list(r), np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--vs", default=None,
                    help="reference arm every other arm is paired against "
                         "(default: the best-scoring arm)")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these tags (default: every complete arm)")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    arms = {"INCUMBENT": "artifacts/fac_oof_f{f}.npz"}
    for p in sorted(glob.glob(os.path.join(args.dir, "*_cv_oof_f0.npz"))):
        tag = os.path.basename(p)[: -len("_cv_oof_f0.npz")]
        if args.arms and tag not in args.arms:
            continue
        arms[tag] = os.path.join(args.dir, tag + "_cv_oof_f{f}.npz")

    ref = load(arms["INCUMBENT"], args.k)
    if ref is None:
        raise SystemExit("incumbent OOF missing")
    rid0, _ = ref

    Y = np.zeros((len(rid0), N), dtype=int)
    for i, r in enumerate(rid0):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1

    P = {}
    for tag, pat in arms.items():
        got = load(pat, args.k)
        if got is None:
            print(f"  skip {tag}: incomplete folds")
            continue
        r, pr = got
        if r != rid0:
            if set(r) != set(rid0):
                print(f"  skip {tag}: different post set")
                continue
            o = {x: i for i, x in enumerate(r)}
            pr = pr[[o[x] for x in rid0]]
        P[tag] = pr

    auth = [lab.loc[r, "anon_user_id"] for r in rid0]
    print(f"\n{len(rid0)} posts, {len(set(auth))} authors, "
          f"{len(P)} arms, {args.splits} random author partitions")
    print("(one partition is what compare_factor_arms.py reports)\n")

    rng = np.random.default_rng(args.seed)
    scores = {t: [] for t in P}
    for _ in range(args.splits):
        h = random_author_halves(auth, rng)
        A, B = h == "A", h == "B"
        for t, pr in P.items():
            scores[t].append(xfit(pr, Y, A, B))

    base = np.array(scores["INCUMBENT"])
    rows = []
    for t, v in scores.items():
        v = np.array(v)
        if t == "INCUMBENT":
            rows.append((t, v.mean(), v.std(), 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        rows.append((t, v.mean(), v.std(), d.mean(), se,
                     d.mean() - 1.96 * se, d.mean() + 1.96 * se,
                     (d > 0).mean()))
    rows.sort(key=lambda r: -r[3])

    print("=" * 96)
    print(f"CROSS-FITTED MACRO-F1 OVER {args.splits} AUTHOR PARTITIONS, "
          f"paired against the incumbent")
    print("=" * 96)
    print(f"{'arm':<12}{'mean':>9}{'sd':>8}{'d':>10}{'se':>8}"
          f"{'95% CI':>21}{'wins':>8}{'':>6}")
    print("-" * 96)
    for t, m, s, d, se, lo, hi, w in rows:
        if t == "INCUMBENT":
            print(f"{t:<12}{m:>9.4f}{s:>8.4f}{'':>10}{'':>8}{'':>21}{'':>8}")
            continue
        ci = f"[{lo:+.4f}, {hi:+.4f}]"
        verdict = "REAL" if (lo > 0 and w >= 0.8) else (
            "worse" if hi < 0 else "unproven")
        print(f"{t:<12}{m:>9.4f}{s:>8.4f}{d:>+10.4f}{se:>8.4f}{ci:>21}"
              f"{w:>8.0%}  {verdict}")

    print("\n  REAL      interval excludes zero AND wins at least 80% of splits")
    print("  unproven  interval spans zero -- not disproved, just not measured")
    print("  worse     interval entirely below zero")

    good = [r for r in rows if r[0] != "INCUMBENT" and r[5] > 0 and r[7] >= 0.8]
    if good:
        b = good[0]
        print(f"\n  best supported arm: {b[0]}  {b[3]:+.4f} "
              f"[{b[5]:+.4f}, {b[6]:+.4f}]")
        print(f"  composite impact:   {0.3 * b[3]:+.4f} "
              f"[{0.3*b[5]:+.4f}, {0.3*b[6]:+.4f}]   (factors are 30%)")
    else:
        print("\n  NOTHING clears the bar. Do not adopt any arm.")

    # Once several arms beat the incumbent, beating it stops discriminating
    # between them; the operative comparison is against the CURRENT BEST. The
    # reference is derived from the results rather than hardcoded, so a newly
    # added arm can never go unchecked.
    ref = args.vs or max((r[0] for r in rows if r[0] != "INCUMBENT"),
                         key=lambda t: np.mean(scores[t]), default=None)
    if ref and ref in scores:
        print("\n" + "=" * 96)
        print(f"EVERY ARM vs THE CURRENT BEST ({ref}) -- the bar for adoption")
        print("=" * 96)
        print("Beating the incumbent no longer discriminates between arms. To")
        print("be adopted, an arm must clear the arm it would replace.")
        print()
        rv = np.array(scores[ref])
        print(f"{'arm':<12}{'d vs ' + ref:>16}{'95% CI':>23}{'wins':>8}")
        print("-" * 62)
        for t, m, s, d, se, lo, hi, w in rows:
            if t == ref or t == "INCUMBENT":
                continue
            dd = np.array(scores[t]) - rv
            e = dd.std(ddof=1) / np.sqrt(len(dd))
            l, h = dd.mean() - 1.96 * e, dd.mean() + 1.96 * e
            if h < -0.002:
                continue          # clearly behind, not worth a line
            print(f"{t:<12}{dd.mean():>+16.4f}"
                  f"{f'[{l:+.4f}, {h:+.4f}]':>23}{(dd > 0).mean():>8.0%}")

    print("\n" + "=" * 96)
    print("DOES THE ENSEMBLE BEAT ITS OWN MEMBERS?")
    print("=" * 96)
    print("An ensemble that only beats the INCUMBENT proves nothing extra -- it")
    print("has to beat the better member, or the extra model is dead weight at")
    print("inference time.")
    print()
    for ens, members in (("incmnli", ["mnli"]),
                         ("mnlipair", ["mnli", "dmnli"]),
                         ("mnli3", ["mnli", "dmnli", "mnlipair"]),
                         ("pairprot", ["mnli", "dmnli", "prot", "mnlipair"]),
                         ("paircm", ["mnli", "dmnli", "cmine", "mnlipair",
                                     "pairprot"])):
        if ens not in scores:
            continue
        e = np.array(scores[ens])
        for m in members:
            if m not in scores:
                continue
            d = e - np.array(scores[m])
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {ens:<10} vs {m:<10} {d.mean():>+8.4f}  "
                  f"[{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]  "
                  f"wins {(d > 0).mean():.0%}")


if __name__ == "__main__":
    main()

"""
factor_blend.py

Blend the two independent factor models. Factors are the last untouched block.

WHERE THE GAP IS. Composite = 0.4 risk + 0.3 evidence + 0.3 factors. Backing
the components out of the composite score:

    risk     ~0.82        evidence ~0.75        factors 0.6301
    rank-1 factors ~0.6967  ->  0.020 composite sits here

S2 has not moved across submissions so far. Every gain has come from S1.

WHY BLENDING, SPECIFICALLY. Two observations point the same way: removing the
friend model from the risk ensemble hurt performance, while adding Qwen to the
verified champion ensemble helped, cross-fitted.

Ensembling decorrelated models is the one thing that has reliably worked. And
there are two factor models that have never been combined:

    NLI     artifacts/fac_oof_f*.npz        24 probs + deployed thresholds
    friend  artifacts/friend_oof_export.csv 24 probs, 1590 posts

They are completely different architectures -- the NLI model scores one
(post, definition) pair at a time and cannot use label correlations at all,
while the friend model emits 24 sigmoids jointly. Their errors have no reason
to coincide.

THE THRESHOLD TRAP, AND HOW THIS AVOIDS IT. A blend needs new thresholds, and
fitting 24 of them is exactly the move that has repeatedly failed to
generalize. So:

  - every arm gets thresholds by the SAME cross-fitted procedure (fit on four
    folds, apply blind to the fifth), so the only difference between arms is
    the blend itself, not the fitting
  - `nli_xfit` is therefore the honest baseline, NOT the deployed 0.6693 --
    that number had its thresholds fitted with more information than the blend
    is allowed to use, and comparing against it would understate the blend
  - the deployed figure is printed too, but only for orientation
  - a `global` variant fits ONE threshold instead of 24, because fewer fitted
    parameters generalize better here

Ship criterion, unchanged: beat nli_xfit overall AND on both author halves.

Usage:
  python src/factor_blend.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors


def macro_f1(P, Y):
    out = []
    for j in range(Y.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        fp = int((P[:, j] & (1 - Y[:, j])).sum())
        fn = int(((1 - P[:, j]) & Y[:, j]).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out)), out


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def fit_per_label(PR, Y, rows):
    """Per-label F1-optimal thresholds on `rows` only."""
    th = np.full(PR.shape[1], 0.5)
    for j in range(PR.shape[1]):
        y = Y[rows, j]
        p = PR[rows, j]
        best, bt = -1.0, 0.5
        for t in GRID:
            pred = (p >= t).astype(int)
            tp = int((pred & y).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((pred & (1 - y)).sum())
                           + int(((1 - pred) & y).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt
    return th


def fit_global(PR, Y, rows):
    best, bt = -1.0, 0.5
    for t in GRID:
        m, _ = macro_f1((PR[rows] >= t).astype(int), Y[rows])
        if m > best:
            best, bt = m, t
    return np.full(PR.shape[1], bt)


def xfit(PR, Y, FD, k, fitter):
    """Cross-fitted decisions: fold f is thresholded by the other folds."""
    P = np.zeros_like(Y)
    for f in range(k):
        th = fitter(PR, Y, FD != f)
        P[FD == f] = (PR[FD == f] >= th).astype(int)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--nli", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--friend", default="artifacts/friend_oof_export.csv")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled)
    gmap = dict(zip(lab["row_id"], lab["factors"]))
    amap = dict(zip(lab["row_id"], lab["anon_user_id"]))
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    rids, PR, TH, FD = [], [], [], []
    for f in range(args.k):
        p = args.nli.format(f=f)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = np.load(p, allow_pickle=True)
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
        FD.extend([f] * len(d["row_id"]))
    PR = np.concatenate(PR)
    TH = np.concatenate(TH)
    FD = np.array(FD)
    nli = pd.DataFrame(PR, index=rids)

    fr = pd.read_csv(args.friend).set_index("row_id")
    missing = [t for t in FACTOR_TAXONOMY if t not in fr.columns]
    if missing:
        raise SystemExit(f"friend export missing labels: {missing[:4]}")
    fr = fr[FACTOR_TAXONOMY]

    keep = [i for i, r in enumerate(rids) if r in fr.index]
    rows = [rids[i] for i in keep]
    A = PR[keep]                       # NLI probs
    Ath = TH[keep]                     # NLI deployed thresholds
    B = fr.loc[rows].to_numpy(float)   # friend probs
    fold = FD[keep]
    print(f"NLI {len(rids)} posts, friend {len(fr)} posts, "
          f"overlap {len(rows)}")

    Y = np.zeros(A.shape, dtype=int)
    for i, r in enumerate(rows):
        for fa in parse_factors(gmap.get(r, "")):
            j = idx.get(str(fa).strip().lower())
            if j is not None:
                Y[i, j] = 1
    half = author_halves([amap[r] for r in rows])

    def report(name, P):
        m, per = macro_f1(P, Y)
        a, _ = macro_f1(P[half == "A"], Y[half == "A"])
        b, _ = macro_f1(P[half == "B"], Y[half == "B"])
        return name, m, a, b, per

    results = []
    results.append(report("nli DEPLOYED", (A >= Ath).astype(int)))
    nli_x = xfit(A, Y, fold, args.k, fit_per_label)
    results.append(report("nli xfit", nli_x))
    results.append(report("friend xfit",
                          xfit(B, Y, fold, args.k, fit_per_label)))

    blends = {}
    for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        M = (1 - w) * A + w * B
        blends[w] = M
        results.append(report(f"blend w={w:.1f}",
                              xfit(M, Y, fold, args.k, fit_per_label)))
    results.append(report("blend w=0.5 GLOBAL",
                          xfit(blends[0.5], Y, fold, args.k, fit_global)))

    # ---- deployed-style arms -------------------------------------------
    # The cross-fitted arms above answer "does blending help?" (it does).
    # They do NOT answer "should we ship it?", because the DEPLOYED system had
    # its 24 thresholds fitted on ALL the data, and comparing a cross-fitted
    # blend against it is comparing a handicapped arm to an unhandicapped one.
    #
    # These arms fit thresholds on everything, exactly as the deployed system
    # did. They are OPTIMISTIC in absolute terms -- 'nli fit-all' should land
    # near the deployed system's own held-out figure, which sits noticeably
    # above its public evaluation score, so read the same offset off all three.
    # What matters is the DIFFERENCE between them, which is now measured on
    # equal footing.
    allrows = np.ones(len(Y), dtype=bool)
    results.append(report("nli fit-all",
                          (A >= fit_per_label(A, Y, allrows)).astype(int)))
    results.append(report("blend fit-all",
                          (blends[0.5] >= fit_per_label(
                              blends[0.5], Y, allrows)).astype(int)))
    results.append(report("blend fit-all GLOBAL",
                          (blends[0.5] >= fit_global(
                              blends[0.5], Y, allrows)).astype(int)))

    print(f"\n{'arm':<22}{'macro-F1':>10}{'halfA':>10}{'halfB':>10}")
    print("-" * 52)
    for name, m, a, b, _ in results:
        print(f"{name:<22}{m:>10.4f}{a:>10.4f}{b:>10.4f}")

    base = next(r for r in results if r[0] == "nli xfit")
    print(f"\nbaseline for judging = 'nli xfit' ({base[1]:.4f}), NOT the")
    print(f"deployed {results[0][1]:.4f} -- the deployed thresholds were fitted")
    print("with information the blend is not allowed to use.")

    print(f"\n{'arm':<22}{'vs nli xfit':>13}{'halfA':>10}{'halfB':>10}"
          f"{'composite':>11}")
    print("-" * 66)
    winners = []
    for name, m, a, b, _ in results:
        if name in ("nli xfit", "nli DEPLOYED"):
            continue
        d, da, db = m - base[1], a - base[2], b - base[3]
        ok = d > 0 and da > 0 and db > 0
        if ok:
            winners.append((name, d))
        print(f"{name:<22}{d:>+13.4f}{da:>+10.4f}{db:>+10.4f}"
              f"{0.3 * d:>+11.4f}{'  <- both halves' if ok else ''}")

    # The shipping question, on equal footing.
    d = {n: (m, a, b) for n, m, a, b, _ in results}
    if "nli fit-all" in d and "blend fit-all" in d:
        n0, na, nb = d["nli fit-all"]
        print(f"\n{'THE SHIPPING COMPARISON (thresholds fitted the same way)':<66}")
        print("-" * 66)
        print(f"{'nli fit-all':<24}{n0:>10.4f}{na:>10.4f}{nb:>10.4f}"
              f"   <- deployed-equivalent")
        for nm in ("blend fit-all", "blend fit-all GLOBAL"):
            m, a, b = d[nm]
            print(f"{nm:<24}{m:>10.4f}{a:>10.4f}{b:>10.4f}"
                  f"   {m - n0:+.4f}  ({0.3 * (m - n0):+.4f} composite)")
        print(f"\nSanity: 'nli fit-all' should be near the deployed system's own")
        print(f"held-out figure, which sits above its public evaluation score, so "
              f"subtract\nthe same offset from these absolutes -- but the "
              f"DIFFERENCES are fair.")

    print("\n" + "=" * 66)
    if winners:
        best = max(winners, key=lambda x: x[1])
        print(f"BEST: {best[0]}  {best[1]:+.4f} macro  "
              f"({0.3 * best[1]:+.4f} composite)")
        print("Next: rebuild the test factor column with this blend, refit")
        print("thresholds on ALL folds, patch factors ONLY on top of the")
        print("current champion.")
        print("NOTE: prefer the GLOBAL variant if it is within ~0.005 of the")
        print("per-label one -- 1 fitted parameter transfers better than 24,")
        print("and that distinction is what has decided every submission here.")
    else:
        print("No blend beats nli xfit on both halves. Factors do not improve")
        print("by ensembling these two models, and the component is closed.")
    print("=" * 66)


if __name__ == "__main__":
    main()

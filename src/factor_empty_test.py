"""
factor_empty_test.py

Does FORCING at least one factor onto every post help or hurt?

THE STRUCTURAL FACT. In gold, 128 of 1635 posts (7.8%) carry NO factors at all.
The deployed submission has ZERO empty factor cells -- every one of 378 test
posts is given at least one label. If the test set has the same 7.8% rate, that
is roughly 29 posts receiving a guaranteed false positive.

This is the last zero-parameter idea remaining, and zero-parameter is the
only category that has worked: more heavily fitted threshold changes have
transferred from val to test only partially, or negatively, and the damage
grows with the number of fitted parameters.

Forcing or not forcing is a single binary convention with nothing fitted.

WHY IT MIGHT STILL LOSE. Macro-F1 is computed per label, so an unnecessary
positive on a RARE label costs much more than one on a common label. Forcing
picks the argmax, which for an all-below-threshold post is usually a common
label like hopelessness -- where one extra false positive among 745 gold
positives is nearly free. So the harm may be smaller than the 29-post count
suggests, and dropping the force could also lose the occasional post where the
argmax happened to be right.

This measures both directions on 1635 OOF posts -- the same data the incumbent
thresholds were fitted on, not the 372-post val split that inverted.

Usage:
  python src/factor_empty_test.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    rids, PR, TH = [], [], []
    for f in range(args.k):
        p = args.oof.format(f=f)
        d = np.load(p, allow_pickle=True)
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
    PR, TH = np.concatenate(PR), np.concatenate(TH)

    Y = np.zeros(PR.shape, dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    half = author_halves([lab.loc[r, "anon_user_id"] for r in rids])

    natural = (PR >= TH).astype(int)
    n_empty_pred = int((natural.sum(1) == 0).sum())
    n_empty_gold = int((Y.sum(1) == 0).sum())
    print(f"{len(rids)} OOF posts")
    print(f"  gold posts with no factors      : {n_empty_gold} "
          f"({100 * n_empty_gold / len(rids):.1f}%)")
    print(f"  natural predictions with none   : {n_empty_pred} "
          f"({100 * n_empty_pred / len(rids):.1f}%)")

    forced = natural.copy()
    for i in range(len(rids)):
        if forced[i].sum() == 0:
            forced[i, int(PR[i].argmax())] = 1

    # how often is the forced label actually right?
    hits = sum(1 for i in range(len(rids))
               if natural[i].sum() == 0 and Y[i, int(PR[i].argmax())] == 1)
    if n_empty_pred:
        print(f"  forced label correct            : {hits}/{n_empty_pred} "
              f"({100 * hits / n_empty_pred:.0f}%)")
        which = {}
        for i in range(len(rids)):
            if natural[i].sum() == 0:
                t = FACTOR_TAXONOMY[int(PR[i].argmax())]
                which[t] = which.get(t, 0) + 1
        print("  forced onto which labels:")
        for t, c in sorted(which.items(), key=lambda x: -x[1]):
            print(f"      {t:<44}{c:>4}   (gold n={int(Y[:, idx[t]].sum())})")

    print(f"\n{'arm':<22}{'macro':>9}{'delta':>9}{'halfA':>9}{'halfB':>9}"
          f"{'composite':>11}")
    print("-" * 69)
    bf, _ = macro_f1(forced, Y)
    ba = macro_f1(forced[half == "A"], Y[half == "A"])[0]
    bb = macro_f1(forced[half == "B"], Y[half == "B"])[0]
    print(f"{'FORCED (deployed)':<22}{bf:>9.4f}{0.0:>+9.4f}{0.0:>+9.4f}"
          f"{0.0:>+9.4f}{0.0:>+11.4f}")
    m, _ = macro_f1(natural, Y)
    a = macro_f1(natural[half == "A"], Y[half == "A"])[0]
    b = macro_f1(natural[half == "B"], Y[half == "B"])[0]
    print(f"{'NATURAL (allow empty)':<22}{m:>9.4f}{m - bf:>+9.4f}"
          f"{a - ba:>+9.4f}{b - bb:>+9.4f}{0.3 * (m - bf):>+11.4f}")

    print("\n" + "=" * 69)
    if m > bf and a > ba and b > bb:
        print("ALLOWING EMPTY WINS, on both halves, with zero fitted")
        print("parameters. Build it by passing --allow-empty to")
        print("src/factor_build_test.py, or by dropping the force step when")
        print("assembling the factor column.")
    else:
        print("Forcing is not the problem. The argmax label on an")
        print("all-below-threshold post lands on a COMMON label, where one")
        print("extra false positive among hundreds of gold positives costs")
        print("almost nothing in macro-F1 -- and it occasionally hits.")
    print("=" * 69)


if __name__ == "__main__":
    main()

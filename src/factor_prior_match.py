"""
factor_prior_match.py

Fix the factor model's systematic over-prediction WITHOUT tuning anything.

THE OBSERVATION. factor_label_diagnosis.py --by-risk shows recall exceeding
precision on nearly every one of the 24 labels, and predicted count exceeding
gold count on 20 of 24:

    psychological capital     P 0.486  R 0.777   pred/gold 1.6
    interpersonal violence    P 0.544  R 0.819   pred/gold 1.5
    mental health issues      P 0.672  R 0.846   pred/gold 1.3
    emotion dysregulation     P 0.647  R 0.838   pred/gold 1.3

That is not 24 independent quirks, it is one systematic bias: the deployed
thresholds sit too low. The same overfitting logic explains why OOF scores
run higher than test: 24 separately fitted cut-offs are 24 chances to
overfit, and labels like `sexual orientation related issues` have EIGHT gold
positives to fit on.

THE FIX, AND WHY IT DOES NOT REPEAT THE PATTERN THAT HAS REPEATEDLY FAILED.

  Threshold changes that SEARCH for cut-offs maximising a score on held-out
  data have consistently failed to generalize. This does not search. For each
  label it sets the threshold
  at the quantile that makes

      (number of posts predicted positive) == (base rate) x (number of posts)

  The base rate comes from the training folds; the quantile comes from the
  probability distribution. There is no objective being maximised and no free
  parameter per label, so there is nothing to overfit. It is the same class of
  change as the evidence consistency fix: enforcing a property the data is
  known to have, rather than fitting a number.

ARMS COMPARED, all cross-fitted (statistics from four folds, applied to the
fifth), all judged on author-disjoint halves:

  current   the deployed per-label thresholds          (24 fitted parameters)
  prior     prior-matched quantile thresholds          (0 fitted parameters)
  global    one threshold shared by all labels         (1 fitted parameter)
  scaled    prior-matched, then relaxed by a factor    (1 fitted parameter)

`scaled` exists because exact prior matching is not necessarily F1-optimal --
for a label the model ranks poorly, predicting slightly MORE than the base rate
can still raise F1. One shared multiplier across all 24 labels is a single
parameter, which is a different risk profile from 24.

Usage:
  python src/factor_prior_match.py
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


def prior_threshold(pr_te, rate):
    """Threshold on this fold's probabilities such that the predicted-positive
    count equals rate * n. Zero fitted parameters: `rate` is measured on the
    other folds, the quantile is read off the distribution."""
    n = len(pr_te)
    k = int(round(rate * n))
    if k <= 0:
        return 1.01                       # predict nothing
    if k >= n:
        return -0.01                      # predict everything
    return float(np.partition(pr_te, n - k)[n - k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--factor-oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled)
    gmap = dict(zip(lab["row_id"], lab["factors"]))
    amap = dict(zip(lab["row_id"], lab["anon_user_id"]))
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    rids, PR, TH, FD = [], [], [], []
    for f in range(args.k):
        p = args.factor_oof.format(f=f)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = np.load(p, allow_pickle=True)
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
        FD.extend([f] * len(d["row_id"]))
    PR, TH, FD = np.concatenate(PR), np.concatenate(TH), np.array(FD)

    Y = np.zeros(PR.shape, dtype=int)
    for i, r in enumerate(rids):
        for fa in parse_factors(gmap.get(r, "")):
            j = idx.get(str(fa).strip().lower())
            if j is not None:
                Y[i, j] = 1
    half = author_halves(np.array([amap[r] for r in rids]))
    L = Y.shape[1]

    arms = {"current": (PR >= TH).astype(int)}

    # ---- prior-matched, and prior-matched x scale -----------------------
    for scale in (1.0, 1.1, 1.2, 1.3):
        P = np.zeros_like(Y)
        for f in range(args.k):
            tr, te = FD != f, FD == f
            for j in range(L):
                rate = float(Y[tr, j].mean()) * scale
                t = prior_threshold(PR[te, j], min(rate, 1.0))
                P[te, j] = (PR[te, j] >= t).astype(int)
        arms["prior" if scale == 1.0 else f"prior x{scale}"] = P

    # ---- one global threshold, chosen on the training folds -------------
    P = np.zeros_like(Y)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    picked = []
    for f in range(args.k):
        tr, te = FD != f, FD == f
        best = max(grid, key=lambda t: macro_f1((PR[tr] >= t).astype(int), Y[tr])[0])
        picked.append(float(best))
        P[te] = (PR[te] >= best).astype(int)
    arms["global"] = P

    print(f"{'arm':<14}{'macro-F1':>10}{'delta':>9}{'halfA':>9}{'halfB':>9}"
          f"{'pred/gold':>11}")
    base, _ = macro_f1(arms["current"], Y)
    ba = macro_f1(arms["current"][half == "A"], Y[half == "A"])[0]
    bb = macro_f1(arms["current"][half == "B"], Y[half == "B"])[0]
    results = {}
    for name, P in arms.items():
        m, per = macro_f1(P, Y)
        a = macro_f1(P[half == "A"], Y[half == "A"])[0]
        b = macro_f1(P[half == "B"], Y[half == "B"])[0]
        results[name] = (m, a - ba, b - bb, per)
        print(f"{name:<14}{m:>10.4f}{m - base:>+9.4f}"
              f"{a - ba:>+9.4f}{b - bb:>+9.4f}{P.sum() / Y.sum():>11.2f}")
    print(f"\nglobal threshold picked per fold: {picked}")

    best = max((n for n in arms if n != "current"), key=lambda n: results[n][0])
    m, da, db, per = results[best]
    print(f"\nbest non-current arm: {best}  ({m - base:+.4f} macro)")
    print(f"  -> {0.3 * (m - base):+.4f} composite if it transfers")

    _, per0 = macro_f1(arms["current"], Y)
    print(f"\nPER-LABEL, current -> {best}")
    print(f"{'factor':<46}{'gold':>6}{'cur':>8}{'new':>8}{'delta':>8}")
    for j, t in enumerate(FACTOR_TAXONOMY):
        print(f"{t:<46}{int(Y[:, j].sum()):>6}{per0[j]:>8.3f}"
              f"{per[j]:>8.3f}{per[j] - per0[j]:>+8.3f}")

    print("\n" + "=" * 78)
    if m > base and da > 0 and db > 0:
        print(f"SHIP CRITERION MET for '{best}': positive overall AND on both "
              f"author halves.")
        print("Next: rebuild the test factor column with this rule and patch")
        print("factors ONLY on top of NIDHI_evidence_consistent.csv (0.7422).")
        if best.startswith("prior") and best != "prior":
            print(f"NOTE: '{best}' has ONE fitted parameter (the scale). "
                  f"'prior' has zero.")
            print(f"      'prior' scores {results['prior'][0]:.4f} vs "
                  f"{m:.4f}. If they are within ~0.005, ship 'prior' -- "
                  f"fewer fitted parameters transfer better, which is the "
                  f"whole reason this arm exists.")
    else:
        print("Ship criterion NOT met. The over-prediction is real but "
              "correcting it does not pay on this metric.")
    print("=" * 78)


if __name__ == "__main__":
    main()

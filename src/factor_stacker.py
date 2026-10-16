"""
factor_stacker.py

Let each factor see the other 23. The NLI model cannot.

THE STRUCTURAL GAP. train_factors_nli scores one (post, definition) pair at a
time and emits one logit. By construction the decision for `low self-esteem`
knows nothing about the score for `hopelessness`, even though those two
co-occur constantly. Every correlation between labels is discarded at exactly
the point it could help.

That is not a hyperparameter or a threshold -- it is a property of the
architecture, and it sits on the ranking of the per-label scores rather than
on where they are cut.

A stacker is the cheapest possible attack on it: a per-label logistic
regression whose inputs are all 24 probabilities. Label j's decision can now
depend on labels 1..24, which is precisely the information the cross-encoder
threw away.

WHY IT MIGHT NOT WORK. The base probabilities already encode most of what the
post says; correlations may add nothing beyond what each label's own score
carries. And a stacker fitted badly is just another 24-parameter overfit --
which is why everything below is cross-fitted.

CONTROLS
  - the stacker is fitted on four folds and applied to the fifth, never on the
    posts it scores
  - the baseline is the incumbent's own deployed decisions on the same posts
  - both author halves must improve, not just the mean
  - `+risk` is reported separately because predicted risk is an extra input the
    NLI model never had, so it is a different experiment from pure stacking

Usage:
  python src/factor_stacker.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, parse_factors, FACTOR_TAXONOMY

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def macro(P, Y):
    out = []
    for j in range(Y.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        out.append(0.0 if tp == 0 else 2 * tp /
                   (2 * tp + int((P[:, j] & (1 - Y[:, j])).sum())
                    + int(((1 - P[:, j]) & Y[:, j]).sum())))
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


def fit_th(PR, Y, rows):
    th = np.full(PR.shape[1], 0.5)
    for j in range(PR.shape[1]):
        best, bt = -1.0, 0.5
        for t in GRID:
            p = (PR[rows, j] >= t).astype(int)
            tp = int((p & Y[rows, j]).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((p & (1 - Y[rows, j])).sum())
                           + int(((1 - p) & Y[rows, j]).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--risk-oof", default="artifacts/nli_risk_oof_all.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    rids, PR, TH, FD = [], [], [], []
    for f in range(args.k):
        d = np.load(args.oof.format(f=f), allow_pickle=True)
        n = len(d["row_id"])
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (n, 1)))
        FD.extend([f] * n)
    PR, TH, FD = np.concatenate(PR), np.concatenate(TH), np.array(FD)
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros(PR.shape, dtype=int)
    for i, r in enumerate(rids):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1
    half = author_halves([lab.loc[r, "anon_user_id"] for r in rids])
    A, B = half == "A", half == "B"
    print(f"{len(rids)} OOF posts, {Y.shape[1]} labels, "
          f"halves {int(A.sum())}/{int(B.sum())}")

    # optional extra input: predicted risk, which the factor model never sees
    R = None
    if os.path.exists(args.risk_oof):
        rk = pd.read_csv(args.risk_oof).set_index("row_id")
        cols = [f"prob_{c}" for c in RISK_LEVELS]
        if all(c in rk.columns for c in cols) and all(r in rk.index for r in rids):
            R = rk.loc[rids, cols].to_numpy(float)
            print(f"risk features available from {args.risk_oof}")
    if R is None:
        print("no risk OOF found -- running probs-only arm")

    base = (PR >= TH).astype(int)
    b0, per0 = macro(base, Y)
    bA = macro(base[A], Y[A])[0]
    bB = macro(base[B], Y[B])[0]

    def stack(X):
        """Cross-fitted: each fold's predictions come from a model fitted on
        the other four."""
        out = np.zeros_like(PR)
        for f in range(args.k):
            tr, te = FD != f, FD == f
            for j in range(PR.shape[1]):
                y = Y[tr, j]
                if y.sum() < 3 or y.sum() == len(y):
                    out[te, j] = PR[te, j]      # too few positives to fit
                    continue
                m = LogisticRegression(max_iter=2000, C=args.C)
                m.fit(X[tr], y)
                out[te, j] = m.predict_proba(X[te])[:, 1]
        return out

    arms = {"stack (24 probs)": stack(PR)}
    if R is not None:
        arms["stack (24 probs + risk)"] = stack(np.hstack([PR, R]))

    print(f"\n{'arm':<26}{'macro':>9}{'delta':>9}{'halfA':>9}{'halfB':>9}")
    print("-" * 62)
    print(f"{'INCUMBENT':<26}{b0:>9.4f}{0.0:>+9.4f}{0.0:>+9.4f}{0.0:>+9.4f}")
    best = None
    for nm, S in arms.items():
        # thresholds for the stacked scores, also cross-fitted by half
        P = np.zeros_like(Y)
        P[A] = (S[A] >= fit_th(S, Y, B)).astype(int)
        P[B] = (S[B] >= fit_th(S, Y, A)).astype(int)
        m, per = macro(P, Y)
        da = macro(P[A], Y[A])[0] - bA
        db = macro(P[B], Y[B])[0] - bB
        ok = m > b0 and da > 0 and db > 0
        if ok and (best is None or m > best[1]):
            best = (nm, m, per)
        print(f"{nm:<26}{m:>9.4f}{m - b0:>+9.4f}{da:>+9.4f}{db:>+9.4f}"
              f"{'  <- both halves' if ok else ''}")

    if best:
        nm, m, per = best
        print(f"\nPER-LABEL, incumbent -> {nm}  (movers only)")
        for j, t in enumerate(FACTOR_TAXONOMY):
            if abs(per[j] - per0[j]) > 0.01:
                print(f"  {t:<44}{per0[j]:>7.3f}{per[j]:>8.3f}"
                      f"{per[j] - per0[j]:>+8.3f}")
        print(f"\n  -> {0.3 * (m - b0):+.4f} composite")
        print("\nDISCOUNT IT. A cross-fitted OOF gain is measured on fold")
        print("models, while deployment uses full-data models, so the margin")
        print("attenuates on held-out data.")
    else:
        print("\nNo stacker arm wins on both halves. Label correlations add")
        print("nothing the per-label scores did not already carry, and the")
        print("architecture's independence assumption is not the binding")
        print("constraint on the factor score.")

    print("\nNOTE: the incumbent row uses thresholds tuned on each fold's own")
    print("held-out set, so it is mildly flattered. The stacker rows are")
    print("cross-fitted by author half. If a stacker wins despite that, the")
    print("margin is real.")


if __name__ == "__main__":
    main()

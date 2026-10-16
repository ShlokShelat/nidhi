"""
factor_cooccurrence_stack.py

The 24 factors are decided INDEPENDENTLY. They do not occur independently.

Gold averages 2.92 factors per post and the labels travel in packs --
`hopelessness`, `low self-esteem` and `poor social support` co-occur far above
chance. A stage-2 model that predicts each label from ALL 24 stage-1
probabilities can use that structure; 24 separate thresholds cannot. This is the
last structural idea on factors that needs no new training data and no GPU: the
OOF dumps already contain 1635 x 24 held-out probabilities, which is exactly the
training set a stacker needs.

WHY IT MIGHT WORK WHERE FIVE OTHER IDEAS DID NOT. Everything tried today
attacked the READER (better definitions, an LLM, rationale spans) or the CUTOFF
(refit, rate-match, shrinkage, plug-in). Both were dead ends -- the reader is
already good on concrete labels and the cutoffs are already near-optimal. This
attacks neither. It changes what INFORMATION each decision gets to use, which is
the one axis nobody has touched.

WHY IT MIGHT NOT. Stage-1 probabilities already encode much of the
co-occurrence, because the encoder read the whole post for every hypothesis. And
a stacker fitted on 1635 posts with 24 inputs per label has real overfitting
capacity -- which is why every number below is cross-fitted over AUTHOR halves,
and why a plain logistic model is offered alongside the gradient-boosted one.

HONEST ABOUT LEAKAGE. The stage-1 probabilities here are out-of-fold, so no post
was ever seen by the model that scored it. The stacker is then fitted on one
author half and applied to the other. Nothing is fitted and scored on the same
posts at any stage.

THE BAR, unchanged. Paired author bootstrap: CI excluding zero, at least 80% of
resamples won, AND clearing the ~0.014 null band measured from the mnlirep
replica. A gain smaller than the difference between two identical training runs
is not a gain.

Usage:
  python src/factor_cooccurrence_stack.py --arm mnli
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def gold_matrix(df):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(df), N), dtype=np.int8)
    for i, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def per_label_f1(P, Y):
    tp = (P & Y).sum(0).astype(float)
    fp = (P & (1 - Y)).sum(0).astype(float)
    fn = ((1 - P) & Y).sum(0).astype(float)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def macro(P, Y):
    return float(per_label_f1(P, Y).mean())


def argmax_threshold(pr, y):
    npos = int(y.sum())
    if npos == 0:
        return 1.01
    o = np.argsort(-pr, kind="stable")
    cum = np.cumsum(y[o])
    ks = np.searchsorted(-pr[o], -GRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    f1 = np.where(ks + npos > 0, 2.0 * tp / (ks + npos), 0.0)
    return float(GRID[int(np.argmax(f1))])


def author_halves(auth):
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        p = "A" if size["A"] <= size["B"] else "B"
        half[a] = p
        size[p] += c[a]
    return s.map(half).to_numpy()


def binarise_xfit(Pr, Y, A, B):
    X = np.zeros_like(Y)
    for src, dst in ((B, A), (A, B)):
        th = np.array([argmax_threshold(Pr[src, j], Y[src, j])
                       for j in range(N)])
        X[dst] = (Pr[dst] >= th).astype(np.int8)
    return X


def stack(Pr, Y, A, B, kind):
    """Stage-2 probabilities, cross-fitted: fit on one author half, predict the
    other. Input for label j is ALL 24 stage-1 probabilities."""
    from sklearn.linear_model import LogisticRegression
    out = np.zeros_like(Pr)
    X = np.log(np.clip(Pr, 1e-6, 1 - 1e-6) /
               (1 - np.clip(Pr, 1e-6, 1 - 1e-6)))     # logits are better inputs
    for src, dst in ((B, A), (A, B)):
        for j in range(N):
            y = Y[src, j]
            if y.sum() < 8 or y.sum() == len(y):
                out[dst, j] = Pr[dst, j]               # too rare to stack
                continue
            if kind == "logistic":
                m = LogisticRegression(max_iter=1000, C=0.25,
                                       class_weight="balanced")
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                m = HistGradientBoostingClassifier(
                    max_iter=120, max_depth=3, learning_rate=0.08,
                    min_samples_leaf=20, l2_regularization=1.0,
                    random_state=0)
            m.fit(X[src], y)
            out[dst, j] = m.predict_proba(X[dst])[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mnli")
    ap.add_argument("--dir", default="artifacts/factor_oof")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Pr, rid = [], []
    for f in range(args.k):
        p = os.path.join(args.dir, f"{args.arm}_f{f}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        z = np.load(p, allow_pickle=True)
        Pr.append(np.asarray(z["probs"], dtype=float))
        rid += list(z["row_id"])
    Pr = np.concatenate(Pr, axis=0)

    lab = pd.read_csv(args.labeled, keep_default_na=False).set_index("row_id")
    ev = lab.loc[rid].reset_index()
    Y = gold_matrix(ev)
    auth = ev["anon_user_id"].to_numpy()
    half = author_halves(auth)
    A, B = half == "A", half == "B"

    print(f"arm {args.arm}   {len(ev)} OOF posts   {pd.unique(auth).size} "
          f"authors   gold {Y.sum(1).mean():.2f} factors/post")

    # how much co-occurrence structure is actually there
    C = np.corrcoef(Y.T)
    np.fill_diagonal(C, 0.0)
    iu = np.triu_indices(N, 1)
    strong = sorted(zip(C[iu], *[x[iu] for x in np.indices((N, N))]),
                    key=lambda t: -abs(t[0]))[:6]
    print("\nstrongest gold label correlations (the structure being exploited):")
    for r, i, j in strong:
        print(f"  {r:+.3f}  {FACTOR_TAXONOMY[i]}  <->  {FACTOR_TAXONOMY[j]}")

    base_P = binarise_xfit(Pr, Y, A, B)
    base = macro(base_P, Y)
    preds = {"stage1": base_P}
    print(f"\n{'variant':<14}{'xfit macro':>12}{'d vs stage1':>13}"
          f"{'fac/post':>10}")
    print("-" * 50)
    print(f"{'stage1':<14}{base:>12.4f}{'':>13}{base_P.sum(1).mean():>10.2f}")

    for kind in ("logistic", "gbm"):
        P2 = stack(Pr, Y, A, B, kind)
        X = binarise_xfit(P2, Y, A, B)
        preds[kind] = X
        print(f"{kind:<14}{macro(X, Y):>12.4f}{macro(X, Y) - base:>+13.4f}"
              f"{X.sum(1).mean():>10.2f}")

    uniq = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uniq}
    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.boot) for k in preds}
    for b in range(args.boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        ix = np.concatenate([by[uniq[k]] for k in pick])
        Yb = Y[ix]
        for k, P in preds.items():
            boot[k][b] = macro(P[ix], Yb)

    print("\n" + "=" * 70)
    print(f"PAIRED delta vs stage1   ({args.boot} author-bootstrap resamples)")
    print("=" * 70)
    print(f"{'variant':<14}{'delta':>9}{'95% CI':>21}{'win%':>8}{'verdict':>16}")
    print("-" * 70)
    for k in preds:
        if k == "stage1":
            continue
        d = boot[k] - boot["stage1"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        win = float((d > 0).mean()) * 100
        pt = macro(preds[k], Y) - base
        ci_ok = lo > 0 and win >= 80
        verdict = ("SHIPPABLE" if ci_ok and pt > 0.014 else
                   "CI ok, under band" if ci_ok else "no")
        print(f"{k:<14}{pt:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>21}{win:>7.1f}%"
              f"{verdict:>16}")
    print("\n'CI ok, under band' means real but smaller than the gap between two")
    print("identical training runs (~0.014) -- not worth a submission slot.")

    best = max((k for k in preds if k != "stage1"),
               key=lambda k: macro(preds[k], Y))
    f1, f2 = per_label_f1(preds["stage1"], Y), per_label_f1(preds[best], Y)
    print("\n" + "=" * 70)
    print(f"PER LABEL -- stage1 vs {best}")
    print("=" * 70)
    print(f"{'factor':<40}{'gold':>6}{'stage1':>9}{best[:8]:>9}{'delta':>8}")
    for j in np.argsort(f2 - f1):
        print(f"{FACTOR_TAXONOMY[j]:<40}{int(Y[:, j].sum()):>6}{f1[j]:>9.3f}"
              f"{f2[j]:>9.3f}{f2[j] - f1[j]:>+8.3f}")


if __name__ == "__main__":
    main()

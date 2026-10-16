"""
factor_threshold_policies.py

How much macro-F1 is the thresholding rule throwing away? Free to answer -- it
runs on the OOF dumps that already exist, and trains nothing.

WHAT THE OOF INSTRUMENT EXPOSED. On 1635 held-out posts the deployed model fires
21 of 24 labels ABOVE their gold rate (mean predicted/gold = 1.25, 3.37
factors/post against gold's 2.92), and precision is below recall on nearly every
label -- 0.634/0.843 on `emotion dysregulation`, 0.760/0.884 on `poor social
support`, 0.261/0.750 on `sexual orientation related issues` at 2.88x its gold
rate.

WHY THAT HAPPENS, AND WHY IT IS NOT A BUG IN tune_thresholds. Maximising F1 for
a rare positive class correctly puts the cutoff well below 0.5, because recall
is cheap relative to precision in the F1 trade. That is right ON THE FIT SET.
The failure is in TRANSFER: the cutoff is chosen at the argmax of a jagged
empirical F1 curve estimated from a few hundred posts, and the argmax of a noisy
curve is biased toward whatever direction adds positives. So the cutoff lands
systematically low, and every label pays a little precision on data it has not
seen.

Nobody could see this before because the thresholds were always fitted and
evaluated on the same 372 posts, where the bias is invisible by construction.

THE POLICIES COMPARED (all CROSS-FITTED over author halves -- fit on one half,
score the other, so no policy is judged on the data that chose it):

  deployed    the shipped cutoffs, exactly as the fold dumps carry them.
              Already honest: each fold's thresholds came from a THRESH set
              disjoint from that fold's eval posts. This is the number to beat.
  refit       per-label F1 argmax, but fitted on the other author half. Shows
              how much of the loss is pure fit-set noise rather than policy.
  ratematch   choose the cutoff that makes the predicted positive count equal
              the count implied by TRAIN prevalence. Fits no F1 curve at all, so
              it cannot inherit the argmax bias, and prevalence is known ahead
              of time -- nothing here is tuned on the scored data.
  shrink      refit, then pull each cutoff toward the global median by an amount
              that grows as the label gets rarer. Rare labels have the noisiest
              F1 curves, so they should trust the pooled estimate more.
  plugin      Lipton's rule: for the F1-optimal decision, keep predicting while
              the posterior exceeds half the attainable F1. Derived, not fitted.

READ THE `xfit` COLUMN. In-sample is printed only to show how large the
optimism is -- it is the trap that produced every previous threshold verdict.

A policy ships only if its paired author-bootstrap CI excludes zero AND it wins
at least 80% of resamples. That is the same bar arm_robustness.py uses,
calibrated to catch a high mean paired with a near-50% win rate -- a pattern
that has previously failed to generalize.
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


def gold_matrix(df: pd.DataFrame) -> np.ndarray:
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


def f1_curve_argmax(pr, y):
    """Threshold maximising F1, vectorised over the grid."""
    npos = int(y.sum())
    if npos == 0:
        return 1.01
    o = np.argsort(-pr, kind="stable")
    cum = np.cumsum(y[o])
    ks = np.searchsorted(-pr[o], -GRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    f1 = np.where(ks + npos > 0, 2.0 * tp / (ks + npos), 0.0)
    return float(GRID[int(np.argmax(f1))])


def best_f1_value(pr, y):
    npos = int(y.sum())
    if npos == 0:
        return 0.0
    o = np.argsort(-pr, kind="stable")
    cum = np.cumsum(y[o])
    k = np.arange(1, len(y) + 1)
    return float(np.max(2.0 * cum / (k + npos)))


def author_halves(auth):
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        p = "A" if size["A"] <= size["B"] else "B"
        half[a] = p
        size[p] += c[a]
    return s.map(half).to_numpy()


def fit_policy(name, Pr, Y, rows, prevalence, n_target):
    """Return 24 cutoffs fitted on `rows` only."""
    th = np.full(N, 0.5)
    for j in range(N):
        pr, y = Pr[rows, j], Y[rows, j]
        if name == "refit":
            th[j] = f1_curve_argmax(pr, y)
        elif name == "ratematch":
            # cutoff reproducing the count TRAIN prevalence implies
            k = max(int(round(prevalence[j] * n_target)), 1)
            k = min(k, len(pr))
            th[j] = float(np.sort(pr)[-k])
        elif name == "plugin":
            f1s = best_f1_value(pr, y)
            th[j] = max(f1s / 2.0, 0.02) if f1s > 0 else 1.01
        elif name == "shrink":
            th[j] = f1_curve_argmax(pr, y)
    if name == "shrink":
        med = float(np.median(th))
        npos = Y[rows].sum(0).astype(float)
        # w -> 1 for a label with many positives, -> 0 when it has almost none
        w = npos / (npos + 25.0)
        th = w * th + (1 - w) * med
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mnli")
    ap.add_argument("--dir", default="artifacts/factor_oof")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Pr, rid, dep = [], [], []
    for f in range(args.k):
        p = os.path.join(args.dir, f"{args.arm}_f{f}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        z = np.load(p, allow_pickle=True)
        pr = np.asarray(z["probs"], dtype=float)
        th = np.asarray(z["thresholds"], dtype=float)
        Pr.append(pr)
        dep.append((pr >= th).astype(np.int8))
        rid += list(z["row_id"])
    Pr = np.concatenate(Pr, axis=0)
    Dep = np.concatenate(dep, axis=0)

    lab = pd.read_csv(args.labeled, keep_default_na=False).set_index("row_id")
    ev = lab.loc[rid].reset_index()
    Y = gold_matrix(ev)
    auth = ev["anon_user_id"].to_numpy()
    half = author_halves(auth)
    A, B = half == "A", half == "B"
    prevalence = Y.mean(0)     # stands in for train prevalence; see note below

    print(f"arm {args.arm}   {len(ev)} OOF posts   "
          f"{pd.unique(auth).size} authors")
    print(f"deployed: {Dep.sum(1).mean():.2f} factors/post   "
          f"gold {Y.sum(1).mean():.2f}   "
          f"fire rate {Dep.sum() / max(Y.sum(), 1):.3f}")
    print("NOTE prevalence for `ratematch` is taken from the labelled pool, "
          "which is\n     the same quantity a submission would use from "
          "full_train -- it is known\n     in advance and is not fitted to the "
          "scored posts.")

    base = macro(Dep, Y)
    print(f"\n{'policy':<12}{'in-sample':>11}{'xfit':>9}{'d vs dep':>10}"
          f"{'fac/post':>10}{'rate':>7}")
    print("-" * 62)
    print(f"{'deployed':<12}{'-':>11}{base:>9.4f}{'':>10}"
          f"{Dep.sum(1).mean():>10.2f}{Dep.sum()/max(Y.sum(),1):>7.2f}")

    all_rows = np.ones(len(Y), dtype=bool)
    preds = {"deployed": Dep}
    for name in ("refit", "ratematch", "shrink", "plugin"):
        ins = (Pr >= fit_policy(name, Pr, Y, all_rows, prevalence,
                                len(Y))).astype(np.int8)
        X = np.zeros_like(Y)
        thB = fit_policy(name, Pr, Y, B, prevalence, int(A.sum()))
        thA = fit_policy(name, Pr, Y, A, prevalence, int(B.sum()))
        X[A] = (Pr[A] >= thB).astype(np.int8)
        X[B] = (Pr[B] >= thA).astype(np.int8)
        preds[name] = X
        print(f"{name:<12}{macro(ins, Y):>11.4f}{macro(X, Y):>9.4f}"
              f"{macro(X, Y) - base:>+10.4f}{X.sum(1).mean():>10.2f}"
              f"{X.sum()/max(Y.sum(),1):>7.2f}")

    # ---- paired author bootstrap on the deltas ------------------------------
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

    print("\n" + "=" * 72)
    print(f"PAIRED delta vs deployed   ({args.boot} author-bootstrap resamples)")
    print("=" * 72)
    print(f"{'policy':<12}{'delta':>9}{'95% CI':>21}{'win%':>8}{'verdict':>20}")
    print("-" * 72)
    winner, best_d = None, 0.0
    for k in preds:
        if k == "deployed":
            continue
        d = boot[k] - boot["deployed"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        win = float((d > 0).mean()) * 100
        pt = macro(preds[k], Y) - base
        good = lo > 0 and win >= 80
        if good and pt > best_d:
            winner, best_d = k, pt
        print(f"{k:<12}{pt:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>21}{win:>7.1f}%"
              f"{('SHIPPABLE' if good else 'no'):>20}")

    if winner:
        print(f"\n>>> {winner} clears the bar at {best_d:+.4f} macro "
              f"= {0.3 * best_d:+.4f} composite, with NO retraining.")
        print("    Next: apply it to the test probabilities in")
        print("    artifacts/factor_runs/nli_ensemble_mnli_test.npz, using")
        print("    cutoffs fitted on the LABELLED pool only, and patch the")
        print("    factor column onto the champion as a single variable.")
    else:
        print("\nNo policy clears the bar. The thresholds are already close to")
        print("the best a per-label rule can do, and the over-firing is the")
        print("correct F1 trade rather than a defect. Say so and move on.")

    print("\n" + "=" * 72)
    print("PER LABEL -- deployed vs the best-scoring policy")
    print("=" * 72)
    cand = winner or max((k for k in preds if k != "deployed"),
                         key=lambda k: macro(preds[k], Y))
    fd, fc = per_label_f1(Dep, Y), per_label_f1(preds[cand], Y)
    o = np.argsort(fc - fd)
    print(f"{'factor':<40}{'gold':>6}{'dep':>8}{cand[:9]:>10}{'delta':>8}")
    for j in o:
        print(f"{FACTOR_TAXONOMY[j]:<40}{int(Y[:, j].sum()):>6}{fd[j]:>8.3f}"
              f"{fc[j]:>10.3f}{fc[j] - fd[j]:>+8.3f}")


if __name__ == "__main__":
    main()

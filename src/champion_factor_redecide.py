"""
champion_factor_redecide.py

Keep the model that scores S2 = 0.6301. Change only how it decides.

THE SUSPECTED BUG. predict_factors_nli.py averages the two runs' PROBABILITIES
and, separately, averages the two runs' THRESHOLDS (lines 63-69), then applies
one to the other. Those are not compatible operations. Averaging two models
shrinks the spread of the probabilities toward their mean, so a threshold
averaged from the individual models sits further out on the combined
distribution than either did on its own -- and the model under-predicts.

That is the same variance-shrinkage failure mode documented elsewhere: fitting
thresholds on cross-fitted probabilities and applying them to pooled in-sample
probabilities narrows the distribution enough that a label can receive ZERO
predictions.

THE PRINCIPLED VERSION. Fit the threshold ON the averaged probabilities. Same
data (val), same models, same everything -- just the operation done in the
right order.

WHAT THIS SCRIPT DOES. Compares four decision policies on the 372-post val
split, where the champion's own thresholds were fitted, and judges them
CROSS-FITTED over author halves so the comparison is honest:

  deployed    average the two thresholds, apply to averaged probs (current)
  refit       fit one threshold per label on the averaged probs
  ratematch   fit on val, then set the test cutoff at the quantile that
              reproduces val's positive RATE -- immune to any calibration
              shift between val and test
  optimalk    keep predicting while marginal precision exceeds F1*/2, the
              E[F1]-maximising stopping rule (Lewis 1995). Derived from the
              model's own ranking; fits nothing.

NOTHING HERE IS TUNED ON TEST. That matters twice over: the public evaluation
is not the final ranking (a separate held-out set decides it), so any policy
fitted to the live test scores would be optimising against the wrong data.

Usage:
  python src/champion_factor_redecide.py                       # compare only
  python src/champion_factor_redecide.py --apply ratematch \
      --out artifacts/nli_ensemble_factors_redecided.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def f1_at(pred, y):
    tp = int((pred & y).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + int((pred & (1 - y)).sum())
                     + int(((1 - pred) & y).sum()))


def macro(P, Y):
    return float(np.mean([f1_at(P[:, j], Y[:, j]) for j in range(N)]))


def best_th(pr, y):
    best, bt = -1.0, 0.5
    for t in GRID:
        v = f1_at((pr >= t).astype(int), y)
        if v > best:
            best, bt = v, t
    return bt, best


def optimal_k_th(pr, y):
    """Stop adding predictions when marginal precision drops below F1*/2.

    Adding the (k+1)-th ranked item changes E[F1] by about (2*pi - F1) / (k+n),
    so it is worth adding while its hit probability pi exceeds F1*/2. F1* is
    estimated from the label's own best achievable F1 on this data."""
    _, f1star = best_th(pr, y)
    if f1star <= 0:
        return 1.0
    o = np.argsort(-pr)
    ys = y[o]
    cum = np.cumsum(ys)
    ks = np.arange(1, len(pr) + 1)
    prec_marginal = cum / ks          # running precision as a proxy for pi
    keep = np.where(prec_marginal >= f1star / 2.0)[0]
    k = int(keep[-1] + 1) if len(keep) else 1
    return float(np.sort(pr)[-k])


def author_halves(auth):
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        p = "A" if size["A"] <= size["B"] else "B"
        half[a] = p
        size[p] += c[a]
    return s.map(half).to_numpy()


def gold_matrix(df):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(df), N), dtype=int)
    for i, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["factors_nli", "factors_nli_s2"])
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--apply", default=None,
                    choices=["deployed", "refit", "ratematch", "optimalk"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    VP, VT, TP, rid_t = [], [], [], None
    for r in args.runs:
        v = np.load(f"{args.dir}/{r}_val.npz", allow_pickle=True)
        VP.append(v["probs"])
        VT.append(v["thresholds"])
        rid_v = list(v["row_id"])
        # Per-run TEST dumps are optional. The comparison that matters -- how a
        # pair of runs scores under the deployed procedure -- lives entirely on
        # val. A newly trained pair may only have an ENSEMBLE test dump, or
        # none at all, and that should not block the val verdict.
        tp = f"{args.dir}/{r}_test.npz"
        if os.path.exists(tp):
            t = np.load(tp, allow_pickle=True)
            TP.append(t["probs"])
            rid_t = list(t["row_id"])
    Pv = np.mean(VP, axis=0)
    th_deployed = np.mean(VT, axis=0)
    have_test = len(TP) == len(args.runs)
    Pt = np.mean(TP, axis=0) if have_test else None
    if not have_test:
        print(f"NOTE: per-run test dumps missing for some of {args.runs}; "
              "reporting the val comparison only.\n")

    va = va.set_index("row_id").loc[rid_v].reset_index()
    Y = gold_matrix(va)
    print(f"val {Pv.shape[0]} posts   "
          f"test {Pt.shape[0] if have_test else 0} posts   "
          f"{len(args.runs)} runs averaged")
    print(f"prob sd  val {Pv.std(0).mean():.3f}"
          + (f"   test {Pt.std(0).mean():.3f}" if have_test else ""))
    print(f"deployed thresholds: min {th_deployed.min():.2f} "
          f"max {th_deployed.max():.2f} median {np.median(th_deployed):.2f}\n")

    half = author_halves(va["anon_user_id"] if "anon_user_id" in va.columns
                         else np.arange(len(va)))
    A, B = half == "A", half == "B"

    def fit(rows, rule):
        th = np.full(N, 0.5)
        for j in range(N):
            pr, y = Pv[rows, j], Y[rows, j]
            if y.sum() == 0:
                continue
            th[j] = best_th(pr, y)[0] if rule == "refit" else optimal_k_th(pr, y)
        return th

    print("=" * 78)
    print("DECISION POLICIES ON VAL, cross-fitted over author halves")
    print("=" * 78)
    results = {}
    Pd = (Pv >= th_deployed).astype(int)
    results["deployed"] = macro(Pd, Y)
    print(f"{'policy':<14}{'in-sample':>12}{'cross-fitted':>14}{'d vs deployed':>16}")
    print("-" * 78)
    print(f"{'deployed':<14}{results['deployed']:>12.4f}"
          f"{results['deployed']:>14.4f}{'':>16}"
          "   (no fitting, so both columns are the same)")

    for rule in ("refit", "optimalk"):
        ins = macro((Pv >= fit(np.ones(len(Y), bool), rule)).astype(int), Y)
        Px = np.zeros_like(Y)
        Px[A] = (Pv[A] >= fit(B, rule)).astype(int)
        Px[B] = (Pv[B] >= fit(A, rule)).astype(int)
        xf = macro(Px, Y)
        results[rule] = xf
        print(f"{rule:<14}{ins:>12.4f}{xf:>14.4f}"
              f"{xf - results['deployed']:>+16.4f}")

    print("\nThe cross-fitted column is the honest one. `deployed` fits nothing")
    print("at this stage -- its thresholds came from each run's own val fit --")
    print("so it is not flattered here the way the others would be in-sample.")

    if not have_test:
        print("\n(no per-run test dumps -- stopping after the val comparison.")
        print(" Compare the `deployed` row above against the champion pair's")
        print(" 0.5755: same data, same averaging, same threshold handling.)")
        return

    print("\n" + "=" * 78)
    print("WHAT EACH POLICY WOULD PREDICT ON TEST")
    print("=" * 78)
    th_refit = fit(np.ones(len(Y), bool), "refit")
    th_optk = fit(np.ones(len(Y), bool), "optimalk")

    # rate-match: reproduce val's positive rate at the refit threshold
    th_rate = th_refit.copy()
    for j in range(N):
        r = float((Pv[:, j] >= th_refit[j]).mean())
        th_rate[j] = (float(np.quantile(Pt[:, j], 1.0 - r)) if r > 0
                      else float(Pt[:, j].max()) + 1.0)

    ths = {"deployed": th_deployed, "refit": th_refit,
           "ratematch": th_rate, "optimalk": th_optk}
    print(f"{'policy':<14}{'mean/post':>11}{'empty':>7}{'zero labels':>13}")
    print("-" * 78)
    for k, th in ths.items():
        on = (Pt >= th).astype(int)
        print(f"{k:<14}{on.sum(1).mean():>11.2f}{int((on.sum(1)==0).sum()):>7}"
              f"{int((on.sum(0)==0).sum()):>13}")
    print("\ngold is 2.92 unique factors/post with 7.8% empty; the deployed")
    print("column scores S2 = 0.6301 at 2.94.")

    print("\n" + "=" * 78)
    print("PER LABEL -- deployed vs the policy that wins cross-fitted")
    print("=" * 78)
    best_rule = max(("refit", "optimalk"), key=lambda r: results[r])
    show = "ratematch" if args.apply == "ratematch" else best_rule
    on_d = (Pt >= th_deployed).astype(int)
    on_n = (Pt >= ths[show]).astype(int)
    lab = pd.read_csv("artifacts/full_train.csv")
    print(f"{'factor':<44}{'gold%':>7}{'depl':>6}{show:>11}{'delta':>7}")
    print("-" * 78)
    for j, t in enumerate(FACTOR_TAXONOMY):
        gp = 100.0 * sum(1 for x in lab["factors"]
                         if t in parse_factors(x)) / len(lab)
        a, b = int(on_d[:, j].sum()), int(on_n[:, j].sum())
        if a != b:
            flag = "  <-- ZERO" if b == 0 else ""
            print(f"{t:<44}{gp:>7.1f}{a:>6}{b:>11}{b-a:>+7}{flag}")

    if args.apply:
        if not args.out:
            raise SystemExit("--apply requires --out")
        th = ths[args.apply]
        on = (Pt >= th).astype(int)
        if (on.sum(0) == 0).any():
            dead = [FACTOR_TAXONOMY[j] for j in range(N) if on[:, j].sum() == 0]
            print(f"\nREFUSING: {args.apply} predicts nothing for {dead}. "
                  "F1 = 0 outright.")
            raise SystemExit(1)
        out = pd.DataFrame({
            "row_id": rid_t,
            "factors": [str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
                        for i in range(len(rid_t))],
        })
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False)
        old = pd.read_csv("artifacts/nli_ensemble_factors.csv",
                          keep_default_na=False).set_index("row_id")
        ch = int(sum(1 for r, f in zip(out.row_id, out.factors)
                     if str(old.loc[r, "factors"]) != f))
        print(f"\nwrote {args.out}")
        print(f"policy: {args.apply}   posts changed vs deployed: {ch} of "
              f"{len(out)}")
        if ch < 40:
            print("  UNDER 40 -- below the resolution of a 378-post test set.")
            print("  Two slots were spent today on changes this small.")


if __name__ == "__main__":
    main()

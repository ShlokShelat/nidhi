"""
average_factor_seeds.py

Average the factor probabilities of several seeds into one arm.

WHY. The risk model averages several seeds; the factor model runs a single seed
per fold, so part of its error is initialisation and negative-sampling variance
rather than anything about the data or the encoder. Averaging seeds cancels the
part of the error that is independent across initialisations.

Seed averaging fits ZERO parameters: nothing is tuned and nothing is selected,
the probabilities are simply meaned. A change with no fitted parameters cannot
transfer worse than it measures for the reason a fitted one can, because there
is nothing in it that was chosen on the evaluation data.

THRESHOLDS. Written per fold, fitted on that fold's own posts, which reproduces
exactly what train_factors_nli.py does at the end of each run. That keeps the
`per-fold` column in compare_factor_arms.py comparable to the incumbent's. The
honest `xfit-half` column is recomputed by that script from the raw
probabilities regardless, so nothing here can flatter the result.

Usage:
  # incumbent (seed 13) + two more seeds
  python src/average_factor_seeds.py --tag seedavg \
      --runs artifacts/fac_oof_f{f}.npz \
             artifacts/factor_runs/s21_cv_oof_f{f}.npz \
             artifacts/factor_runs/s37_cv_oof_f{f}.npz

  python src/compare_factor_arms.py     # picks up `seedavg` automatically
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
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def tune(P, Y):
    th = np.full(N, 0.5)
    for j in range(N):
        best, bt = -1.0, 0.5
        for t in GRID:
            p = (P[:, j] >= t).astype(int)
            tp = int((p & Y[:, j]).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((p & (1 - Y[:, j])).sum())
                           + int(((1 - p) & Y[:, j]).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="OOF patterns containing {f}, one per seed")
    ap.add_argument("--tag", default="seedavg")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--outdir", default="artifacts/factor_runs")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    os.makedirs(args.outdir, exist_ok=True)

    for f in range(args.k):
        paths = [p.format(f=f) for p in args.runs]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"fold {f}: missing {missing}")

        base = np.load(paths[0], allow_pickle=True)
        rid = list(base["row_id"])
        acc = base["probs"].astype(np.float64).copy()

        for p in paths[1:]:
            d = np.load(p, allow_pickle=True)
            r = list(d["row_id"])
            if set(r) != set(rid):
                raise SystemExit(f"fold {f}: {p} covers a different post set")
            # align by row_id -- fold files are not guaranteed to share an order
            o = {x: i for i, x in enumerate(r)}
            acc += d["probs"].astype(np.float64)[[o[x] for x in rid]]

        P = (acc / len(paths)).astype(np.float32)

        Y = np.zeros((len(rid), N), dtype=int)
        for i, r in enumerate(rid):
            for x in parse_factors(lab.loc[r, "factors"]):
                j = idx.get(str(x).strip().lower())
                if j is not None:
                    Y[i, j] = 1

        th = tune(P, Y)
        out = os.path.join(args.outdir, f"{args.tag}_cv_oof_f{f}.npz")
        np.savez(out, probs=P, thresholds=th,
                 row_id=np.array(rid, dtype=object))

        # per-seed vs averaged, on this fold, at each one's own tuned thresholds
        def macro(Pm):
            t = tune(Pm, Y)
            b = (Pm >= t).astype(int)
            v = []
            for j in range(N):
                tp = int((b[:, j] & Y[:, j]).sum())
                v.append(0.0 if tp == 0 else 2 * tp /
                         (2 * tp + int((b[:, j] & (1 - Y[:, j])).sum())
                          + int(((1 - b[:, j]) & Y[:, j]).sum())))
            return float(np.mean(v))

        singles = []
        for p in paths:
            d = np.load(p, allow_pickle=True)
            o = {x: i for i, x in enumerate(list(d["row_id"]))}
            singles.append(macro(d["probs"][[o[x] for x in rid]]))
        print(f"fold {f}: seeds " + " ".join(f"{s:.4f}" for s in singles)
              + f"   mean {np.mean(singles):.4f}   averaged {macro(P):.4f}"
              + f"   -> {out}")

    print(f"\nwrote {args.k} folds as `{args.tag}`.")
    print("Now run:  python src/compare_factor_arms.py")
    print("\nJudge it on the xfit-half column only. `averaged` beating `mean`")
    print("above is expected and proves nothing -- both are tuned in-sample.")


if __name__ == "__main__":
    main()

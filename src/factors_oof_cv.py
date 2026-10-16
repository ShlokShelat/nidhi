"""
factors_oof_cv.py

Author-disjoint K-fold cross-validation for the NLI factor model.

WHY NOT A PLAIN FULL-DATA RETRAIN. train_factors_nli.py requires --val and
uses it for two things: checkpoint selection (best macro-F1 epoch) and
THRESHOLD TUNING (tune_thresholds(vp, va_gold) -> thresholds.npy). If the
model trains on all 1635 posts, val is inside the training set, so both are
contaminated -- the run would emit a model with no valid thresholds. Risk
models survive this (runs/risk_full_* just take the last epoch, best_val
-1.0), but factors cannot: without thresholds there is no prediction at all.

WHAT K-FOLD OOF FIXES, all at once:

  * thresholds get tuned on 1635 out-of-fold posts instead of 372. The
    measured overfitting gap for per-label tuning was 0.0684 macro-F1 on 372
    posts (factor_threshold_strategies.py); the standard error scales roughly
    as 1/sqrt(n), so ~2.1x more data should shrink it substantially. This is
    the single largest quantified defect found in the whole system.
  * every fold model trains on (K-1)/K of 1635, i.e. ~1308 posts at K=5.
    That is only +3.5% over the 1263 the deployed models saw -- NOT the +29%
    a true full-data model would get. Do not sell this as the extra-data win;
    the win here is the thresholds and the honest evaluation. To also capture
    the +29%, train one final model on all 1635 and apply the OOF thresholds
    to it (K+1 trainings total).
  * the pooled OOF score is an honest estimate over 1635 posts and ~153
    authors, instead of 372 posts and 32 authors. Four factor labels that are
    unmeasurable on val (0, 1, 2 and 4 positives) become measurable.
  * K fold models form a natural ensemble at inference, replacing the 2-seed
    average.

Folds are split BY AUTHOR. Splitting by post would put the same author on
both sides and reproduce exactly the leakage that made our 372-post val
optimistic.

One honest caveat: each fold's checkpoint is selected on that fold's held-out
data, so the pooled OOF score is mildly optimistic. Properly removing that
needs a nested split. It is still far better than the status quo.

Usage:
  # 1. CPU -- write the fold CSVs and print the GPU commands
  python src/factors_oof_cv.py --make-folds --k 5

  # 2. GPU -- run the printed loop (train + dump OOF probs per fold)

  # 3. CPU -- pool the OOF probabilities, tune and compare thresholds
  python src/factors_oof_cv.py --collect --k 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors
from sklearn.metrics import f1_score

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
FOLD_DIR = "artifacts/factor_cv"


def author_folds(df, k, seed=13):
    """Deterministic author-disjoint folds, balanced on post count."""
    counts = df.groupby("anon_user_id").size().sort_values(
        ascending=False, kind="mergesort")
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    sizes = {i: 0 for i in range(k)}
    assign = {}
    for a in order:
        f = min(sizes, key=lambda i: sizes[i])
        assign[a] = f
        sizes[f] += counts[a]
    return df["anon_user_id"].map(assign), sizes


def macro(Y, P):
    return float(f1_score(Y, P, average=None, zero_division=0).mean())


def tune(Y, P, rows=None, mode="per_label"):
    sel = slice(None) if rows is None else rows
    if mode == "global":
        best_t, best_f = 0.5, -1.0
        for t in GRID:
            f = macro(Y[sel], (P[sel] >= t).astype(int))
            if f > best_f:
                best_t, best_f = t, f
        return np.full(P.shape[1], best_t)
    ths = np.full(P.shape[1], 0.5)
    for j in range(P.shape[1]):
        y, p = Y[sel, j], P[sel, j]
        if y.sum() == 0:
            continue
        best_t, best_f = 0.5, -1.0
        for t in GRID:
            f = f1_score(y, (p >= t).astype(int), zero_division=0)
            if f > best_f:
                best_t, best_f = t, f
        ths[j] = best_t
    return ths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--make-folds", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-thresholds", default="artifacts/factor_thresholds_oof.npy")
    args = ap.parse_args()

    df = pd.read_csv(args.labeled)
    need = {"row_id", "anon_user_id", "post", "factors"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{args.labeled} needs columns {need}; has {list(df.columns)}")

    fold, sizes = author_folds(df, args.k, args.seed)
    df = df.assign(_fold=fold.values)

    if args.make_folds:
        os.makedirs(FOLD_DIR, exist_ok=True)
        print(f"{len(df)} posts, {df.anon_user_id.nunique()} authors, k={args.k}")
        for f in range(args.k):
            tr = df[df._fold != f].drop(columns=["_fold"])
            va = df[df._fold == f].drop(columns=["_fold"])
            tr.to_csv(f"{FOLD_DIR}/train_f{f}.csv", index=False)
            va.to_csv(f"{FOLD_DIR}/val_f{f}.csv", index=False)
            print(f"  fold {f}: train {len(tr):>5} posts / "
                  f"{tr.anon_user_id.nunique():>3} authors   "
                  f"held-out {len(va):>4} posts / {va.anon_user_id.nunique():>3} authors")
        overlap = any(
            set(df[df._fold == a].anon_user_id) & set(df[df._fold == b].anon_user_id)
            for a in range(args.k) for b in range(a + 1, args.k))
        print(f"\nauthor overlap between folds: {'YES -- BUG' if overlap else 'none'}")
        print(f"\n--- GPU commands ---")
        for f in range(args.k):
            print(f"python src/train_factors_nli.py "
                  f"--train {FOLD_DIR}/train_f{f}.csv "
                  f"--val {FOLD_DIR}/val_f{f}.csv "
                  f"--outdir runs/factors_nli_cv_f{f} --seed {args.seed}")
            print(f"python src/predict_factors_nli.py "
                  f"--test {FOLD_DIR}/val_f{f}.csv "
                  f"--nli-run runs/factors_nli_cv_f{f} "
                  f"--out artifacts/_oof_f{f}.csv "
                  f"--dump-probs artifacts/fac_oof_f{f}.npz")
        return

    if not args.collect:
        raise SystemExit("pass --make-folds or --collect")

    tax = FACTOR_TAXONOMY
    idx = {t: i for i, t in enumerate(tax)}
    P = np.full((len(df), len(tax)), np.nan)
    pos = {r: i for i, r in enumerate(df["row_id"])}
    got = 0
    for f in range(args.k):
        p = f"artifacts/fac_oof_f{f}.npz"
        if not os.path.exists(p):
            print(f"  MISSING {p} -- run the GPU step for fold {f}")
            continue
        d = np.load(p, allow_pickle=True)
        for rid, row in zip(d["row_id"], d["probs"]):
            if rid in pos:
                P[pos[rid]] = row
                got += 1
    print(f"OOF rows filled: {got}/{len(df)}")
    if got < len(df):
        raise SystemExit("incomplete OOF -- cannot evaluate")

    Y = np.zeros_like(P, dtype=int)
    for i, x in enumerate(df["factors"]):
        for l in parse_factors(x):
            if l.strip().lower() in idx:
                Y[i, idx[l.strip().lower()]] = 1

    print(f"\nlabel support over {len(df)} OOF posts (val had 4 labels with <5):")
    thin = [(tax[j], int(Y[:, j].sum())) for j in range(len(tax))
            if Y[:, j].sum() < 20]
    for n, c in sorted(thin, key=lambda z: z[1]):
        print(f"  {n:<44}{c:>5}")

    # cross-fitted comparison, now on 1635 posts instead of 372
    fold_arr = df["_fold"].values
    print(f"\n{'strategy':<22}{'in-sample':>11}{'CROSS-FIT':>11}{'gap':>9}")
    for mode in ("per_label", "global"):
        ins = macro(Y, (P >= tune(Y, P, mode=mode)).astype(int))
        pred = np.zeros_like(Y)
        for f in range(args.k):
            tr = np.where(fold_arr != f)[0]
            te = np.where(fold_arr == f)[0]
            th = tune(Y, P, rows=tr, mode=mode)
            pred[te] = (P[te] >= th).astype(int)
        cf = macro(Y, pred)
        print(f"{mode:<22}{ins:>11.4f}{cf:>11.4f}{ins - cf:>9.4f}")
        if mode == "per_label":
            best_mode, best_cf, best_th = mode, cf, tune(Y, P, mode=mode)
        elif cf > best_cf:
            best_mode, best_cf, best_th = mode, cf, tune(Y, P, mode=mode)

    print(f"\nbest strategy on OOF: {best_mode}  (cross-fitted {best_cf:.4f})")
    print(f"for reference, the same comparison on the 372-post val gave")
    print(f"  per_label 0.5537 cross-fitted, gap 0.0684")
    print(f"  global    0.5643 cross-fitted, gap 0.0049")
    np.save(args.out_thresholds, best_th)
    print(f"\nwrote {args.out_thresholds}")


if __name__ == "__main__":
    main()

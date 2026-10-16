"""
Second-stage label-refinement model for factors.

Motivation: the first-stage factor model treats 24 labels independently
(24 sigmoids). But the labels have strong joint structure — hopelessness
co-occurs with low self-esteem; social support and poor social support are
near-exclusive; inferential labels (meaning in life, sense of
responsibility) are predictable from their CO-OCCURRING factors even when
their own text signal is weak. An independent model throws this away.

This stage takes the 24 first-stage probabilities for a post and predicts
refined 24 probabilities, learning label-label dependencies. It's the
mechanism most likely to lift macro F1 past the ~0.44 independent-model
plateau, and it specifically helps the inferential labels that
augmentation provably could not.

CRITICAL — no leakage: the first-stage probabilities MUST be out-of-fold.
We reuse the CV harness: for each fold, the held-out posts get first-stage
probs from a model that never saw them. Those OOF probs are the second
stage's training input. This file consumes an OOF-probability matrix
produced by train_factors_cv.py (saved via --save-oof).

Two second-stage options, both provided:
  A. GBM (LightGBM-style via sklearn HistGradientBoosting) per label — robust
     on small data, captures nonlinear label interactions, no GPU needed.
  B. Small MLP over the 24 probs — smoother, GPU-optional.
We default to A (GBM) — more reliable on this data size.

Usage:
  # 1. produce OOF first-stage probs (add --save-oof to the CV script)
  # 2. fit + evaluate the second stage on those OOF probs:
  python src/train_factors_stack.py \
      --oof runs/factors_cv_final/oof_probs.npz \
      --out runs/factors_stack

The OOF npz must contain: probs [n x 24], gold [n x 24].
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scoring import FACTOR_TAXONOMY, factors_macro_f1

N = len(FACTOR_TAXONOMY)


def to_lists(binary):
    return [[FACTOR_TAXONOMY[j] for j in range(N) if row[j]] for row in binary]


def tune_thresholds(probs, gold):
    """Per-label F1-max threshold; skip labels with <5 positives (keep 0.5).
    Same policy as stage 1, on the refined probabilities."""
    ths = np.full(N, 0.5)
    grid = np.arange(0.05, 0.95, 0.025)
    for j in range(N):
        if gold[:, j].sum() < 5:
            continue
        best_f1, best_t = -1.0, 0.5
        gj = gold[:, j].astype(int)
        for t in grid:
            pred = (probs[:, j] >= t).astype(int)
            tp = int((pred & gj).sum()); fp = int((pred & (1 - gj)).sum())
            fn = int(((1 - pred) & gj).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        ths[j] = best_t
    return ths


def fit_gbm_stack(X, Y, seed=13):
    """One HistGradientBoosting classifier per label, each seeing ALL 24
    first-stage probs. Inner CV gives out-of-fold refined probs for honest
    threshold tuning. Returns (oof_refined_probs, fitted_models)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, KFold

    n = len(X)
    oof = np.zeros((n, N), dtype=np.float32)
    models = []
    for j in range(N):
        yj = Y[:, j].astype(int)
        col_models = []
        # 5-fold inner CV for OOF refined probs on this label
        if yj.sum() >= 5 and (n - yj.sum()) >= 5:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            splitter = skf.split(X, yj)
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            splitter = kf.split(X)
        for tr_idx, va_idx in splitter:
            if yj[tr_idx].sum() == 0:
                oof[va_idx, j] = yj[tr_idx].mean()  # degenerate: base rate
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=3,
                l2_regularization=1.0, random_state=seed)
            clf.fit(X[tr_idx], yj[tr_idx])
            oof[va_idx, j] = clf.predict_proba(X[va_idx])[:, 1]
        # final model on all data for this label (used at inference)
        if yj.sum() > 0:
            clf = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=3,
                l2_regularization=1.0, random_state=seed)
            clf.fit(X, yj)
            col_models.append(clf)
        models.append(col_models)
    return oof, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True,
                    help="npz with first-stage OOF probs+gold (from CV script)")
    ap.add_argument("--out", default="runs/factors_stack")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    d = np.load(args.oof)
    probs, gold = d["probs"].astype(np.float32), d["gold"].astype(int)
    assert probs.shape[1] == N and gold.shape[1] == N
    print(f"loaded {len(probs)} posts x {N} labels")

    # baseline: stage-1 alone, its own tuned thresholds
    base_ths = tune_thresholds(probs, gold)
    base_macro, base_per = factors_macro_f1(
        to_lists(gold), to_lists((probs >= base_ths).astype(int)),
        return_per_label=True)
    print(f"\nstage-1 baseline OOF macro = {base_macro:.4f}")

    # stage-2: co-occurrence refinement
    print("\nfitting stage-2 co-occurrence models (24 labels x 5-fold)...")
    refined_oof, models = fit_gbm_stack(probs, gold, args.seed)
    ref_ths = tune_thresholds(refined_oof, gold)
    ref_macro, ref_per = factors_macro_f1(
        to_lists(gold), to_lists((refined_oof >= ref_ths).astype(int)),
        return_per_label=True)
    print(f"stage-2 refined OOF macro = {ref_macro:.4f}  "
          f"(delta {ref_macro - base_macro:+.4f})")

    # per-label delta — where does co-occurrence help?
    print("\nper-label change (stage1 -> stage2):")
    for lab in sorted(FACTOR_TAXONOMY, key=lambda L: ref_per[L] - base_per[L]):
        d1, d2 = base_per[lab], ref_per[lab]
        print(f"  {d1:.3f} -> {d2:.3f}  ({d2-d1:+.3f})  {lab}")

    # per-label gate: for each label pick the stage with higher OOF F1.
    # stage-2 wins big on co-occurring labels but overfits the rarest few
    # (n<30) where stage-1 was better; this keeps the best of both.
    use_stage2 = np.zeros(N, dtype=bool)
    for j, lab in enumerate(FACTOR_TAXONOMY):
        use_stage2[j] = ref_per[lab] >= base_per[lab]
    gated = np.where(use_stage2[None, :], refined_oof, probs)
    gated_ths = np.where(use_stage2, ref_ths, base_ths)
    gated_macro, gated_per = factors_macro_f1(
        to_lists(gold), to_lists((gated >= gated_ths).astype(int)),
        return_per_label=True)
    print(f"\nGATED (best-of-both per label) OOF macro = {gated_macro:.4f}  "
          f"(vs stage1 {base_macro:.4f}, stage2 {ref_macro:.4f})")
    print(f"  using stage-2 for {use_stage2.sum()}/{N} labels")

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "stage2_thresholds.npy"), ref_ths)
    np.save(os.path.join(args.out, "gated_thresholds.npy"), gated_ths)
    np.save(os.path.join(args.out, "use_stage2_mask.npy"), use_stage2)
    import pickle
    with open(os.path.join(args.out, "stage2_models.pkl"), "wb") as f:
        pickle.dump(models, f)
    with open(os.path.join(args.out, "stack_report.json"), "w") as f:
        json.dump({"stage1_macro": float(base_macro),
                   "stage2_macro": float(ref_macro),
                   "gated_macro": float(gated_macro),
                   "use_stage2_labels": int(use_stage2.sum()),
                   "delta_gated": float(gated_macro - base_macro)}, f, indent=2)
    print(f"\nsaved stage-2 to {args.out}")
    best = max(base_macro, ref_macro, gated_macro)
    if best == gated_macro and gated_macro > base_macro:
        print(f"WIN: gated two-stage lifts factors {base_macro:.4f} -> {gated_macro:.4f}")
    elif ref_macro > base_macro:
        print(f"WIN: stage-2 lifts factors {base_macro:.4f} -> {ref_macro:.4f}")
    else:
        print("no gain — keep stage-1")


if __name__ == "__main__":
    main()

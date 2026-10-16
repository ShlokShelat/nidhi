"""
factor_loss_metric_gap.py

HOW BADLY DOES THE TRAINING LOSS MISMATCH THE SCORING METRIC?

The factor arms are trained with BCE on flattened (post, factor) pairs and
scored with MACRO-F1 over 24 labels. Those objectives disagree in a specific,
measurable way:

  * under BCE a label with 8 positives out of 1635 contributes ~0.5% of the
    gradient
  * under macro-F1 it is worth 1/24 = 4.2% of the score

That is an 8x under-weighting of exactly the rare labels that score worst.

THE TELL. If the probabilities were shaped for macro-F1, the F1-optimal
per-label thresholds would CLUSTER near a common value. Instead they span
roughly 0.12 to 0.97, and the whole `tfit` machinery exists to patch the
mismatch AFTER training. This quantifies it three ways:

  1. macro-F1 at a single global threshold, swept -- the best a
     metric-agnostic decode can do
  2. macro-F1 with per-label `tfit` thresholds -- the deployed decode
  3. the gap between them, which is how much of the factor score is
     recovered by decode-time patching rather than learned

A LARGE gap means the probabilities are badly shaped for the metric, and a
metric-matched loss (soft macro-F1) has real room. A SMALL gap means the
probabilities are already fine and only the thresholds mattered, in which case
retraining buys nothing: reweighting the loss toward rare labels then only
removes the regularisation the uniform weighting was providing.

Also reports, per label, how far its threshold sits from the global optimum and
how much F1 that distance is worth -- so the diagnosis names the labels a
metric-matched loss would have to fix.

Usage:
  python src/factor_loss_metric_gap.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_structure import (BAG8, best_thr, cells,  # noqa: E402
                              load_oof, logit_mean)
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)


def per_label_f1(Pb, Y):
    tp = (Pb & Y).sum(0)
    den = 2 * tp + (Pb & ~Y).sum(0) + (~Pb & Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--arms", nargs="+", default=BAG8)
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--splits", type=int, default=100)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n).astype(bool)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    sup = Y.sum(0)
    print(f"OOF {n} posts / {len(pd.unique(auth))} authors, {N} labels\n")

    print("=" * 74)
    print("1. A SINGLE GLOBAL THRESHOLD vs PER-LABEL tfit")
    print("=" * 74)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    gm = [float(np.mean(per_label_f1(P >= t, Y))) for t in grid]
    bg = grid[int(np.argmax(gm))]
    tfit = np.array([best_thr(P[:, j], Y[:, j].astype(int)) or 0.5
                     for j in range(N)])
    mt = float(np.mean(per_label_f1(P >= tfit, Y)))
    print(f"  best GLOBAL threshold {bg:.2f}   macro {max(gm):.4f}")
    print(f"  per-label tfit                   macro {mt:.4f}")
    print(f"  gap recovered by decode-time patching  {mt - max(gm):+.4f}")
    print(f"  threshold spread: min {tfit.min():.2f}  median "
          f"{np.median(tfit):.2f}  max {tfit.max():.2f}  sd {tfit.std():.2f}")
    print("\n  A well-shaped model needs ONE threshold. This one needs 24")
    print("  spread across the range, and the spread is worth the gap above.")
    print("  That gap is the size of the loss/metric mismatch.")

    print("\n" + "=" * 74)
    print("2. WHICH LABELS CARRY THE MISMATCH?")
    print("=" * 74)
    f_glob = per_label_f1(P >= bg, Y)
    f_tfit = per_label_f1(P >= tfit, Y)
    order = np.argsort(-(f_tfit - f_glob))
    print(f"  {'label':<38}{'pos':>5}{'thr':>6}{'F1@global':>11}"
          f"{'F1@tfit':>9}{'gain':>8}")
    for j in order[:10]:
        print(f"  {FACTOR_TAXONOMY[j][:37]:<38}{int(sup[j]):>5}"
              f"{tfit[j]:>6.2f}{f_glob[j]:>11.3f}{f_tfit[j]:>9.3f}"
              f"{f_tfit[j]-f_glob[j]:>+8.3f}")
    lo = sup < 60
    print(f"\n  mean gain, LOW support (<60): "
          f"{np.mean((f_tfit - f_glob)[lo]):+.3f}")
    print(f"  mean gain, HIGH support:      "
          f"{np.mean((f_tfit - f_glob)[~lo]):+.3f}")
    print("\n  If the gain concentrates on LOW-support labels, the model is")
    print("  under-training them exactly as the gradient-share argument says,")
    print("  and a metric-matched loss has a specific target.")

    print("\n" + "=" * 74)
    print("3. IS THE MISMATCH IN THE PROBABILITIES OR JUST THE CUTS?")
    print("=" * 74)
    # rank quality is threshold-free: if AP is already high, the ordering is
    # fine and only the cut is wrong -- a loss change would not help ranking
    def ap(p, y):
        o = np.argsort(-p, kind="stable")
        ys = y[o]
        return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum()
                     / max(ys.sum(), 1))
    aps = np.array([ap(P[:, j], Y[:, j]) for j in range(N)])
    print(f"  mean AP (ranking, threshold-free)  {aps.mean():.4f}")
    print(f"  mean F1 at tfit                    {f_tfit.mean():.4f}")
    print(f"  mean F1 at the best global cut     {f_glob.mean():.4f}")
    print(f"\n  {'label':<38}{'AP':>7}{'F1@tfit':>9}{'AP-F1':>8}")
    for j in np.argsort(-(aps - f_tfit))[:8]:
        print(f"  {FACTOR_TAXONOMY[j][:37]:<38}{aps[j]:>7.3f}"
              f"{f_tfit[j]:>9.3f}{aps[j]-f_tfit[j]:>+8.3f}")
    print("\n  AP measures ORDERING and ignores thresholds. Where AP is much")
    print("  higher than F1, the ranking is fine and only the cut is wrong --")
    print("  a metric-matched LOSS cannot help those, only better THRESHOLDS")
    print("  can. Where AP is LOW, the ordering itself is bad and that is what")
    print("  a new loss must fix.")


if __name__ == "__main__":
    main()

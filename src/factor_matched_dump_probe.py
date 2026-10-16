#!/usr/bin/env python
"""WHICH TEST DUMP IS MATCHED TO THE OOF? An offline diagnostic.

THE SITUATION. Several arms (mnli, dmnli, prot) have MORE THAN ONE test dump on
disk, and those dumps disagree substantially in probability. `resolve_test`
picks between them by FILENAME PRECEDENCE:

    {arm}_cv_test.npz  ->  {arm}_test.npz  ->  nli_ensemble_{arm}_test.npz

so which dump an arm actually decodes depends on which filenames happen to
exist, not on which dump matches its OOF.

WHY IT MATTERS. The factor decode fits per-label thresholds on the OOF (5
held-out folds) and applies them to the test dump. If the test dump was built
by a DIFFERENT construction than the OOF, those thresholds are calibrated for
the wrong distribution and the decode silently over- or under-fires on the test
set. Matched OOF/test construction is a precondition for any threshold fitted
offline to mean anything at inference time.

THE TEST. Fit each arm's per-label thresholds on ITS OWN OOF, then apply them
to each candidate test dump. A dump built the same way as the OOF should fire
at roughly the OOF's own rate. A mismatched one systematically over- or
under-fires, because the thresholds sit in the wrong place on its distribution.

    firing ratio = (test factors/post) / (OOF factors/post)

    ratio near 1.00  -> consistent with matched construction
    ratio far from 1 -> the thresholds do not transfer to this dump

This is a DIAGNOSTIC, not a proof: prevalence really can differ between the
labelled pool and the test set. Read it together with the per-label
distribution distance, and treat a large disagreement as a reason to
investigate the dump's provenance, not as a result in itself.

Read-only.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_threshold_decode import (  # noqa: E402
    N, best_threshold_and_f1, cells, logit_mean)
from scoring import FACTOR_TAXONOMY  # noqa: E402

CANDIDATES = ("{d}/{a}_cv_test.npz",
              "{d}/{a}_test.npz",
              "{d}/nli_ensemble_{a}_test.npz",
              "{d}/_shadowed/{a}_cv_test.npz",
              "{d}/_shadowed/nli_ensemble_{a}_test.npz",
              # written by scripts/factor_matched_redump.sh, deliberately NOT
              # a name resolve_test reaches, so probing cannot change shipping
              "{d}/{a}_matched_test.npz")


def arm_oof(d, a, k=5):
    """Concatenate one arm's k held-out folds, in fold order."""
    pr, ids = [], []
    for f in range(k):
        p = f"{d}/{a}_cv_oof_f{f}.npz"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        z = np.load(p, allow_pickle=True)
        pr.append(z["probs"].astype(float))
        ids.extend(str(x) for x in z["row_id"])
    return np.concatenate(pr), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["mnli", "dmnli", "prot"])
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    for a in args.arms:
        paths = [c.format(d=args.dir, a=a) for c in CANDIDATES
                 if os.path.exists(c.format(d=args.dir, a=a))]
        print("=" * 74)
        print(f"ARM {a}   {len(paths)} test dump(s) on disk")
        print("=" * 74)
        if not paths:
            print("  none found\n")
            continue

        P, rid = arm_oof(args.dir, a)
        Y = cells([lab.loc[r, "factors"] for r in rid], len(rid))
        th = np.zeros(N)
        for j in range(N):
            th[j], _ = best_threshold_and_f1(P[:, j], Y[:, j])

        oof_fire = float((P >= th[None, :]).sum() / len(rid))
        gold_fire = float(Y.sum() / len(rid))
        print(f"  OOF ({len(rid)} posts): thresholds fire "
              f"{oof_fire:.3f} factors/post   gold {gold_fire:.3f}")
        print()
        print(f"  {'dump':<42} {'f/post':>7} {'ratio':>7} {'meanD':>7}")
        print(f"  {'-'*42} {'-'*7} {'-'*7} {'-'*7}")

        oof_mean = P.mean(0)
        for p in paths:
            z = np.load(p, allow_pickle=True)
            T = z["probs"].astype(float)
            fire = float((T >= th[None, :]).sum() / len(T))
            ratio = fire / oof_fire if oof_fire else float("nan")
            # mean |per-label mean prob difference| vs the OOF
            md = float(np.abs(T.mean(0) - oof_mean).mean())
            flag = ""
            if not (0.85 <= ratio <= 1.15):
                flag = "  <-- thresholds do not transfer"
            name = p.replace(args.dir + "/", "")
            print(f"  {name:<42} {fire:7.3f} {ratio:7.3f} {md:7.4f}{flag}")

        # the per-label story, for the two most divergent labels
        if len(paths) >= 2:
            A = np.load(paths[0], allow_pickle=True)["probs"].astype(float)
            B = np.load(paths[1], allow_pickle=True)["probs"].astype(float)
            if A.shape == B.shape:
                perlab = np.abs(A.mean(0) - B.mean(0))
                worst = np.argsort(-perlab)[:3]
                print()
                print("  labels where the two dumps disagree most "
                      "(mean prob):")
                for j in worst:
                    print(f"    {FACTOR_TAXONOMY[j]:<44} "
                          f"{A[:, j].mean():.3f} vs {B[:, j].mean():.3f}   "
                          f"OOF {oof_mean[j]:.3f}")
        print()

    print("=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("The dump whose firing ratio is nearest 1.00 is the one the OOF-fitted")
    print("thresholds actually transfer to. If that is NOT the dump filename")
    print("precedence selects, those arms are being decoded with thresholds")
    print("calibrated on a different distribution, and the resolution order or")
    print("the dump itself should be fixed before anything else is changed.")
    print()
    print("Caveat that keeps this honest: real prevalence differences between")
    print("the labelled pool and the test set also move the ratio. This ranks")
    print("two candidates against each other; it does not prove either correct.")


if __name__ == "__main__":
    main()

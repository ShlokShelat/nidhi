"""
rethreshold_test.py

Re-threshold an existing test probability dump using thresholds fitted in the
SAME regime the test probabilities were produced in.

WHY. A threshold fitted on OOF probabilities where each post is scored by 3
checkpoints does not transfer to test probabilities averaged over 15.
Averaging five times as many models narrows the distribution, so a cutoff
calibrated in the wider regime sits too far out in the narrower one.

The damage lands on the rare labels. A label whose predicted rate is pushed to
ZERO scores F1 0 no matter how well it ranks, and on a 24-label macro average
one dead label removes 1/24 of the factor score by itself -- enough to outweigh
a broad improvement across the other 23.

scripts/factors_deploy_oof.sh fixes the measurement side: it re-scores every
labelled post with all checkpoints EXCEPT the ones that trained on it, giving
12-checkpoint averages against the test set's 15. This script fixes the
deployment side -- fit on those, apply to the test dump already on disk.

No GPU: the test probabilities do not change, only the cutoffs.

A GUARD THAT SHOULD HAVE EXISTED. Any label the model predicts out-of-fold but
predicts ZERO times on test scores F1 = 0, a guaranteed 1/24 of macro thrown
away. The earlier sanity check skipped labels under 1% gold prevalence, which
is exactly where this bites. --min-predictions lowers such a label's threshold
to the smallest value that emits at least one prediction, and reports every
time it fires. Set it to 0 to disable.

Usage:
  python src/rethreshold_test.py --oof-tag deploy_mdc3 --test-tag paircm \
      --out artifacts/paircm_regime_test_factors.csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-tag", required=True,
                    help="tag of the OOF npz set to fit thresholds on -- use "
                         "the deployment-regime one, e.g. deploy_mdc3")
    ap.add_argument("--test-tag", required=True,
                    help="tag of the existing test dump, e.g. paircm")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-predictions", type=int, default=1,
                    help="floor on predictions per label; 0 disables")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rid, P = [], []
    for f in range(args.k):
        p = f"artifacts/factor_runs/{args.oof_tag}_cv_oof_f{f}.npz"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run scripts/factors_deploy_oof.sh")
        d = np.load(p, allow_pickle=True)
        rid.extend(list(d["row_id"]))
        P.append(d["probs"])
    P = np.concatenate(P)

    te = np.load(f"artifacts/factor_runs/{args.test_tag}_test.npz",
                 allow_pickle=True)
    TP, trid = te["probs"], list(te["row_id"])

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rid), N), dtype=int)
    for i, r in enumerate(rid):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1

    print(f"fit on  {args.oof_tag}: {len(rid)} posts   "
          f"mean prob sd {P.std(0).mean():.3f}")
    print(f"apply to {args.test_tag}: {len(trid)} posts   "
          f"mean prob sd {TP.std(0).mean():.3f}")
    print("(the two sd figures should now be close; the shipped mismatch had "
          "them apart)\n")

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

    on = (TP >= th).astype(int)
    floored = []
    if args.min_predictions > 0:
        for j in range(N):
            oof_rate = float((P[:, j] >= th[j]).mean())
            if on[:, j].sum() < args.min_predictions and oof_rate > 0:
                k = args.min_predictions
                cut = float(np.sort(TP[:, j])[-k])
                floored.append((FACTOR_TAXONOMY[j], th[j], cut,
                                int(on[:, j].sum()), k))
                th[j] = cut
                on[:, j] = (TP[:, j] >= cut).astype(int)

    if floored:
        print("FLOOR APPLIED -- these would have scored F1 = 0 outright:")
        for t, old, new, had, want in floored:
            print(f"  {t:<44} threshold {old:.2f} -> {new:.2f}   "
                  f"predictions {had} -> {want}")
        print()

    print(f"{'factor':<44}{'th':>6}{'gold%':>7}{'oof%':>7}{'test%':>7}{'t/o':>6}")
    print("-" * 78)
    for j, t in enumerate(FACTOR_TAXONOMY):
        o = float((P[:, j] >= th[j]).mean())
        s = float(on[:, j].mean())
        g = float(Y[:, j].mean())
        r = s / o if o > 1e-9 else float("nan")
        flag = "  <--" if (s == 0 or (not np.isnan(r) and (r < 0.5 or r > 2.0))) else ""
        print(f"{t:<44}{th[j]:>6.2f}{100*g:>7.1f}{100*o:>7.1f}{100*s:>7.1f}"
              f"{r:>6.2f}{flag}")

    cnt = on.sum(1)
    guniq = [len({str(f).strip().lower() for f in parse_factors(x)}
                 & set(FACTOR_TAXONOMY)) for x in lab["factors"]]
    print(f"\ntest  mean {cnt.mean():.2f} factors/post   empty {int((cnt == 0).sum())}"
          f" ({100.0 * (cnt == 0).mean():.1f}%)   labels never predicted "
          f"{int((on.sum(0) == 0).sum())}")
    print(f"gold  mean {np.mean(guniq):.2f} factors/post   "
          f"empty {sum(1 for g in guniq if g == 0)} "
          f"({100.0 * sum(1 for g in guniq if g == 0) / len(guniq):.1f}%)")

    out = pd.DataFrame({
        "row_id": trid,
        "factors": [str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
                    for i in range(len(trid))],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    np.save(f"artifacts/factor_thresholds_{args.test_tag}_regime.npy", th)
    print(f"\nwrote {args.out}")
    print(f"wrote artifacts/factor_thresholds_{args.test_tag}_regime.npy")


if __name__ == "__main__":
    main()

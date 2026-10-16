"""
threshold_regime_check.py

Can a threshold decode measure a consistent, positive gain offline and still
move the WRONG WAY on submission? (Case examined: the `paircm` tag.)

THE RISK THIS CHECKS FOR. An offline measurement can show a positive gain on
100% of many author partitions and still reverse in sign once applied on a
different probability regime than it was fitted on. Cross-fitting alone does
not catch this -- the resampling machinery can be entirely valid and the
result still flip. Something can differ between the OOF regime thresholds are
fitted in and the TEST regime they are applied in.

THE SUSPECT. Count the checkpoints scoring each post:

    OOF   post is in exactly one fold's val, so it is scored by ONE checkpoint
          per arm -- 3 for paircm (mnli, dmnli, cmine at that fold).
    TEST  every post is scored by ALL 15 checkpoints and averaged.

Averaging 15 models instead of 3 shrinks the variance of each probability
toward its mean. A per-label threshold fitted on the wider distribution then
sits too far out on the narrower one, so FEWER posts clear it. On a common
label that trims a little recall. On a rare label, whose positives sit in the
thin upper tail, it can mean predicting the label ZERO times -- and a label
predicted zero times scores F1 = 0, which costs 1/24 of macro outright.

`sexual orientation related issues` was predicted 0.0% on test against 0.5% in
gold. Its incumbent F1 was 0.480. That single label alone is 0.480/24 = 0.020
of macro -- enough on its own to erase an otherwise-positive change.

WHAT THIS MEASURES. For each label, at the shipped thresholds:
  - the positive RATE on OOF vs on TEST
  - the probability distribution (mean, sd, and upper quantiles) in each regime
  - how far the test threshold would have to move to reproduce the OOF rate
If test rates are systematically below OOF rates and test probabilities are
visibly narrower, the diagnosis is confirmed.

THE FIX IT POINTS TO. Transfer the OPERATING POINT rather than the raw
threshold: pick the test threshold per label so the predicted positive rate on
test matches the rate the fitted threshold produced on OOF. That is rank-based,
so it is immune to any monotone calibration shift from averaging more models,
and it fits NOTHING on labelled data -- it only enforces that the two regimes
behave the same, which the raw threshold silently assumed.

Usage:
  python src/threshold_regime_check.py --tag paircm
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="paircm")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None,
                    help="write rate-matched thresholds here (.npy)")
    ap.add_argument("--apply", default=None,
                    help="also write the factor column produced by those "
                         "thresholds (.csv), ready for patch_submission_risk")
    args = ap.parse_args()

    th = np.load(f"artifacts/factor_thresholds_{args.tag}_ship.npy")
    te = np.load(f"artifacts/factor_runs/{args.tag}_test.npz", allow_pickle=True)
    TP = te["probs"]

    rid, OP = [], []
    for f in range(args.k):
        d = np.load(f"artifacts/factor_runs/{args.tag}_cv_oof_f{f}.npz",
                    allow_pickle=True)
        rid.extend(list(d["row_id"]))
        OP.append(d["probs"])
    OP = np.concatenate(OP)

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rid), N), dtype=int)
    for i, r in enumerate(rid):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1

    print(f"OOF {OP.shape[0]} posts (each scored by ONE checkpoint per arm)")
    print(f"TEST {TP.shape[0]} posts (each scored by ALL checkpoints, averaged)\n")

    print("=" * 104)
    print("POSITIVE RATE AT THE SHIPPED THRESHOLD, and the distribution behind it")
    print("=" * 104)
    print(f"{'factor':<44}{'th':>6}{'gold%':>7}{'oof%':>7}{'test%':>7}"
          f"{'t/o':>6}{'sd oof':>8}{'sd test':>8}{'shrink':>8}")
    print("-" * 104)

    new_th = th.copy()
    rows = []
    for j, t in enumerate(FACTOR_TAXONOMY):
        o_rate = float((OP[:, j] >= th[j]).mean())
        t_rate = float((TP[:, j] >= th[j]).mean())
        g_rate = float(Y[:, j].mean())
        so, st = float(OP[:, j].std()), float(TP[:, j].std())
        shrink = st / so if so > 1e-9 else float("nan")
        ratio = t_rate / o_rate if o_rate > 1e-9 else float("nan")

        # threshold that reproduces the OOF operating point on test
        if o_rate > 0:
            q = float(np.quantile(TP[:, j], 1.0 - o_rate))
            new_th[j] = q
        flag = ""
        if t_rate == 0 and o_rate > 0:
            flag = "  <-- PREDICTS ZERO"
        elif not np.isnan(ratio) and (ratio < 0.6 or ratio > 1.6):
            flag = "  <--"
        print(f"{t:<44}{th[j]:>6.2f}{100*g_rate:>7.1f}{100*o_rate:>7.1f}"
              f"{100*t_rate:>7.1f}{ratio:>6.2f}{so:>8.3f}{st:>8.3f}"
              f"{shrink:>8.2f}{flag}")
        rows.append(dict(factor=t, th=th[j], gold=g_rate, oof=o_rate,
                         test=t_rate, ratio=ratio, sd_oof=so, sd_test=st,
                         matched_th=new_th[j]))

    df = pd.DataFrame(rows)
    print("\n" + "=" * 104)
    print("VERDICT")
    print("=" * 104)
    print(f"  mean test/oof rate ratio      {df.ratio.mean():.3f} "
          f"(1.00 = the two regimes agree)")
    print(f"  labels under-predicted (<0.8) {int((df.ratio < 0.8).sum())} of {N}")
    print(f"  labels predicting ZERO        {int((df.test == 0).sum())}")
    print(f"  mean sd shrink test/oof       {df.sd_test.mean()/df.sd_oof.mean():.3f}"
          f"  (<1 means test probabilities are narrower)")
    print()
    print("  A mean ratio well below 1 with narrower test probabilities is the")
    print("  diagnosis: thresholds fitted where each post saw 3 checkpoints,")
    print("  applied where each post saw 15. Every label predicting zero costs")
    print("  1/24 of macro-F1 outright.")

    zero = df[df.test == 0]
    if len(zero):
        print("\n  labels lost entirely:")
        for _, r in zero.iterrows():
            print(f"    {r.factor:<44} gold {100*r.gold:>4.1f}%   "
                  f"rate-matched threshold would be {r.matched_th:.2f} "
                  f"(shipped {r.th:.2f})")

    print("\n" + "=" * 104)
    print("RATE-MATCHED THRESHOLDS -- transfer the operating point, not the number")
    print("=" * 104)
    df["move"] = (df.matched_th - df.th).abs()
    moved = df[df.move > 0.01].sort_values("move", ascending=False)
    print(f"{len(moved)} of {N} thresholds move by more than 0.01")
    for _, r in moved.iterrows():
        print(f"  {r.factor:<44}{r.th:>6.2f} -> {r.matched_th:>5.2f}"
              f"   test rate {100*r.test:>5.1f}% -> {100*r.oof:>5.1f}%")

    if args.out:
        np.save(args.out, new_th)
        print(f"\nwrote {args.out}")
    else:
        print("\nPass --out artifacts/factor_thresholds_"
              f"{args.tag}_matched.npy to write these.")

    if args.apply:
        # Write the factor column directly, so the shipped cut-offs are the
        # ones this script just justified rather than a separate re-fit.
        on = (TP >= new_th).astype(int)
        trid = list(te["row_id"])
        out = pd.DataFrame({
            "row_id": trid,
            "factors": [str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
                        for i in range(len(trid))],
        })
        os.makedirs(os.path.dirname(args.apply) or ".", exist_ok=True)
        out.to_csv(args.apply, index=False)
        cnt = on.sum(1)
        guniq = [len({str(f).strip().lower() for f in parse_factors(x)}
                     & set(FACTOR_TAXONOMY)) for x in lab["factors"]]
        print(f"\nwrote {args.apply}")
        print(f"  test  mean {cnt.mean():.2f} factors/post   "
              f"empty {int((cnt == 0).sum())}   "
              f"labels never predicted {int((on.sum(0) == 0).sum())}")
        print(f"  gold  mean {np.mean(guniq):.2f} factors/post   "
              f"(the champion, which scored 0.6301 on S2, predicts 2.94)")
        print("\n  The point of rate-matching is to land on the OOF operating")
        print("  point instead of wherever an in-sample threshold happens to")
        print("  fall on a differently-scaled test distribution. If the mean")
        print("  above is not close to the OOF/gold figure, something else is")
        print("  wrong and this should not be shipped.")

    print("\nNOTE this fits nothing on labelled data. It only asserts that the")
    print("test set should be predicted at the same per-label rate the fitted")
    print("threshold produced out-of-fold -- the assumption the raw threshold")
    print("was making silently and incorrectly.")


if __name__ == "__main__":
    main()

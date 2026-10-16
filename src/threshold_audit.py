"""
threshold_audit.py

Is the champion's factor column using the wrong thresholds?

WHAT WAS FOUND. predict_factors_nli.py loads each run's own thresholds.npy
unless --thresholds is passed, and train_factors_nli.py writes those after
tuning on the 372-post val split. scripts/rebuild_and_apply_rules.sh builds the
submission from artifacts/nli_ensemble_factors.csv, which was produced WITHOUT
the override. So the deployed factor column decides 24 labels with 24 cut-offs
fitted on 372 posts:

    0.85 0.85 0.70 0.85 0.80 0.85 0.70 0.30 0.45 0.80 0.45 0.55
    0.75 0.60 0.85 0.85 0.85 0.85 0.60 0.40 0.85 0.25 0.35 0.20

Meanwhile artifacts/factor_thresholds_oof.npy holds a SINGLE value, 0.63, for
all 24 -- chosen by factors_oof_cv.py because the global strategy beat
per-label under cross-fitting (0.5643 vs 0.5537, with a 14x smaller in-sample
gap: 0.0049 vs 0.0684).

That matters because parameter count has predicted transfer on every
submission: 0-parameter changes over-deliver from val to test, a single
fitted parameter transfers only partially, and more heavily parameterised
changes transfer negatively.

24 cut-offs on 372 posts is the same shape as changes that transfer
negatively.

WHY THIS AUDIT IS NOT TRIVIAL. The obvious test -- score both threshold sets on
val -- is rigged. The deployed cut-offs were TUNED on val, so val is in-sample
for them and out-of-sample for everything else. Any comparison there flatters
the incumbent.

So the honest test is cross-fitted on author-disjoint halves: fit on half A,
apply to half B, and the reverse, for each strategy, using the DEPLOYED models'
own probabilities. Both strategies then face identical conditions.

Three arms:
    per-label   24 cut-offs, cross-fitted        (what ships, made honest)
    global      1 cut-off, cross-fitted
    fixed 0.63  0 cut-offs fitted here at all    (comes from the 1635-post CV
                                                  run, fully out-of-sample)

If global or fixed beats per-label under cross-fitting, the champion is losing
points to threshold overfitting and the fix costs nothing.

Usage:
  python src/threshold_audit.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def macro(P, Y):
    out = []
    for j in range(Y.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        fp = int((P[:, j] & (1 - Y[:, j])).sum())
        fn = int(((1 - P[:, j]) & Y[:, j]).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out)), out


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def fit_per_label(PR, Y, rows):
    th = np.full(PR.shape[1], 0.5)
    for j in range(PR.shape[1]):
        best, bt = -1.0, 0.5
        for t in GRID:
            p = (PR[rows, j] >= t).astype(int)
            tp = int((p & Y[rows, j]).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((p & (1 - Y[rows, j])).sum())
                           + int(((1 - p) & Y[rows, j]).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt
    return th


def fit_global(PR, Y, rows):
    best, bt = -1.0, 0.5
    for t in GRID:
        m, _ = macro((PR[rows] >= t).astype(int), Y[rows])
        if m > best:
            best, bt = m, t
    return np.full(PR.shape[1], bt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--runs", default="factors_nli,factors_nli_s2")
    ap.add_argument("--oof-th", default="artifacts/factor_thresholds_oof.npy")
    args = ap.parse_args()

    names = [n.strip() for n in args.runs.split(",")]
    rid0, mats, deployed = None, [], []
    for n in names:
        d = np.load(os.path.join(args.dir, f"{n}_val.npz"), allow_pickle=True)
        rids = list(d["row_id"])
        if rid0 is None:
            rid0 = rids
        elif rids != rid0:
            o = {r: i for i, r in enumerate(rids)}
            d = {k: d[k] for k in d.files}
            d["probs"] = d["probs"][[o[r] for r in rid0]]
        mats.append(np.asarray(d["probs"], float))
        deployed.append(np.asarray(d["thresholds"], float))
        print(f"  {n}: {mats[-1].shape}")
    PR = np.mean(mats, axis=0)
    TH_DEPLOYED = np.mean(deployed, axis=0)

    lab = pd.read_csv(args.val).set_index("row_id")
    keep = [i for i, r in enumerate(rid0) if r in lab.index]
    rows = [rid0[i] for i in keep]
    PR = PR[keep]
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros(PR.shape, dtype=int)
    for i, r in enumerate(rows):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    half = author_halves(list(lab.loc[rows, "anon_user_id"]))
    A, B = half == "A", half == "B"
    print(f"\n{len(rows)} val posts   halves {int(A.sum())}/{int(B.sum())}")
    print(f"deployed thresholds (seed-averaged): "
          f"min {TH_DEPLOYED.min():.2f} max {TH_DEPLOYED.max():.2f}")

    fixed = np.load(args.oof_th) if os.path.exists(args.oof_th) else None
    if fixed is not None:
        print(f"OOF thresholds: {len(set(np.round(fixed,4)))} distinct value(s)"
              f", {fixed[0]:.2f}")

    print(f"\n{'arm':<34}{'macro':>9}{'halfA':>9}{'halfB':>9}   note")
    print("-" * 76)

    # 1. what ships -- and note val is IN-SAMPLE for these cut-offs
    P = (PR >= TH_DEPLOYED).astype(int)
    m0, _ = macro(P, Y)
    print(f"{'DEPLOYED (24 th, tuned on val)':<34}{m0:>9.4f}"
          f"{macro(P[A], Y[A])[0]:>9.4f}{macro(P[B], Y[B])[0]:>9.4f}"
          f"   IN-SAMPLE, flattered")

    # 2/3. cross-fitted -- identical conditions for both strategies
    res = {}
    for nm, fn in (("per-label", fit_per_label), ("global", fit_global)):
        P = np.zeros_like(Y)
        P[A] = (PR[A] >= fn(PR, Y, B)).astype(int)
        P[B] = (PR[B] >= fn(PR, Y, A)).astype(int)
        m, _ = macro(P, Y)
        res[nm] = m
        print(f"{nm + ' (cross-fitted)':<34}{m:>9.4f}"
              f"{macro(P[A], Y[A])[0]:>9.4f}{macro(P[B], Y[B])[0]:>9.4f}"
              f"   honest")

    if fixed is not None:
        P = (PR >= fixed).astype(int)
        m, _ = macro(P, Y)
        res["fixed"] = m
        print(f"{'fixed 0.63 (from 1635 OOF)':<34}{m:>9.4f}"
              f"{macro(P[A], Y[A])[0]:>9.4f}{macro(P[B], Y[B])[0]:>9.4f}"
              f"   fully out-of-sample")

    print("\n" + "=" * 76)
    print("READ THE CROSS-FITTED ROWS AGAINST EACH OTHER, not against DEPLOYED")
    print("-- val is in-sample for the deployed cut-offs, so that row is")
    print("inflated and cannot be compared to the others.")
    pl, gl = res["per-label"], res["global"]
    print(f"\nper-label {pl:.4f}  vs  global {gl:.4f}   ({gl - pl:+.4f})")
    if gl > pl:
        print("\nGLOBAL WINS under identical conditions. The champion is using")
        print("24 cut-offs fitted on 372 posts where 1 does better. Swapping")
        print("them fits NOTHING new -- 0.63 already exists, chosen on 1635")
        print("posts by cross-fitted comparison.")
        if fixed is not None and res["fixed"] > pl:
            print(f"The fixed 0.63 also beats per-label ({res['fixed']:.4f}),")
            print("and it never saw these posts at all. That is the strongest")
            print("form of the result.")
    else:
        print("\nPer-label holds up under cross-fitting. The deployed cut-offs")
        print("are defensible and this is not the bug it looked like.")
    print("=" * 76)


if __name__ == "__main__":
    main()

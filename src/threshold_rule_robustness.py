"""
threshold_rule_robustness.py

Is the shrinkage threshold gain real, or one lucky author split?

WHAT IS AVAILABLE. src/threshold_rules.py, on the incumbent's 1635 out-of-fold
posts, one author split, shows both `shrink` and `prevalence` reducing the
honest cross-fit drop relative to `argmax`, with `shrink` the larger gain.

The gain from `shrink` is an order of magnitude above the best of twelve
model arms tested, and the gain sits entirely on the rare labels, which is
the signature predicted BEFORE the run rather than found afterwards.

WHY IT IS NOT YET SHIPPABLE. Two unknowns, both of which have caused false
positives in this kind of measurement before.

  1. ONE SPLIT. This gain came from a single arbitrary partition of authors
     into halves. The Qwen factor pilot looked like a solid gain until its two
     folds were separated and disagreed in sign. This script repeats the whole
     measurement over many independent random author partitions and reports the
     spread, not just the mean.

  2. ONE VALUE OF k. Shrinkage weight is n/(n+k) with k=50, chosen a priori as
     roughly where a per-label estimate starts to be worth trusting. If the
     rule only wins at 50, that is a knife edge and the win is noise. If it
     wins across 20..150, the mechanism is real.

     CHECKING ROBUSTNESS ACROSS k IS NOT THE SAME AS PICKING k. We keep 50
     whatever this prints. The sweep exists to see whether the surface is flat,
     not to find its peak -- selecting the best k here would fit a parameter on
     validation data, and the last one-parameter change transferred at 17%.

WHAT WOULD FALSIFY IT. shrink winning on fewer than ~80% of splits, or a mean
gain smaller than its own standard deviation, or the win vanishing away from
k=50. Any of those and this goes in the same drawer as the other twelve.

Usage:
  python src/threshold_rule_robustness.py
  python src/threshold_rule_robustness.py --splits 40
  python src/threshold_rule_robustness.py --arm artifacts/factor_runs/mnli_cv_oof_f{f}.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors
from threshold_rules import (GRID, N, f1_at, fit, macro)

PROT = {"social support", "coping strategy", "psychological capital",
        "sense of responsibility", "meaning in life"}


def random_author_halves(auth, rng, frac=0.5):
    """Partition AUTHORS (never posts) into a fitting side A and a scored side
    B, with roughly `frac` of the POSTS on the scored side.

    frac=0.5 is the symmetric split used by compare_factor_arms.py, which lets
    both halves be scored and every post contribute.

    frac=0.2 mimics the SUBMISSION instead: thresholds fitted on ~1300 posts
    and applied to ~330, against the real 1635-fit / 378-test ratio. This
    matters because shrinkage repairs noise in the per-label estimate, and that
    noise falls as the fitting set grows -- a 50/50 split fits on ~800 and
    therefore OVERSTATES how much shrinkage will be worth when shipped."""
    s = pd.Series(auth)
    counts = s.value_counts()
    authors = list(counts.index)
    rng.shuffle(authors)
    target = frac * len(s)
    half, nb = {}, 0
    for a in authors:
        if nb < target:
            half[a] = "B"
            nb += counts[a]
        else:
            half[a] = "A"
    return s.map(half).to_numpy()


def honest(PR, Y, A, B, rule, tol, k):
    Px = np.zeros_like(Y)
    Px[A] = (PR[A] >= fit(PR, Y, B, rule, tol, k)).astype(int)
    Px[B] = (PR[B] >= fit(PR, Y, A, rule, tol, k)).astype(int)
    return macro(Px, Y)


def one_way(PR, Y, A, B, rule, tol, k):
    """Fit on A, score B only. Used for the asymmetric submission-scale split,
    where A is deliberately much larger and is never scored."""
    th = fit(PR, Y, A, rule, tol, k)
    return macro((PR[B] >= th).astype(int), Y[B])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=25)
    ap.add_argument("--sub-splits", type=int, default=100,
                    help="splits for the submission-scale test. Higher than "
                         "--splits because only ~20%% of posts are scored "
                         "there, so each estimate is far noisier.")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--shrink-k", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    rid, P = [], []
    for f in range(args.k):
        p = args.arm.format(f=f)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = np.load(p, allow_pickle=True)
        rid.extend(list(d["row_id"]))
        P.append(d["probs"])
    PR = np.concatenate(P)

    Y = np.zeros((len(rid), N), dtype=int)
    for i, r in enumerate(rid):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1
    nlab = Y.sum(0)
    auth = [lab.loc[r, "anon_user_id"] for r in rid]

    print(f"{args.arm}")
    print(f"{len(rid)} posts, {len(set(auth))} authors, "
          f"{args.splits} random author partitions\n")

    rules = ["argmax", "plateau", "prevalence", "shrink"]
    acc = {r: [] for r in rules}
    per_acc = {r: [] for r in rules}
    rng = np.random.default_rng(args.seed)

    for s in range(args.splits):
        h = random_author_halves(auth, rng)
        A, B = h == "A", h == "B"
        for r in rules:
            m, per = honest(PR, Y, A, B, r, args.tol, args.shrink_k)
            acc[r].append(m)
            per_acc[r].append(per)

    print("=" * 82)
    print(f"HONEST MACRO-F1 OVER {args.splits} INDEPENDENT AUTHOR PARTITIONS")
    print("=" * 82)
    base = np.array(acc["argmax"])
    print(f"{'rule':<14}{'mean':>9}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'d vs argmax':>13}{'wins':>8}")
    print("-" * 82)
    for r in rules:
        v = np.array(acc[r])
        d = v - base
        wins = "" if r == "argmax" else f"{int((d > 0).sum())}/{args.splits}"
        dd = "" if r == "argmax" else f"{d.mean():>+13.4f}"
        print(f"{r:<14}{v.mean():>9.4f}{v.std():>8.4f}{v.min():>9.4f}"
              f"{v.max():>9.4f}{dd}{wins:>8}")

    d = np.array(acc["shrink"]) - base
    print(f"\nshrink vs argmax: mean {d.mean():+.4f}, sd {d.std():.4f}, "
          f"worst split {d.min():+.4f}, best {d.max():+.4f}")
    if d.mean() > 2 * d.std() and (d > 0).mean() >= 0.8:
        print("  -> the mean is more than twice its own spread and it wins on")
        print("     at least 80% of partitions. This is not split luck.")
    else:
        print("  -> NOT clean. The gain is comparable to its own variability;")
        print("     treat it as unproven.")

    print("\n" + "=" * 82)
    print("SENSITIVITY TO k -- robustness check ONLY. k stays at 50.")
    print("=" * 82)
    print("A flat surface means the mechanism is real. A single peak at 50")
    print("means we found noise. Do NOT adopt the best row.")
    print()
    rng2 = np.random.default_rng(args.seed)
    splits = [random_author_halves(auth, rng2)
              for _ in range(min(args.splits, 10))]
    print(f"{'k':>7}{'mean honest':>14}{'d vs argmax':>14}{'wins':>8}")
    print("-" * 50)
    for kk in (10, 20, 35, 50, 75, 100, 150, 300):
        vals = []
        for h in splits:
            A, B = h == "A", h == "B"
            m, _ = honest(PR, Y, A, B, "shrink", args.tol, float(kk))
            vals.append(m)
        vals = np.array(vals)
        bb = base[:len(vals)]
        mark = "   <- shipped" if kk == 50 else ""
        print(f"{kk:>7}{vals.mean():>14.4f}{vals.mean()-bb.mean():>+14.4f}"
              f"{int((vals - bb > 0).sum()):>5}/{len(vals)}{mark}")

    print("\n" + "=" * 82)
    print("SUBMISSION-SCALE ESTIMATE -- fit on ~80% of authors, score the rest")
    print("=" * 82)
    print("The 50/50 measurement above fits thresholds on ~800 posts. The real")
    print("submission fits on all 1635 and scores 378. Shrinkage repairs noise")
    print("in the per-label estimate, and that noise shrinks as the fitting set")
    print("grows -- so 50/50 OVERSTATES what shipping will actually gain.")
    print("This ratio matches the real one and is the number to believe.")
    print()
    rng3 = np.random.default_rng(args.seed + 1)
    nsub = max(args.splits, args.sub_splits)
    sub = {r: [] for r in rules}
    sub_per = {r: [] for r in rules}
    sub_has = []
    for _ in range(nsub):
        h = random_author_halves(auth, rng3, frac=0.2)
        A, B = h == "A", h == "B"
        sub_has.append(Y[B].sum(0) > 0)
        for r in rules:
            m, per = one_way(PR, Y, A, B, r, args.tol, args.shrink_k)
            sub[r].append(m)
            sub_per[r].append(per)
    sb = np.array(sub["argmax"])
    print(f"{'rule':<14}{'mean':>9}{'sd':>8}{'d vs argmax':>14}{'wins':>8}")
    print("-" * 60)
    for r in rules:
        v = np.array(sub[r])
        dd = "" if r == "argmax" else f"{(v - sb).mean():>+14.4f}"
        w = "" if r == "argmax" else f"{int((v - sb > 0).sum())}/{nsub}"
        print(f"{r:<14}{v.mean():>9.4f}{v.std():>8.4f}{dd}{w:>8}")

    # The PAIRED spread is what decides this, not the spread of each arm --
    # both rules see the identical posts on every split, so their shared
    # split-to-split variation cancels in the difference.
    ds = np.array(sub["shrink"]) - sb
    se = ds.std(ddof=1) / np.sqrt(len(ds))
    lo, hi = ds.mean() - 1.96 * se, ds.mean() + 1.96 * se
    print(f"\n  paired difference: mean {ds.mean():+.4f}  sd {ds.std(ddof=1):.4f}"
          f"  se {se:.4f}")
    print(f"  95% interval:      [{lo:+.4f}, {hi:+.4f}]   over {nsub} splits")
    print(f"  wins {int((ds > 0).sum())}/{nsub}")
    print(f"  -> composite impact: {0.3 * ds.mean():+.4f} "
          f"[{0.3*lo:+.4f}, {0.3*hi:+.4f}]")
    if lo > 0:
        print("\n  Interval excludes zero -> the gain is small but real at")
        print("  submission scale. Ship it.")
    elif hi < 0:
        print("\n  Interval is entirely negative -> shrinkage HURTS at")
        print("  submission scale. Do not ship.")
    else:
        print("\n  Interval spans zero -> UNDERPOWERED, not disproved. Only")
        print("  ~330 posts are scored per split, so the rarest labels carry")
        print("  one or two positives each. Re-run with --sub-splits 200 to")
        print("  narrow it before concluding anything.")

    print("\n  PER LABEL AT SUBMISSION SCALE  (averaged over splits where the")
    print("  label has at least one positive in the scored portion)")
    print(f"  {'factor':<42}{'n':>5}{'seen':>6}{'argmax':>9}{'shrink':>9}"
          f"{'delta':>9}")
    print("  " + "-" * 80)
    HA = np.array(sub_has)
    PA = np.array(sub_per["argmax"])
    PS = np.array(sub_per["shrink"])
    srare = scom = 0.0
    for j in sorted(range(N), key=lambda x: nlab[x]):
        m = HA[:, j]
        if m.sum() == 0:
            continue
        a_, b_ = PA[m, j].mean(), PS[m, j].mean()
        dl = b_ - a_
        if nlab[j] <= 60:
            srare += dl
        else:
            scom += dl
        if abs(dl) > 0.005:
            print(f"  {FACTOR_TAXONOMY[j]:<42}{int(nlab[j]):>5}{int(m.sum()):>6}"
                  f"{a_:>9.3f}{b_:>9.3f}{dl:>+9.3f}"
                  f"{'  prot' if FACTOR_TAXONOMY[j] in PROT else ''}")
    print(f"\n  rare (n <= 60)   {srare:>+8.3f}      common (n > 60) {scom:>+8.3f}")
    print("  Compare these against the 50/50 table below. If the rare-label")
    print("  gain holds here too, the macro average is being diluted by the")
    print("  common labels rather than the effect disappearing.")

    print("\n" + "=" * 82)
    print("PER LABEL, averaged over all partitions  (argmax -> shrink)")
    print("=" * 82)
    a = np.array(per_acc["argmax"]).mean(0)
    b = np.array(per_acc["shrink"]).mean(0)
    order = sorted(range(N), key=lambda j: nlab[j])
    print(f"{'factor':<44}{'n':>5}{'argmax':>9}{'shrink':>9}{'delta':>9}")
    print("-" * 80)
    rare = com = 0.0
    for j in order:
        dl = b[j] - a[j]
        if nlab[j] <= 60:
            rare += dl
        else:
            com += dl
        if abs(dl) > 0.005:
            print(f"{FACTOR_TAXONOMY[j]:<44}{int(nlab[j]):>5}{a[j]:>9.3f}"
                  f"{b[j]:>9.3f}{dl:>+9.3f}"
                  f"{'  prot' if FACTOR_TAXONOMY[j] in PROT else ''}")
    print(f"\n  total delta on RARE labels (n <= 60)   {rare:>+8.3f}")
    print(f"  total delta on COMMON labels (n > 60)  {com:>+8.3f}")
    print("\n  Shrinkage should help where per-label estimates are noisy and do")
    print("  nothing elsewhere. Rare positive and common near zero is the")
    print("  mechanism working. Gains on COMMON labels would be unexplained,")
    print("  and an unexplained gain should not be trusted on submission.")


if __name__ == "__main__":
    main()

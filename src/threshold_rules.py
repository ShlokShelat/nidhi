"""
threshold_rules.py

The threshold RULE is worth more than most of the arms tested so far.

THE SIZE OF THE PRIZE. compare_factor_arms.py measures what each arm loses when
thresholds are fitted honestly instead of on the posts being scored:

    DeBERTa incumbent   pooled 0.6614 -> xfit 0.6083     drop 0.0531

That 0.053 is larger than every gain the last twelve arms produced put
together. Part of it is irreducible -- `meaning in life` has 45 positives in
1635 posts, so any cutoff for it rests on a couple of dozen examples. But part
of it is the RULE, and the rule we use is the crudest one available:

    for t in 0.05 .. 0.95: keep whichever t scores highest

On a label with 20 positives the F1-vs-threshold curve is a sawtooth. Moving t
by 0.01 flips one post, and one post out of 20 moves F1 by several points. So
argmax routinely lands on a spike created by a single example that happened to
fall in the tuning half. That spike is not there in the other half.

FOUR RULES, ALL SCORED THE SAME WAY. Fit on one author half, apply to the
other, both directions. No rule ever sees the posts it is scored on.

  argmax       what is currently used. The baseline.
  plateau      take every threshold within `--tol` of the best F1 and use the
               MIDDLE of that range. If the top of the curve is a broad
               plateau, the middle of it survives resampling; the exact peak
               does not.
  prevalence   set the cutoff so the model predicts the factor about as often
               as it actually occurs in the tuning half. Ignores F1 entirely,
               so it cannot chase a spike. Strong when positives are few.
  shrink       fit the per-label threshold, then pull it toward the single
               global threshold by a weight that depends on how few positives
               the label has: w = n / (n + k). A label with 400 positives keeps
               its own estimate; one with 8 mostly inherits the global. This is
               ordinary shrinkage -- trust a noisy estimate less.

WHY THIS IS WORTH RUNNING BEFORE ANY MORE GPU. It fits NOTHING new. The rule
is chosen once, structurally, not tuned per label -- so it lands in the
zero-fitted-parameter category, the only category that has ever over-delivered,
against more heavily parameterised strategies that transfer negatively.

HONEST WARNING. `--tol` and `--shrink-k` are knobs. Picking them by looking at
the output below IS fitting parameters, at which point the transfer law applies
and you should expect roughly a sixth of what you see. The defaults are set
from first principles -- tol at one F1 point, k at 50 positives, which is
roughly where per-label estimates start to be worth trusting -- and they should
be left alone. A rule that only wins at one setting is noise.

Usage:
  python src/threshold_rules.py
  python src/threshold_rules.py --arm artifacts/factor_runs/mnli_cv_oof_f{f}.npz
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
PROT = {"social support", "coping strategy", "psychological capital",
        "sense of responsibility", "meaning in life"}


def f1_at(p, y):
    tp = int((p & y).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + int((p & (1 - y)).sum()) + int(((1 - p) & y).sum()))


def curve(pr, y):
    return np.array([f1_at((pr >= t).astype(int), y) for t in GRID])


def th_argmax(pr, y):
    return float(GRID[int(np.argmax(curve(pr, y)))])


def th_plateau(pr, y, tol):
    c = curve(pr, y)
    best = c.max()
    if best <= 0:
        return 0.5
    keep = GRID[c >= best - tol]
    return float(np.median(keep))


def th_prevalence(pr, y):
    """Cutoff reproducing the observed positive rate. Never looks at F1, so a
    single lucky post cannot move it."""
    k = int(y.sum())
    if k == 0:
        return 0.5
    q = 1.0 - k / len(y)
    return float(np.clip(np.quantile(pr, q), GRID[0], GRID[-1]))


def fit(PR, Y, rows, rule, tol, kshrink):
    """One threshold per label, fitted on `rows` only."""
    th = np.full(N, 0.5)
    for j in range(N):
        pr, y = PR[rows, j], Y[rows, j]
        if y.sum() == 0:
            continue
        if rule == "argmax":
            th[j] = th_argmax(pr, y)
        elif rule == "plateau":
            th[j] = th_plateau(pr, y, tol)
        elif rule == "prevalence":
            th[j] = th_prevalence(pr, y)
        elif rule == "shrink":
            th[j] = th_argmax(pr, y)
        else:
            raise ValueError(rule)

    if rule == "shrink":
        # one global threshold over all labels pooled, then pull each per-label
        # estimate toward it with weight n/(n+k)
        flat_p = PR[rows].ravel()
        flat_y = Y[rows].ravel()
        g = float(GRID[int(np.argmax([f1_at((flat_p >= t).astype(int), flat_y)
                                      for t in GRID]))])
        n = Y[rows].sum(0)
        w = n / (n + kshrink)
        th = w * th + (1 - w) * g
    return th


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(counts.index, key=lambda x: (-counts[x], str(x))):
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def macro(P, Y):
    v = [f1_at(P[:, j], Y[:, j]) for j in range(N)]
    return float(np.mean(v)), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="plateau width, in F1. Default one F1 point.")
    ap.add_argument("--shrink-k", type=float, default=50.0,
                    help="positives at which a label half-trusts its own "
                         "threshold. Default 50.")
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
    n = Y.sum(0)

    half = author_halves([lab.loc[r, "anon_user_id"] for r in rid])
    A, B = half == "A", half == "B"
    print(f"{args.arm}")
    print(f"{len(rid)} posts, author halves {int(A.sum())}/{int(B.sum())}\n")

    rules = ["argmax", "plateau", "prevalence", "shrink"]
    print("=" * 78)
    print("THRESHOLD RULE, scored honestly (fit on one author half, apply to")
    print("the other). `cheating` fits on the same posts it scores, for")
    print("reference only -- it is what the old summaries reported.")
    print("=" * 78)
    print(f"{'rule':<14}{'cheating':>11}{'honest':>10}{'drop':>9}{'d vs argmax':>14}")
    print("-" * 78)

    per_label = {}
    base = None
    for r in rules:
        ch, _ = macro((PR >= fit(PR, Y, np.ones(len(Y), bool),
                                 r, args.tol, args.shrink_k)).astype(int), Y)
        Px = np.zeros_like(Y)
        Px[A] = (PR[A] >= fit(PR, Y, B, r, args.tol, args.shrink_k)).astype(int)
        Px[B] = (PR[B] >= fit(PR, Y, A, r, args.tol, args.shrink_k)).astype(int)
        ho, per = macro(Px, Y)
        per_label[r] = per
        if base is None:
            base = ho
        print(f"{r:<14}{ch:>11.4f}{ho:>10.4f}{ch-ho:>9.4f}"
              f"{'' if r == 'argmax' else f'{ho-base:>+14.4f}'}")

    best = max(rules, key=lambda r: np.mean(per_label[r]))
    print(f"\nbest honest rule: {best}")

    print("\n" + "=" * 78)
    print(f"PER LABEL, argmax -> {best}   (movers only, sorted by rarity)")
    print("=" * 78)
    a, b = per_label["argmax"], per_label[best]
    order = sorted(range(N), key=lambda j: n[j])
    print(f"{'factor':<44}{'n':>5}{'argmax':>9}{best:>11}{'delta':>9}")
    print("-" * 80)
    for j in order:
        if abs(b[j] - a[j]) > 0.005:
            print(f"{FACTOR_TAXONOMY[j]:<44}{int(n[j]):>5}{a[j]:>9.3f}"
                  f"{b[j]:>11.3f}{b[j]-a[j]:>+9.3f}"
                  f"{'  prot' if FACTOR_TAXONOMY[j] in PROT else ''}")

    print("\n" + "=" * 78)
    print("HOW TO READ IT")
    print("=" * 78)
    print("  The `drop` column is the cost of threshold noise. A rule that")
    print("  shrinks it is worth more than an arm that raises the cheating")
    print("  column, because only the honest column reaches a submission.")
    print()
    print("  Expect gains concentrated in the RARE labels -- that is where the")
    print("  F1 curve is a sawtooth and argmax lands on a spike. If a rule wins")
    print("  only on common labels, it is probably noise.")
    print()
    print("  Do NOT tune --tol or --shrink-k from this output. Picking them")
    print("  here fits parameters on validation data, and the transfer law says")
    print("  you would keep about a sixth of what you see.")


if __name__ == "__main__":
    main()

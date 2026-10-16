"""
factor_per_label_arm.py

Per-LABEL arm selection: use whichever factor arm is best at each label.

WHY NOW. Per-CLASS routing on risk is a zero-parameter rule: hand one class
to whichever model is better at it. Factors is the exact analogue -- 24
labels instead of 4 classes, and ~20
trained arms instead of 2 models -- and it has never been tried. Factors is
also 0.30 of the composite, has not moved across recent submissions (0.6519),
and is where the leaders are furthest ahead.

WHY IT IS NOT THE CLOSED THRESHOLD LANE. Every threshold policy failed, best
case, with a CI spanning zero, and rank-averaging failed the same way.
Those change the DECISION on a fixed ranking. This changes WHICH MODEL'S RANKING
is used, per label, which is a different degree of freedom.

THRESHOLDING IS REMOVED BY CONSTRUCTION. Each label keeps the incumbent's
F1-optimal fire COUNT, and the top-k of whichever arm is selected is taken. So
density, per-label precision/recall trade and zero-prediction safety are all held
fixed, and only the ranking moves -- the same isolation `--match-counts` gives.

THE CAPACITY PROBLEM IS REAL AND IS PRICED. Choosing 1 of A arms for each of 24
labels is a lot of fitted choice, and the measured transfer law is that
0 params over-deliver, 1 fitted parameter transfers only partially, and
heavily parameterised changes transfer negatively. So:
  - every selection is CROSS-FITTED over author halves (chosen on one half,
    applied to the other), so no label is scored where it was fitted
  - a 0-parameter control (mean of the shipped arms) and a low-capacity variant
    (one arm for ALL labels) are reported alongside
  - the number of labels whose choice DISAGREES between the two halves is
    printed: if the choices are unstable across halves, the rule is fitting
    noise however good its cross-fitted score looks

BAR: beat the shipped mean with the 95% paired-author CI excluding zero, AND
>= 60% of per-label choices stable across halves. Below the stability bar it
is noise dressed as selection.

Usage:
  python src/factor_per_label_arm.py
  python src/factor_per_label_arm.py --arms mnli,dmnli,cmine,prot,d4,d6,defs,qnli
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

D = "artifacts/factor_runs"


def load_arm(arm: str, rows: list[str]) -> np.ndarray | None:
    """OOF probs for one arm aligned to rows. Fold dumps are DISJOINT
    (330/330/323/323/329) and CONCATENATE to 1635 -- a post is scored
    out-of-fold by exactly one model, so averaging them is wrong."""
    fs = sorted(glob.glob(f"{D}/{arm}_cv_oof_f*.npz"))
    if not fs:
        return None
    chunks, ids = [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        if "row_id" not in z.files:
            return None
        chunks.append(np.asarray(z["probs"], float))
        ids += [str(x) for x in z["row_id"]]
    P = np.concatenate(chunks, axis=0)
    idx = {}
    for i, r in enumerate(ids):
        idx.setdefault(r, i)
    if any(r not in idx for r in rows):
        return None
    return P[[idx[r] for r in rows]]


def f1_at_count(score, y, k):
    if k <= 0:
        return 0.0 if y.sum() else 1.0
    k = min(k, len(score))
    order = np.argsort(-score, kind="stable")
    pred = np.zeros(len(score), bool)
    pred[order[:k]] = True
    tp = int((pred & (y == 1)).sum())
    if tp == 0:
        return 0.0
    pr, rc = tp / k, tp / max(int(y.sum()), 1)
    return 2 * pr * rc / (pr + rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="mnli,dmnli,cmine,prot,d4")
    ap.add_argument("--extra-arms", default="d6,defs,qnli,lnli,mnli7,hyb,aspv2,"
                                           "syn,synapi,synsty,robmnli",
                    help="candidate arms beyond the shipped bag; skipped "
                         "silently if their OOF dumps are absent")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = list(lab.index)
    labels = list(FACTOR_TAXONOMY)
    Y = np.array([[f in set(parse_factors(lab.loc[r, "factors"]))
                   for f in labels] for r in rows], int)
    author = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    print(f"{len(rows)} posts / {len(set(author))} authors / {len(labels)} labels")

    shipped = [a.strip() for a in args.arms.split(",") if a.strip()]
    extra = [a.strip() for a in args.extra_arms.split(",") if a.strip()]
    M, used_ship, used_all = {}, [], []
    for a in shipped + extra:
        P = load_arm(a, rows)
        if P is None:
            continue
        if P.shape[1] != len(labels):
            print(f"  skip {a}: {P.shape[1]} columns, expected {len(labels)}")
            continue
        M[a] = P
        used_all.append(a)
        if a in shipped:
            used_ship.append(a)
    if len(used_ship) < 2:
        raise SystemExit(f"need >=2 shipped arms with OOF dumps in {D}/; "
                         f"found {used_ship}")
    print(f"  shipped arms loaded : {used_ship}")
    print(f"  extra arms loaded   : "
          f"{[a for a in used_all if a not in used_ship]}")

    INC = np.mean([M[a] for a in used_ship], axis=0)

    # counts: the incumbent's own F1-optimal fire count per label, held fixed
    counts = {}
    step = max(1, len(rows) // 300)
    for j, f in enumerate(labels):
        best_k, best = 0, -1.0
        for k in range(0, len(rows) + 1, step):
            v = f1_at_count(INC[:, j], Y[:, j], k)
            if v > best:
                best, best_k = v, k
        counts[f] = best_k

    def score_matrix(pick):
        """pick: label index -> probability column. Returns per-label F1."""
        return np.array([f1_at_count(pick[j], Y[:, j], counts[labels[j]])
                         for j in range(len(labels))])

    base_f1 = score_matrix([INC[:, j] for j in range(len(labels))])
    print(f"\nshipped mean of {len(used_ship)} arms: macro "
          f"{base_f1.mean():.4f}")

    rng = np.random.default_rng(args.seed)
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}
    perm = rng.permutation(len(auths))
    hA = {auths[i] for i in perm[:len(auths) // 2]}
    inA = np.array([a in hA for a in author])

    def best_arm_on(mask, j, pool):
        best, bf = pool[0], -1.0
        kk = max(1, int(round(counts[labels[j]] * mask.sum() / len(rows))))
        for a in pool:
            v = f1_at_count(M[a][mask, j], Y[mask, j], kk)
            if v > bf:
                bf, best = v, a
        return best

    def xfit_per_label(pool):
        """Choose an arm per label on one half, apply to the other."""
        cols = [np.zeros(len(rows)) for _ in labels]
        choices = []
        for tune, appl in ((inA, ~inA), (~inA, inA)):
            ch = []
            for j in range(len(labels)):
                a = best_arm_on(tune, j, pool)
                ch.append(a)
                cols[j][appl] = M[a][appl, j]
            choices.append(ch)
        stable = sum(1 for x, y_ in zip(*choices) if x == y_)
        return cols, choices, stable

    def ci(per_label_cols):
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            a_, b_ = [], []
            for j in range(len(labels)):
                kk = max(1, int(round(counts[labels[j]] * len(sel) / len(rows))))
                a_.append(f1_at_count(per_label_cols[j][sel], Y[sel, j], kk))
                b_.append(f1_at_count(INC[sel, j], Y[sel, j], kk))
            draws.append(float(np.mean(a_) - np.mean(b_)))
        return np.percentile(draws, [2.5, 97.5])

    print(f"\n{'rule':<40}{'macro':>9}{'delta':>9}{'95% CI':>22}{'stable':>9}")
    print("-" * 89)
    print(f"{'shipped mean (baseline)':<40}{base_f1.mean():>9.4f}")

    # 0-parameter control: mean of EVERY loaded arm
    if len(used_all) > len(used_ship):
        allmean = np.mean([M[a] for a in used_all], axis=0)
        cols = [allmean[:, j] for j in range(len(labels))]
        f = score_matrix(cols)
        lo, hi = ci(cols)
        print(f"{f'mean of all {len(used_all)} arms (0 par)':<40}"
              f"{f.mean():>9.4f}{f.mean() - base_f1.mean():>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{'n/a':>9}")

    # low capacity: ONE arm for every label, cross-fitted
    one = []
    for tune, appl in ((inA, ~inA), (~inA, inA)):
        best, bf = used_all[0], -1.0
        for a in used_all:
            v = np.mean([f1_at_count(
                M[a][tune, j], Y[tune, j],
                max(1, int(round(counts[labels[j]] * tune.sum() / len(rows)))))
                for j in range(len(labels))])
            if v > bf:
                bf, best = v, a
        one.append((best, appl))
    cols = [np.zeros(len(rows)) for _ in labels]
    for a, appl in one:
        for j in range(len(labels)):
            cols[j][appl] = M[a][appl, j]
    f = score_matrix(cols)
    lo, hi = ci(cols)
    print(f"{f'single best arm xfit {[a for a,_ in one]}'[:39]:<40}"
          f"{f.mean():>9.4f}{f.mean() - base_f1.mean():>+9.4f}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{'-':>9}")

    # the real rule: per-label arm, cross-fitted, over two pools
    for nm, pool in (("shipped pool", used_ship), ("all arms", used_all)):
        cols, choices, stable = xfit_per_label(pool)
        f = score_matrix(cols)
        lo, hi = ci(cols)
        ok = (f.mean() - base_f1.mean()) >= 0.010 and lo > 0 and \
             stable / len(labels) >= 0.60
        print(f"{f'per-label arm xfit ({nm}, {len(pool)})':<40}"
              f"{f.mean():>9.4f}{f.mean() - base_f1.mean():>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}"
              f"{stable}/{len(labels):<4}{'  PASSES' if ok else ''}")

    print("\nBAR: beats the shipped mean AND CI excluding zero AND >= 60% of per-label")
    print("choices identical across the two author halves. Unstable choices")
    print("mean the rule is fitting noise, whatever its cross-fitted score.")
    print("Composite impact = 0.3 x delta.")


if __name__ == "__main__":
    main()

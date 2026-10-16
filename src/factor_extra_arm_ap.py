"""
factor_extra_arm_ap.py

Do the SYNTHETIC / EXTERNAL arms help the STARVED labels, per label?

WHY RE-ASK. The synthetic arms were previously rejected on one number: the
MACRO (all 24) row came out net negative, under the stated rule "SHIP ONLY IF
the MACRO row beats the incumbent." But artifacts/external_factor_aug.csv adds
gold to only FIVE labels and nothing to the other nineteen:

    substance use                 33 gold -> +64   (+194%)
    exposure to others' suicide   14      -> +17   (+121%)
    poor school performance       16      -> +13    (+81%)
    traumatic experience          64      -> +36    (+56%)
    low socio-economic status     54      -> +29    (+54%)

A net-negative macro is exactly what "helped 5, slightly hurt 19" looks like.
Under a MACRO-averaged scorer those five are worth 5/24 of the score
regardless of how rare they are, and their combined deficit is enough to
outweigh gains on the other nineteen.

Closing that on an average is the same mistake that wrongly closed a risk
definitions lane (a model that IMPROVED indicator and ideation was closed on
an aggregate loss) and the factor-LLM lane. This is the per-label check.

AND A DIFFERENT INSTRUMENT. The earlier rejection judged on tuned-threshold
F1, which tangles ranking quality with fitted thresholds. AP is
threshold-free, so it answers the
only question that matters here: does the extra data make the RANKING better on
the labels it actually fed?

WHAT IT REPORTS, per label, all paired by author bootstrap
  * AP of the shipped 8-arm pool
  * AP of each extra arm alone
  * AP of pool+arm (a 9-arm logit-mean) -- the shippable shape, since adding an
    arm for SOME labels only is a per-label source choice with 0 parameters
  * the delta and CI for pool+arm vs pool, flagged on the starved labels

DECISION. An extra arm earns a label only if pool+arm beats pool on that label
with the CI clear of zero. Anything else and the July closure stands.

Usage:
  python src/factor_extra_arm_ap.py
  python src/factor_extra_arm_ap.py --extra syn synapi synsty --boot 2000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]

# labels the external file actually feeds, with how much gold it adds
STARVED = {
    "substance use": "+194%",
    "exposure to others' suicide": "+121%",
    "poor school performance": "+81%",
    "traumatic experience": "+56%",
    "low socio-economic status": "+54%",
}


def load_arm(d, a, k):
    pr, ids = [], []
    for f in range(k):
        p = f"{d}/{a}_cv_oof_f{f}.npz"
        if not os.path.exists(p):
            return None, None
        z = np.load(p, allow_pickle=True)
        pr.append(z["probs"].astype(float))
        ids.extend(str(x) for x in z["row_id"])
    return np.concatenate(pr), ids


def align(pr, ids, rid):
    if ids == rid:
        return pr
    o = {x: i for i, x in enumerate(ids)}
    missing = [r for r in rid if r not in o]
    if missing:
        return None
    return pr[[o[r] for r in rid]]


def logit_mean(S):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def ap_per_label(P, Y):
    return np.array([average_precision_score(Y[:, j], P[:, j])
                     if Y[:, j].sum() else 0.0 for j in range(N)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--extra", nargs="+", default=["syn", "synapi", "synsty"])
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    base, rid = [], None
    for a in args.arms:
        pr, ids = load_arm(args.dir, a, args.k)
        if pr is None:
            raise SystemExit(f"missing OOF dumps for base arm {a}")
        if rid is None:
            rid = ids
            base.append(pr)
        else:
            al = align(pr, ids, rid)
            if al is None:
                raise SystemExit(f"{a} does not cover the same posts")
            base.append(al)
    S = np.stack(base)
    P = logit_mean(S)
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = np.zeros((n, N), dtype=np.int8)
    for i, r in enumerate(rid):
        for x in set(parse_factors(str(lab.loc[r, "factors"]))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([by[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]

    print(f"OOF {n} posts / {len(uauth)} authors")
    print(f"base pool: {args.arms}")

    ap_pool = ap_per_label(P, Y)

    extras = {}
    for a in args.extra:
        pr, ids = load_arm(args.dir, a, args.k)
        if pr is None:
            print(f"  {a}: no OOF dumps, skipping")
            continue
        al = align(pr, ids, rid)
        if al is None:
            print(f"  {a}: does not cover the same posts, skipping")
            continue
        extras[a] = al
    if not extras:
        raise SystemExit("no usable extra arms")
    print(f"extra arms: {list(extras)}\n")

    for a, E in extras.items():
        Pplus = logit_mean(np.concatenate([S, E[None]], axis=0))
        ap_alone = ap_per_label(E, Y)
        ap_plus = ap_per_label(Pplus, Y)

        D = np.zeros((args.boot, N))
        for b, ix in enumerate(IXS):
            D[b] = ap_per_label(Pplus[ix], Y[ix]) - ap_per_label(P[ix], Y[ix])
        lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
        win = (D > 0).mean(0)

        print("=" * 104)
        print(f"EXTRA ARM: {a}")
        print("=" * 104)
        print(f"{'label':<44}{'gold':>6}{'pool':>7}{'alone':>7}{'pool+':>7}"
              f"{'d':>8}{'95% CI':>18}{'win':>6}")
        earned = []
        for j in np.argsort(-(ap_plus - ap_pool)):
            t = FACTOR_TAXONOMY[j]
            mark = ""
            if lo[j] > 0:
                mark = " EARNS IT"
                earned.append(j)
            elif hi[j] < 0:
                mark = " (hurts)"
            tag = ("> " + t) if t in STARVED else "  " + t
            print(f"{tag:<44}{int(Y[:, j].sum()):>6}{ap_pool[j]:>7.3f}"
                  f"{ap_alone[j]:>7.3f}{ap_plus[j]:>7.3f}"
                  f"{ap_plus[j] - ap_pool[j]:>+8.3f}"
                  f"{f'[{lo[j]:+.3f},{hi[j]:+.3f}]':>18}{win[j]:>6.0%}{mark}")
        print("  > = a label the external file actually feeds")

        print(f"\n  mean AP  pool {ap_pool.mean():.4f}   "
              f"pool+{a} {ap_plus.mean():.4f} "
              f"({ap_plus.mean() - ap_pool.mean():+.4f})")
        st = [IDX[t] for t in STARVED if t in IDX]
        print(f"  on the 5 STARVED labels only:  pool {ap_pool[st].mean():.4f}"
              f"   pool+{a} {ap_plus[st].mean():.4f} "
              f"({ap_plus[st].mean() - ap_pool[st].mean():+.4f})")
        print(f"  labels where it EARNS its place (CI > 0): {len(earned)}"
              f"  -> {[FACTOR_TAXONOMY[j] for j in earned]}\n")

    print("=" * 104)
    print("DECISION")
    print("=" * 104)
    print("An extra arm earns a label only where pool+arm beats pool with the "
          "CI clear of zero.")
    print("Adding an arm for SOME labels and not others is a per-label source "
          "choice with ZERO")
    print("fitted parameters -- the only category that has transferred above "
          "its estimate here.")
    print("If nothing is earned, especially on the five starved labels, the "
          "July closure stands")
    print("and it stands for the right reason this time.")


if __name__ == "__main__":
    main()

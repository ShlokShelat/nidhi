"""
risk_cascade_decode.py

THE FOUR RISK CLASSES ARE NOT FOUR CATEGORIES. THEY ARE THREE NESTED QUESTIONS.

Read the organizers' definitions:

    indicator   no explicit suicide mention
    ideation    explicit expression, no plan
    behavior    explicit + plan or self-harm
    attempt     a past or recent attempt

Each class adds ONE condition to the previous. So the label is really the answer
to three yes/no questions asked in order:

    q1  any explicit suicidal content?     611 vs 1024   balanced
    q2  plan or self-harm present?         519 vs  505   almost exactly even
    q3  was there an actual attempt?       391 vs  114   22.6% positive

Compare that to the 4-way softmax, where `attempt` is 7% of the data. The
cascade turns one badly-imbalanced problem into three well-balanced ones, and q2
lands exactly on the ideation<->behavior boundary that the disagreement matrix
shows is the largest confusion (29/37 and 22/16 on the two directions).

THIS SCRIPT IS THE CHEAP HALF. The real idea is to TRAIN three binary
classifiers, which needs a GPU. But the three questions can be derived exactly
from the 4-way probabilities we already have:

    q1 = p_ide + p_beh + p_att
    q2 = (p_beh + p_att) / (p_ide + p_beh + p_att)
    q3 = p_att / (p_beh + p_att)

so the DECODE half is free. If walking the three questions beats argmax on
probabilities already available, the training version is worth 12 GPU-hours.
If it does not, the sequential structure alone buys nothing and only
better-balanced TRAINING could -- which is a different and weaker bet.

The same ladder logic (cheap decode-only check before a costly retrain) was
used to save GPU-hours on the factor fine-tune.

NOT THE SAME AS THE ORDINAL HEAD, which was found to be harmful. An ordinal
head forces a single monotone score through all four classes. A cascade makes
three independent cuts, each free to sit wherever it likes.

DECODES COMPARED, all scored as weighted F1 with a paired AUTHOR bootstrap:
    argmax      the deployed decode
    cascade50   the three questions at 0.5 -- ZERO fitted parameters
    cascade_fit thresholds tuned for wF1 -- 3 parameters, cross-fitted so they
                pay their own estimation cost
    argmax_mult per-class multipliers, for reference: this was already found
                harmful out of sample, and it is the closest relative of
                cascade_fit. If cascade_fit merely matches it, the sequential
                structure is adding nothing beyond reparameterisation.

Usage:
  python src/risk_cascade_decode.py
  python src/risk_cascade_decode.py --enc artifacts/lora_ql70_oof.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

PC = [f"prob_{c}" for c in RISK_LEVELS]
GRID = np.round(np.arange(0.10, 0.91, 0.02), 2)


def cascade_probs(P):
    """q1 = not indicator, q2 = plan | not indicator, q3 = attempt | plan."""
    p_ind, p_ide, p_beh, p_att = (P[:, i] for i in range(4))
    q1 = p_ide + p_beh + p_att
    q2 = (p_beh + p_att) / np.maximum(q1, 1e-9)
    q3 = p_att / np.maximum(p_beh + p_att, 1e-9)
    return q1, q2, q3


def cascade_pred(P, t):
    q1, q2, q3 = cascade_probs(P)
    out = np.empty(len(P), dtype=object)
    out[:] = "indicator"
    m1 = q1 >= t[0]
    out[m1 & (q2 < t[1])] = "ideation"
    m2 = m1 & (q2 >= t[1])
    out[m2 & (q3 < t[2])] = "behavior"
    out[m2 & (q3 >= t[2])] = "attempt"
    return out


def fit_cascade(P, gold):
    """Tune the three cuts IN ORDER -- q1 first, since q2 and q3 only apply to
    the posts q1 lets through. Tuning them jointly would search a space the
    cascade cannot actually reach."""
    t = [0.5, 0.5, 0.5]
    for _ in range(2):
        for i in range(3):
            best, bt = -1.0, t[i]
            for v in GRID:
                c = list(t)
                c[i] = float(v)
                s = risk_weighted_f1(list(gold), list(cascade_pred(P, c)))
                if s > best:
                    best, bt = s, float(v)
            t[i] = bt
    return t


def fit_mult(P, gold):
    w = np.ones(4)
    best = risk_weighted_f1(
        list(gold), [RISK_LEVELS[i] for i in P.argmax(1)])
    for _ in range(2):
        for c in range(4):
            for m in np.arange(0.5, 2.01, 0.05):
                t = w.copy()
                t[c] = m
                s = risk_weighted_f1(
                    list(gold),
                    [RISK_LEVELS[i] for i in (P * t[None, :]).argmax(1)])
                if s > best:
                    best, w = s, t
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--splits", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    d = pd.read_csv(args.enc, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d.drop_duplicates("row_id").set_index("row_id")
    if not all(c in d.columns for c in PC):
        raise SystemExit(f"{args.enc} lacks prob_* columns")

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in d.index if r in lab.index]
    P = d.loc[rows, PC].to_numpy(float)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    n = len(rows)

    print(f"{args.enc}: {n} posts / {pd.unique(auth).size} authors")
    gi = np.array([RISK_LEVELS.index(g) for g in gold])
    print(f"\nthe three questions, on gold:")
    print(f"  q1 any explicit content   "
          f"{int((gi > 0).sum())} yes / {int((gi == 0).sum())} no")
    print(f"  q2 plan or self-harm      "
          f"{int((gi >= 2).sum())} yes / {int((gi == 1).sum())} no"
          f"   (of the {int((gi > 0).sum())} that pass q1)")
    print(f"  q3 actual attempt         "
          f"{int((gi == 3).sum())} yes / {int((gi == 2).sum())} no"
          f"   (of the {int((gi >= 2).sum())} that pass q2)")

    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([by[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]

    am = np.array([RISK_LEVELS[i] for i in P.argmax(1)])
    c50 = cascade_pred(P, [0.5, 0.5, 0.5])
    tfit = fit_cascade(P, gold)
    cfit = cascade_pred(P, tfit)
    wfit = fit_mult(P, gold)
    mfit = np.array([RISK_LEVELS[i] for i in (P * wfit[None, :]).argmax(1)])

    base = risk_weighted_f1(list(gold), list(am))
    print(f"\ncascade thresholds fitted in-sample: "
          f"q1={tfit[0]:.2f} q2={tfit[1]:.2f} q3={tfit[2]:.2f}")
    print("\n" + "=" * 94)
    print("IN-SAMPLE weighted-F1 (optimistic -- the cross-fit below is the one)")
    print("=" * 94)
    print(f"{'decode':<13}{'params':>8}{'wF1':>9}{'d vs argmax':>13}"
          f"{'95% CI':>20}{'win':>7}")
    for name, pr, npar in (("argmax", am, 0), ("cascade50", c50, 0),
                           ("cascade_fit", cfit, 3), ("argmax_mult", mfit, 3)):
        v = risk_weighted_f1(list(gold), list(pr))
        if name == "argmax":
            print(f"{name:<13}{npar:>8}{v:>9.4f}{'--':>13}{'--':>20}{'--':>7}")
            continue
        dd = np.array([risk_weighted_f1(list(gold[ix]), list(pr[ix]))
                       - risk_weighted_f1(list(gold[ix]), list(am[ix]))
                       for ix in IXS])
        lo, hi = np.percentile(dd, [2.5, 97.5])
        print(f"{name:<13}{npar:>8}{v:>9.4f}{v-base:>+13.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{(dd > 0).mean():>7.0%}")

    print("\n" + "=" * 94)
    print("CROSS-FIT -- thresholds from FIT authors, scored on ~378 held-out")
    print("=" * 94)
    rng2 = np.random.default_rng(args.seed + 7)
    eval_frac = 378.0 / (378.0 + n)
    keep = {"cascade50": [], "cascade_fit": [], "argmax_mult": []}
    for _ in range(args.splits):
        order = list(uauth)
        rng2.shuffle(order)
        ev, got = [], 0
        for a in order:
            if got >= eval_frac * n:
                break
            ev.append(a)
            got += len(by[a])
        E = np.concatenate([by[a] for a in ev])
        F = np.setdiff1d(np.arange(n), E)
        b = risk_weighted_f1(list(gold[E]),
                             [RISK_LEVELS[i] for i in P[E].argmax(1)])
        keep["cascade50"].append(
            risk_weighted_f1(list(gold[E]),
                             list(cascade_pred(P[E], [0.5, 0.5, 0.5]))) - b)
        t = fit_cascade(P[F], gold[F])
        keep["cascade_fit"].append(
            risk_weighted_f1(list(gold[E]), list(cascade_pred(P[E], t))) - b)
        w = fit_mult(P[F], gold[F])
        keep["argmax_mult"].append(
            risk_weighted_f1(
                list(gold[E]),
                [RISK_LEVELS[i] for i in (P[E] * w[None, :]).argmax(1)]) - b)
    for k in ("cascade50", "cascade_fit", "argmax_mult"):
        v = np.array(keep[k])
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f"  {k:<13}{v.mean():>+9.4f}  "
              f"[{v.mean()-1.96*se:+.4f},{v.mean()+1.96*se:+.4f}]  "
              f"win {(v > 0).mean():>4.0%}   composite {0.4*v.mean():+.4f}")

    print("\n" + "=" * 94)
    print("HOW TO READ IT")
    print("=" * 94)
    print("`argmax_mult` is the control, and cascade_fit's closest relative -- both")
    print("re-cut the same probabilities with 3 numbers. If cascade_fit only")
    print("matches it, the SEQUENTIAL STRUCTURE adds nothing on top of")
    print("reparameterisation, and the decode half of this idea is dead.")
    print("")
    print("`cascade50` is the one to hope for: zero fitted parameters, and the")
    print("only category that has ever transferred above its estimate here.")
    print("")
    print("EITHER WAY this does not settle the TRAINING version. Three binary")
    print("classifiers see balanced problems (611/1024, 519/505, 391/114)")
    print("instead of one 4-way with attempt at 7%, and that changes the")
    print("PROBABILITIES, which no decode can. A negative here only means the")
    print("free half is worthless -- it makes the 12-GPU-hour version a weaker")
    print("bet, not a dead one.")


if __name__ == "__main__":
    main()

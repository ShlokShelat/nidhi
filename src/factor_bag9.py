"""
factor_bag9.py

Is bag8 + fenum better than bag8, on the scorer's own terms?

WHY fenum AND NOTHING ELSE. Seven unused arms were re-tested per label on AP
with a paired author bootstrap. Six of them scatter, mixing wins and harms
with every mean AP negative. `fenum` alone comes out positive and at a high
win rate on substance use, psychological capital, coping strategy, and
sense of responsibility; meaning in life is a near miss; social support is
the one label it harms. Mean AP improves overall.

FOUR OF THE FIVE PROTECTIVE LABELS IMPROVE, addressing the persistent
protective-factor gap, and `fenum` was already flagged as the strongest arm
in per-fold screening, pending an honest measurement. This is it.

THE MULTIPLE-COMPARISON HONESTY. 7 arms x 24 labels is 168 tests; chance alone
gives ~4 "CI clear of zero" at 95%, and 5 were observed. So NO INDIVIDUAL LABEL RESULT
IS SAFE. What survives is the ARM-level pattern -- a noise arm scatters and
lands negative, and six did exactly that. That is why this script tests the
whole-bag change and NOT a per-label cherry-pick: the evidence is about fenum,
not about substance use.

AND AP IS NOT THE SCORER. Everything above is ranking quality. The scorer is
macro-F1 after a quantile decode at fixed per-label counts, so a ranking gain
can vanish entirely. This measures the thing that actually scores.

WHAT IT REPORTS
  bag8   the deployed pool                       (reproduces the shipped column)
  bag9   bag8 + fenum, all 24 labels             (0 parameters -- the candidate)
  sel    bag8, with bag9 used ONLY on the labels fenum earned. Printed for
         contrast and flagged as multiple-comparison-tainted; prefer bag9.
all decoded at the CHAMPION'S per-label rates, paired author bootstrap.

CALIBRATE YOUR EXPECTATIONS. Offline predictions for this kind of change tend
to overstate the gain a submission actually delivers. Apply the same discount
here before deciding a submission slot is worth it.

Usage:
  python src/factor_bag9.py
  python src/factor_bag9.py --build-test artifacts/bag9_factors.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
PROTECTIVE = ["social support", "coping strategy", "psychological capital",
              "sense of responsibility", "meaning in life"]
PROT_J = [IDX[t] for t in PROTECTIVE]
EARNED = ["substance use", "psychological capital", "coping strategy"]


def load_arm(d, a, k):
    pr, ids = [], []
    for f in range(k):
        p = f"{d}/{a}_cv_oof_f{f}.npz"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        z = np.load(p, allow_pickle=True)
        pr.append(z["probs"].astype(float))
        ids.extend(str(x) for x in z["row_id"])
    return np.concatenate(pr), ids


def align(pr, ids, rid):
    if ids == rid:
        return pr
    o = {x: i for i, x in enumerate(ids)}
    return pr[[o[r] for r in rid]]


def resolve_test(d, a):
    for c in (f"{d}/{a}_cv_test.npz", f"{d}/{a}_test.npz",
              f"{d}/nli_ensemble_{a}_test.npz"):
        if os.path.exists(c):
            return c
    return None


def logit_mean(S):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, cell in enumerate(col):
        for x in set(parse_factors(str(cell))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def topk(P, rates):
    n = len(P)
    on = np.zeros((n, N), dtype=np.int8)
    for j in range(N):
        k = min(max(int(round(rates[j] * n)), 1), n - 1)
        cut = np.sort(P[:, j])[-k]
        on[:, j] = (P[:, j] >= cut).astype(np.int8)
    return on


def f1pl(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--add", nargs="+", default=["fenum"],
                    help="one or more arms to ADD as extra members. 17 "
                         "candidates have now failed this test, and the ceiling "
                         "check showed additions LOWER the achievable macro. "
                         "Prefer --replace for a retrained variant of an arm "
                         "already in the bag.")
    ap.add_argument("--replace", nargs="+", metavar="OLD NEW", default=None,
                    help="swap an arm ALREADY IN the bag for an improved "
                         "version of itself, keeping the bag at 8 members. "
                         "THIS IS THE RIGHT TEST for a retrained variant "
                         "(mnli -> fenumb, mnli -> ml512): an ADDITION pays a "
                         "fixed per-arm dilution tax that every add-candidate "
                         "so far has failed to overcome, while a replacement "
                         "pays none. Overrides --add.")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--build-test", default=None,
                    help="write the bag9 factor column for the 378 test posts")
    args = ap.parse_args()

    base, rid = [], None
    for a in args.arms:
        pr, ids = load_arm(args.dir, a, args.k)
        if rid is None:
            rid = ids
        base.append(align(pr, ids, rid))
    S8 = np.stack(base)
    P8 = logit_mean(S8)
    n = len(rid)

    if args.replace:
        if len(args.replace) % 2:
            raise SystemExit("--replace takes OLD NEW pairs, got an odd count")
        pairs = list(zip(args.replace[::2], args.replace[1::2]))
        seen = set()
        for old, _ in pairs:
            if old not in args.arms:
                raise SystemExit(
                    f"--replace: {old} is not in the bag {args.arms}")
            if old in seen:
                raise SystemExit(
                    f"--replace: {old} targeted twice; each slot takes one arm")
            seen.add(old)
        S9 = S8.copy()
        arms9 = list(args.arms)
        for old, new in pairs:
            pr, ids = load_arm(args.dir, new, args.k)
            j = list(args.arms).index(old)
            S9[j] = align(pr, ids, rid)
            arms9[j] = new
        P9 = logit_mean(S9)
        what = ("bag8 with " + ", ".join(f"{o} -> {n}" for o, n in pairs) +
                f"  (still {len(args.arms)} members, NO dilution)")
    else:
        extra = []
        for a in args.add:
            pr, ids = load_arm(args.dir, a, args.k)
            extra.append(align(pr, ids, rid))
        P9 = logit_mean(np.concatenate([S8, np.stack(extra)], axis=0))
        arms9 = list(args.arms) + list(args.add)
        what = f"bag8 + {'+'.join(args.add)}  (dilution tax applies)"

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])

    ch = pd.read_csv(args.champion, keep_default_na=False)
    champ = cells(ch["factors"], len(ch))
    rates = champ.sum(0) / len(ch)

    Psel = P8.copy()
    for t in EARNED:
        Psel[:, IDX[t]] = P9[:, IDX[t]]

    pred = {"bag8": topk(P8, rates), "bag9": topk(P9, rates),
            "sel": topk(Psel, rates)}
    per = {k: f1pl(v, Y) for k, v in pred.items()}

    print(f"OOF {n} posts / {pd.unique(auth).size} authors")
    print(f"bag8 {args.arms}")
    print(f"bag9 = {what}\n")

    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([by[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]

    print(f"{'candidate':<8}{'macro':>9}{'protective':>12}{'risk':>8}"
          f"{'d vs bag8':>11}{'95% CI':>20}{'win':>7}")
    for k in ("bag8", "bag9", "sel"):
        p = per[k]
        risk_j = [j for j in range(N) if j not in PROT_J]
        if k == "bag8":
            print(f"{k:<8}{p.mean():>9.4f}{p[PROT_J].mean():>12.4f}"
                  f"{p[risk_j].mean():>8.4f}{'--':>11}{'--':>20}{'--':>7}")
            continue
        d = np.array([f1pl(pred[k][ix], Y[ix]).mean()
                      - f1pl(pred["bag8"][ix], Y[ix]).mean() for ix in IXS])
        lo, hi = np.percentile(d, [2.5, 97.5])
        note = "   <- multiple-comparison tainted" if k == "sel" else ""
        print(f"{k:<8}{p.mean():>9.4f}{p[PROT_J].mean():>12.4f}"
              f"{p[risk_j].mean():>8.4f}{p.mean() - per['bag8'].mean():>+11.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{(d > 0).mean():>7.0%}{note}")

    print(f"\ncomposite = 0.3 x the macro delta")
    print(f"\n{'label':<44}{'bag8':>8}{'bag9':>8}{'delta':>8}")
    d9 = per["bag9"] - per["bag8"]
    for j in np.argsort(-d9):
        if abs(d9[j]) < 0.002:
            continue
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{per['bag8'][j]:>8.3f}"
              f"{per['bag9'][j]:>8.3f}{d9[j]:>+8.3f}")
    print("  * = protective")

    # ---- test side ---------------------------------------------------------
    if args.build_test:
        T, tid = [], None
        for a in list(args.arms) + [a for a in arms9 if a not in args.arms]:
            p = resolve_test(args.dir, a)
            if p is None:
                raise SystemExit(f"no test dump for {a}")
            z = np.load(p, allow_pickle=True)
            T.append(z["probs"].astype(float))
            tid = [str(x) for x in z["row_id"]]
        T = np.stack(T)
        nT = len(tid)
        loaded = list(args.arms) + [a for a in arms9 if a not in args.arms]
        # bag8 is always the FIRST len(args.arms) rows, whatever mode we are in
        P8t = logit_mean(T[:len(args.arms)])
        # bag9 selects by NAME, so --replace swaps in place instead of appending
        P9t = logit_mean(np.stack([T[loaded.index(a)] for a in arms9]))
        on8 = topk(P8t, rates)
        on9 = topk(P9t, rates)

        order = {r: i for i, r in enumerate(ch["row_id"].astype(str))}
        perm = [order[r] for r in tid]
        repro = int((on8 != champ[perm]).sum())
        print(f"\nPROVENANCE: bag8 reproduction vs the shipped column -- "
              f"{repro} of {nT * N} cells differ")
        if repro:
            raise SystemExit(
                "REFUSING to write. The bag8 reconstruction does not match the "
                "shipped factor column, so the bag9 diff would mix this bug "
                "with the actual change -- that is exactly how bag3_ev lost "
                "0.0169. Fix the arm list first.")

        out = ch.copy()
        newcol = list(out["factors"])
        moved = 0
        for i, row in enumerate(tid):
            labs = str([FACTOR_TAXONOMY[j] for j in range(N) if on9[i, j]])
            k = order[row]
            if str(newcol[k]) != labs:
                moved += 1
            newcol[k] = labs
        out["factors"] = newcol
        os.makedirs(os.path.dirname(args.build_test) or ".", exist_ok=True)
        out.to_csv(args.build_test, index=False)
        print(f"cells changed vs shipped: {int((on9 != on8).sum())}   "
              f"posts changed: {moved}")
        print(f"factors/post {on9.sum() / nT:.2f} (shipped "
              f"{champ.sum() / len(ch):.2f})")
        print(f"wrote {args.build_test}")
        print("\nThis file changes ONLY the factors column. Risk and evidence "
              "are copied from the")
        print("champion untouched, so S1 must come back bit-identical -- if it "
              "does not, something")
        print("else moved and the result is not interpretable.")

    print("\nDECIDE ON bag9, NOT sel. The evidence is that fenum is a good arm, "
          "not that any")
    print("single label is special -- 168 tests produce ~4 spurious per-label "
          "wins by chance.")
    print("And discount hard: offline predictions for this kind of change "
          "tend to overstate the gain a submission actually delivers.")


if __name__ == "__main__":
    main()

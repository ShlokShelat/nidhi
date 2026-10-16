"""
factor_prevalence_decode.py

THE DECODE WAS CALIBRATED AGAINST ITS OWN OUTPUT.

`champ_rate = champion's predicted counts / 378`. That is circular: the fire
rate used to calibrate each label was set by how often the model already
fired it. Against the data:

    champion predictions   3.44 factors/post
    gold                    2.92 factors/post        -- 18% over-fire

And the F1-optimal rate sits next to GOLD PREVALENCE, not next to ours:

    label                    champ%   best%   gold%
    substance use              5.6     2.2     2.0
    traumatic experience       7.4     3.6     3.9
    poor school performance    2.4     0.6     1.0
    stressful life event      16.4     8.4    11.1

On a label with 16 golds in 1635, firing at 2.4% instead of 1.0% caps precision
near 40% no matter how good the ranking is. Rare labels are exactly where
score is lost, and under MACRO-F1 each of them costs as much as
`hopelessness`.

WHY THIS IS NOT THE RATE-RELEASE LANE AGAIN. That tested champion rates against
rates FITTED on the data -- 24 fitted parameters, and that many fitted
parameters transfer negatively as a rule. Gold prevalence is not fitted. It is measured on
1635 labelled posts, and the scorer decomposition already established that
TEST FACTOR PREVALENCE MATCHES TRAIN. So this is a ZERO-PARAMETER change, the
only category that has ever transferred above its estimate here
(route_attempt, evcond, bag8).

WHY IT MATTERS. The factor component (S2) is the weakest of the two halves of
the composite, and weak by a wider margin than the risk-and-evidence half (S1).
Under the 0.7*S1 + 0.3*S2 split, S1 is already close to the range its
components can reach, so effort spent on S2 buys more composite per unit of
improvement than the same effort spent on S1.

WHAT IT COMPARES, all decoded as quantile top-k, all scored as macro-F1 with a
paired author bootstrap over 153 authors:
    champ   the shipped rates                          (current, circular)
    gold    train gold prevalence                      (0 parameters, the candidate)
    shrunk  halfway between, for the case where test prevalence drifts
    fitted  per-label F1-optimal, fitted in-sample     (ceiling, NEVER ship)

Also reports a per-post floor: posts predicted with ZERO factors. Gold posts
almost always carry at least one, so any post we empty out is a guaranteed miss
on every label it should have had.

Usage:
  python src/factor_prevalence_decode.py
  python src/factor_prevalence_decode.py --build-test artifacts/factors_prev_test.csv
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


def load_oof(d, arms, k):
    O, rid = [], None
    for a in arms:
        pr, ids = [], []
        for f in range(k):
            p = f"{d}/{a}_cv_oof_f{f}.npz"
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            z = np.load(p, allow_pickle=True)
            pr.append(z["probs"].astype(float))
            ids.extend(str(x) for x in z["row_id"])
        pr = np.concatenate(pr)
        if rid is None:
            rid = ids
        elif ids != rid:
            o = {x: i for i, x in enumerate(ids)}
            pr = pr[[o[x] for x in rid]]
        O.append(pr)
    return rid, np.stack(O)


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


def fitted_rates(P, Y):
    grid = np.round(np.arange(0.002, 0.60, 0.002), 3)
    n = len(P)
    out = np.zeros(N)
    for j in range(N):
        best, br = -1.0, Y[:, j].mean()
        for r in grid:
            k = max(1, int(round(r * n)))
            if k >= n:
                continue
            cut = np.sort(P[:, j])[-k]
            pred = P[:, j] >= cut
            tp = int((pred & (Y[:, j] == 1)).sum())
            f = 0.0 if tp == 0 else 2.0 * tp / (int(pred.sum()) + int(Y[:, j].sum()))
            if f > best:
                best, br = f, r
        out[j] = br
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--build-test", default=None)
    ap.add_argument("--use", default="gold", choices=["gold", "shrunk"],
                    help="which rate set --build-test writes")
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])

    ch = pd.read_csv(args.champion, keep_default_na=False)
    champ = cells(ch["factors"], len(ch))

    R = {
        "champ": champ.sum(0) / len(ch),
        "gold": Y.sum(0) / n,
    }
    R["shrunk"] = 0.5 * R["champ"] + 0.5 * R["gold"]
    R["fitted"] = fitted_rates(P, Y)

    print(f"OOF {n} posts / {pd.unique(auth).size} authors")
    print(f"factors/post   champ {R['champ'].sum():.2f}   "
          f"gold {R['gold'].sum():.2f}   shrunk {R['shrunk'].sum():.2f}   "
          f"fitted {R['fitted'].sum():.2f}")

    pred = {k: topk(P, v) for k, v in R.items()}
    per = {k: f1pl(v, Y) for k, v in pred.items()}

    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([by[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]

    print("\n" + "=" * 100)
    print("DECODE RATE SOURCE -- macro-F1, paired author bootstrap")
    print("=" * 100)
    print(f"{'rates':<9}{'params':>8}{'macro':>9}{'protective':>12}"
          f"{'d vs champ':>12}{'95% CI':>20}{'win':>7}{'0-fac posts':>13}")
    params = {"champ": 0, "gold": 0, "shrunk": 0, "fitted": 24}
    for k in ("champ", "gold", "shrunk", "fitted"):
        p = per[k]
        zero = int((pred[k].sum(1) == 0).sum())
        if k == "champ":
            print(f"{k:<9}{params[k]:>8}{p.mean():>9.4f}"
                  f"{p[PROT_J].mean():>12.4f}{'--':>12}{'--':>20}{'--':>7}"
                  f"{zero:>13}")
            continue
        d = np.array([f1pl(pred[k][ix], Y[ix]).mean()
                      - f1pl(pred["champ"][ix], Y[ix]).mean() for ix in IXS])
        lo, hi = np.percentile(d, [2.5, 97.5])
        note = "  CEILING, never ship" if k == "fitted" else ""
        print(f"{k:<9}{params[k]:>8}{p.mean():>9.4f}{p[PROT_J].mean():>12.4f}"
              f"{p.mean() - per['champ'].mean():>+12.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{(d > 0).mean():>7.0%}"
              f"{zero:>13}{note}")
    print("\n  composite = 0.3 x the macro delta")
    print("  gold prevalence is MEASURED on 1635 posts, not fitted -- and the "
          "scorer")
    print("  decomposition established that test factor prevalence matches "
          "train.")

    print("\n" + "=" * 100)
    print("PER LABEL -- champ vs gold prevalence")
    print("=" * 100)
    d = per["gold"] - per["champ"]
    print(f"{'label':<44}{'gold n':>7}{'champ%':>8}{'gold%':>7}{'fit%':>7}"
          f"{'F1 champ':>10}{'F1 gold':>9}{'delta':>8}")
    for j in np.argsort(-d):
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{int(Y[:, j].sum()):>7}"
              f"{100 * R['champ'][j]:>7.1f}%{100 * R['gold'][j]:>6.1f}%"
              f"{100 * R['fitted'][j]:>6.1f}%{per['champ'][j]:>10.3f}"
              f"{per['gold'][j]:>9.3f}{d[j]:>+8.3f}")
    print("  * = protective")

    # ---------------- deployment-faithful cross-fit -------------------------
    # The table above computes gold prevalence on the SAME posts it scores. In
    # deployment we estimate it from 1635 train posts and apply it to 378 test
    # posts whose true prevalence differs by sampling noise -- for a 2% label
    # that is 7.6 +/- 2.7 expected positives, a ~35% swing on exactly the rare
    # labels carrying the gain. This reproduces that situation: prevalence from
    # the fit authors, decode and score the held-out ~378.
    print("\n" + "=" * 100)
    print("DEPLOYMENT-FAITHFUL CROSS-FIT -- prevalence from FIT authors, "
          "scored on ~378 held-out posts")
    print("=" * 100)
    rng2 = np.random.default_rng(args.seed + 7)
    eval_frac = len(ch) / (len(ch) + n)
    keep = {"gold": [], "shrunk": []}
    for _ in range(args.splits):
        order = list(uauth)
        rng2.shuffle(order)
        ev, got = [], 0
        target = eval_frac * n
        for a in order:
            if got >= target:
                break
            ev.append(a)
            got += len(by[a])
        E = np.concatenate([by[a] for a in ev])
        F = np.setdiff1d(np.arange(n), E)
        nE = len(E)
        r_gold = Y[F].sum(0) / len(F)
        r_shr = 0.5 * R["champ"] + 0.5 * r_gold
        b = f1pl(topk(P[E], R["champ"]), Y[E]).mean()
        keep["gold"].append(f1pl(topk(P[E], r_gold), Y[E]).mean() - b)
        keep["shrunk"].append(f1pl(topk(P[E], r_shr), Y[E]).mean() - b)
    for k in ("gold", "shrunk"):
        v = np.array(keep[k])
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f"  {k:<8}{v.mean():>+9.4f}  "
              f"[{v.mean() - 1.96 * se:+.4f},{v.mean() + 1.96 * se:+.4f}]  "
              f"win {(v > 0).mean():>4.0%}   composite {0.3 * v.mean():+.4f}")
    print(f"  ({args.splits} author partitions, eval {eval_frac:.1%} of posts "
          f"-- the real 1635-fit / 378-test ratio)")
    print("\n  THIS is the number to believe, not the in-sample table above.")

    # ---------------- test side --------------------------------------------
    if args.build_test:
        T, tid = [], None
        for a in args.arms:
            p = resolve_test(args.dir, a)
            if p is None:
                raise SystemExit(f"no test dump for {a}")
            z = np.load(p, allow_pickle=True)
            T.append(z["probs"].astype(float))
            tid = [str(x) for x in z["row_id"]]
        Pt = logit_mean(np.stack(T))
        nT = len(tid)
        order = {r: i for i, r in enumerate(ch["row_id"].astype(str))}
        perm = [order[r] for r in tid]

        repro = int((topk(Pt, R["champ"]) != champ[perm]).sum())
        print(f"\nPROVENANCE: champ-rate reconstruction vs the shipped column "
              f"-- {repro} of {nT * N} cells differ")
        if repro:
            raise SystemExit(
                "REFUSING to write. The reconstruction does not match the "
                "shipped factor column, so the new column would mix this bug "
                "with the intended change and silently corrupt the result.")

        on = topk(Pt, R[args.use])
        out = ch.copy()
        newcol = list(out["factors"])
        moved = 0
        for i, row in enumerate(tid):
            labs = str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
            k = order[row]
            if str(newcol[k]) != labs:
                moved += 1
            newcol[k] = labs
        out["factors"] = newcol
        os.makedirs(os.path.dirname(args.build_test) or ".", exist_ok=True)
        out.to_csv(args.build_test, index=False)
        print(f"rates used: {args.use}   {on.sum() / nT:.2f} factors/post "
              f"(shipped {champ.sum() / len(ch):.2f})")
        print(f"cells changed {int((on != champ[perm]).sum())}   "
              f"posts changed {moved}   "
              f"posts with 0 factors {int((on.sum(1) == 0).sum())}")
        print(f"wrote {args.build_test}")
        print("\nONLY the factors column changes. Risk and evidence are copied "
              "from the champion,")
        print("so S1 must come back BIT-IDENTICAL at 0.8061. If it moves, "
              "something else did too")
        print("and the result cannot be attributed.")


if __name__ == "__main__":
    main()

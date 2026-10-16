"""
factor_pool_diagnostic.py

IS THE UNWEIGHTED LOGIT-MEAN ACTUALLY GIVING EACH ARM AN EQUAL VOTE?

The champion pools 11 arms with `logit_mean` -- the arithmetic mean of
log-odds. Everyone has read that as "equal weight per arm". IT IS NOT. A mean
of logits gives each arm influence in proportion to the SPREAD of its logits.
An arm whose probabilities live in [0.2, 0.6] contributes a logit range of
about 1.8; an arm that saturates to 0.001/0.999 contributes about 14. The
second arm outvotes the first ~8:1 on every post, and no threshold can undo it
because the pooled ranking is already set.

WHY THIS IS NOT THE ARM-WEIGHT LANE. `factor_arm_weights.py` asked "should
some arm count MORE?" and answered no. That is a question about arm QUALITY,
and both of its schemes were scalars chosen against the scored data (solo
macro, or coordinate ascent on gold). This asks a different question: are the
nominal weights already equal to the EFFECTIVE weights? The fix, if there is
one, uses no gold at all -- each arm's logits are standardised by its own
dispersion, so it is a 0-fitted-parameter reparameterisation, not a fitted
weight.

It is also not rank-averaging. Rank-averaging throws away magnitude entirely;
standardisation keeps magnitude and equalises only scale.

WHAT IT PRINTS

  1. per-arm logit dispersion and saturation -- the raw scale defect, if any
  2. INFLUENCE SHARE: cov(L_arm, L_pool)/var(L_pool), which sums to 1 across
     arms. Equal influence is 1/n_arms. This is the number the whole script
     exists for.
  3. pairwise logit correlation + the effective member count, so a
     near-duplicate pair that the mean double-counts shows up
  4. per-label AP vs F1-at-tfit, cross-fitted by author -- high AP with low F1
     is a decision-rule problem, low AP is a ranking problem
  5. `zpool`, the standardised-logit pool, scored against the shipped pool on
     the same held-out authors

PRE-REGISTERED READS, so this cannot be talked into a positive afterwards:

  * If every arm's influence share is inside 1/n +- 20%, the pooling rule is
    already doing what it claims and section 5 should be a coin flip. Say so
    and close the lane.
  * `zpool` ships only at >= 90% win over >= 200 author splits, judged on the
    win rate and per-split sd, never the CI of the mean -- a clean confidence
    interval is not sufficient, since a positive mean can still come from a
    few large splits rather than a consistent effect.

Usage:
  python src/factor_pool_diagnostic.py
  python src/factor_pool_diagnostic.py --splits 400
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
SHIPPED11 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli",
             "ltail", "cbnsynapi", "friendfac"]
PROTECTIVE = {"social support", "coping strategy", "psychological capital",
              "sense of responsibility", "meaning in life"}
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)
EPS = 1e-6


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


def load_test(d, arms):
    """Same precedence as factor_threshold_stabilise.resolve_test, so this
    reports on the dumps the champion is actually built from."""
    T, tid, src = [], None, {}
    for a in arms:
        p = resolve_test(d, a)
        if p is None:
            return None, None, None
        src[a] = os.path.basename(p)
        z = np.load(p, allow_pickle=True)
        ids = [str(x) for x in z["row_id"]]
        pr = z["probs"].astype(float)
        if tid is None:
            tid = ids
        elif ids != tid:
            if sorted(ids) != sorted(tid):
                return None, None, None
            o = {r: i for i, r in enumerate(ids)}
            pr = pr[[o[r] for r in tid]]
        T.append(pr)
    return tid, np.stack(T), src


def lg(P):
    X = np.clip(np.asarray(P, dtype=float), EPS, 1 - EPS)
    return np.log(X / (1 - X))


def sig(L):
    return 1.0 / (1.0 + np.exp(-L))


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, cell in enumerate(col):
        for x in set(parse_factors(str(cell))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def f1_curve(p, y):
    npos = int(y.sum())
    if npos == 0:
        return np.zeros(len(TGRID))
    o = np.argsort(-p, kind="stable")
    ys, ps = y[o], p[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    return 2.0 * tp / np.maximum(ks + npos, 1)


def thr_argmax(p, y):
    """`tfit` -- exactly what ships."""
    return float(TGRID[int(np.argmax(f1_curve(p, y)))])


def f1_per_label(P, Y, th):
    Pb = P >= th[None, :]
    tp = (Pb & (Y == 1)).sum(0)
    den = 2 * tp + (Pb & (Y == 0)).sum(0) + (~Pb & (Y == 1)).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def ap_score(p, y):
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--test-dir", default=None)
    a_.add_argument("--arms", nargs="+", default=SHIPPED11)
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    arms = list(args.arms)
    na = len(arms)
    rid, S = load_oof(args.dir, arms, args.k)
    L = lg(S)                                   # (arms, posts, labels)
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    auth = np.array([str(lab.loc[r, acol]) for r in rid])
    ua = np.array(sorted(set(auth)))
    Lp = L.mean(0)
    P = sig(Lp)
    print(f"OOF {n} posts / {len(ua)} authors / {na} arms   equal share = "
          f"{1.0/na:.4f}\n")

    # ---- 1. per-arm logit dispersion --------------------------------------
    print("=" * 100)
    print("1. PER-ARM LOGIT SCALE -- a mean of logits weights by DISPERSION, "
          "not by intent")
    print("=" * 100)
    sdL = np.array([L[a].std() for a in range(na)])
    med = float(np.median(sdL))
    print(f"{'arm':<12}{'mean p':>9}{'mean L':>9}{'sd L':>9}{'p1 L':>9}"
          f"{'p99 L':>9}{'|L|>6':>8}{'clipped':>9}{'sd/median':>11}")
    for a in range(na):
        La = L[a]
        clip = float(np.mean((S[a] <= EPS) | (S[a] >= 1 - EPS)))
        print(f"{arms[a]:<12}{S[a].mean():>9.4f}{La.mean():>9.3f}"
              f"{La.std():>9.3f}{np.percentile(La, 1):>9.2f}"
              f"{np.percentile(La, 99):>9.2f}"
              f"{float(np.mean(np.abs(La) > 6)):>8.1%}{clip:>9.2%}"
              f"{La.std()/max(med, 1e-9):>11.2f}")
    print(f"\n  sd spread across arms: min {sdL.min():.3f} "
          f"max {sdL.max():.3f}  ratio {sdL.max()/max(sdL.min(),1e-9):.2f}x")
    print("  A ratio near 1 means the arms really are on one scale. A ratio "
          "above ~2\n  means the logit-mean is a WEIGHTED mean whose weights "
          "nobody chose.")

    # ---- 2. influence share ------------------------------------------------
    print("\n" + "=" * 100)
    print("2. INFLUENCE SHARE  cov(L_arm, L_pool) / var(L_pool)  -- sums to "
          "1.0 across arms")
    print("=" * 100)
    share = np.zeros((na, N))
    for j in range(N):
        lp = Lp[:, j]
        v = lp.var()
        for a in range(na):
            share[a, j] = (np.cov(L[a][:, j], lp, ddof=0)[0, 1] / v
                           if v > 0 else np.nan)
    ms = np.nanmean(share, axis=1)
    order = np.argsort(-ms)
    print(f"{'arm':<12}{'mean share':>12}{'x equal':>9}{'min lbl':>9}"
          f"{'max lbl':>9}   {'label of max':<38}")
    for a in order:
        jmx = int(np.nanargmax(share[a]))
        print(f"{arms[a]:<12}{ms[a]:>12.4f}{ms[a]*na:>9.2f}"
              f"{np.nanmin(share[a]):>9.3f}{np.nanmax(share[a]):>9.3f}   "
              f"{FACTOR_TAXONOMY[jmx][:37]:<38}")
    lo, hi = ms.min() * na, ms.max() * na
    print(f"\n  effective-vs-nominal weight runs {lo:.2f}x to {hi:.2f}x equal.")
    print("  PRE-REGISTERED: inside 0.8x-1.2x for every arm => the pooling "
          "rule is\n  sound and section 5 must be treated as a coin flip.")

    # ---- 3. correlation / duplication -------------------------------------
    print("\n" + "=" * 100)
    print("3. PAIRWISE LOGIT CORRELATION (mean over the 24 labels)")
    print("=" * 100)
    R = np.eye(na)
    for a in range(na):
        for b in range(a + 1, na):
            r = np.nanmean([np.corrcoef(L[a][:, j], L[b][:, j])[0, 1]
                            for j in range(N)])
            R[a, b] = R[b, a] = r
    print(f"{'':<12}" + "".join(f"{x[:7]:>8}" for x in arms))
    for a in range(na):
        print(f"{arms[a]:<12}" + "".join(f"{R[a, b]:>8.3f}"
                                         for b in range(na)))
    pairs = sorted(((R[a, b], arms[a], arms[b])
                    for a in range(na) for b in range(a + 1, na)),
                   reverse=True)
    print("\n  most duplicated pairs:")
    for r, x, y in pairs[:5]:
        print(f"    {x:<12}{y:<12}{r:>7.3f}")
    print("  least duplicated pairs:")
    for r, x, y in pairs[-3:]:
        print(f"    {x:<12}{y:<12}{r:>7.3f}")
    sdv = np.array([L[a].std() for a in range(na)])
    var_sum = float(np.sum(np.outer(sdv, sdv) * R))
    n_eff = (sdv.sum() ** 2) / max(var_sum, 1e-12)
    print(f"\n  EFFECTIVE MEMBER COUNT {n_eff:.2f} of {na} nominal.")
    print("  (n_eff = (sum sd)^2 / var(sum L); it is n for uncorrelated equal-"
          "sd arms\n  and 1 for identical ones. A pair at rho > 0.97 is one "
          "member voting twice.)")

    # ---- 4. per-label anatomy ---------------------------------------------
    print("\n" + "=" * 100)
    print("4. PER-LABEL: AP vs F1 AT tfit, cross-fitted by author "
          f"({args.splits} splits)")
    print("=" * 100)
    rng = np.random.default_rng(args.seed)
    TP = np.zeros(N)
    FP = np.zeros(N)
    FN = np.zeros(N)
    THS = [[] for _ in range(N)]
    base_macro, z_macro = [], []
    # zpool: per-arm, per-label standardisation of the logits. Uses no gold.
    Z = np.empty_like(L)
    for a in range(na):
        mu = L[a].mean(0, keepdims=True)
        sd = L[a].std(0, keepdims=True)
        Z[a] = (L[a] - mu) / np.maximum(sd, 1e-9)
    Pz = sig(Z.mean(0) * Lp.std(0, keepdims=True) + Lp.mean(0, keepdims=True))

    for _ in range(args.splits):
        perm = rng.permutation(len(ua))
        fa = set(ua[perm[:len(ua) // 2]])
        fm = np.array([a in fa for a in auth])
        if fm.all() or not fm.any():
            continue
        th = np.array([thr_argmax(P[fm, j], Y[fm, j]) for j in range(N)])
        thz = np.array([thr_argmax(Pz[fm, j], Y[fm, j]) for j in range(N)])
        Pb = P[~fm] >= th[None, :]
        Yh = Y[~fm]
        TP += (Pb & (Yh == 1)).sum(0)
        FP += (Pb & (Yh == 0)).sum(0)
        FN += (~Pb & (Yh == 1)).sum(0)
        for j in range(N):
            THS[j].append(th[j])
        base_macro.append(float(f1_per_label(P[~fm], Yh, th).mean()))
        z_macro.append(float(f1_per_label(Pz[~fm], Yh, thz).mean()))

    f1 = np.where(2 * TP + FP + FN == 0, 0.0,
                  2 * TP / np.maximum(2 * TP + FP + FN, 1))
    ap = np.array([ap_score(P[:, j], Y[:, j]) for j in range(N)])
    rows = sorted(range(N), key=lambda j: -(ap[j] - f1[j]))
    print(f"  {'label':<40}{'sup':>5}{'F1':>7}{'AP':>7}{'AP-F1':>8}"
          f"{'prec':>7}{'rec':>7}{'thr':>13}{'fires%':>8}{'gold%':>7}")
    for j in rows:
        pr = TP[j] / max(TP[j] + FP[j], 1)
        rc = TP[j] / max(TP[j] + FN[j], 1)
        firepct = float(np.mean(P[:, j] >= np.mean(THS[j])))
        tag = "*" if FACTOR_TAXONOMY[j] in PROTECTIVE else " "
        print(f" {tag}{FACTOR_TAXONOMY[j][:39]:<40}{int(Y[:, j].sum()):>5}"
              f"{f1[j]:>7.3f}{ap[j]:>7.3f}{ap[j]-f1[j]:>+8.3f}"
              f"{pr:>7.2f}{rc:>7.2f}"
              f"{('%.2f+-%.2f' % (np.mean(THS[j]), np.std(THS[j]))):>13}"
              f"{firepct:>8.1%}{Y[:, j].mean():>7.1%}")
    print(f"\n  macro {f1.mean():.4f}    protective "
          f"{np.mean([f1[IDX[t]] for t in PROTECTIVE]):.4f}    risk "
          f"{np.mean([f1[j] for j in range(N) if FACTOR_TAXONOMY[j] not in PROTECTIVE]):.4f}")
    print("  Sorted by AP-F1. A LARGE gap with a HIGH AP is a decision-rule "
          "failure and is\n  free to fix. A large gap with a LOW AP is just "
          "the arithmetic of a rare label.")

    # ---- 5. zpool ----------------------------------------------------------
    b = np.array(base_macro)
    z = np.array(z_macro)
    d = z - b
    print("\n" + "=" * 100)
    print("5. zpool -- per-arm per-label logit standardisation before the mean")
    print("=" * 100)
    print(f"  shipped logit-mean   {b.mean():.4f}")
    print(f"  zpool                {z.mean():.4f}   d {d.mean():+.4f}   "
          f"win {(d > 0).mean():.0%}   sd {d.std(ddof=1):.4f}   "
          f"composite {0.3*d.mean():+.4f}")
    if d.mean() > 0 and (d > 0).mean() >= 0.90:
        print("  CLEARS the 90% bar. Rebuild via factor_threshold_stabilise.py")
        print("  with the same standardisation applied to the TEST dumps using")
        print("  THEIR OWN mean/sd, prove the unchanged pool still reproduces")
        print("  the champion at 0 of 9072 cells, then hold S1 at 0.8061.")
    else:
        print("  DOES NOT CLEAR. Equal-scale pooling is not the limitation and")
        print("  this lane closes with the rest of the factor pool.")

    # ---- 6. regime shift in influence -------------------------------------
    tdir = args.test_dir or args.dir
    tid, ST, src = load_test(tdir, arms)
    if ST is None:
        print("\n(no complete set of test dumps under "
              f"{tdir}; skipping section 6)")
        return
    LT = lg(ST)
    print("\n" + "=" * 100)
    print("6. DOES EACH ARM CARRY THE SAME INFLUENCE ON TEST AS ON OOF?")
    print("=" * 100)
    print("  The OOF scores every post with ONE fold model; the test dump "
          "AVERAGES the\n  folds. That compresses each arm's logits by an "
          "ARM-SPECIFIC amount, so the\n  effective pooling weights on test "
          "need not be the OOF ones the thresholds\n  were fitted under. "
          "Pooled-level checks (factor_test_calibration.py, sd ratio\n  0.972) "
          "cannot see this because they look at the pool, not the members.")
    LpT = LT.mean(0)
    print(f"\n{'arm':<12}{'sd oof':>9}{'sd test':>9}{'ratio':>8}"
          f"{'share oof':>11}{'share test':>12}{'shift':>8}   {'dump':<34}")
    for a in range(na):
        so, st = L[a].std(), LT[a].std()
        sho = ms[a]
        sht = np.nanmean([np.cov(LT[a][:, j], LpT[:, j], ddof=0)[0, 1]
                          / max(LpT[:, j].var(), 1e-12) for j in range(N)])
        print(f"{arms[a]:<12}{so:>9.3f}{st:>9.3f}{st/max(so,1e-9):>8.3f}"
              f"{sho:>11.4f}{sht:>12.4f}{sht-sho:>+8.4f}   {src[arms[a]]:<34}")
    print("\n  A shift of more than about +-0.02 in share means that arm votes "
          "with a\n  different strength on the posts we score than on the "
          "posts the 24\n  thresholds were fitted on. That is a real "
          "OOF/deployment mismatch and it is\n  invisible to every pooled-"
          "level diagnostic run so far.")


if __name__ == "__main__":
    main()

"""
factor_threshold_stabilise.py

Estimate the 24 per-label decision thresholds of the factor pool, compare four
estimators of them under cross-fitting, and optionally build the factors column
for the test set.

THE QUANTITY BEING ESTIMATED. The pool emits a probability per (post, factor);
the factors column is produced by thresholding each label independently, so
decoding is exactly 24 numbers. Held-out macro-F1 under oracle thresholds is
0.7028 against 0.6774 under fitted ones, so that 0.025 is ESTIMATION error
rather than missing information: the pool already ranks well enough to reach
the higher score if the 24 cut points were known.

WHY THE ESTIMATE IS NOISY. A threshold is chosen by argmax of the
F1-vs-threshold curve on a finite sample. For a label with only eight
positives that curve is jagged and its peak is frequently a single lucky grid
point, so the argmax has high variance even though the underlying curve is
smooth.

FOUR ESTIMATORS OF THE SAME 24 NUMBERS. None adds a parameter; they differ only
in how the same quantity is estimated:

  tfit      argmax of the raw F1-vs-threshold curve on the fit data.
  tbag      mean argmax over B bootstrap resamples of the fit data. Bagging
            reduces the variance of an unstable estimator: the average of many
            noisy argmaxes is steadier than any one of them.
  tsmooth   argmax of the F1 curve after moving-average smoothing, so the peak
            sits where F1 is high across a neighbourhood of thresholds rather
            than at an isolated spike.
  tboth     bagging and smoothing together.

HOW THEY ARE COMPARED. Each estimator sees the SAME fit data and is scored on
the SAME held-out posts, over author partitions at the real 1635-fit /
378-eval ratio, with `tfit` as the reference. Partitions are by AUTHOR because
posts by one author are not independent, and the evaluation set the column is
built for contains authors held out entirely.

ACCEPT RULE: positive mean delta AND win rate >= 80% across splits. The mean
alone is insufficient -- a positive mean driven by a few large splits describes
an estimator that is better on average and worse most of the time, and only one
column is ever delivered.

Usage:
  python src/factor_threshold_stabilise.py
  python src/factor_threshold_stabilise.py --splits 300 --tbag 60
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
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)


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


def _apply_floor(on, Pt, Poof, thr, k):
    """Never let a label emit ZERO predictions when its OOF threshold does fire.

    Macro-F1 averages 24 per-label F1s, so a label that predicts nothing scores
    a hard 0.000 and costs 1/24 = 0.0417 macro = **0.0125 composite** on its
    own, however well the other 23 labels do.

    Three labels currently fire 1-2 times on 378 posts (`sexual orientation`,
    `exposure to others' suicide`, `poor school performance`). Scaled to the
    502-post final set that is 1.3-2.7 expected firings; under Poisson each has
    a 7-27% chance of landing on zero. Expected cost across the three is about
    -0.004 composite with a tail to -0.012.

    The floor lowers such a label's threshold just enough to emit its top-k
    candidates. It only fires when the label DOES fire on OOF -- a label the
    pool genuinely never supports is left alone rather than forced to guess.
    Set --min-predictions 0 to disable.
    """
    if k <= 0:
        return on
    floored = []
    for j in range(on.shape[1]):
        if on[:, j].sum() >= k:
            continue
        if float((Poof[:, j] >= thr[j]).mean()) <= 0:
            continue          # pool never supports it; forcing would be noise
        kk = min(k, on.shape[0])
        cut = float(np.sort(Pt[:, j])[-kk])
        floored.append((FACTOR_TAXONOMY[j], int(on[:, j].sum()),
                        float(thr[j]), cut))
        on[:, j] = (Pt[:, j] >= cut).astype(np.int8)
    if floored:
        print(f"  ZERO-PREDICTION FLOOR fired on {len(floored)} label(s) "
              f"-- each would have scored F1 0.000 outright:")
        for t, had, old, new in floored:
            print(f"    {t:<38} had {had}, threshold {old:.3f} -> {new:.3f}")
    return on


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


def f1pl(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def f1_curve(p, y):
    """F1 at every grid threshold, in one pass."""
    npos = int(y.sum())
    if npos == 0:
        return np.zeros(len(TGRID))
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    ps = p[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    return 2.0 * tp / np.maximum(ks + npos, 1)


def thr_argmax(p, y):
    c = f1_curve(p, y)
    return float(TGRID[int(np.argmax(c))])


def thr_smooth(p, y, w=9):
    """Argmax of a MOVING-AVERAGE-smoothed curve. A raw curve over a handful of
    positives is jagged; its peak is often one lucky grid point."""
    c = f1_curve(p, y)
    if w > 1:
        k = np.ones(w) / w
        c = np.convolve(c, k, mode="same")
    return float(TGRID[int(np.argmax(c))])


def thr_bag(p, y, rng, B, smooth=0):
    """Mean argmax threshold over B bootstrap resamples of the fit data."""
    n = len(p)
    out = []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        yy = y[ix]
        if yy.sum() < 1:
            continue
        out.append(thr_smooth(p[ix], yy, smooth) if smooth
                   else thr_argmax(p[ix], yy))
    return float(np.mean(out)) if out else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--tbag", type=int, default=40)
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--build-test", default=None,
                    help="write the factors column for the 378 test posts")
    # THE DEFAULT IS THE ESTIMATOR THE DELIVERED COLUMN IS BUILT WITH, so an
    # invocation that omits --use reproduces that column rather than quietly
    # decoding with a different estimator. `tfit` selects each threshold by
    # maximising per-label F1 on the out-of-fold predictions.
    ap.add_argument("--use", default="tfit",
                    choices=["tfit", "tsmooth", "tbag", "tboth"])
    ap.add_argument("--test-dir", default=None,
                    help="directory holding the TEST dumps, when they are not "
                         "beside the OOF dumps. --dir always supplies the OOF "
                         "(which comes from real training and never moves); "
                         "this overrides only where {arm}_cv_test.npz is "
                         "looked up. Needed by the cold-run drill, which "
                         "writes fresh test dumps into its own namespace so no "
                         "skip-guard can hand it a stale file.")
    ap.add_argument("--cold-test", default=None,
                    help="COLD-START MODE. Path to a fresh test CSV (needs a "
                         "row_id column). THIS script builds the delivered "
                         "factor column, so this is the path a rerun on unseen "
                         "authors takes. Without it, "
                         "--build-test reads --shipped for row order, for the "
                         "provenance gate and as the frame it copies, and dies "
                         "on any unseen row_id. Writes row_id,factors for "
                         "build_submission.py to merge.")
    ap.add_argument("--expect-change", type=float, default=None,
                    help="fraction of cells allowed to differ from the "
                         "reference column (default 0.02). Raise ONLY when an input "
                         "artefact was changed deliberately AND the same "
                         "command on the old inputs returned 0 differing.")
    ap.add_argument("--min-predictions", type=int, default=1,
                    help="floor: a label whose OOF threshold fires must emit "
                         "at least this many predictions on test. A label "
                         "predicting nothing scores F1 0.000 for that label, "
                         "which costs 0.0125 composite. 0 disables.")
    ap.add_argument("--match-oof-scale-sd-only", action="store_true",
                    help="with --match-oof-scale: correct DISPERSION only, "
                         "keeping each arm's test mean. Separates the two "
                         "effects of the correction -- restoring an arm's "
                         "influence in the pool (sd) versus restoring the "
                         "operating point the thresholds were fitted at (mean).")
    ap.add_argument("--match-oof-scale-shrink", type=float, default=None,
                    help="with --match-oof-scale: interpolate the scale factor "
                         "geometrically between arm-level (0.0) and per-label "
                         "(1.0), trading the lower variance of one factor per "
                         "arm against the finer fit of one factor per "
                         "(arm, label).")
    ap.add_argument("--match-oof-scale-perlabel", action="store_true",
                    help="as --match-oof-scale but a separate factor per "
                         "(arm, label): fold-averaging compresses a label the "
                         "folds disagree about harder than one they agree on, "
                         "and a single arm-level factor averages that away. "
                         "Still zero gold, still transductive.")
    ap.add_argument("--match-oof-scale", action="store_true",
                    help="affinely map each arm's TEST logits onto its own OOF "
                         "mean and sd before pooling, so every arm carries the "
                         "influence the 24 thresholds were fitted under. Zero "
                         "fitted parameters, no gold. Off leaves the pooled "
                         "probabilities exactly as the arms emit them.")
    ap.add_argument("--shipped", default="artifacts/factors_tfit_test.csv",
                    help="the CURRENT factors column -- the newly built one is "
                         "diffed against it, and reconstructing it with tfit "
                         "is the provenance check")
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    eval_frac = 378.0 / (378.0 + n)

    print(f"OOF {n} posts / {len(uauth)} authors   arms {args.arms}")
    print(f"{args.splits} author partitions, eval {eval_frac:.1%}   "
          f"tbag B={args.tbag}, smooth window={args.smooth}")
    print("all four estimators see the SAME fit data and are scored on the "
          "SAME held-out posts\n")

    names = ["tfit", "tsmooth", "tbag", "tboth"]
    res = {k: [] for k in names}
    prot = {k: [] for k in names}
    dens = {k: [] for k in names}
    rng = np.random.default_rng(args.seed)
    brng = np.random.default_rng(args.seed + 1)

    for _ in range(args.splits):
        order = list(uauth)
        rng.shuffle(order)
        ev, got = [], 0
        for a in order:
            if got >= eval_frac * n:
                break
            ev.append(a)
            got += len(by[a])
        E = np.concatenate([by[a] for a in ev])
        F = np.setdiff1d(np.arange(n), E)
        PF, YF, PE, YE = P[F], Y[F], P[E], Y[E]

        th = {k: np.zeros(N) for k in names}
        for j in range(N):
            th["tfit"][j] = thr_argmax(PF[:, j], YF[:, j])
            th["tsmooth"][j] = thr_smooth(PF[:, j], YF[:, j], args.smooth)
            th["tbag"][j] = thr_bag(PF[:, j], YF[:, j], brng, args.tbag, 0)
            th["tboth"][j] = thr_bag(PF[:, j], YF[:, j], brng, args.tbag,
                                     args.smooth)
        for k in names:
            pr = (PE >= th[k][None, :]).astype(np.int8)
            f = f1pl(pr, YE)
            res[k].append(f.mean())
            prot[k].append(f[PROT_J].mean())
            dens[k].append(pr.sum() / len(E))

    ref = np.array(res["tfit"])
    print("=" * 96)
    print("HELD-OUT macro-F1 -- vs tfit, which is what ships")
    print("=" * 96)
    print(f"{'estimator':<10}{'macro':>9}{'protective':>12}{'d vs tfit':>11}"
          f"{'95% CI':>20}{'win':>6}{'fac/post':>10}")
    for k in names:
        v = np.array(res[k])
        if k == "tfit":
            print(f"{k:<10}{v.mean():>9.4f}{np.mean(prot[k]):>12.4f}"
                  f"{'--':>11}{'--':>20}{'--':>6}{np.mean(dens[k]):>10.2f}")
            continue
        d = v - ref
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{k:<10}{v.mean():>9.4f}{np.mean(prot[k]):>12.4f}"
              f"{d.mean():>+11.4f}"
              f"{f'[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]':>20}"
              f"{(d > 0).mean():>6.0%}{np.mean(dens[k]):>10.2f}")

    best = max(names, key=lambda k: np.mean(res[k]))
    db = np.array(res[best]) - ref
    print(f"\n  best: {best}   vs tfit {db.mean():+.4f}   "
          f"win {(db > 0).mean():.0%}   composite {0.3*db.mean():+.4f}")

    # ---- the full-data thresholds, for the build ---------------------------
    print("\n" + "=" * 96)
    print("FULL-DATA THRESHOLDS -- what each estimator would ship")
    print("=" * 96)
    full = {}
    for k in names:
        t = np.zeros(N)
        for j in range(N):
            if k == "tfit":
                t[j] = thr_argmax(P[:, j], Y[:, j])
            elif k == "tsmooth":
                t[j] = thr_smooth(P[:, j], Y[:, j], args.smooth)
            elif k == "tbag":
                t[j] = thr_bag(P[:, j], Y[:, j], brng, args.tbag, 0)
            else:
                t[j] = thr_bag(P[:, j], Y[:, j], brng, args.tbag, args.smooth)
        full[k] = t
    print(f"{'label':<44}{'gold n':>7}" + "".join(f"{k:>9}" for k in names))
    for j in np.argsort([Y[:, x].sum() for x in range(N)]):
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{int(Y[:, j].sum()):>7}" +
              "".join(f"{full[k][j]:>9.3f}" for k in names))
    print("\n  Sorted by gold count. The RAREST labels are where argmax is most")
    print("  unstable, so that is where the estimators should diverge most --")
    print("  if they agree everywhere, there was no variance to remove and this")
    print("  lane is closed.")

    # ---- build ------------------------------------------------------------
    if args.build_test:
        def resolve_test(d, a):
            for c in (f"{d}/{a}_cv_test.npz", f"{d}/{a}_test.npz",
                      f"{d}/nli_ensemble_{a}_test.npz"):
                if os.path.exists(c):
                    return c
            return None

        tdir = args.test_dir or args.dir
        if args.test_dir:
            print(f"test dumps from {tdir} (OOF still from {args.dir})")
        T, tid = [], None
        for a in args.arms:
            p = resolve_test(tdir, a)
            if p is None:
                raise SystemExit(f"no test dump for {a} under {tdir}")
            z = np.load(p, allow_pickle=True)
            ids = [str(x) for x in z["row_id"]]
            # Every arm must agree on row order before the dumps are stacked.
            # Stacking without checking would pool one arm's score for a post
            # with another arm's score for a different post; the resulting
            # column can still look plausible, and a partial scramble can stay
            # under the provenance tolerance below.
            if tid is None:
                tid = ids
            elif ids != tid:
                if sorted(ids) != sorted(tid):
                    raise SystemExit(
                        f"{a} covers a different ROW SET than {args.arms[0]} "
                        f"({len(ids)} vs {len(tid)} rows)")
                o = {r: i for i, r in enumerate(ids)}
                z_probs = z["probs"].astype(float)[[o[r] for r in tid]]
                T.append(z_probs)
                print(f"  {a}: row order differed, realigned to "
                      f"{args.arms[0]}")
                continue
            T.append(z["probs"].astype(float))
        Tarr = np.stack(T)

        # RESTORE EACH ARM'S OWN OOF LOGIT SCALE ON TEST.
        #
        # `logit_mean` weights an arm by the DISPERSION of its log-odds, so an
        # arm whose test logits are compressed relative to its OOF logits votes
        # more weakly on the posts being scored than on the posts the 24
        # thresholds were fitted on. Measured by src/factor_pool_diagnostic.py
        # across the 11 pooled arms (influence share =
        # cov(L_arm, L_pool)/var(L_pool), equal share = 1.0):
        #
        #     arm     sd oof  sd test  ratio   share oof -> test    shift
        #     d4       3.374   2.714   0.804     1.2371 -> 1.0517  -0.1854
        #     dmnli    3.612   3.486   0.965     1.2419 -> 1.3345  +0.0926
        #     qnli     3.599   3.144   0.874     1.2778 -> 1.1886  -0.0892
        #     mnli     3.179   3.083   0.970     1.1779 -> 1.2448  +0.0669
        #     lnli     3.917   3.716   0.949     1.3979 -> 1.4551  +0.0573
        #
        # Five arms move by more than +-0.02 and `d4` by nine times that. The
        # cause is construction: OOF scores each post with ONE fold model while
        # the test dump AVERAGES folds, and averaging compresses by an
        # ARM-SPECIFIC amount.
        #
        # This is NOT an equalisation of the arms against EACH OTHER on test,
        # and NOT factor_test_calibration.py, which standardises the
        # single POOLED logit. Neither compares an arm's TEST distribution
        # against that arm's OWN OOF distribution, which is the only comparison
        # that can see an arm-specific OOF->test compression. Uses no gold and
        # fits nothing: each arm's test logits are affinely mapped onto its own
        # OOF mean and sd.
        if args.match_oof_scale or args.match_oof_scale_perlabel:
            def _lg(p):
                q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
                return np.log(q / (1 - q))
            Lo, Lt = _lg(S), _lg(Tarr)
            if args.match_oof_scale_perlabel:
                # PER-LABEL refinement of the arm-level correction above.
                # Fold-averaging need not compress every label equally: a label
                # the folds disagree about is compressed harder than one they
                # agree on, and a single arm-level factor averages that
                # difference away. Still transductive and still zero gold --
                # 11x24 scale factors read off the two logit distributions,
                # each estimated from 1635 OOF and 378 test rows.
                print("  --match-oof-scale-perlabel: per (arm, label) scale")
                worst = []
                for ai, a in enumerate(args.arms):
                    for j in range(N):
                        so = float(Lo[ai, :, j].std())
                        st = float(Lt[ai, :, j].std())
                        if st < 1e-9 or so < 1e-9:
                            continue
                        mo = float(Lo[ai, :, j].mean())
                        mt = float(Lt[ai, :, j].mean())
                        Lt[ai, :, j] = mo + (Lt[ai, :, j] - mt) * (so / st)
                        worst.append((abs(np.log(so / st)), a,
                                      FACTOR_TAXONOMY[j], so / st))
                worst.sort(reverse=True)
                print("    largest (arm, label) corrections:")
                for _, a, lab, r in worst[:8]:
                    print(f"      {a:<12}{lab:<34} x{r:.3f}")
            else:
                # --sd-only keeps each arm's TEST mean and corrects only the
                # DISPERSION. The two halves of the correction act through
                # different channels: influence in a logit-mean pool is driven
                # by sd, while the mean sets where the absolute thresholds
                # bite. Correcting only the sd therefore separates the effect
                # of restoring an arm's weight from the effect of restoring the
                # operating point the thresholds were fitted at.
                #
                # --shrink L interpolates in log-ratio space between the
                # arm-level factor (L=0) and the per-label factor (L=1),
                # trading the lower estimation variance of one factor per arm
                # against the finer resolution of one per (arm, label).
                lam = args.match_oof_scale_shrink
                tag = ("sd-only" if args.match_oof_scale_sd_only else "mean+sd")
                if lam:
                    tag += f", per-label shrink lambda={lam}"
                print(f"  --match-oof-scale ({tag})")
                for ai, a in enumerate(args.arms):
                    mo, so = float(Lo[ai].mean()), float(Lo[ai].std())
                    mt, st = float(Lt[ai].mean()), float(Lt[ai].std())
                    if st < 1e-9:
                        print(f"    {a:<12} test sd ~0, skipped")
                        continue
                    r_arm = so / st
                    tgt = mt if args.match_oof_scale_sd_only else mo
                    if lam:
                        for j in range(N):
                            soj = float(Lo[ai, :, j].std())
                            stj = float(Lt[ai, :, j].std())
                            if stj < 1e-9 or soj < 1e-9:
                                rj = r_arm
                            else:
                                # geometric interpolation: lam=0 -> arm factor
                                rj = r_arm * ((soj / stj) / r_arm) ** lam
                            mj = float(Lt[ai, :, j].mean())
                            base = (mj if args.match_oof_scale_sd_only
                                    else float(Lo[ai, :, j].mean()))
                            Lt[ai, :, j] = base + (Lt[ai, :, j] - mj) * rj
                    else:
                        Lt[ai] = tgt + (Lt[ai] - mt) * r_arm
                    print(f"    {a:<12} sd {st:.3f} -> {so:.3f}  "
                          f"(x{r_arm:.3f})")
            Tarr = 1.0 / (1.0 + np.exp(-Lt))

        Pt = logit_mean(Tarr)
        nT = len(tid)

        # ---- COLD START: no shipped column exists for these rows ----------
        # THIS is the script that builds the delivered factor column, so this
        # is the path a rerun on 48 unseen authors actually takes. Without
        # --cold-test it reads --shipped for row order, for the
        # provenance gate AND as the frame it copies -- so on any row_id it has
        # not seen before it dies with a KeyError, and the factor column (30%
        # of the composite) never gets built.
        if args.cold_test:
            ct = pd.read_csv(args.cold_test, keep_default_na=False)
            if "row_id" not in ct.columns:
                raise SystemExit(f"{args.cold_test} has no row_id column")
            test_ids = [str(x) for x in ct["row_id"]]
            pos = {r: i for i, r in enumerate(test_ids)}
            missing = [r for r in tid if r not in pos]
            if missing:
                raise SystemExit(
                    f"{len(missing)} scored row_ids are absent from "
                    f"{args.cold_test} (first: {missing[:3]}) -- the factor "
                    f"dumps and the test set disagree about which posts exist.")
            holes = [r for r in test_ids if r not in set(tid)]
            if holes:
                raise SystemExit(
                    f"{len(holes)} test rows have NO factor prediction "
                    f"(first: {holes[:3]}). Refusing to write a column with "
                    f"holes -- a silently short factors file is how three "
                    f"columns failed on a fresh test set.")
            on = (Pt >= full[args.use][None, :]).astype(np.int8)
            on = _apply_floor(on, Pt, P, full[args.use],
                              args.min_predictions)
            fac = {r: str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
                   for i, r in enumerate(tid)}
            out = pd.DataFrame({"row_id": test_ids,
                                "factors": [fac[r] for r in test_ids]})
            os.makedirs(os.path.dirname(args.build_test) or ".", exist_ok=True)
            out.to_csv(args.build_test, index=False)
            print("\nPROVENANCE GATE: SKIPPED -- cold start, there is no "
                  "shipped column for these rows to reconstruct.")
            print("  The gate catches REBUILD DRIFT on rows already scored; it "
                  "cannot apply to unseen ones. Validate the path instead by "
                  "running WITHOUT --cold-test on data/test.csv and confirming "
                  "0 of N cells differ.")
            print(f"rule {args.use}: {on.sum()/nT:.2f} factors/post "
                  f"(gold 2.92)")
            print(f"0-factor posts {int((on.sum(1) == 0).sum())} of {nT} "
                  f"(gold rate 7.8%)")
            print(f"wrote {args.build_test}  (row_id,factors -- merge with "
                  f"build_submission.py)")
            return

        ship = pd.read_csv(args.shipped, keep_default_na=False)
        ship["row_id"] = ship["row_id"].astype(str)
        order = {r: i for i, r in enumerate(ship["row_id"])}
        cur = cells(ship["factors"], len(ship))
        perm = [order[r] for r in tid]

        # PROVENANCE: tfit on these arms must reproduce the shipped column.
        # A mismatch means the diff below would mix rebuild drift with the
        # intended change, so the measured effect could not be attributed to
        # the one variable under test.
        #
        # THE ONE CASE THIS GATE GETS WRONG. Its premise is that the INPUTS are
        # unchanged, so any difference must be rebuild drift. That premise
        # fails when the intended change IS an input artefact -- e.g. swapping
        # an arm's test dump for a construction-matched one. Then the gate
        # blocks the very experiment it is being run for.
        #
        # `--expect-change` is for exactly that, and it is deliberately
        # awkward: you must FIRST prove the path is clean by running this same
        # command with the OLD inputs and seeing "0 of N cells differ", then
        # state the tolerance explicitly. Never raise it to silence a surprise.
        repro = (Pt >= full["tfit"][None, :]).astype(np.int8)
        diff = int((repro != cur[perm]).sum())
        print(f"\nPROVENANCE: tfit reconstruction vs the shipped column -- "
              f"{diff} of {nT*N} cells differ")
        tol = args.expect_change if args.expect_change is not None else 0.02
        if args.expect_change is not None:
            print(f"  --expect-change {tol:.3f}: the gate is RELAXED because an "
                  f"input artefact was changed on purpose.")
            print(f"  This is only valid if the SAME command on the OLD inputs "
                  f"returned 0 cells differing.")
        if diff > tol * nT * N:
            raise SystemExit(
                f"REFUSING to write: {diff} of {nT*N} cells differ, above the "
                f"{tol:.3f} tolerance. Either this build is not a clean "
                f"single-variable change, or the intended input change is "
                f"larger than declared -- decide which before raising "
                f"--expect-change.")

        on = (Pt >= full[args.use][None, :]).astype(np.int8)
        on = _apply_floor(on, Pt, P, full[args.use], args.min_predictions)
        out = ship.copy()
        newcol = list(out["factors"])
        # WARM-PATH COVERAGE GUARD, the counterpart of the `holes` check on the
        # cold path. `newcol` is seeded from the SHIPPED column and only rows
        # present in `tid` are overwritten, so a test dump short of the full
        # row set would leave STALE factors on the rows it misses, with no
        # visible symptom: the provenance tolerance above is a fraction of all
        # cells and absorbs a handful of untouched rows. A column that is part
        # old and part new is not the model being validated, so refuse instead.
        _uncov = [r for r in order if r not in set(tid)]
        if _uncov:
            raise SystemExit(
                f"REFUSING: the test dumps cover {len(tid)} rows but the "
                f"shipped column has {len(order)}; {len(_uncov)} would keep "
                f"their OLD factors while the rest are rebuilt. First few: "
                f"{_uncov[:5]}. Rebuild the missing dumps, or pass "
                f"--cold-test if this is genuinely a new row set.")

        moved = 0
        for i, r in enumerate(tid):
            labs = str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
            kk = order[r]
            if str(newcol[kk]) != labs:
                moved += 1
            newcol[kk] = labs
        out["factors"] = newcol
        os.makedirs(os.path.dirname(args.build_test) or ".", exist_ok=True)
        out.to_csv(args.build_test, index=False)
        print(f"rule {args.use}: {on.sum()/nT:.2f} factors/post "
              f"(shipped {cur.sum()/len(ship):.2f}, gold 2.92)")
        print(f"cells changed {int((on != cur[perm]).sum())}   "
              f"posts changed {moved}   "
              f"0-factor posts {int((on.sum(1) == 0).sum())} of {nT} "
              f"(gold rate 7.8%)")
        print(f"wrote {args.build_test}")
        print("\nOnly the factors column changes -- S1 must return 0.8061 and "
              "the whole\ndelta must land in S2.")

    print("\n" + "=" * 96)
    print("WHY THIS IS DIFFERENT FROM EVERY OTHER DECODE IDEA")
    print("=" * 96)
    print("tshr/tsupp/platt/ecount/trisk/prot5 all changed WHAT is estimated.")
    print("This changes only HOW WELL the same 24 numbers are estimated, and")
    print("the cross-fit that killed those ideas measures exactly estimation")
    print("error -- so a lower-variance estimator should show up here directly.")
    print("")
    print("If nothing beats tfit, the 0.025 to the ceiling is irreducible with")
    print("this pool and factors are finished at 0.6774.")


if __name__ == "__main__":
    main()

"""
factor_threshold_decode.py

Compares quantile factor decodes against probability-threshold decodes, and
optionally builds the test factor column from the chosen rule.

A quantile decode fires exactly k = rate x n posts per label, with the rate
estimated from training. That FORCES the output count regardless of what is
actually in the eval set: if the eval set holds more `substance use` posts than
average, the assumed number fires and the rest are missed by construction. The
eval set's true prevalence wanders from the training estimate and a quantile
decode has no way to react.

A PROBABILITY THRESHOLD ADAPTS. Fire every post above t and the count moves with
the content. Prevalence drift stops being fatal; calibration drift becomes the
new exposure instead. Which of the two costs more is what this measures.

FIVE RULES, every one estimated on FIT authors and scored on ~378 held-out
posts, so the comparison is apples to apples and matches the real 1635/378 ratio:

  champ      quantile at the rates of the --champion column (0 params)
  shrunk     quantile at (champ + gold prevalence)/2        (0 params)
  qfit       quantile at per-label F1-optimal rates
  tfit       THRESHOLD at per-label F1-optimal probability
  thalf      THRESHOLD at F1*/2 -- the classic plug-in result that the
             F1-optimal threshold is half the achievable F1. Less estimation
             than tfit: it needs one number per label that is more stable than
             a rate estimated off a handful of gold examples.

The comparison of record is against `shrunk`, the strongest zero-parameter
quantile rule, rather than against `champ`.

Usage:
  python src/factor_threshold_decode.py
  python src/factor_threshold_decode.py --build-test artifacts/factors_thr_test.csv --use tfit
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


def quantile_pred(P, rates):
    n = len(P)
    on = np.zeros((n, N), dtype=np.int8)
    for j in range(N):
        k = min(max(int(round(rates[j] * n)), 1), n - 1)
        cut = np.sort(P[:, j])[-k]
        on[:, j] = (P[:, j] >= cut).astype(np.int8)
    return on


def thresh_pred(P, th):
    return (P >= th[None, :]).astype(np.int8)


def f1pl(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def best_threshold_and_f1(p, y):
    """argmax-F1 probability threshold for one label, and that F1."""
    npos = int(y.sum())
    if npos == 0:
        return 0.5, 0.0
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    f1 = np.where(ks + npos > 0, 2.0 * tp / np.maximum(ks + npos, 1), 0.0)
    b = int(np.argmax(f1))
    return float(TGRID[b]), float(f1[b])


def best_rate(p, y):
    """argmax-F1 FIRE RATE for one label (the quantile equivalent)."""
    n = len(p)
    t, _ = best_threshold_and_f1(p, y)
    return max(1, int((p >= t).sum())) / n


def fit_rules(PF, YF, champ_rate):
    """Everything a rule needs, estimated on the fit part only."""
    th = np.zeros(N)
    f1s = np.zeros(N)
    qr = np.zeros(N)
    for j in range(N):
        th[j], f1s[j] = best_threshold_and_f1(PF[:, j], YF[:, j])
        qr[j] = best_rate(PF[:, j], YF[:, j])
    gold = YF.mean(0)
    return {
        "champ": ("q", champ_rate),
        "shrunk": ("q", 0.5 * champ_rate + 0.5 * gold),
        "qfit": ("q", qr),
        "tfit": ("t", th),
        # the classic plug-in: the F1-optimal threshold is half the achievable
        # F1. Needs one F1 per label, which is steadier than a rate estimated
        # off 8 positives.
        "thalf": ("t", np.clip(f1s / 2.0, 0.02, 0.98)),
    }


def apply_rule(kind, par, P):
    return quantile_pred(P, par) if kind == "q" else thresh_pred(P, par)


def author_split(uauth, by, rng, eval_frac, n):
    order = list(uauth)
    rng.shuffle(order)
    ev, got = [], 0
    target = eval_frac * n
    for a in order:
        if got >= target:
            break
        ev.append(a)
        got += len(by[a])
    E = np.concatenate([by[a] for a in ev])
    return E, np.setdiff1d(np.arange(n), E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--cold-test", default=None,
                    help="COLD-START MODE. Path to a fresh test CSV (needs a "
                         "row_id column). Use this when there is NO champion "
                         "submission for the rows being scored -- which is the "
                         "situation on the organizers' 48 unseen authors, "
                         "where this code is re-run from scratch. Without it "
                         "this script indexes into --champion for row order, "
                         "for the provenance gate and for the output frame, "
                         "and dies with a KeyError on any unseen row_id. "
                         "Writes a row_id,factors CSV for build_submission.py "
                         "to merge, rather than a full submission.")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--build-test", default=None)
    ap.add_argument("--use", default="tfit",
                    choices=["shrunk", "qfit", "tfit", "thalf"])
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

    COLD = args.cold_test is not None
    if COLD:
        ct = pd.read_csv(args.cold_test, keep_default_na=False)
        if "row_id" not in ct.columns:
            raise SystemExit(f"{args.cold_test} has no row_id column")
        test_ids = [str(x) for x in ct["row_id"]]
        ch, champ = None, None
        # The `champ` and `shrunk` rules need a per-label target rate. With no
        # champion column to read it from, use the GOLD prevalence of the
        # labelled data. That is the sounder reference in any case: a rate read
        # off a predicted column calibrates the decode against its own output,
        # so any firing bias it already carries is preserved rather than
        # corrected.
        champ_rate = Y.mean(0)
        eval_frac = len(test_ids) / (len(test_ids) + n)
        print(f"COLD-START MODE: {len(test_ids)} test rows from "
              f"{args.cold_test}, no champion.")
        print("  champ/shrunk target rates come from GOLD prevalence, not from "
              "a shipped column.")
        print("  NOTE: `shrunk` is 0.5*champ_rate + 0.5*gold, so with "
              "champ_rate := gold the `champ` and `shrunk` rows of the table "
              "below are IDENTICAL by construction. That is expected in cold "
              "mode and is not a bug; the deployed rule is `tfit`, which is "
              "unaffected.")
    else:
        ch = pd.read_csv(args.champion, keep_default_na=False)
        champ = cells(ch["factors"], len(ch))
        champ_rate = champ.sum(0) / len(ch)
        eval_frac = len(ch) / (len(ch) + n)

    print(f"OOF {n} posts / {len(uauth)} authors")
    print(f"{args.splits} author partitions, eval {eval_frac:.1%} "
          f"(the real 1635-fit / 378-test ratio)")
    print("every rule is ESTIMATED ON FIT and SCORED ON HELD-OUT\n")

    names = ["champ", "shrunk", "qfit", "tfit", "thalf"]
    res = {k: [] for k in names}
    dens = {k: [] for k in names}
    zero = {k: [] for k in names}
    rng = np.random.default_rng(args.seed)
    for _ in range(args.splits):
        E, F = author_split(uauth, by, rng, eval_frac, n)
        rules = fit_rules(P[F], Y[F], champ_rate)
        for k in names:
            kind, par = rules[k]
            pr = apply_rule(kind, par, P[E])
            res[k].append(f1pl(pr, Y[E]).mean())
            dens[k].append(pr.sum() / len(E))
            zero[k].append((pr.sum(1) == 0).mean())

    base = np.array(res["champ"])
    shr = np.array(res["shrunk"])
    print("=" * 104)
    print("DECODE RULE -- held-out macro-F1")
    print("=" * 104)
    print(f"{'rule':<8}{'kind':>6}{'macro':>9}{'d vs champ':>12}{'95% CI':>20}"
          f"{'d vs shrunk':>13}{'win':>6}{'fac/post':>10}{'0-fac':>7}")
    for k in names:
        v = np.array(res[k])
        kind = "quant" if k in ("champ", "shrunk", "qfit") else "THRESH"
        if k == "champ":
            print(f"{k:<8}{kind:>6}{v.mean():>9.4f}{'--':>12}{'--':>20}"
                  f"{'--':>13}{'--':>6}{np.mean(dens[k]):>10.2f}"
                  f"{np.mean(zero[k]):>7.1%}")
            continue
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        ds = v - shr
        print(f"{k:<8}{kind:>6}{v.mean():>9.4f}{d.mean():>+12.4f}"
              f"{f'[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]':>20}"
              f"{ds.mean():>+13.4f}{(ds > 0).mean():>6.0%}"
              f"{np.mean(dens[k]):>10.2f}{np.mean(zero[k]):>7.1%}")
    print("\n  'win' and 'd vs shrunk' are against the strongest "
          "zero-parameter quantile")
    print("  rule, which is the bar a new decode has to clear.")
    print("  composite = 0.3 x the macro delta")

    best = max((k for k in names if k != "champ"),
               key=lambda k: np.mean(res[k]))
    dbest = np.array(res[best]) - shr
    print(f"\n  best rule: {best}   vs shrunk {dbest.mean():+.4f}  "
          f"win {(dbest > 0).mean():.0%}  composite {0.3 * dbest.mean():+.4f}")
    if best in ("tfit", "thalf"):
        print("  -> a THRESHOLD decode beats the quantile decode out-of-sample. "
              "The count adapting")
        print("     to the test set is worth more than pinning it to a "
              "training estimate.")
    else:
        print("  -> quantile still wins. The count constraint is helping, not "
              "hurting, and the")
        print("     remaining decode gap is irreducible estimation noise.")

    # ---- full-data rules, for the build ------------------------------------
    full = fit_rules(P, Y, champ_rate)
    print(f"\n{'label':<44}{'gold n':>7}{'champ%':>8}{'tfit t':>8}"
          f"{'tfit%':>7}{'thalf t':>9}")
    on_t = apply_rule(*full["tfit"], P)
    for j in range(N):
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{int(Y[:, j].sum()):>7}"
              f"{100*champ_rate[j]:>7.1f}%{full['tfit'][1][j]:>8.3f}"
              f"{100*on_t[:, j].mean():>6.1f}%{full['thalf'][1][j]:>9.3f}")

    # ---- test build --------------------------------------------------------
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
        on = apply_rule(*full[args.use], Pt)
        os.makedirs(os.path.dirname(args.build_test) or ".", exist_ok=True)

        if COLD:
            # No champion column exists for these rows, so there is nothing to
            # reproduce and nothing to copy. Emit row_id,factors and let
            # build_submission.py merge it.
            order = {r: i for i, r in enumerate(test_ids)}
            missing = [r for r in tid if r not in order]
            if missing:
                raise SystemExit(
                    f"{len(missing)} scored row_ids are absent from "
                    f"{args.cold_test} (first: {missing[:3]}). The factor "
                    f"dumps and the test set disagree about which posts exist.")
            uncovered = [r for r in test_ids if r not in set(tid)]
            if uncovered:
                raise SystemExit(
                    f"{len(uncovered)} test rows have NO factor prediction "
                    f"(first: {uncovered[:3]}). Refusing to write a column "
                    f"with holes -- a silently short factors file is how three "
                    f"columns failed on a fresh test set before the coverage "
                    f"guards went into build_submission.py.")
            print("\nPROVENANCE GATE: SKIPPED -- cold start, there is no "
                  "shipped column for these rows to reconstruct.")
            print("  The gate protects against REBUILD DRIFT on rows we have "
                  "already scored; it cannot apply to unseen ones. Validate "
                  "instead by running this on data/test.csv WITHOUT --cold-test "
                  "and confirming 0 cells differ.")
            fac = {}
            for i, row in enumerate(tid):
                fac[row] = str([FACTOR_TAXONOMY[j]
                                for j in range(N) if on[i, j]])
            out = pd.DataFrame({"row_id": test_ids,
                                "factors": [fac[r] for r in test_ids]})
            out.to_csv(args.build_test, index=False)
            print(f"rule {args.use}: {on.sum()/nT:.2f} factors/post "
                  f"(gold 2.92)")
            print(f"0-factor posts {int((on.sum(1) == 0).sum())} of {nT} "
                  f"(gold rate 7.8%)")
            print(f"wrote {args.build_test}  (row_id,factors -- merge with "
                  f"build_submission.py)")
            return

        order = {r: i for i, r in enumerate(ch["row_id"].astype(str))}
        perm = [order[r] for r in tid]

        repro = int((quantile_pred(Pt, champ_rate) != champ[perm]).sum())
        print(f"\nPROVENANCE: champ-rate reconstruction vs the shipped column "
              f"-- {repro} of {nT*N} cells differ")
        if repro:
            raise SystemExit(
                "REFUSING to write -- the reconstruction does not match the "
                "shipped column, so the new column would mix a rebuild bug "
                "with the intended change and silently corrupt the result.")

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
        out.to_csv(args.build_test, index=False)
        print(f"rule {args.use}: {on.sum()/nT:.2f} factors/post "
              f"(shipped {champ.sum()/len(ch):.2f}, gold 2.92)")
        print(f"cells changed {int((on != champ[perm]).sum())}   "
              f"posts changed {moved}   "
              f"0-factor posts {int((on.sum(1) == 0).sum())} of {nT} "
              f"(gold rate 7.8%)")
        print(f"wrote {args.build_test}")
        print("\nOnly the factors column changes -- S1 must return "
              "bit-identical at 0.8061.")


if __name__ == "__main__":
    main()

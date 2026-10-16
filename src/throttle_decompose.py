"""
throttle_decompose.py

Is the --match-counts constraint on factor_bag_ship.py THROTTLING the 8-arm
bag, or is the refit-xfit instrument simply over-optimistic at 8 arms?

THE QUESTION, IN NUMBERS. bag6q shipped count-matched with almost all of its
predicted gain delivered. bag8 shipped count-matched with only a small
fraction of its predicted gain delivered. The 8-arm bag's own F1-optimal
operating point is 3.12 factors/post, but --match-counts pins it to the
champion's 3.44, so most of its cells never move. Either the gain is locked
behind the count constraint, or it never existed outside the instrument.

WHAT THIS MEASURES. On the 1635 OOF posts, over --splits random author
partitions at SUBMISSION SCALE (fit ~81% of authors, score the held-out ~19%,
matching the real 1635-fit / 378-test ratio -- a 50/50 protocol was found to
flatter threshold shrinkage), each bag is decoded on the held-out part under:

    refit   absolute per-label argmax-F1 thresholds fitted on the fit part
    champ   the champion's per-label RATES (counts/378 from --champion,
            0 fitted parameters) applied as quantile counts -- this is what
            --match-counts ships
    own     the bag's own F1-optimal FIRE RATES measured on the fit part,
            applied as quantile counts -- this is what factor_bag_ship ships
            WITHOUT --match-counts
    w=0.25/0.50/0.75   per-label rates linearly between own (w=0) and champ

    own - champ IS THE THROTTLE. If it is ~0, the constraint is not binding
    and the lane is a mirage.

Then, because a 24-number release may transfer like 24 parameters, the
PER-LABEL release table: champ everywhere except ONE label at its own rate,
paired against champ across splits. Labels that win >=80% of splits are the
safe subset. The cumulative-release curve is ordered on the first half of the
splits and EVALUATED ONLY ON THE SECOND HALF, so the curve is not ranked and
scored on the same noise.

Finally the TEST side: how many cells each option would actually move against
the shipped bag8 column, and the per-label count deltas.

Usage (cluster, CPU, ~2-4 min):
  python src/throttle_decompose.py \
      --arms mnli dmnli cmine prot d4 qnli mnli7 lnli \
      --ref-arms mnli dmnli cmine prot d4 qnli \
      --champion submissions/NIDHI_cluster.csv
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
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


# ---------------------------------------------------------------- loading

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


# ---------------------------------------------------------------- scoring

def best_threshold(pr, y):
    """argmax-F1 threshold in one pass (from arm_robustness.py)."""
    npos = int(y.sum())
    if npos == 0:
        return 0.5
    o = np.argsort(-pr, kind="stable")
    ps, ys = pr[o], y[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -GRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    f1 = np.where(ks + npos > 0, 2.0 * tp / (ks + npos), 0.0)
    return float(GRID[int(np.argmax(f1))])


def macro(P, Y):
    v = []
    for j in range(N):
        tp = int((P[:, j] & Y[:, j]).sum())
        v.append(0.0 if tp == 0 else 2 * tp /
                 (2 * tp + int((P[:, j] & (1 - Y[:, j])).sum())
                  + int(((1 - P[:, j]) & Y[:, j]).sum())))
    return float(np.mean(v))


def topk_pred(PE, k_vec, nE):
    """Quantile decode: per label fire on the top-k probabilities (ties fire
    together, exactly as factor_bag_ship.py decodes the test set)."""
    on = np.zeros((nE, N), dtype=np.int8)
    for j in range(N):
        k = min(max(int(k_vec[j]), 1), nE - 1)
        cut = np.sort(PE[:, j])[-k]
        on[:, j] = (PE[:, j] >= cut).astype(np.int8)
    return on


def fit_rates(PF, YF):
    """Per-label F1-optimal threshold and its fire rate, on the fit part."""
    th = np.array([best_threshold(PF[:, j], YF[:, j]) for j in range(N)])
    rate = np.array([(PF[:, j] >= th[j]).mean() for j in range(N)])
    return th, rate


def submission_scale_split(auth, rng, eval_frac):
    """Assign whole authors to the eval side until ~eval_frac of posts."""
    s = pd.Series(auth)
    counts = s.value_counts()
    order = list(counts.index)
    rng.shuffle(order)
    target = eval_frac * len(auth)
    ev, got = set(), 0
    for a in order:
        if got >= target:
            break
        ev.add(a)
        got += counts[a]
    return s.isin(ev).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="the bag under test (bag8)")
    ap.add_argument("--ref-arms", nargs="+", default=None,
                    help="reference bag (bag6q) -- calibrates the instrument "
                         "against the reference bag's own delivered gain")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_cluster.csv")
    ap.add_argument("--shipped-bag8", default="submissions/NIDHI_bag8.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    O = logit_mean(S)
    Oref = None
    if args.ref_arms:
        rid2, S2 = load_oof(args.dir, args.ref_arms, args.k)
        if rid2 != rid:
            o = {x: i for i, x in enumerate(rid2)}
            S2 = S2[:, [o[x] for x in rid], :]
        Oref = logit_mean(S2)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rid), N), dtype=np.int8)
    for i, r in enumerate(rid):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1
    auth = lab.loc[rid, "anon_user_id"].to_numpy()

    # champion rates: 24 numbers, fixed, never fitted in this script
    ch = pd.read_csv(args.champion, keep_default_na=False)
    champ_cnt = np.zeros(N)
    for cell in ch["factors"]:
        for x in set(parse_factors(cell)):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                champ_cnt[j] += 1
    champ_rate = champ_cnt / len(ch)
    eval_frac = len(ch) / (len(ch) + len(rid))   # 378 / 2013 = 0.188

    print(f"bag {args.arms} on {len(rid)} OOF posts, "
          f"{pd.unique(auth).size} authors")
    print(f"champion column: {args.champion}  "
          f"{champ_cnt.sum() / len(ch):.2f} factors/post")
    print(f"{args.splits} author partitions at submission scale "
          f"(eval {eval_frac:.1%} of posts)\n")

    conds = ["refit", "champ", "own", "w25", "w50", "w75"]
    res = {c: [] for c in conds}
    res_ref = {c: [] for c in conds}
    single = np.zeros((args.splits, N))     # per-label release delta vs champ
    dens = {c: [] for c in conds}

    rng = np.random.default_rng(args.seed)
    for s in range(args.splits):
        E = submission_scale_split(auth, rng, eval_frac)
        F = ~E
        nE = int(E.sum())
        for P, out in ((O, res), (Oref, res_ref)):
            if P is None:
                continue
            th, own_rate = fit_rates(P[F], Y[F])
            pe = P[E]
            ye = Y[E]
            preds = {
                "refit": (pe >= th).astype(np.int8),
                "champ": topk_pred(pe, np.round(champ_rate * nE), nE),
                "own":   topk_pred(pe, np.round(own_rate * nE), nE),
            }
            for w, name in ((0.25, "w25"), (0.5, "w50"), (0.75, "w75")):
                r = (1 - w) * own_rate + w * champ_rate
                preds[name] = topk_pred(pe, np.round(r * nE), nE)
            for c in conds:
                out[c].append(macro(preds[c], ye))
                if out is res:
                    dens[c].append(preds[c].sum() / nE)
            if out is res:
                base = preds["champ"]
                for j in range(N):
                    m = base.copy()
                    k = min(max(int(round(own_rate[j] * nE)), 1), nE - 1)
                    cut = np.sort(pe[:, j])[-k]
                    m[:, j] = (pe[:, j] >= cut).astype(np.int8)
                    single[s, j] = macro(m, ye) - res["champ"][-1]

    def line(tag, v, ref):
        v = np.array(v)
        d = v - np.array(ref)
        se = d.std(ddof=1) / np.sqrt(len(d))
        return (f"  {tag:<8} {v.mean():.4f}   d vs champ {d.mean():+.4f} "
                f"[{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}] "
                f"win {(d > 0).mean():.0%}   {np.mean(dens[tag]):.2f} fac/post"
                if tag in dens else "")

    print("=" * 78)
    print("BAG UNDER TEST -- held-out macro-F1 by decode policy")
    print("(composite = 0.3 x these deltas)")
    print("=" * 78)
    for c in conds:
        print(line(c, res[c], res["champ"]))

    t = np.array(res["own"]) - np.array(res["champ"])
    se = t.std(ddof=1) / np.sqrt(len(t))
    print(f"\nTHE THROTTLE (own - champ): {t.mean():+.4f} macro-F1 "
          f"[{t.mean() - 1.96 * se:+.4f}, {t.mean() + 1.96 * se:+.4f}]  "
          f"win {(t > 0).mean():.0%}   composite {0.3 * t.mean():+.4f}")

    if Oref is not None:
        d = np.array(res["champ"]) - np.array(res_ref["champ"])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\nCALIBRATION  bag - refbag, BOTH at champion counts: "
              f"{d.mean():+.4f} [{d.mean() - 1.96 * se:+.4f}, "
              f"{d.mean() + 1.96 * se:+.4f}]")
        print("  This exact contrast can be separately verified against the")
        print("  official score. If the instrument says much more than that,")
        print("  divide every number above by the same optimism factor")
        print("  before believing it.")

    # ---------------- per-label release, paired vs champ across splits ------
    print("\n" + "=" * 78)
    print("PER-LABEL RELEASE: champion counts everywhere, ONE label at its "
          "own rate")
    print("=" * 78)
    mean_d = single.mean(0)
    win = (single > 0).mean(0)
    order = np.argsort(-mean_d)
    print(f"{'label':<45}{'champ%':>7}{'d':>9}{'win':>6}")
    for j in order:
        if abs(mean_d[j]) < 1e-5 and win[j] == 0:
            continue
        print(f"{FACTOR_TAXONOMY[j]:<45}{100 * champ_rate[j]:>6.1f}%"
              f"{mean_d[j]:>+9.4f}{win[j]:>6.0%}")

    # cumulative release: rank on first half of splits, score on second half
    h = args.splits // 2
    rank = np.argsort(-single[:h].mean(0))
    cum = []
    sec = single[h:]
    # additive approximation is wrong; recompute properly needs the split
    # predictions, so approximate the curve by the sum of paired deltas --
    # per-label decodes are independent (per-label F1), so the macro delta of
    # releasing a set IS the sum of its single-label deltas within a split.
    run = np.zeros(sec.shape[0])
    print("\ncumulative release, ranked on splits[:%d], scored on splits[%d:]"
          % (h, h))
    print(f"{'m':>3}  {'label added':<45}{'cum d':>9}{'win':>6}")
    for m, j in enumerate(rank, 1):
        run = run + sec[:, j]
        cum.append(run.mean())
        if m <= 12:
            print(f"{m:>3}  {FACTOR_TAXONOMY[j]:<45}{run.mean():>+9.4f}"
                  f"{(run > 0).mean():>6.0%}")
    best_m = int(np.argmax(cum)) + 1
    print(f"\nbest m on held-out splits: {best_m}  "
          f"(cum {max(cum):+.4f} macro-F1, composite {0.3 * max(cum):+.4f})")

    # ---------------- test side: what would actually move -------------------
    print("\n" + "=" * 78)
    print("TEST SIDE -- counts and cells each option would move")
    print("=" * 78)
    T = []
    ok = True
    for a in args.arms:
        p = resolve_test(args.dir, a)
        if p is None:
            print(f"  {a}: no test dump -- skipping test-side section")
            ok = False
            break
        z = np.load(p, allow_pickle=True)
        T.append(z["probs"].astype(float))
        tid = [str(x) for x in z["row_id"]]
    if ok:
        T = logit_mean(np.stack(T))
        nT = len(tid)
        th_full, rate_full = fit_rates(O, Y)   # full-OOF fit, as ship would
        ship = pd.read_csv(args.shipped_bag8, keep_default_na=False)
        cur = np.zeros((nT, N), dtype=np.int8)
        for i, cell in enumerate(ship["factors"]):
            for x in set(parse_factors(cell)):
                j = idx.get(str(x).strip().lower())
                if j is not None:
                    cur[i, j] = 1
        for name, r in (("own", rate_full),
                        ("w50", 0.5 * rate_full + 0.5 * champ_rate)):
            on = topk_pred(T, np.round(r * nT), nT)
            print(f"  {name:<5} {on.sum() / nT:.2f} fac/post   "
                  f"cells vs shipped bag8: {int((on != cur).sum())}   "
                  f"posts affected: {int((on != cur).any(1).sum())}")
        print(f"\n{'label':<45}{'champ':>6}{'own-k':>6}{'delta':>7}")
        for j in range(N):
            k_own = max(1, int(round(rate_full[j] * nT)))
            print(f"{FACTOR_TAXONOMY[j]:<45}{int(champ_cnt[j]):>6}"
                  f"{k_own:>6}{k_own - int(champ_cnt[j]):>+7}")

    print("\nDECISION RULE, PRE-STATED. The throttle is REAL only if "
          "own-champ > 0 with win >= 80%")
    print("AND the calibration contrast does not show the instrument "
          "over-predicting the")
    print("count-matched delta by more than ~2x. Otherwise this lane is "
          "closed -- say so.")


if __name__ == "__main__":
    main()

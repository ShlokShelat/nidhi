"""
factor_rate_decompose.py

For each of the 24 factor labels, split its weakness into the two causes that
need OPPOSITE fixes:

  DECODE   the probability ranking is fine, we just fire the label the wrong
           number of times. `sexual orientation` sits at P 0.27 / R 0.75 and
           `substance use` at P 0.34 / R 0.94 -- both fire far too often. Fixed
           by changing a count. Costs a fitted parameter.
  RANKING  the model cannot tell the label apart at all, so no count fixes it.
           `meaning in life` at P 0.32 / R 0.16 is the suspect. Needs a better
           model or better definitions.

AVERAGE PRECISION IS THE INSTRUMENT THAT SEPARATES THEM, because it is computed
on the ranking alone and is completely independent of where the decode
thresholds. High
AP with low F1 means decode. Low AP means ranking, and no amount of re-decoding
will help. This is the check that should have been run before the factor-LLM
lane was built on a ranking claim.

THREE F1 NUMBERS PER LABEL, and the gap between them is the whole point:
  cur    F1 at the CHAMPION's rate -- what is currently shipped, 0 fitted parameters
  best   F1 at the best rate fitted ON THE SAME DATA -- the decode ceiling, and
         optimistic by construction. Never ship this number.
  xfit   F1 at a rate fitted on held-out authors -- what a rate change would
         ACTUALLY deliver, paired against `cur` on the same splits

RATES ARE PARAMETERS AND PARAMETERS DO NOT TRANSFER HERE. Releasing all 24 is a
27-parameter-shaped move, and moves of that shape transfer negatively as a
rule. So this
reports the per-label release the way throttle_decompose.py does: a label counts
only if it wins >= WINBAR of splits, and the cumulative curve is ORDERED on the
first half of the splits and SCORED ONLY ON THE SECOND HALF, so the same noise
cannot both rank and reward a label.

Usage (cluster, CPU, ~2-4 min):
  python src/factor_rate_decompose.py
  python src/factor_rate_decompose.py --splits 100 --winbar 0.8
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
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
PROTECTIVE = {
    "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
}


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


def logit_mean(S):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def cells_from_factor_col(col, n):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    M = np.zeros((n, N), dtype=np.int8)
    for i, cell in enumerate(col):
        for x in set(parse_factors(cell)):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def f1_at_k(p, y, k):
    """F1 when the top-k probabilities fire (ties fire together)."""
    n = len(p)
    k = min(max(int(k), 1), n - 1)
    cut = np.sort(p)[-k]
    pred = p >= cut
    tp = int((pred & (y == 1)).sum())
    if tp == 0:
        return 0.0
    return 2.0 * tp / (int(pred.sum()) + int(y.sum()))


def best_k(p, y, grid):
    """Fire-count maximising F1, searched on a coarse rate grid."""
    n = len(p)
    best, bk = -1.0, max(1, int(y.sum()))
    for r in grid:
        k = max(1, int(round(r * n)))
        if k >= n:
            continue
        f = f1_at_k(p, y, k)
        if f > best:
            best, bk = f, k
    return bk, best


def submission_scale_split(auth, rng, eval_frac):
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
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--winbar", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    O = logit_mean(S)
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells_from_factor_col([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])

    ch = pd.read_csv(args.champion, keep_default_na=False)
    champ = cells_from_factor_col(ch["factors"], len(ch))
    champ_rate = champ.sum(0) / len(ch)
    eval_frac = len(ch) / (len(ch) + n)

    print(f"OOF {n} posts / {pd.unique(auth).size} authors   arms {args.arms}")
    print(f"champion rates from {args.champion} "
          f"({champ.sum() / len(ch):.2f} factors/post)")
    print(f"{args.splits} author partitions at submission scale "
          f"(eval {eval_frac:.1%})\n")

    grid = np.round(np.arange(0.002, 0.60, 0.002), 3)

    # ---------------- in-sample diagnosis: AP, current F1, decode ceiling ----
    APs = np.zeros(N)
    cur = np.zeros(N)
    bestf = np.zeros(N)
    bestr = np.zeros(N)
    for j in range(N):
        y = Y[:, j]
        p = O[:, j]
        APs[j] = average_precision_score(y, p) if y.sum() else 0.0
        cur[j] = f1_at_k(p, y, round(champ_rate[j] * n))
        bk, bf = best_k(p, y, grid)
        bestf[j] = bf
        bestr[j] = bk / n

    # ---------------- honest cross-fit: fit the rate on held-out authors -----
    rng = np.random.default_rng(args.seed)
    dx = np.zeros((args.splits, N))
    for s in range(args.splits):
        E = submission_scale_split(auth, rng, eval_frac)
        F = ~E
        nE = int(E.sum())
        for j in range(N):
            kf, _ = best_k(O[F, j], Y[F, j], grid)
            rate_own = kf / int(F.sum())
            a = f1_at_k(O[E, j], Y[E, j], round(rate_own * nE))
            b = f1_at_k(O[E, j], Y[E, j], round(champ_rate[j] * nE))
            dx[s, j] = a - b

    mean_d = dx.mean(0)
    win = (dx > 0).mean(0)
    se = dx.std(0, ddof=1) / np.sqrt(args.splits)

    print("=" * 108)
    print("PER-LABEL DIAGNOSIS -- is the weakness DECODE or RANKING?")
    print("=" * 108)
    print(f"{'label':<44}{'sup':>5}{'AP':>7}{'curF1':>7}{'ceil':>7}"
          f"{'champ%':>8}{'best%':>7}{'xfit d':>9}{'win':>6}  why")
    order = np.argsort(cur)
    for j in order:
        # AP well above current F1 means the ranking carries information the
        # decode is throwing away. Low AP means there is nothing to recover.
        if APs[j] < 0.35:
            why = "RANKING -- decode cannot fix"
        elif bestf[j] - cur[j] >= 0.05:
            why = "DECODE -- firing the wrong count"
        else:
            why = "near its ceiling"
        tag = FACTOR_TAXONOMY[j]
        if tag in PROTECTIVE:
            tag = "* " + tag
        print(f"{tag:<44}{int(Y[:, j].sum()):>5}{APs[j]:>7.3f}{cur[j]:>7.3f}"
              f"{bestf[j]:>7.3f}{100 * champ_rate[j]:>7.1f}%"
              f"{100 * bestr[j]:>6.1f}%{mean_d[j]:>+9.4f}{win[j]:>6.0%}  {why}")
    print("  * = protective factor    ceil = decode ceiling, fitted in-sample "
          "(optimistic -- never ship it)")

    # ---------------- what an honest release is worth ------------------------
    print("\n" + "=" * 108)
    print("WHAT A RATE CHANGE ACTUALLY BUYS")
    print("=" * 108)
    print(f"  current macro-F1                      {cur.mean():.4f}")
    print(f"  in-sample decode ceiling              {bestf.mean():.4f}   "
          f"({bestf.mean() - cur.mean():+.4f}, composite "
          f"{0.3 * (bestf.mean() - cur.mean()):+.4f})   OPTIMISTIC")

    sel = [j for j in range(N) if win[j] >= args.winbar and mean_d[j] > 0]
    gain = float(mean_d[sel].sum()) / N if sel else 0.0
    print(f"\n  labels clearing win >= {args.winbar:.0%}: {len(sel)}")
    print(f"  honest cross-fitted macro gain        {gain:+.4f}   "
          f"composite {0.3 * gain:+.4f}")
    for j in sorted(sel, key=lambda x: -mean_d[x]):
        print(f"    {FACTOR_TAXONOMY[j]:<44}{mean_d[j]:>+8.4f}  "
              f"[{mean_d[j] - 1.96 * se[j]:+.4f},{mean_d[j] + 1.96 * se[j]:+.4f}]"
              f"  {100 * champ_rate[j]:.1f}% -> {100 * bestr[j]:.1f}%")

    # cumulative release, ranked on the first half of splits, scored on second
    h = args.splits // 2
    rank = np.argsort(-dx[:h].mean(0))
    sec = dx[h:]
    run = np.zeros(sec.shape[0])
    cumv = []
    print(f"\n  cumulative release, ranked on splits[:{h}], "
          f"SCORED on splits[{h}:]")
    print(f"  {'m':>3}  {'label added':<44}{'cum macro':>11}{'win':>6}")
    for m, j in enumerate(rank, 1):
        run = run + sec[:, j] / N
        cumv.append(run.mean())
        if m <= 10:
            print(f"  {m:>3}  {FACTOR_TAXONOMY[j]:<44}{run.mean():>+11.4f}"
                  f"{(run > 0).mean():>6.0%}")
    bm = int(np.argmax(cumv)) + 1
    print(f"\n  best m on held-out splits: {bm}   cum {max(cumv):+.4f} macro   "
          f"composite {0.3 * max(cumv):+.4f}")

    print("\n" + "=" * 108)
    print("HOW TO READ THIS")
    print("=" * 108)
    print("A label marked RANKING is not a decode problem and no count change "
          "will move it.")
    print("Those are the ones that justify a fine-tuned factor model or better "
          "definitions.")
    print("A label marked DECODE is free score IF its xfit delta survives -- "
          "but each released")
    print("rate is a fitted parameter, and releasing all 24 transfers "
          "negatively as a rule. Ship the")
    print("smallest set that clears the win bar, never the full 24.")


if __name__ == "__main__":
    main()

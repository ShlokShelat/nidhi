"""
risk_prior_decode.py

THE FACTOR LENS, POINTED AT RISK.

Factors moved 0.6519 -> 0.6774 on nothing but the decode: each label was
firing 18% too often because the counts had been calibrated against the
model's own output instead of the data. Risk has never been examined the
same way. Its
decode is plain argmax over four probabilities and nobody has asked whether the
resulting class distribution matches the prevalence we know the test set has.

WHAT MAKES THIS CHECKABLE. The scorer decomposition established that test
prevalence matches train. Train gold is

    indicator 611 (37.4%)   ideation 519 (31.7%)
    behavior  391 (23.9%)   attempt  114 ( 7.0%)

so on 378 test posts the expected mix is roughly 141 / 120 / 90 / 26. Any
systematic departure from that is a decode error, not a property of the posts.

WHY ARGMAX CAN BE WRONG HERE. Argmax is Bayes-optimal for 0-1 loss. The scorer
is WEIGHTED F1, which is not 0-1 loss: it is a per-class F1 averaged by support,
so a class the model systematically under-calls loses recall on every one of its
posts at once. A model can be perfectly calibrated post-by-post and still put
out a skewed marginal, and argmax has no mechanism to notice.

FOUR DECODES, all scored as weighted F1 with a paired AUTHOR bootstrap:
  argmax    the deployed decode
  prior     per-class multipliers chosen so the ARGMAX MARGINAL matches train
            prevalence. The multipliers are determined by matching a KNOWN
            distribution, not fitted to a score -- the same sense in which the
            factor prevalence fix was zero-parameter.
  shrunk    half way between argmax and prior, because on factors the
            half-way version BEAT the full correction out of sample. Rate
            estimates wander on 378 posts; shrinkage hedges.
  fitted    per-class logit offsets tuned for weighted F1 -- 3 free parameters,
            the CEILING, never shippable.

AND A DEPLOYMENT-FAITHFUL CROSS-FIT, because the in-sample version of exactly
this measurement was inflated roughly 4x on factors. Multipliers are derived
from FIT authors and scored on ~378 held-out posts.

THE HONEST BLOCKER, stated up front. The deployed champion's risk
PROBABILITIES on test are not directly available -- `nli_risk_test_probs.csv`
is a component whose argmax agrees with the champion on only 0.897, and the
chain ends in symbolic rules and route_attempt that produce hard labels. So a
probability-level correction cannot be applied to the champion as things
stand. This measures whether the effect EXISTS and is large enough to justify
reconstructing that path. If it comes back near zero,
risk decoding is closed and no reconstruction is needed.

Usage:
  python src/risk_prior_decode.py
  python src/risk_prior_decode.py --enc artifacts/lora_ql70_oof.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

K = len(RISK_LEVELS)
PC = [f"prob_{c}" for c in RISK_LEVELS]


def load_probs(path, rows=None):
    d = pd.read_csv(path, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d.drop_duplicates("row_id").set_index("row_id")
    if not all(c in d.columns for c in PC):
        raise SystemExit(f"{path} lacks prob_* columns; this needs "
                         f"probabilities, not labels")
    idx = rows if rows is not None else list(d.index)
    return idx, d.loc[idx, PC].to_numpy(float)


def marginal(P, w):
    lab = (P * w[None, :]).argmax(1)
    return np.bincount(lab, minlength=K) / len(lab)


def fit_prior_weights(P, target, iters=400):
    """Multipliers so the ARGMAX marginal matches `target`. Iterative
    proportional adjustment -- determined by the target, not by any score."""
    w = np.ones(K)
    for _ in range(iters):
        m = marginal(P, w)
        m = np.maximum(m, 1e-4)
        w = w * (target / m) ** 0.25
        w = w / w.max()
    return w


def wf1(P, w, gold):
    pred = [RISK_LEVELS[i] for i in (P * w[None, :]).argmax(1)]
    return risk_weighted_f1(gold, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="artifacts/factors_tfit_test.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    rows, P = load_probs(args.enc)
    rows = [r for r in rows if r in lab.index]
    _, P = load_probs(args.enc, rows)
    n = len(rows)
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    gi = np.array([RISK_LEVELS.index(g) for g in gold])
    target = np.bincount(gi, minlength=K) / n

    print(f"{args.enc}: {n} posts / {pd.unique(auth).size} authors")
    print(f"\n{'class':<12}{'gold %':>9}{'argmax %':>10}{'diff':>8}"
          f"{'gold n/378':>12}{'argmax n/378':>14}")
    m0 = marginal(P, np.ones(K))
    for c in range(K):
        print(f"{RISK_LEVELS[c]:<12}{100*target[c]:>8.1f}%{100*m0[c]:>9.1f}%"
              f"{100*(m0[c]-target[c]):>+8.1f}{target[c]*378:>12.0f}"
              f"{m0[c]*378:>14.0f}")

    # the shipped column's own marginal, for orientation
    if os.path.exists(args.champion):
        ch = pd.read_csv(args.champion, keep_default_na=False)
        cm = pd.Series([normalize_label(v) for v in ch["risk_level"]]
                       ).value_counts()
        print(f"\nSHIPPED test column {os.path.basename(args.champion)}:")
        for c in RISK_LEVELS:
            got = int(cm.get(c, 0))
            exp = target[RISK_LEVELS.index(c)] * len(ch)
            print(f"  {c:<12}{got:>5}   expected {exp:>5.0f}"
                  f"   {got-exp:>+6.0f}")
        print("  (expected = train prevalence x 378; the scorer decomposition")
        print("   established that test prevalence matches train)")

    w_prior = fit_prior_weights(P, target)
    w_shr = np.ones(K) ** 0.5 * w_prior ** 0.5
    print(f"\nprior multipliers: "
          f"{dict(zip(RISK_LEVELS, np.round(w_prior, 3)))}")

    # ---- in-sample -------------------------------------------------------
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([by[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]

    def fitted_weights(Q, g):
        w = np.ones(K)
        best = wf1(Q, w, g)
        for _ in range(3):
            for c in range(K):
                for mlt in np.arange(0.5, 2.01, 0.02):
                    t = w.copy()
                    t[c] = mlt
                    v = wf1(Q, t, g)
                    if v > best:
                        best, w = v, t
        return w

    w_fit = fitted_weights(P, gold)
    cands = {"argmax": np.ones(K), "prior": w_prior, "shrunk": w_shr,
             "fitted": w_fit}
    print("\n" + "=" * 92)
    print("IN-SAMPLE weighted-F1 (optimistic -- see the cross-fit below)")
    print("=" * 92)
    print(f"{'decode':<9}{'params':>8}{'wF1':>9}{'d vs argmax':>13}"
          f"{'95% CI':>20}{'win':>7}")
    base = wf1(P, np.ones(K), gold)
    npar = {"argmax": 0, "prior": 0, "shrunk": 0, "fitted": 3}
    for k, w in cands.items():
        v = wf1(P, w, gold)
        if k == "argmax":
            print(f"{k:<9}{npar[k]:>8}{v:>9.4f}{'--':>13}{'--':>20}{'--':>7}")
            continue
        d = np.array([wf1(P[ix], w, gold[ix]) - wf1(P[ix], np.ones(K), gold[ix])
                      for ix in IXS])
        lo, hi = np.percentile(d, [2.5, 97.5])
        note = "   CEILING" if k == "fitted" else ""
        print(f"{k:<9}{npar[k]:>8}{v:>9.4f}{v-base:>+13.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{(d > 0).mean():>7.0%}{note}")

    # ---- deployment-faithful cross-fit ------------------------------------
    print("\n" + "=" * 92)
    print("CROSS-FIT -- multipliers from FIT authors, scored on ~378 held-out")
    print("=" * 92)
    rng2 = np.random.default_rng(args.seed + 7)
    eval_frac = 378.0 / (378.0 + n)
    keep = {"prior": [], "shrunk": [], "fitted": []}
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
        tF = np.bincount([RISK_LEVELS.index(g) for g in gold[F]],
                         minlength=K) / len(F)
        wp = fit_prior_weights(P[F], tF)
        b = wf1(P[E], np.ones(K), gold[E])
        keep["prior"].append(wf1(P[E], wp, gold[E]) - b)
        keep["shrunk"].append(wf1(P[E], np.sqrt(wp), gold[E]) - b)
        keep["fitted"].append(
            wf1(P[E], fitted_weights(P[F], gold[F]), gold[E]) - b)
    for k in ("prior", "shrunk", "fitted"):
        v = np.array(keep[k])
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f"  {k:<8}{v.mean():>+9.4f}  "
              f"[{v.mean()-1.96*se:+.4f},{v.mean()+1.96*se:+.4f}]  "
              f"win {(v > 0).mean():>4.0%}   composite {0.4*v.mean():+.4f}")
    print("\n  THIS is the number to believe. On factors the in-sample table")
    print("  overstated by 4x and REVERSED the ranking of prior vs shrunk.")
    print("\n  composite = 0.4 x the wF1 delta (risk is the largest weight)")
    print("\n  If nothing clears ~+0.005, risk decoding is closed and there is")
    print("  no reason to reconstruct the champion's probability path.")


if __name__ == "__main__":
    main()

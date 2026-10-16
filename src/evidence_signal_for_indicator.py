"""
evidence_signal_for_indicator.py

Can the EVIDENCE tagger's silence improve the RISK indicator decision?

THE OPENING. `evidence_suppression_rule.py` measured, on 1590 OOF posts:

    P(gold evidence empty | gold risk == indicator) = 0.961
    P(gold evidence empty | gold risk != indicator) = 0.000

Evidence-emptiness and the indicator label are the SAME EVENT. The whole
pipeline has only ever used that implication in one direction -- predict risk,
then suppress evidence. This runs it BACKWARDS: a span tagger that scans a post
and finds nothing worth extracting is casting an independent vote for
`indicator`, produced by a different architecture with a different objective
than the risk classifier. That is exactly the condition under which two
predictors carry complementary information.

WHY IT MIGHT PAY 1.75x. An indicator error costs twice -- once in risk_wF1
(weight 0.40) and again in evidence, because the suppression rule keys off the
risk label (weight 0.30). Nothing else measured in this project moves two
components at once.

WHY IT MIGHT BE NOTHING. Both models were trained on the same posts and the
same annotations. The tagger's silence may be a restatement of what the risk
head already knows, in which case the added features are redundant and the
honest answer is zero. That is a real possibility and this script is built to
detect it rather than to confirm the hypothesis.

FALSIFIER, FIXED BEFORE LOOKING. This lane is worth pursuing only if adding the
evidence features lifts indicator F1 by >= 0.01 with a 95% author-bootstrap CI
excluding zero. Anything smaller is inside the noise that has produced nine
losses, and the lane closes.

DESIGN. Logistic regression on risk log-odds (baseline) versus risk log-odds
plus evidence-silence features. Cross-fitted over AUTHOR-disjoint halves -- fit
on A, predict B, and vice versa -- so no post is scored by a model that saw its
author. Bootstrap resamples AUTHORS, not posts.

Usage:
  python src/evidence_signal_for_indicator.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_vote_oof import author_halves  # noqa: E402
from scoring import normalize_label  # noqa: E402

PCOLS = ["Indicator", "Ideation", "Behavior", "Attempt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--risk", default="artifacts/friend_oof_export.csv")
    ap.add_argument("--cands",
                    default="artifacts/evidence_oof/friend_oof_candidates.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    rk = pd.read_csv(args.risk)
    rk["row_id"] = rk["row_id"].astype(str)
    rk = rk.set_index("row_id")

    cd = pd.read_csv(args.cands)
    cd["row_id"] = cd["row_id"].astype(str)
    # deliberately UNFILTERED by min_conf: the question is how much signal the
    # raw confidence surface carries, and thresholding throws that away
    agg = cd.groupby("row_id")["conf"].agg(["count", "max", "mean", "sum"])

    rows = sorted(set(lab.index) & set(rk.index))
    y = np.array([normalize_label(lab.loc[r, "risk_level"]) == "indicator"
                  for r in rows], dtype=int)
    auth = np.array([lab.loc[r, "anon_user_id"] for r in rows])

    P = rk.loc[rows, PCOLS].to_numpy(dtype=float)
    P = np.clip(P, 1e-6, 1 - 1e-6)
    Xb = np.log(P / (1 - P))

    n_c = np.array([float(agg["count"].get(r, 0.0)) for r in rows])
    mx = np.array([float(agg["max"].get(r, 0.0)) for r in rows])
    mn = np.array([float(agg["mean"].get(r, 0.0)) for r in rows])
    sm = np.array([float(agg["sum"].get(r, 0.0)) for r in rows])
    silent = (n_c == 0).astype(float)
    Xe = np.column_stack([n_c, mx, mn, sm, silent, np.log1p(n_c)])
    Xp = np.column_stack([Xb, Xe])

    half = author_halves(auth)
    A, B = half == "A", half == "B"

    print(f"{len(rows)} OOF posts, {pd.unique(auth).size} authors, "
          f"{int(y.sum())} indicator ({y.mean():.3f})")
    print(f"tagger silent on {int(silent.sum())} posts")
    print(f"\nraw association, before any model:")
    print(f"  P(indicator | tagger silent)     = {y[silent == 1].mean():.3f}"
          f"   ({int((silent == 1).sum())} posts)")
    print(f"  P(indicator | tagger spoke)      = {y[silent == 0].mean():.3f}"
          f"   ({int((silent == 0).sum())} posts)")

    def xfit(X):
        out = np.zeros(len(rows))
        for src, dst in ((B, A), (A, B)):
            m = LogisticRegression(max_iter=2000, C=1.0)
            mu, sd = X[src].mean(0), X[src].std(0) + 1e-9
            m.fit((X[src] - mu) / sd, y[src])
            out[dst] = m.predict_proba((X[dst] - mu) / sd)[:, 1]
        return out

    pb, pp = xfit(Xb), xfit(Xp)
    # argmax of the risk head alone, the shipped decision, as a reference point
    arg = (P.argmax(1) == 0).astype(int)

    def best_f1(p):
        ts = np.unique(np.round(p, 3))
        return max(f1_score(y, (p >= t).astype(int), zero_division=0)
                   for t in ts)

    print(f"\n{'model':<34}{'AUC':>8}{'best F1':>10}")
    print("-" * 52)
    print(f"{'risk argmax (shipped decision)':<34}{'-':>8}"
          f"{f1_score(y, arg, zero_division=0):>10.4f}")
    print(f"{'LR on risk log-odds':<34}{roc_auc_score(y, pb):>8.4f}"
          f"{best_f1(pb):>10.4f}")
    print(f"{'LR + evidence silence features':<34}{roc_auc_score(y, pp):>8.4f}"
          f"{best_f1(pp):>10.4f}")

    # A single shared operating point, chosen on the OTHER half, so the F1
    # comparison is not two different thresholds flattering two models.
    def xfit_thresh(p):
        out = np.zeros(len(rows), dtype=int)
        for src, dst in ((B, A), (A, B)):
            ts = np.unique(np.round(p[src], 3))
            t = max(ts, key=lambda t: f1_score(y[src], (p[src] >= t).astype(int),
                                               zero_division=0))
            out[dst] = (p[dst] >= t).astype(int)
        return out

    db, dp = xfit_thresh(pb), xfit_thresh(pp)
    fb = f1_score(y, db, zero_division=0)
    fp = f1_score(y, dp, zero_division=0)
    print(f"\nhonest cross-fitted threshold (no peeking):")
    print(f"  risk only          {fb:.4f}")
    print(f"  + evidence signal  {fp:.4f}   delta {fp - fb:+.4f}")

    uniq = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uniq}
    rng = np.random.default_rng(args.seed)
    d = np.empty(args.boot)
    for i in range(args.boot):
        pk = rng.choice(len(uniq), size=len(uniq), replace=True)
        ix = np.concatenate([by[uniq[j]] for j in pk])
        d[i] = (f1_score(y[ix], dp[ix], zero_division=0)
                - f1_score(y[ix], db[ix], zero_division=0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\npaired author bootstrap  delta {fp - fb:+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  win {100 * (d > 0).mean():.1f}%")

    ok = (fp - fb) >= 0.01 and lo > 0
    print("\n" + "=" * 64)
    if ok:
        print("PASSES the pre-stated falsifier. The evidence tagger carries")
        print("indicator information the risk head does not. Worth building.")
    else:
        print("FAILS the pre-stated falsifier (needed >= +0.0100 with CI")
        print("excluding zero). The tagger's silence restates what the risk")
        print("head already knows. LANE CLOSED -- do not spend a slot.")
    print("=" * 64)


if __name__ == "__main__":
    main()

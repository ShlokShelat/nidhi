"""
compare_cv_vs_deployed.py

Decides whether the 5-fold CV factor models are actually BETTER than the
deployed 2-seed NLI ensemble, on a like-for-like basis.

The OOF table cannot answer this on its own: OOF macro-F1 is measured over
1635 posts while the deployed model was measured over 372, and the 372-post
val is intrinsically harder because two labels have 0 and 1 gold positives
there and are forced to F1=0. Comparing the two headline numbers directly
would be comparing different exams.

The clean comparison is the 372 VAL POSTS ONLY, where both systems are
genuinely author-out-of-sample:

  deployed : trained on train_split (1263 posts), never saw any val author
  CV fold k: trained on the other four folds (~1305 posts). Folds are
             AUTHOR-disjoint, so for a val post in fold k the model has never
             seen that post's author either. It has seen other val authors
             from other folds -- but those are different authors, which is
             extra training data, not leakage on the post being scored.

So both are honest predictions for the same 372 posts, and the only
difference is what each was trained on, which is the comparison the question
asks for.

Reported with author halves, and with the fold-to-fold spread in view.
Per-fold macro-F1 varies by roughly a tenth across folds at this sample
size, so any difference smaller than ~0.02 is unresolved rather than real.

Usage (CPU, after the OOF job):
  python src/compare_cv_vs_deployed.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors
from rule_ablation_v2 import author_split

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def macro(Y, P):
    return float(f1_score(Y, P, average=None, zero_division=0).mean())


def gold_matrix(df, tax):
    idx = {t: i for i, t in enumerate(tax)}
    Y = np.zeros((len(df), len(tax)), dtype=int)
    for i, x in enumerate(df["factors"]):
        for l in parse_factors(x):
            if l.strip().lower() in idx:
                Y[i, idx[l.strip().lower()]] = 1
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--deployed-probs", default="artifacts/nli_factors_val_probs.npz")
    ap.add_argument("--oof-thresholds", default="artifacts/factor_thresholds_oof.npy")
    args = ap.parse_args()

    va = pd.read_csv(args.val)
    tax = FACTOR_TAXONOMY
    Y = gold_matrix(va, tax)
    rid = list(va["row_id"])
    pos = {r: i for i, r in enumerate(rid)}

    # ---- CV predictions restricted to the val posts ----------------------
    P_cv = np.full((len(va), len(tax)), np.nan)
    for f in range(args.k):
        p = f"artifacts/fac_oof_f{f}.npz"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = np.load(p, allow_pickle=True)
        for r, row in zip(d["row_id"], d["probs"]):
            if r in pos:
                P_cv[pos[r]] = row
    n_have = int((~np.isnan(P_cv[:, 0])).sum())
    print(f"val posts {len(va)}   covered by OOF folds: {n_have}")
    if n_have != len(va):
        raise SystemExit("val posts not fully covered by the OOF folds")

    # ---- deployed 2-seed predictions on the same posts -------------------
    if not os.path.exists(args.deployed_probs):
        raise SystemExit(f"missing {args.deployed_probs} -- regenerate with "
                         f"predict_factors_nli.py --dump-probs")
    dep = np.load(args.deployed_probs, allow_pickle=True)
    order = [list(dep["row_id"]).index(r) for r in rid]
    P_dep = dep["probs"][order]
    th_dep = dep["thresholds"]

    half = author_split(va)[0].values
    A = np.where(half == "A")[0]
    B = np.where(half == "B")[0]

    def report(name, P, th, note=""):
        s = macro(Y, (P >= th).astype(int))
        a = macro(Y[A], (P[A] >= th).astype(int))
        b = macro(Y[B], (P[B] >= th).astype(int))
        print(f"{name:<44}{s:>9.4f}{a:>9.4f}{b:>9.4f}   {note}")
        return s

    print(f"\n{'system (all on the SAME 372 val posts)':<44}"
          f"{'macro-F1':>9}{'halfA':>9}{'halfB':>9}")

    dep_s = report("deployed 2-seed, its own thresholds", P_dep, th_dep,
                   "thresholds tuned ON val -- flatters it")

    th_oof = np.load(args.oof_thresholds) if os.path.exists(args.oof_thresholds) \
        else None
    if th_oof is not None:
        cv_s = report("CV 5-fold OOF, OOF global thresholds", P_cv, th_oof,
                      "thresholds never saw these posts' folds")
    else:
        cv_s = None
        print("  (no OOF thresholds file -- run factors_oof_cv.py --collect)")

    # same thresholds on both, to isolate the MODEL from the thresholds
    print(f"\nisolating the model from the thresholds (identical cut on both):")
    for t in (0.3, 0.4, 0.5, 0.6):
        th = np.full(len(tax), t)
        d_ = macro(Y, (P_dep >= th).astype(int))
        c_ = macro(Y, (P_cv >= th).astype(int))
        print(f"  global t={t:.1f}   deployed {d_:.4f}   CV {c_:.4f}   "
              f"{c_ - d_:+.4f}")

    if cv_s is not None:
        d = cv_s - dep_s
        print(f"\n{'=' * 72}")
        print(f"CV minus deployed on the same posts: {d:+.4f} macro-F1 "
              f"= {0.3 * d:+.4f} composite")
        print(f"\nfold-to-fold spread at this sample size is a large fraction")
        print(f"of a point, so a difference under ~0.02 here is NOT resolved")
        print(f"by this evidence.")
        if d > 0.02:
            print(f"VERDICT: CV models look genuinely better. Worth building a")
            print(f"         test-set prediction from the 5 fold models.")
        elif d < -0.02:
            print(f"VERDICT: CV models are worse. Keep the deployed 2-seed.")
        else:
            print(f"VERDICT: unresolved by this evidence.")


if __name__ == "__main__":
    main()

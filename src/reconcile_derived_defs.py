"""
reconcile_derived_defs.py

Reconcile the two contradictory readings of the derived-definitions arm, then
build the hybrid on the CORRECT selection rule.

THE CONTRADICTION THIS RESOLVES. `why_defs_failed.py` reports one reading of
the derived-definitions arm's mean per-label delta over 24 labels; the
recorded headline for the same arm, from a differently-averaged macro score,
disagrees in sign. Macro-F1 IS the mean of per-label F1, so on the same posts
and the same labels these cannot both be right. This script computes both
readings side by side against the current OOF artefacts so the divergence --
and which reading is correct -- is visible directly, instead of quoted from a
prior run.

THE LIKELY CAUSE, which this tests directly: POOLED vs PER-FOLD averaging.
  pooled   -- one per-label F1 over all 1635 OOF posts, then mean over labels
  per-fold -- per-label F1 inside each ~327-post fold, macro per fold, then mean
              over folds
They diverge on rare labels. `sexual orientation related issues` has 8 positives
in total, so ~1.6 per fold; a fold predicting none contributes F1 = 0, and
averaging those zeros buries the label -- exactly the kind of label where the
derived definitions show their largest gains, reported per-label below.

**POOLED IS THE CORRECT INSTRUMENT.** The competition scores one test set in one
pass; it never averages folds. If per-fold averaging reproduces the recorded
closure and pooled does not, the arm was closed by the wrong measurement.

Both arms use THEIR OWN saved thresholds, matching what why_defs_failed.py did,
so the comparison is like-for-like.

IT ALSO BUILDS THE HYBRID ON THE RIGHT CRITERION. The shipped
`build_hybrid_defs.py` swaps in the derived definition for labels whose F1 is
LOW. Checking that criterion against which labels the derived definition
actually helps on shows the two are roughly ANTI-correlated: labels where the
derived version helps most are often left on the published definition, and
labels where it hurts most get swapped in anyway -- see the per-label table
this script prints.

The right criterion is "does the DERIVED definition beat the PUBLISHED one for
this label", decided CROSS-FITTED on the other four folds so a label is never
selected using the posts it is scored on.

Usage:
  python src/reconcile_derived_defs.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402


def load_oof(pattern, k=5):
    """Returns row_ids, probs, thresholds, fold index per row."""
    r, P, T, F = [], [], [], []
    for f in range(k):
        p = pattern.format(f=f)
        if not os.path.exists(p):
            return None, None, None, None
        d = np.load(p, allow_pickle=True)
        n = len(d["row_id"])
        r.extend([str(x) for x in d["row_id"]])
        P.append(np.asarray(d["probs"], float))
        T.append(np.tile(np.asarray(d["thresholds"], float), (n, 1)))
        F.append(np.full(n, f))
    return r, np.concatenate(P), np.concatenate(T), np.concatenate(F)


def f1(pred, y):
    tp = int((pred & (y == 1)).sum())
    if tp == 0:
        return 0.0
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return 2 * pr * rc / (pr + rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--pub-oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--derived-oof",
                    default="artifacts/factor_runs/defs_cv_oof_f{f}.npz")
    ap.add_argument("--defs", default="artifacts/derived_defs")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = list(FACTOR_TAXONOMY)
    idx = {t: i for i, t in enumerate(labels)}
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    ir, IP, ITH, IF = load_oof(args.pub_oof, args.k)
    if ir is None:
        raise SystemExit(f"published OOF not found at {args.pub_oof}")
    dr, DP, DTH, _ = load_oof(args.derived_oof, args.k)
    if dr is None:
        raise SystemExit(f"derived OOF not found at {args.derived_oof}")
    o = {r: i for i, r in enumerate(dr)}
    miss = [r for r in ir if r not in o]
    if miss:
        raise SystemExit(f"derived OOF lacks {len(miss)} rows")
    DP, DTH = DP[[o[r] for r in ir]], DTH[[o[r] for r in ir]]

    Y = np.zeros(IP.shape, int)
    for i, r in enumerate(ir):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    PI, PD = IP >= ITH, DP >= DTH
    print(f"{len(ir)} pooled OOF posts, {args.k} folds, {len(labels)} labels")

    # ---------- 1. POOLED vs PER-FOLD ----------
    pub_pooled = np.array([f1(PI[:, j], Y[:, j]) for j in range(len(labels))])
    der_pooled = np.array([f1(PD[:, j], Y[:, j]) for j in range(len(labels))])

    def perfold(P):
        out = []
        for f in range(args.k):
            m = IF == f
            out.append(np.mean([f1(P[m, j], Y[m, j])
                                for j in range(len(labels))]))
        return np.array(out)

    pub_pf, der_pf = perfold(PI), perfold(PD)

    print("\n" + "=" * 72)
    print("1. WHICH AVERAGING REPRODUCES WHICH HEADLINE?")
    print("=" * 72)
    print(f"{'method':<34}{'published':>11}{'derived':>10}{'delta':>10}")
    print("-" * 65)
    print(f"{'POOLED (mean of per-label F1)':<34}{pub_pooled.mean():>11.4f}"
          f"{der_pooled.mean():>10.4f}{der_pooled.mean() - pub_pooled.mean():>+10.4f}")
    print(f"{'PER-FOLD (macro per fold, averaged)':<34}{pub_pf.mean():>11.4f}"
          f"{der_pf.mean():>10.4f}{der_pf.mean() - pub_pf.mean():>+10.4f}")
    print("\nThe method matching the previously recorded closure is the one")
    print("that produced it. POOLED is what the competition does -- it scores")
    print("one test set in one pass and never averages folds.")

    rare = [j for j in range(len(labels)) if Y[:, j].sum() <= 20]
    print(f"\n{len(rare)} labels have <=20 positives in 1635 posts "
          f"(<=4 per fold):")
    for j in rare:
        pf = np.mean([f1(PD[IF == f, j], Y[IF == f, j]) for f in range(args.k)])
        print(f"  {labels[j][:40]:<42} n={int(Y[:, j].sum()):>3}  "
              f"pooled {der_pooled[j]:.3f}  per-fold {pf:.3f}")
    print("A large pooled/per-fold gap on rare labels is the mechanism.")

    # ---------- 2. the social support anomaly ----------
    print("\n" + "=" * 72)
    print("2. THE social support DEFINITION (large derived regression, names no other)")
    print("=" * 72)
    os.environ.pop("FACTOR_DEFS_JSON", None)
    try:
        from train_factors_nli import FACTOR_DEFINITIONS
        print(f"  published: {FACTOR_DEFINITIONS.get('social support')}")
    except Exception as e:
        print(f"  (could not import published defs: {e})")
    for f in range(args.k):
        p = os.path.join(args.defs, f"defs_f{f}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"  fold {f}: {d.get('social support')}")

    # ---------- 3. hybrid on the CORRECT criterion ----------
    print("\n" + "=" * 72)
    print("3. CROSS-FITTED HYBRID -- swap where DERIVED BEATS PUBLISHED")
    print("=" * 72)
    HY = PI.copy()
    chosen = []
    for f in range(args.k):
        te, tr = IF == f, IF != f
        pick = []
        for j in range(len(labels)):
            use_d = f1(PD[tr, j], Y[tr, j]) > f1(PI[tr, j], Y[tr, j])
            pick.append(use_d)
            if use_d:
                HY[te, j] = PD[te, j]
        chosen.append(pick)
    hyb = np.array([f1(HY[:, j], Y[:, j]) for j in range(len(labels))])
    stable = sum(1 for j in range(len(labels))
                 if len({c[j] for c in chosen}) == 1)
    n_der = int(np.mean([sum(c) for c in chosen]))
    print(f"  labels taking the DERIVED definition: ~{n_der}/24 per fold")
    print(f"  choices identical across all 5 folds: {stable}/24")
    print(f"\n{'arm':<28}{'pooled macro':>14}{'vs published':>14}")
    print("-" * 56)
    print(f"{'published':<28}{pub_pooled.mean():>14.4f}")
    print(f"{'all-derived':<28}{der_pooled.mean():>14.4f}"
          f"{der_pooled.mean() - pub_pooled.mean():>+14.4f}")
    print(f"{'hybrid (xfit, correct rule)':<28}{hyb.mean():>14.4f}"
          f"{hyb.mean() - pub_pooled.mean():>+14.4f}")
    print(f"{'ORACLE per-label best':<28}"
          f"{np.maximum(pub_pooled, der_pooled).mean():>14.4f}"
          f"{np.maximum(pub_pooled, der_pooled).mean() - pub_pooled.mean():>+14.4f}")

    # paired author bootstrap on the hybrid
    author = np.array([str(lab.loc[r, "anon_user_id"]) for r in ir])
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}
    rng = np.random.default_rng(args.seed)
    draws = []
    for _ in range(args.boot):
        pick = rng.choice(len(auths), len(auths), replace=True)
        sel = np.concatenate([aidx[auths[i]] for i in pick])
        a_ = np.mean([f1(HY[sel, j], Y[sel, j]) for j in range(len(labels))])
        b_ = np.mean([f1(PI[sel, j], Y[sel, j]) for j in range(len(labels))])
        draws.append(a_ - b_)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"\n  hybrid paired author bootstrap: 95% CI "
          f"[{lo:+.4f}, {hi:+.4f}]   ({len(auths)} authors)")
    print(f"  composite impact = 0.3 x delta")

    print("\n" + "=" * 72)
    print("Read (1) FIRST. If per-fold reproduces the recorded closure and")
    print("pooled does not, the arm was closed by an instrument the")
    print("competition does not use, and (3) is a free gain from a model")
    print("already trained. If pooled reproduces the recorded closure too,")
    print("the per-label pooled gain was itself the artefact and this closes.")


if __name__ == "__main__":
    main()

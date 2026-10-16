"""
factor_bag_ship.py

Build a multi-arm factor column using FIRE-RATE TRANSFER.

WHAT THIS SHIPS. The member-count trend measured by `factor_rank_average.py`
over all 57 subsets of six arms, 1635 out-of-fold posts, is POSITIVE per
added member under both a refit (cross-fitted, honest) policy and a
ratematch (fits nothing) policy.

Positive under BOTH policies -- a policy-consistency check (requiring a
positive slope under both a refit and a ratematch procedure) catches a sign
that flips between fitting procedures before it reaches a submission. Going
from 2 to 6 members is a net gain. It is a slope over every subset, not a
selected maximum.

TWO FAILURE MODES THIS GUARDS AGAINST.

1. AN ABSOLUTE-PROBABILITY CUTOFF FAILS UNDER DISTRIBUTION SHIFT. A threshold
   fitted on OOF probabilities (fewer checkpoints/post) and applied to test
   probabilities averaged over more checkpoints sits on a narrower
   distribution. In the rare-label tail a slightly-too-high ABSOLUTE cutoff
   means predicting nothing at all -- a single zero-fire label in a 24-label
   macro can erase more score than the rest of an ensemble's gains combined.

   FIX: never transfer an absolute probability. Transfer a RATE and place the
   cutoff at the corresponding QUANTILE of the test distribution. A quantile is
   invariant to any monotone narrowing, so this failure mode cannot fire. And
   k is floored at 1, so a zero-prediction label is impossible.

2. Plain prevalence-matching is ALSO wrong, and would repeat a different
   failure. The champion predicts 3.44 factors/post against gold's 2.92. That
   1.16 fire rate is not miscalibration -- it is the correct F1 trade for rare
   positives, and moving toward gold prevalence has repeatedly cost score when
   tried.

   FIX: transfer the F1-OPTIMAL FIRE RATE measured out-of-fold, not the gold
   prevalence. Per label, find the threshold maximising OOF F1, record what
   fraction of posts it fires on, and reproduce THAT fraction on test.

So the quantity carried from OOF to test is dimensionless and F1-optimal. The
OOF fit uses 1635 posts that are disjoint from test, so nothing is tuned on the
data being scored.

Usage:
  python src/factor_bag_ship.py --arms mnli dmnli prot d4 --rule mean
  python src/factor_bag_ship.py --arms mnli dmnli prot d4 --rule logit \
      --out artifacts/bag4_factors.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def f1_at(pred, y):
    tp = int((pred & y).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + int((pred & (1 - y)).sum())
                     + int(((1 - pred) & y).sum()))


# Architecture families, decided by TAXONOMY and not fitted from data, so
# `flogit` remains a 0-parameter rule -- the class of change most likely to
# over-deliver relative to more heavily parameterised alternatives, which
# tend to transfer only partially, or negatively.
DECODER_LLM = {"qnli", "mnli7", "lnli", "mist", "qnli7", "llama", "mistral"}


def combine(S, rule, tags=None):
    if rule == "mean":
        return np.mean(S, axis=0)
    if rule == "median":
        return np.median(S, axis=0)
    if rule == "logit":
        X = np.clip(np.asarray(S), 1e-6, 1 - 1e-6)
        return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))
    if rule == "flogit":
        # Logit-average WITHIN each architecture family, then average the
        # family means with EQUAL weight.
        #
        # WHY. Ensemble gain here comes from cross-family diversity, not member
        # count: adding Qwen-14B helped more than adding additional DeBERTa
        # variants did. Flat averaging hands the DeBERTa
        # bloc 5 of 6 votes, muting the member that carries the marginal value.
        # NOT a rank-average approach, which equalises MEMBERS across a set
        # full of DeBERTa duplicates; this equalises FAMILIES, the variable
        # the qnli/aspv2/d6 comparison actually isolated.
        if tags is None:
            raise SystemExit("flogit needs arm tags")
        X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
        L = np.log(X / (1 - X))
        fams = {}
        for i, t in enumerate(tags):
            fams.setdefault("llm" if t in DECODER_LLM else "encoder",
                            []).append(i)
        if len(fams) < 2:
            raise SystemExit(
                f"flogit needs at least two families; all arms resolved to "
                f"{list(fams)}. With one family it is identical to logit.")
        print(f"  families: " + "  ".join(
            f"{k}={[tags[i] for i in v]}" for k, v in fams.items()))
        return 1.0 / (1.0 + np.exp(
            -np.mean([L[ix].mean(axis=0) for ix in fams.values()], axis=0)))
    raise SystemExit(f"unknown rule {rule}")


def resolve_test(d, a):
    for c in (f"{d}/{a}_cv_test.npz", f"{d}/{a}_test.npz",
              f"{d}/nli_ensemble_{a}_test.npz"):
        if os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--rule", default="mean",
                    choices=["mean", "logit", "median", "flogit"])
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--match-counts", default=None,
                    help="A submission CSV whose PER-LABEL PREDICTION COUNTS "
                         "this column must reproduce exactly. Point it at the "
                         "champion to make the bag a true single-variable "
                         "test: identical density, identical per-label "
                         "precision/recall trade, ONLY the ranking changes. "
                         "Overrides the fire rates derived from OOF.")
    ap.add_argument("--release-labels", nargs="*", default=None,
                    help="With --match-counts: these labels IGNORE the "
                         "champion count and use the OOF fire rate instead. "
                         "A partial release of the count constraint -- ship "
                         "only the labels src/throttle_decompose.py showed "
                         "winning >=80%% of held-out splits, keep the "
                         "held-out-validated counts everywhere else.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.release_labels:
        bad = [x for x in args.release_labels if x not in FACTOR_TAXONOMY]
        if bad:
            raise SystemExit(f"--release-labels not in taxonomy: {bad}")
        if not args.match_counts:
            raise SystemExit("--release-labels only means something WITH "
                             "--match-counts (without it every label already "
                             "uses its own rate)")
    out = args.out or f"artifacts/bag{len(args.arms)}_{args.rule}_factors.csv"

    # ---------------------------------------------------------- OOF side
    O, rid = [], None
    for a in args.arms:
        pr, ids = [], []
        for f in range(args.k):
            p = f"{args.dir}/{a}_cv_oof_f{f}.npz"
            if not os.path.exists(p):
                raise SystemExit(f"{a}: missing {p}")
            z = np.load(p, allow_pickle=True)
            pr.append(z["probs"].astype(float))
            ids.extend(str(x) for x in z["row_id"])
        pr = np.concatenate(pr)
        if rid is None:
            rid = ids
        elif ids != rid:
            if set(ids) != set(rid):
                raise SystemExit(f"{a}: OOF row set differs")
            o = {k: i for i, k in enumerate(ids)}
            pr = pr[[o[k] for k in rid]]
        O.append(pr)
    O = combine(np.stack(O), args.rule, args.arms)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rid), N), dtype=int)
    for i, r in enumerate(rid):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1

    # ------------------------------------------- TEST side, same combination
    T, tid = [], None
    for a in args.arms:
        p = resolve_test(args.dir, a)
        if p is None:
            raise SystemExit(f"{a}: no test dump. Build it with "
                             f"scripts/factor_arm_test_dumps.sh")
        z = np.load(p, allow_pickle=True)
        pr = z["probs"].astype(float)
        ids = [str(x) for x in z["row_id"]]
        if tid is None:
            tid = ids
        elif ids != tid:
            if set(ids) != set(tid):
                raise SystemExit(f"{a}: TEST row set differs -- cannot bag")
            o = {k: i for i, k in enumerate(ids)}
            pr = pr[[o[k] for k in ids]]
        T.append(pr)
        print(f"  {a:<10} {os.path.basename(p)}  sd={pr.std():.4f}")
    T = combine(np.stack(T), args.rule, args.arms)
    print(f"\n{len(args.arms)} arms, rule={args.rule}   "
          f"OOF {O.shape}  TEST {T.shape}")
    print(f"OOF sd {O.std():.4f}   TEST sd {T.std():.4f}   "
          f"ratio {T.std() / O.std():.3f}  (a quantile cutoff ignores this)")

    # -------------------------------- fire rates: F1-optimal, measured on OOF
    rate = np.zeros(N)
    oof_f1 = np.zeros(N)
    for j in range(N):
        best, bt = -1.0, 0.5
        for t in GRID:
            v = f1_at((O[:, j] >= t).astype(int), Y[:, j])
            if v > best:
                best, bt = v, t
        rate[j] = float((O[:, j] >= bt).mean())
        oof_f1[j] = best
    prev = Y.mean(axis=0)
    print(f"\nOOF pooled macro-F1 at those thresholds {oof_f1.mean():.4f} "
          f"(IN-SAMPLE, inflated -- the honest xfit figure is lower)")
    print(f"fire rate / prevalence, mean {np.mean(rate / np.maximum(prev, 1e-9)):.3f}"
          f"  -- the F1-optimal over-prediction that mnliall and mnlidef "
          f"destroyed by moving it toward 1.0")

    # --------------------------------- apply as QUANTILES on the test set
    nT = len(tid)
    K = np.array([int(round(rate[j] * nT)) for j in range(N)])

    if args.match_counts:
        # THE RECORD THIS GUARDS AGAINST. Every submission that reduced
        # factors/post has lost: mnlidef at 3.29, mnliall at 2.92, against
        # the champion's 3.44. The OOF-derived fire rate
        # here comes out at 3.14 -- inside that losing range. There is a
        # respectable reason for it (a better-ranked model needs less
        # over-prediction to maximise F1), but that is the same argument that
        # would have justified mnliall, and mnliall lost.
        #
        # Copying the champion's per-label COUNTS removes density from the
        # experiment entirely. Each label keeps its exact precision/recall
        # trade; the only thing that changes is WHICH posts receive it. That
        # isolates the one effect actually measured -- better member ranking --
        # from the one that has failed three times out of three.
        ch = pd.read_csv(args.match_counts, keep_default_na=False)
        cnt = np.zeros(N, dtype=int)
        for cell in ch["factors"]:
            for x in parse_factors(cell):
                j = idx.get(str(x).strip().lower())
                if j is not None:
                    cnt[j] += 1
        if len(ch) != nT:
            raise SystemExit(f"--match-counts has {len(ch)} rows, test has {nT}")
        print(f"\nmatching per-label counts from {args.match_counts}")
        print(f"  champion {cnt.sum() / nT:.2f} factors/post, "
              f"OOF fire rate would have given {K.sum() / nT:.2f}")
        if args.release_labels:
            rel = [FACTOR_TAXONOMY.index(x) for x in args.release_labels]
            merged = cnt.copy()
            for j in rel:
                merged[j] = K[j]
                print(f"  RELEASED {FACTOR_TAXONOMY[j]}: champion {cnt[j]} "
                      f"-> own rate {K[j]}")
            print(f"  partial release: {len(rel)} of {N} labels at own rate, "
                  f"{merged.sum() / nT:.2f} factors/post")
            K = merged
        else:
            K = cnt

    on = np.zeros((nT, N), dtype=int)
    for j in range(N):
        k = min(max(int(K[j]), 1), nT - 1)  # >=1 makes a zero-prediction label impossible
        on[:, j] = (T[:, j] >= np.sort(T[:, j])[-k]).astype(int)

    zero = [FACTOR_TAXONOMY[j] for j in range(N) if on[:, j].sum() == 0]
    if zero:
        raise SystemExit(f"IMPOSSIBLE: zero-prediction labels {zero}")

    fac = [[FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]]
           for i in range(nT)]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame({"row_id": tid,
                  "factors": [str(list(f)) for f in fac]}).to_csv(out,
                                                                  index=False)

    if args.match_counts:
        # How much actually moved? A single-variable test is only informative
        # if the variable moved a meaningful amount -- and only safe if it did
        # not move so much that something else changed too.
        cur = np.zeros((nT, N), dtype=int)
        for i, cell in enumerate(ch["factors"]):
            for x in parse_factors(cell):
                j = idx.get(str(x).strip().lower())
                if j is not None:
                    cur[i, j] = 1
        diff = int((on != cur).sum())
        posts = int((on != cur).any(axis=1).sum())
        print(f"\nCHANGE vs the champion column")
        print(f"  cells differing   {diff} of {nT * N} ({100 * diff / (nT * N):.2f}%)")
        print(f"  posts affected    {posts} of {nT} ({100 * posts / nT:.1f}%)")
        if args.release_labels:
            print(f"  per-label counts  match the champion EXCEPT the "
                  f"released labels (max abs diff "
                  f"{int(np.abs(on.sum(0) - cur.sum(0)).max())})")
        else:
            print(f"  per-label counts  identical by construction "
                  f"(max abs diff {int(np.abs(on.sum(0) - cur.sum(0)).max())})")
        if diff == 0:
            print("  NOTHING CHANGED -- the bag ranks identically to the "
                  "champion. Do not spend a slot.")

    print(f"\nwrote {out}")
    print(f"  {on.sum(1).mean():.2f} factors/post   "
          f"(gold {Y.sum(1).mean():.2f}, champion 3.44)")
    print(f"  posts with 0 factors: {int((on.sum(1) == 0).sum())}/{nT}")
    print(f"  labels with 0 predictions: 0  (floored by construction)")

    print(f"\n{'factor':<44}{'test %':>8}{'oof %':>8}{'gold %':>8}{'oofF1':>8}")
    print("-" * 76)
    for j in range(N):
        print(f"{FACTOR_TAXONOMY[j]:<44}{100 * on[:, j].mean():>8.1f}"
              f"{100 * rate[j]:>8.1f}{100 * prev[j]:>8.1f}{oof_f1[j]:>8.3f}")

    if args.match_counts:
        print("\nThe test %% column shows the CHAMPION's rates, which is the")
        print("point of --match-counts, so it deliberately diverges from oof %%.")
        print("Divergence here is not a tie-breaking bug -- it is the density")
        print("variable being held fixed on purpose.")
    else:
        print("\nThe test %% column should track the oof %% column closely -- that")
        print("is the whole point of transferring a rate. Any large gap means")
        print("the quantile landed on a tie in the probability vector.")
    print("\nThis is the FACTOR COLUMN ONLY. Put it on the champion with")
    print("src/swap_factor_column.py, which VERIFIES risk and evidence are")
    print("unchanged. (Not splice_factor_columns.py -- that merges two npz")
    print("probability matrices and does not take a submission CSV.)")


if __name__ == "__main__":
    main()

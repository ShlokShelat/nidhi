"""
make_factor_oof_splits.py

Build author-disjoint TRAIN / THRESH / EVAL splits for the factor OOF instrument.

WHAT THE DEPLOYED RECIPE DOES, and therefore what must be replicated:
  train_factors_nli.py trains on --train and uses --val for TWO fitting
  operations -- it picks the best EPOCH by tuned macro-F1 on --val, and it fits
  the 24 per-label THRESHOLDS on --val. Then predict_factors_nli.py averages the
  two runs' probabilities and, separately, averages the two runs' thresholds.
  Deployment is train_split (1263 posts) / val_split (372) / test (378).

So `--val` is a FITTING set, not a clean evaluation set. Any post used for
epoch selection or threshold fitting is contaminated for evaluation. This script
therefore produces THREE disjoint author groups per fold, not two.

  EVAL_k    the held-out fold. Never trained on, never fitted on. Scored.
  THRESH_k  epoch selection + threshold fitting. Disjoint from EVAL_k.
  TRAIN_k   gradient updates. Disjoint from both.

THE BUG THIS EXISTS TO PREVENT. scripts/factors_deploy_oof.sh used the mask
`own != f`, which keeps every post whose fold is not k -- including the posts the
checkpoint had trained on. It scored macro 0.8673 against an annotation ceiling
of roughly 0.80-0.83, and every threshold verdict computed on it was discarded.
Assertions below make that class of mistake impossible rather than unlikely.

THE COST, STATED PLAINLY. With K=5 the held-out fold takes 20% of the data, so
TRAIN_k gets about 1011 posts against deployment's 1263 -- 80% of the training
scale, the price of 4.3x the evaluation data. A uniform shortfall mostly cancels
in a PAIRED comparison, but it is not neutral for every arm: transfer-learning
gains shrink as training data grows, so an MNLI-pretrained arm will look
somewhat BETTER here than it will at full scale. Treat MNLI-encoder deltas from
this instrument as an upper bound. Raising --k narrows the gap at linear cost
(K=8 gives TRAIN about 1106).

Usage:
  python src/make_factor_oof_splits.py --k 5
  python src/make_factor_oof_splits.py --k 5 --reuse-folds artifacts/folds
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# deployment's train/val ratio: 1263 / (1263 + 372)
DEPLOY_TRAIN_FRAC = 1263.0 / 1635.0


def balanced_author_folds(df: pd.DataFrame, k: int, seed: int) -> dict:
    """Assign whole authors to k folds, greedily balancing POST counts.

    Balancing on posts rather than authors matters because author post counts
    are very uneven; an author-count split would give folds of unequal size and
    make per-fold numbers incomparable."""
    counts = df.groupby("anon_user_id").size()
    rng = np.random.default_rng(seed)
    # sort by size desc, break ties randomly but reproducibly
    jitter = rng.random(len(counts))
    order = sorted(range(len(counts)),
                   key=lambda i: (-counts.iloc[i], jitter[i]))
    load = np.zeros(k, dtype=int)
    assign = {}
    for i in order:
        f = int(np.argmin(load))
        assign[counts.index[i]] = f
        load[f] += int(counts.iloc[i])
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--outdir", default="artifacts/factor_folds")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--reuse-folds", default=None,
                    help="a directory of fold{k}_val.csv files (e.g. the risk "
                         "instrument's artifacts/folds) so both instruments "
                         "partition authors identically and their results are "
                         "directly comparable")
    args = ap.parse_args()

    df = pd.read_csv(args.labeled, keep_default_na=False)
    for c in ("row_id", "anon_user_id", "factors"):
        if c not in df.columns:
            raise SystemExit(f"{args.labeled} lacks column {c!r}")
    os.makedirs(args.outdir, exist_ok=True)

    # ---- fold assignment ----------------------------------------------------
    assign = None
    if args.reuse_folds:
        got = {}
        for f in range(args.k):
            for pat in (f"fold{f}_val.csv", f"fold_{f}_val.csv",
                        f"fold{f}.csv"):
                p = os.path.join(args.reuse_folds, pat)
                if os.path.exists(p):
                    ids = set(pd.read_csv(p, keep_default_na=False)["row_id"])
                    for a in df.loc[df.row_id.isin(ids), "anon_user_id"]:
                        got[a] = f
                    break
        if len(got) >= df.anon_user_id.nunique():
            assign = got
            print(f"reusing the author partition from {args.reuse_folds}")
        else:
            print(f"NOTE {args.reuse_folds} covered only {len(got)} of "
                  f"{df.anon_user_id.nunique()} authors -- building a fresh "
                  f"partition instead")
    if assign is None:
        assign = balanced_author_folds(df, args.k, args.seed)
        print(f"built a fresh balanced author partition (seed {args.seed})")

    df["_fold"] = df["anon_user_id"].map(assign)
    if df["_fold"].isna().any():
        raise SystemExit("some authors were not assigned to a fold")
    df["_fold"] = df["_fold"].astype(int)

    rng = np.random.default_rng(args.seed + 1)
    print(f"\n{'fold':>5}{'eval':>8}{'train':>8}{'thresh':>8}"
          f"{'ev.auth':>9}{'gold+/lab':>11}")
    print("-" * 52)

    for f in range(args.k):
        ev = df[df._fold == f]
        rest = df[df._fold != f]

        # split the remaining authors into TRAIN / THRESH at deployment's ratio
        ra = list(pd.unique(rest["anon_user_id"]))
        rng.shuffle(ra)
        target = DEPLOY_TRAIN_FRAC * len(rest)
        tr_auth, got = set(), 0
        sizes = rest.groupby("anon_user_id").size()
        for a in ra:
            if got >= target:
                break
            tr_auth.add(a)
            got += int(sizes[a])
        tr = rest[rest.anon_user_id.isin(tr_auth)]
        th = rest[~rest.anon_user_id.isin(tr_auth)]

        # ---- the assertions that make the old leak impossible ---------------
        A_ev = set(ev.anon_user_id)
        A_tr = set(tr.anon_user_id)
        A_th = set(th.anon_user_id)
        assert not (A_ev & A_tr), f"fold {f}: EVAL author leaked into TRAIN"
        assert not (A_ev & A_th), f"fold {f}: EVAL author leaked into THRESH"
        assert not (A_tr & A_th), f"fold {f}: TRAIN author leaked into THRESH"
        assert len(ev) + len(tr) + len(th) == len(df), f"fold {f}: lost rows"
        assert len(th) > 0, f"fold {f}: empty THRESH set"

        for name, part in (("eval", ev), ("train", tr), ("thresh", th)):
            part.drop(columns=["_fold"]).to_csv(
                os.path.join(args.outdir, f"f{f}_{name}.csv"), index=False)

        # TRAIN + THRESH, for the "use all the labelled data" arm. Deployment
        # trains on 1263 of 1635 posts; val_split's 372 only pick an epoch and
        # fit thresholds, never producing a gradient. This file is the fold-level
        # equivalent of reclaiming them: 1306 posts instead of 1015, still
        # author-disjoint from EVAL, so the comparison stays honest.
        # It is 80% of the 1635 a real submission would train on, the same ratio
        # as 1015/1263 today -- so any gain measured here UNDERSTATES the
        # deployed one.
        alld = pd.concat([tr, th]).drop(columns=["_fold"])
        assert not (set(alld.anon_user_id) & A_ev), \
            f"fold {f}: EVAL author leaked into the all-data train file"
        alld.to_csv(os.path.join(args.outdir, f"f{f}_alldata.csv"), index=False)

        # how many gold positives the rarest label has in this eval fold --
        # the direct driver of macro-F1 noise
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from scoring import FACTOR_TAXONOMY, parse_factors
        cnt = [sum(1 for x in ev["factors"] if t in parse_factors(x))
               for t in FACTOR_TAXONOMY]
        print(f"{f:>5}{len(ev):>8}{len(tr):>8}{len(th):>8}"
              f"{len(A_ev):>9}{f'{min(cnt)}-{max(cnt)}':>11}")

    # ---- coverage check ------------------------------------------------------
    seen = set()
    for f in range(args.k):
        ids = set(pd.read_csv(os.path.join(args.outdir, f"f{f}_eval.csv"),
                              keep_default_na=False)["row_id"])
        assert not (seen & ids), f"fold {f} eval overlaps an earlier fold"
        seen |= ids
    assert seen == set(df.row_id), "the eval folds do not cover every post"
    print(f"\nOK: {len(seen)} posts covered exactly once across "
          f"{args.k} eval folds, no author in two roles.")
    print(f"wrote {args.outdir}/f<k>_{{train,thresh,eval}}.csv")
    print("\nTRAIN is about 80% of deployment's 1263 posts. That is the price of")
    print("4.3x the evaluation data, it cancels in PAIRED deltas, and it makes")
    print("MNLI-pretrained arms look somewhat better than they will at scale.")


if __name__ == "__main__":
    main()

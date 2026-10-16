"""
fold_runs_to_bag.py

Publish per-fold `val_probs.npz` files into the bag's naming, with checks.

WHY THIS IS NOT JUST `cp`. `scripts/factors_qwen_nli.sh` writes
`runs/factors_<tag>_f{f}/val_probs.npz`, while the bag tooling
(`factor_rank_average.py`, `factor_bag_ship.py`) reads
`artifacts/factor_runs/<tag>_cv_oof_f{f}.npz`. The payload is already the right
shape -- keys `probs`, `thresholds`, `row_id` -- so the risk here is not the
format, it is publishing a set of folds that does not actually tile the corpus.

A silently misaligned or incomplete arm is the one failure this pipeline exists
to prevent: `factor_rank_average.py` reindexes members onto a common row order
and would happily compare an arm covering 983 posts against arms covering 1635
if the row sets merely overlapped. So this refuses unless:

  - every requested fold is present (a partial arm is a DIFFERENT model)
  - the folds are pairwise DISJOINT (no post scored twice)
  - the union matches the reference corpus exactly (default 1635 OOF posts)

Usage:
  python src/fold_runs_to_bag.py --tag qnli
  python src/fold_runs_to_bag.py --tag qnli --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src", default="runs/factors_{tag}_f{f}/val_probs.npz")
    ap.add_argument("--dst", default="artifacts/factor_runs/{tag}_cv_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    srcs = [args.src.format(tag=args.tag, f=f) for f in range(args.k)]
    missing = [p for p in srcs if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {args.k} folds missing:\n  "
            + "\n  ".join(missing)
            + "\n\nA partial arm is a DIFFERENT model from the one measured. "
              "Finish the remaining folds before publishing.")

    seen, total = {}, 0
    for f, p in enumerate(srcs):
        z = np.load(p, allow_pickle=True)
        need = {"probs", "thresholds", "row_id"}
        if not need.issubset(set(z.files)):
            raise SystemExit(f"{p}: keys {z.files}, need {sorted(need)}")
        ids = [str(x) for x in z["row_id"]]
        if len(ids) != z["probs"].shape[0]:
            raise SystemExit(f"{p}: {len(ids)} row_ids vs "
                             f"{z['probs'].shape[0]} prob rows")
        dup = {r for r in ids if r in seen}
        if dup:
            ex = sorted(dup)[:3]
            raise SystemExit(f"fold {f} overlaps fold(s) "
                             f"{sorted({seen[r] for r in dup})} on "
                             f"{len(dup)} posts (e.g. {ex}) -- folds must be "
                             f"DISJOINT or posts get scored twice")
        for r in ids:
            seen[r] = f
        total += len(ids)
        pr = z["probs"]
        print(f"  fold {f}  {pr.shape}  sd={pr.std():.4f}  "
              f"thr[{z['thresholds'].min():.2f},{z['thresholds'].max():.2f}]")

    ref = pd.read_csv(args.labeled, keep_default_na=False)
    ref = {str(x) for x in ref["row_id"]}
    print(f"\nunion {total} posts across {args.k} folds; "
          f"reference corpus {len(ref)}")
    extra, absent = set(seen) - ref, ref - set(seen)
    if extra or absent:
        raise SystemExit(
            f"corpus mismatch: {len(extra)} posts not in {args.labeled}, "
            f"{len(absent)} reference posts uncovered. The bag compares arms "
            f"post-for-post, so this arm is not comparable.")
    print("folds are disjoint and tile the corpus exactly")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    os.makedirs("artifacts/factor_runs", exist_ok=True)
    for f, p in enumerate(srcs):
        d = args.dst.format(tag=args.tag, f=f)
        shutil.copyfile(p, d)
        print(f"  wrote {d}")
    print(f"\n{args.tag} is now a bag arm. Next:")
    print(f"  python src/factor_rank_average.py --runs mnli dmnli cmine "
          f"prot d4 {args.tag}")
    print("Judge it under RATEMATCH -- the rank-only policy. Qwen's recorded")
    print("defect was thresholding, not ranking, and ratematch ignores")
    print("calibration entirely. An AP win that does not survive there closes")
    print("the large-model lane.")


if __name__ == "__main__":
    main()

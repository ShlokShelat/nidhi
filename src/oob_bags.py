"""
oob_bags.py

MAKE THE INSTRUMENT MATCH DEPLOYMENT.

THE DEFECT. Deployment scores each TEST post with an ENSEMBLE of 5 fold
models. The OOF instrument scores each post with the SINGLE model that held
it out. With standard k-fold there is no way around this: post i is excluded
from exactly one model, so a clean multi-model ensemble over post i cannot be
built.

That asymmetry is not cosmetic: submissions that look good on the fold-model
instrument can still underperform against the deployed ensemble, because
components measured this way read systematically lower than what the
deployed multi-model ensemble actually achieves.

THE FIX -- OUT-OF-BAG BAGGING. Instead of 5 disjoint folds, draw B bags, each
a random `--frac` of AUTHORS. Train one model per bag. For post i, average only
the models whose bag EXCLUDED i's author. With frac=0.8 each post is excluded
from ~0.2*B models, so B=25 gives every post a clean ~5-model ensemble --
structurally identical to what deployment does.

This removes the asymmetry rather than correcting for it. The models are
trained on the same 80% fraction as the fold models, so the only thing that
changes is that the OOF prediction is now an ENSEMBLE, like the shipped one.

AUTHORS, NOT POSTS. Bags are drawn over anon_user_id, because posts by one
author are near-duplicates and a post-level bag would leak.

Writes, for each bag b:
    artifacts/oob/train_b{b}.csv   the ~frac of authors in the bag
    artifacts/oob/val_b{b}.csv     the authors it excluded
    artifacts/oob/bags.json        author lists + the coverage report

Usage:
  python src/oob_bags.py --bags 25 --frac 0.8
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--out", default="artifacts/oob")
    a_.add_argument("--bags", type=int, default=25)
    a_.add_argument("--frac", type=float, default=0.8,
                    help="fraction of AUTHORS per bag. 0.8 matches the "
                         "training fraction of a 5-fold model, so the only "
                         "difference from deployment is the ensembling.")
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    d = pd.read_csv(args.labeled, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    auth = d["anon_user_id"].astype(str)
    ua = pd.unique(auth)
    n_keep = int(round(args.frac * len(ua)))
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    bags, excl_count = {}, np.zeros(len(d), dtype=int)
    for b in range(args.bags):
        keep = set(rng.choice(ua, size=n_keep, replace=False).tolist())
        m = auth.isin(keep).to_numpy()
        d[m].to_csv(f"{args.out}/train_b{b}.csv", index=False)
        d[~m].to_csv(f"{args.out}/val_b{b}.csv", index=False)
        excl_count += (~m).astype(int)
        bags[b] = sorted(keep)

    print(f"{len(d)} posts / {len(ua)} authors, {args.bags} bags at "
          f"{args.frac:.0%} of authors")
    print(f"clean models per post: min {excl_count.min()}  "
          f"median {int(np.median(excl_count))}  mean {excl_count.mean():.1f}  "
          f"max {excl_count.max()}")
    zero = int((excl_count == 0).sum())
    print(f"posts with ZERO clean models: {zero}"
          + ("" if not zero else "  <- these have no honest prediction and "
                                 "must be dropped from any score"))
    print(f"deployment ensembles 5 models, so a median near 5 is the target.")
    if excl_count.mean() < 3:
        print("  WARNING: too few clean models per post. Raise --bags.")

    with open(f"{args.out}/bags.json", "w") as f:
        json.dump({"frac": args.frac, "seed": args.seed,
                   "bags": {str(k): v for k, v in bags.items()},
                   "clean_models_per_post": excl_count.tolist(),
                   "row_id": d["row_id"].tolist()}, f)
    print(f"\nwrote {args.out}/bags.json and {args.bags} train/val pairs")


if __name__ == "__main__":
    main()

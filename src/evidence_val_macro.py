"""
evidence_val_macro.py

Score evidence CSVs on val, MACRO, with a paired author bootstrap.

WHY THIS IS NEEDED BEFORE UPLOADING THE CONDITIONED ENSEMBLE. Everything
currently known about --risk-cond on the eight shipped taggers is either

  - an ORACLE number: every conditioned run's val phrase-F1 was measured with
    GOLD risk levels (the trainer prints exactly that warning), so those
    matched-pair gains are ceilings, not deliveries; or
  - a sanity check: 447 spans vs the champion's 456, empty-evidence 137 vs 138,
    which says the pipeline is healthy and says nothing about score.

The one measured shippable figure came from a 5-fold study of ONE tagger per
fold. The champion runs EIGHT with cluster voting, and voting
already discards spans the taggers disagree about -- the same errors
conditioning discards. The gains may not add. They may also compound. Unknown
until measured, and the val candidate dumps for both versions are already on
disk, so measuring costs minutes of CPU.

ALSO: train_evidence.py scores with phrase_f1() at its DEFAULT, which is MICRO.
The official metric is MACRO over posts. Every per-tagger number in the training
logs is therefore in the wrong average as well as on the wrong risk levels.

HONESTY NOTES
  - Both arms chose their per-run min_conf on THIS val split, so both are
    optimistic. The comparison is fair; the absolute level is not.
  - The bootstrap resamples AUTHORS, not posts: posts from one author are not
    independent, and a post-level interval would be far too narrow.

Usage:
  python src/evidence_val_macro.py \
      --pred base=artifacts/evvote_val_base.csv \
             cond=artifacts/evvote_val_cond.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True,
                    help="name=path.csv (row_id,evidence). First is baseline.")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv",
                    help="source of anon_user_id for the author bootstrap")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from data_audit import load
    va = load(args.val)
    va["row_id"] = va["row_id"].astype(str)
    rows = list(va["row_id"])
    gold = [parse_evidence(e) for e in va["evidence"]]
    n_empty = sum(1 for g in gold if not g)
    print(f"{len(rows)} val posts, {n_empty} with empty gold "
          f"({n_empty / len(rows):.1%}) -- those are free 1.0s under MACRO")

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    amap = dict(zip(lab["row_id"], lab["anon_user_id"].astype(str)))
    author = np.array([amap.get(r, r) for r in rows])
    print(f"{len(set(author))} authors")

    arms = {}
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit(f"--pred wants name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        d = pd.read_csv(path, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        m = dict(zip(d["row_id"], d["evidence"]))
        absent = [r for r in rows if r not in m]
        if absent:
            raise SystemExit(f"{path} lacks {len(absent)} val rows")
        arms[name] = [parse_evidence(m[r]) for r in rows]

    print(f"\n{'arm':<12}{'macro':>9}{'micro':>9}{'P':>8}{'R':>8}{'spans':>8}")
    print("-" * 54)
    per_post = {}
    for name, pred in arms.items():
        ma = phrase_f1(gold, pred, average="macro")
        mi = phrase_f1(gold, pred)
        per_post[name] = np.array([p["f1"] for p in ma.per_post])
        print(f"{name:<12}{ma.f1:>9.4f}{mi.f1:>9.4f}{mi.precision:>8.3f}"
              f"{mi.recall:>8.3f}{sum(len(p) for p in pred):>8}")

    names = list(arms)
    base = names[0]
    if len(names) < 2:
        return

    rng = np.random.default_rng(args.seed)
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}
    print("\n" + "=" * 64)
    print(f"PAIRED AUTHOR BOOTSTRAP vs {base}   ({args.boot} resamples)")
    print("=" * 64)
    for name in names[1:]:
        diff = per_post[name] - per_post[base]
        obs = float(diff.mean())
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            draws.append(float(diff[sel].mean()))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        pwin = float(np.mean(np.array(draws) > 0))
        better = int((diff > 0).sum())
        worse = int((diff < 0).sum())
        print(f"{name} - {base}:  {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"    composite {0.3 * obs:+.4f}   "
              f"P(better) {pwin:.3f}   posts better/worse {better}/{worse}")
        if lo > 0:
            print("    CI excludes zero -- a real gain on this split")
        else:
            print("    CI includes zero -- cannot distinguish from noise")
    print("=" * 64)
    print("\nA change worth a submission slot should have the CI clear of zero.")
    print("Changes whose CI does not clear zero have consistently underperformed")
    print("their predicted gain once submitted.")


if __name__ == "__main__":
    main()

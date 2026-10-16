"""
factor_inventory.py

Map what is on disk onto the six factor arms, before building any submission.

WHY THIS EXISTS. The member-count trend measured by `factor_rank_average.py`
was computed on `*_cv_oof_f*.npz`, where each post is scored by ONE fold model
per arm. A test dump may average FIVE fold models per arm. That regime shift is
not cosmetic: averaging more checkpoints narrows the probability distribution,
and in the rare-label tail a slightly-too-high fixed cutoff means predicting
NOTHING at all. A single zero-fire label can cost more macro-F1 than an entire
change was worth.

So before shipping anything, three things must be checked:
  1. a TEST dump exists for every arm
  2. all dumps cover the same rows, in the same order
  3. how much narrower the test distribution is than the OOF one

Point 3 decides the threshold policy. `ratematch` places the cutoff at a
QUANTILE, so it is invariant to any monotone narrowing and structurally cannot
emit a zero-prediction label. `refit` transfers an absolute probability value
and is exactly the policy that lost.

Usage:
  python src/factor_inventory.py
  python src/factor_inventory.py --arms mnli dmnli cmine prot d4 incmnli
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

D_DEFAULT = "artifacts/factor_runs"


def info(p):
    try:
        z = np.load(p, allow_pickle=True)
        pr = z["probs"]
        n = len(z["row_id"]) if "row_id" in z.files else pr.shape[0]
        th = "thr" if "thresholds" in z.files else "NO-thr"
        return f"{str(pr.shape):<14} rows={n:<6} sd={pr.std():.4f}  {th}"
    except Exception as e:                                    # noqa: BLE001
        return f"UNREADABLE {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*",
                    default=["mnli", "dmnli", "cmine", "prot", "d4", "incmnli"])
    ap.add_argument("--dir", default=D_DEFAULT)
    args = ap.parse_args()
    D = args.dir

    for kind in ("test", "val"):
        fs = sorted(glob.glob(f"{D}/*_{kind}.npz"))
        print(f"=== ALL *_{kind}.npz  ({len(fs)}) ===")
        for p in fs:
            print(f"  {os.path.basename(p):<40} {info(p)}")
        print()

    print("=== ARM -> TEST dump resolution ===")
    resolved = {}
    for a in args.arms:
        cands = [f"{D}/{a}_cv_test.npz", f"{D}/{a}_test.npz",
                 f"{D}/nli_ensemble_{a}_test.npz"]
        hit = next((c for c in cands if os.path.exists(c)), None)
        parts = sorted(glob.glob(f"{D}/{a}_test_part_f*.npz"))
        if hit:
            resolved[a] = hit
        print(f"  {a:<10} "
              f"{('OK  ' + os.path.basename(hit)) if hit else 'MISSING'}"
              f"{('   (+%d fold parts)' % len(parts)) if parts else ''}")

    print(f"\nresolved {len(resolved)} of {len(args.arms)} arms")

    if len(resolved) > 1:
        print("\n=== row alignment across resolved arms ===")
        ref = None
        for a, p in resolved.items():
            z = np.load(p, allow_pickle=True)
            ids = [str(x) for x in z["row_id"]] if "row_id" in z.files else None
            if ids is None:
                print(f"  {a:<10} NO row_id -- cannot verify alignment")
                continue
            if ref is None:
                ref, refa = ids, a
                print(f"  {a:<10} reference, {len(ids)} rows")
            elif ids == ref:
                print(f"  {a:<10} identical order to {refa}")
            elif set(ids) == set(ref):
                print(f"  {a:<10} same rows, DIFFERENT ORDER -- must reindex")
            else:
                print(f"  {a:<10} DIFFERENT ROW SET -- cannot bag with {refa}")

    print("\n=== distribution narrowing: OOF (1 ckpt/arm) vs TEST ===")
    print("ratio < 1 means the test dump is narrower, i.e. it averages more")
    print("checkpoints. That narrowing can zero out a rare label under a fixed")
    print("cutoff. Use ratematch, not refit.")
    for a in args.arms:
        o = f"{D}/{a}_cv_oof_f0.npz"
        t = resolved.get(a)
        if os.path.exists(o) and t:
            so = float(np.load(o, allow_pickle=True)["probs"].std())
            st = float(np.load(t, allow_pickle=True)["probs"].std())
            flag = "  <- narrower" if st < 0.95 * so else ""
            print(f"  {a:<10} oof sd {so:.4f}   test sd {st:.4f}   "
                  f"ratio {st / so:.3f}{flag}")


if __name__ == "__main__":
    main()

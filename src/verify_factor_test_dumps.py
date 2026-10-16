#!/usr/bin/env python
"""Pre-flight for any factor-column rebuild. Catches two silent hazards in
`factor_threshold_stabilise.py --build-test` that nothing else checks.

HAZARD 1 -- SHADOWING. resolve_test() (factor_threshold_stabilise.py:285) picks
the FIRST path that exists, in this order:

    {arm}_cv_test.npz  ->  {arm}_test.npz  ->  nli_ensemble_{arm}_test.npz

A job wrote fresh `{arm}_cv_test.npz` files for nine parents,
including `mnli` and `dmnli` -- both SHIPPED BAG8 members that were almost
certainly resolving to `nli_ensemble_{arm}_test.npz` before, i.e. the THIRD
name. A new file at the FIRST name silently changes which artefact a shipped
arm contributes, with no diff and no warning. If the two disagree, the next
rebuild is not the single-variable change it claims to be.

HAZARD 2 -- ROW ORDER. The build stacks every arm's probs into one array and
keeps only the LAST arm's row_id list (line 292-301); it never checks that the
arms agree on row order. A dump written in a different order is stacked
misaligned, so every post gets another post's factors. The provenance gate
tolerates 2% of cells differing, so a partial misalignment can slip through.

Neither hazard is hypothetical: submissions have lost score to a
measured-one-artefact / shipped-another mismatch of exactly this shape.

Read-only. Prints a verdict and exits non-zero if anything needs a decision.
"""
import argparse
import os
import sys

import numpy as np

CANDIDATE_ORDER = ("{d}/{a}_cv_test.npz",
                   "{d}/{a}_test.npz",
                   "{d}/nli_ensemble_{a}_test.npz")


def existing_candidates(d, a):
    """Every candidate path that exists, in resolve_test's own order."""
    return [c.format(d=d, a=a) for c in CANDIDATE_ORDER
            if os.path.exists(c.format(d=d, a=a))]


def load(p):
    z = np.load(p, allow_pickle=True)
    return z["probs"].astype(float), [str(x) for x in z["row_id"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="the arm set you are about to build with")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max |prob diff| below which two dumps are identical")
    args = ap.parse_args()

    problems = []

    # ---- HAZARD 1: shadowing ------------------------------------------------
    print("=" * 74)
    print("HAZARD 1 -- does any arm have MORE THAN ONE test dump on disk?")
    print("=" * 74)
    chosen = {}
    for a in args.arms:
        cands = existing_candidates(args.dir, a)
        if not cands:
            print(f"  {a:<12} NO TEST DUMP -- the build will refuse")
            problems.append(f"{a}: no test dump")
            continue
        chosen[a] = cands[0]
        tag = os.path.basename(cands[0])
        if len(cands) == 1:
            print(f"  {a:<12} 1 dump   -> {tag}")
            continue

        # More than one exists. resolve_test takes cands[0]; check the others
        # agree with it, because if they do not, which one ships is decided by
        # filename precedence rather than by any deliberate choice.
        print(f"  {a:<12} {len(cands)} dumps -> USES {tag}")
        base_p, base_ids = load(cands[0])
        for other in cands[1:]:
            op, oids = load(other)
            name = os.path.basename(other)
            if base_p.shape != op.shape:
                print(f"      vs {name:<34} SHAPE DIFFERS "
                      f"{base_p.shape} vs {op.shape}  <-- SHADOWED")
                problems.append(f"{a}: {tag} shadows {name} (shape)")
            elif base_ids != oids:
                print(f"      vs {name:<34} ROW ORDER DIFFERS  <-- SHADOWED")
                problems.append(f"{a}: {tag} shadows {name} (row order)")
            else:
                md = float(np.abs(base_p - op).max())
                if md <= args.tol:
                    print(f"      vs {name:<34} identical (max diff {md:.2e})")
                else:
                    print(f"      vs {name:<34} MAX DIFF {md:.6f}"
                          f"  <-- SHADOWED, these are different models")
                    problems.append(f"{a}: {tag} shadows {name} "
                                    f"(max diff {md:.6f})")

    # ---- HAZARD 2: row order ------------------------------------------------
    print()
    print("=" * 74)
    print("HAZARD 2 -- do all arms agree on row_id ORDER?")
    print("=" * 74)
    orders = {}
    for a, p in chosen.items():
        _, ids = load(p)
        orders[a] = ids
    if not orders:
        print("  nothing to compare")
    else:
        ref_arm, ref_ids = next(iter(orders.items()))
        print(f"  reference: {ref_arm}  ({len(ref_ids)} rows)")
        all_ok = True
        for a, ids in orders.items():
            if ids == ref_ids:
                continue
            all_ok = False
            if sorted(ids) == sorted(ref_ids):
                nmis = sum(1 for x, y in zip(ids, ref_ids) if x != y)
                print(f"  {a:<12} SAME ROWS, DIFFERENT ORDER "
                      f"({nmis} positions) <-- probs would be MISALIGNED")
                problems.append(f"{a}: row order differs from {ref_arm}")
            else:
                print(f"  {a:<12} DIFFERENT ROW SET "
                      f"({len(ids)} vs {len(ref_ids)} rows)")
                problems.append(f"{a}: row set differs from {ref_arm}")
        if all_ok:
            print(f"  all {len(orders)} arms share one row order -- stacking "
                  f"is aligned")

    # ---- verdict ------------------------------------------------------------
    print()
    print("=" * 74)
    if problems:
        print(f"VERDICT: {len(problems)} PROBLEM(S) -- do not rebuild yet")
        for p in problems:
            print(f"  - {p}")
        print()
        print("For a SHADOWED arm the fix is a decision, not a delete: pick the")
        print("dump that the CURRENT champion was actually built from and make")
        print("resolve_test reach it (rename the newcomer aside). Deleting the")
        print("wrong one silently changes a shipped column.")
        print("=" * 74)
        return 1
    print("VERDICT: CLEAN -- no shadowing, one row order. Safe to rebuild.")
    print("Still run the provenance gate (0 of N cells differ on the")
    print("PREVIOUS arm set) before trusting any new column.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())

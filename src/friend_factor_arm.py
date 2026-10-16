"""
friend_factor_arm.py

THE FRIEND HAS A COMPLETE FACTOR ARM AND IT HAS NEVER BEEN IN THE POOL.

`src/friend_pipeline` is the MentalRoBERTa tri-head system. Its
`outputs/oof_export.csv` carries **all 24 taxonomy labels for 1590 OOF posts**
and `outputs/factor_probs.csv` the same 24 for the 378 test posts. Neither has
ever been converted to the `{arm}_cv_oof_f{f}.npz` / `{arm}_cv_test.npz` shape,
and `factor_member_forward.py` discovers arms by globbing exactly that pattern
-- so every membership audit run so far was blind to it.

WHY IT IS WORTH A LOOK EVEN THOUGH THE POOL IS "CLOSED". The pool is closed
among the arms tried so far: adding `prot5d`, `synapi` or `synapi+hyb` to
BAG8+ltail all scored WORSE than BAG8+ltail alone, and every single-member
drop was negative too. But every one of those candidates is a DeBERTa-v3-large
NLI cross-encoder trained on the same folds -- variations on one recipe. The
friend is:

  * a different backbone -- `mental/mental-roberta-base`, 125M and
    DOMAIN-PRETRAINED, against the 435M general encoders used here;
  * a different formulation -- a multi-task TriHead (risk + factors + evidence
    tokens on ONE backbone) rather than 24 independent entailment questions;
  * independently strong -- the same model's RISK head measurably helps the
    deployed blend, from a model a quarter the size of the encoders used
    here.

`ltail`'s gain comes from decorrelation, not from its long-tail
mechanism (a protective-factor specialist gained nothing), and
decorrelation is exactly what a different architecture buys.

**THE CONSTRUCTION MATCHES, WHICH IS THE PART THAT USUALLY GOES WRONG.**
Verified by reading the teammate's code:
  * `friend_export_oof.py` concatenates `oof.npy`, i.e. every post scored by
    the ONE fold model that held it out;
  * `stage4_predict.py` averages ALL FIVE fold models over the test set.
That is precisely `predict_factors_nli.py`'s convention -- held-out fold for
OOF, 5-fold bag for test -- and matched construction is the single property
those two successful changes share.

CAVEAT, STATED PLAINLY. The friend's folds are ITS OWN user-grouped partition,
not ours. Each row's OOF probability is honest under the friend's split (the
scoring model never saw that post), which is what matters for pooling. But the
fold FILE a row lands in here is assigned by OUR `artifacts/folds/val_f*.csv`
purely so `load_oof` can find it; do not read the per-file grouping as
meaningful. Rows the friend lacks are reported and dropped rather than filled.

Usage (CPU, seconds):
  python src/friend_factor_arm.py \
      --oof  /path/to/pipeline/outputs/oof_export.csv \
      --test /path/to/pipeline/outputs/factor_probs.csv
  python src/factor_member_forward.py --extra ltail --splits 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)


def matrix_from(df: pd.DataFrame, rows: list[str]) -> np.ndarray:
    """24-column probability matrix in TAXONOMY ORDER, indexed by row_id.

    Column order is the single easiest thing to get wrong here: every consumer
    assumes FACTOR_TAXONOMY order, and the friend's CSV is in its own order.
    Matching is case-insensitive on the stripped name and every label must
    resolve or the run refuses -- a silently mis-ordered column would look
    like a plausible arm and score like noise.
    """
    look = {str(c).strip().lower(): c for c in df.columns}
    cols = []
    for t in FACTOR_TAXONOMY:
        c = look.get(t.strip().lower())
        if c is None:
            raise SystemExit(f"friend export has no column for {t!r}; "
                             f"has {list(df.columns)[:8]}...")
        cols.append(c)
    d = df.set_index(df["row_id"].astype(str))
    miss = [r for r in rows if r not in d.index]
    if miss:
        raise SystemExit(f"friend export lacks {len(miss)} of {len(rows)} "
                         f"rows (e.g. {miss[:5]})")
    return d.loc[rows, cols].to_numpy(float)


def matrix_or_abstain(df, rows, fallback):
    """Same, but rows the friend lacks take the REFERENCE POOL's own value.

    The friend covers 1590 of our 1635 labelled posts. Emitting only the rows
    it has produces an arm with a different length, and every pooling helper
    concatenates arms into one (n_arms, n_posts, 24) block -- so it simply
    cannot be pooled (that is the ValueError this replaced).

    The two ways to fill are not equally honest. A constant 0.5 maps to logit
    0, which drags the logit-mean toward zero on those 45 rows and invents an
    opinion the friend never had. Filling with the REFERENCE POOL's own
    probability makes the friend a no-op exactly where it abstains: the pooled
    logit-mean on those rows is then identical whether or not the friend is a
    member. That is what "this arm has no view here" should mean, and it keeps
    the 2.8% of rows from silently deciding the comparison.
    """
    look = {str(c).strip().lower(): c for c in df.columns}
    cols = [look[t.strip().lower()] for t in FACTOR_TAXONOMY]
    d = df.set_index(df["row_id"].astype(str))
    out = np.array(fallback, dtype=float, copy=True)
    have = [i for i, r in enumerate(rows) if r in d.index]
    idx = [rows[i] for i in have]
    out[have] = d.loc[idx, cols].to_numpy(float)
    return out, len(rows) - len(have)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", required=True,
                    help="pipeline/outputs/oof_export.csv (1590 rows)")
    a_.add_argument("--test", required=True,
                    help="pipeline/outputs/factor_probs.csv (378 rows)")
    a_.add_argument("--folds", default="artifacts/folds/val_f{f}.csv")
    a_.add_argument("--test-csv", default="data/test.csv")
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--tag", default="friendfac")
    a_.add_argument("--k", type=int, default=5)
    args = a_.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    oof = pd.read_csv(args.oof, keep_default_na=False)
    tst = pd.read_csv(args.test, keep_default_na=False)
    print(f"friend OOF {len(oof)} rows, TEST {len(tst)} rows")

    # Reference pool, used only to fill rows the friend abstains on. BAG8 is
    # the standing pool every audit compares against, so filling from it makes
    # the friend exactly neutral there.
    from factor_structure import BAG8, load_oof, logit_mean
    rid_ref, S_ref = load_oof(args.dir, list(BAG8), args.k)
    ref = {r: i for i, r in enumerate(rid_ref)}
    REF = logit_mean(S_ref)
    print(f"reference pool BAG8: {REF.shape} rows for abstain-fill")

    total = n_abstain = 0
    for f in range(args.k):
        p = args.folds.format(f=f)
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing -- fold row_ids are needed only to "
                             f"place rows into the files load_oof expects")
        v = pd.read_csv(p, keep_default_na=False)
        rows = [str(x) for x in v["row_id"]]
        fb = np.stack([REF[ref[r]] if r in ref else np.full(N, 0.5)
                       for r in rows], axis=0)
        P, nmiss = matrix_or_abstain(oof, rows, fb)
        n_abstain += nmiss
        if nmiss:
            print(f"  fold {f}: friend covers {len(rows) - nmiss}/{len(rows)}"
                  f" -- {nmiss} filled with the BAG8 pool value (no-op)")
        dst = f"{args.dir}/{args.tag}_cv_oof_f{f}.npz"
        np.savez(dst, probs=P,
                 row_id=np.array(rows, dtype=object),
                 thresholds=np.full(N, 0.5))
        total += len(rows)
        print(f"  wrote {dst}  {P.shape}")

    te = pd.read_csv(args.test_csv, keep_default_na=False)
    trows = [str(x) for x in te["row_id"]]
    T = matrix_from(tst, trows)
    dst = f"{args.dir}/{args.tag}_cv_test.npz"
    np.savez(dst, probs=T, row_id=np.array(trows, dtype=object),
             thresholds=np.full(N, 0.5))
    print(f"  wrote {dst}  {T.shape}")

    print(f"\n{total} OOF rows placed ({n_abstain} abstained -> BAG8 value, "
          f"{n_abstain / max(total, 1):.1%}). Discoverable as `{args.tag}`.")
    if n_abstain > 0.05 * total:
        print("  WARNING: over 5% abstained. The arm is then partly a copy of "
              "BAG8 on those rows, which BIASES it toward looking harmless "
              "rather than helpful -- read a null result with that in mind.")
    print("  Judge it exactly like any other candidate -- nothing about its "
          "provenance earns it a pass:")
    print("    python src/factor_member_forward.py --extra ltail --splits 200")
    print("    python src/factor_member_forward.py --extra ltail,"
          f"{args.tag} --splits 200")
    print("  The bar is the champion's own set, BAG8+ltail. Every added arm "
          "pays a fixed per-arm dilution tax, so a candidate has to clear "
          "that tax before it clears zero.")
    print("  The thresholds written here are placeholder 0.5 -- every consumer "
          "refits them, and factor_threshold_stabilise.py must be run with "
          "--use tfit, never the default tboth.")


if __name__ == "__main__":
    main()

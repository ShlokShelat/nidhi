"""
friend_coverage_check.py

DOES THE FRIEND'S RISK FILE ACTUALLY COVER EVERY TEST ROW?

`ensemble_risk.py` blends our seed-ensemble with the teammate's probabilities
at `--weight-ours`. The cold-run drill flagged:

    s1_ensembled  DEGRADED  friend member missing on 18 rows
                            (silent per-row fallback in ensemble_risk.py)

On the drill that is expected -- the friend's probs are test-only and the drill
scores val rows. But the fallback is SILENT, so if any real test row is
uncovered it is blended at an effective w_ours = 1.0 while the log still says
0.40, and every conclusion drawn about the weight is wrong for those rows.

This is not a hypothetical failure mode. champion_oof_twin silently dropped the
friend on 35% of the blend and nobody noticed until it had cost a submission.

Exits 1 if coverage is incomplete or the probabilities are malformed.
"""
import argparse
import sys

import pandas as pd

# the 12 rows whose label moves at w_ours=0.40 (job 36555)
CHANGED_040 = ["P00008", "P00009", "P00010", "P00011", "P00012", "P00708",
               "P01058", "P01479", "P01653", "P02280", "P02440", "P02467"]
CROSSERS_040 = ["P00008", "P00009", "P00010", "P00011", "P00012"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--friend", default="artifacts/friend_risk_probs.csv")
    args = ap.parse_args()

    t = pd.read_csv(args.test)
    f = pd.read_csv(args.friend)
    ti = set(t.row_id.astype(str))
    fi = set(f.row_id.astype(str))
    missing = sorted(ti - fi)

    print(f"test rows            {len(ti)}")
    print(f"friend rows          {len(fi)}")
    print(f"covered              {len(ti & fi)}")
    print(f"MISSING from friend  {len(missing)}")
    if missing:
        print(f"  {missing[:25]}")
        if len(missing) > 25:
            print(f"  ... and {len(missing) - 25} more")

    # ensemble_risk.py:60 -- FRIEND_COLS, the official capitalized names,
    # with no p_ prefix. It reads them via fp.loc[rid, FRIEND_COLS].
    FRIEND_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]
    pc = [c for c in FRIEND_COLS if c in f.columns]
    print(f"\nprob columns         {pc}")
    if len(pc) != 4:
        print(f"  EXPECTED {FRIEND_COLS}")
        print(f"  GOT      {list(f.columns)}")
    bad = 0
    if len(pc) == 4:
        nan = int(f[pc].isna().any(axis=1).sum())
        off = int((f[pc].sum(axis=1).sub(1.0).abs() > 0.01).sum())
        print(f"rows with NaN        {nan}")
        print(f"rows not summing 1   {off}")
        bad = nan + off
    else:
        print("  cannot validate the distribution without all four columns")
        bad = 1

    # duplicate row_ids would make the merge ambiguous
    dup = int(f.row_id.astype(str).duplicated().sum())
    print(f"duplicate row_ids    {dup}")

    print(f"\nof the 12 rows that move at w_ours=0.40, friend covers "
          f"{sum(r in fi for r in CHANGED_040)} of 12")
    unc = [r for r in CHANGED_040 if r not in fi]
    if unc:
        print(f"  UNCOVERED: {unc}")
        print("  -> those labels did NOT move because of the weight; they")
        print("     moved for some other reason. Re-derive before shipping A.")
    print(f"of the 5 indicator crossers, friend covers "
          f"{sum(r in fi for r in CROSSERS_040)} of 5")

    ok = not missing and not bad and not dup and not unc
    print("\n" + ("PASS -- the weight applies uniformly; wsplit's read holds"
                  if ok else
                  "FAIL -- the blend is not what the logs claim"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

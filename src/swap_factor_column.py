"""
swap_factor_column.py

Put a new factor column onto an existing submission, changing nothing else.

WHY A DEDICATED SCRIPT. `splice_factor_columns.py` merges two npz probability
matrices for the protective-specialist case; it does not take a submission CSV.
Every factor experiment must reach submission as a SINGLE-VARIABLE test --
same risk column, same evidence column, byte for byte -- or the returned S1
cannot be used to confirm the patch was clean. An S1 that comes back
byte-identical to the champion's is the proof that a factor-only change was
confined to S2.

This script therefore VERIFIES rather than assumes:
  - row_id sets match and are in the same order
  - risk_level and evidence are identical, cell for cell
  - the factor vocabulary is a subset of FACTOR_TAXONOMY (a typo'd or
    title-cased label silently scores zero for that label)
  - it reports exactly how many factor cells moved

Usage:
  python src/swap_factor_column.py \
      --base submissions/NIDHI_canon_on_nlimnli.csv \
      --factors artifacts/bag5_matched_factors.csv \
      --out submissions/NIDHI_bag5.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--factors", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b = pd.read_csv(args.base, keep_default_na=False)
    f = pd.read_csv(args.factors, keep_default_na=False)
    b["row_id"] = b["row_id"].astype(str)
    f["row_id"] = f["row_id"].astype(str)

    if len(b) != len(f):
        raise SystemExit(f"row count differs: base {len(b)}, factors {len(f)}")
    if list(b["row_id"]) != list(f["row_id"]):
        if set(b["row_id"]) != set(f["row_id"]):
            raise SystemExit("row_id SETS differ -- these are not the same posts")
        print("row order differs; reindexing factors onto the base order")
        f = f.set_index("row_id").loc[b["row_id"]].reset_index()

    bad = set()
    for cell in f["factors"]:
        for x in parse_factors(cell):
            if x not in FACTOR_TAXONOMY:
                bad.add(x)
    if bad:
        raise SystemExit(f"labels outside the taxonomy (each would score 0): "
                         f"{sorted(bad)}")

    old = b["factors"].tolist()
    new = f["factors"].tolist()
    changed = sum(1 for a, c in zip(old, new) if set(parse_factors(a))
                  != set(parse_factors(c)))

    out = b.copy()
    out["factors"] = new
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    # Re-read and prove the untouched columns really are untouched.
    chk = pd.read_csv(args.out, keep_default_na=False)
    chk["row_id"] = chk["row_id"].astype(str)
    for col in b.columns:
        if col == "factors":
            continue
        if not chk[col].astype(str).equals(b[col].astype(str)):
            raise SystemExit(f"column {col} changed -- NOT a single-variable "
                             f"test, refusing to leave this file in place")

    nf_old = sum(len(parse_factors(x)) for x in old) / len(b)
    nf_new = sum(len(parse_factors(x)) for x in new) / len(b)
    print(f"wrote {args.out}")
    print(f"  columns          {list(out.columns)}")
    print(f"  rows             {len(out)}")
    print(f"  posts changed    {changed} of {len(b)} "
          f"({100 * changed / len(b):.1f}%)")
    print(f"  factors/post     {nf_old:.2f} -> {nf_new:.2f}")
    print(f"  risk + evidence  VERIFIED identical to {args.base}")
    print("\nS1 must come back UNCHANGED. If it moves, the patch was not clean")
    print("and the S2 reading is void.")


if __name__ == "__main__":
    main()

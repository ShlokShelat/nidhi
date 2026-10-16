"""
merge_confirmed_wins.py

Combine two previously-verified single-column changes into one submission.

WHY THIS IS SAFE, and it is measured rather than assumed. Two single-variable
files were scored, and each one's UNTOUCHED component came back bit-identical
to the champion:

    champion  NIDHI_cluster.csv          S1 0.7982   S2 0.6496   0.7536
    factors   NIDHI_bag8_on_cluster.csv  S1 0.7982   S2 0.6519   0.7543
    evidence  NIDHI_evcond.csv           S1 0.7990   S2 0.6496   0.7542

bag8 moved S2 only; evcond moved S1 only. That is the additivity test passing
exactly -- no interaction between the evidence and factor columns, which is what
lets these be merged without re-measuring.

PREDICTION, stated before upload: S1 0.7990, S2 0.6519,
composite 0.7*0.7990 + 0.3*0.6519 = 0.7549.
If the official scorer returns anything else, the columns interact after all
and the additivity conclusion above is wrong -- which is itself worth knowing.

The risk column is the champion's in all three files and is copied through
untouched. This script verifies that rather than trusting it.

Usage:
  python src/merge_confirmed_wins.py
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="submissions/NIDHI_cluster.csv")
    ap.add_argument("--evidence-from", default="submissions/NIDHI_evcond.csv")
    ap.add_argument("--factors-from",
                    default="submissions/NIDHI_bag8_on_cluster.csv")
    ap.add_argument("--out", default="submissions/NIDHI_evcond_bag8.csv")
    args = ap.parse_args()

    ch = pd.read_csv(args.champion, keep_default_na=False)
    ev = pd.read_csv(args.evidence_from, keep_default_na=False)
    fa = pd.read_csv(args.factors_from, keep_default_na=False)
    for name, d in (("champion", ch), ("evidence", ev), ("factors", fa)):
        d["row_id"] = d["row_id"].astype(str)
        print(f"{name:<10}{len(d):>5} rows   columns {list(d.columns)}")

    if not (len(ch) == len(ev) == len(fa)):
        raise SystemExit("row counts differ")
    order = list(ch["row_id"])
    for name, d in (("evidence", ev), ("factors", fa)):
        if list(d["row_id"]) != order:
            print(f"  {name}: reordering to the champion's row order")
            d.set_index("row_id", inplace=True)
            d = d.loc[order].reset_index()
            if name == "evidence":
                ev = d
            else:
                fa = d

    # VERIFY the premise: each donor must differ from the champion in exactly
    # the one column being taken from it, and match everywhere else.
    ok = True
    for name, d, take in (("evidence", ev, "evidence"),
                          ("factors", fa, "factors")):
        for col in ("risk_level", "evidence", "factors"):
            if col not in d.columns:
                continue
            n_same = int((ch[col].astype(str) == d[col].astype(str)).sum())
            tag = "TAKING" if col == take else "must match"
            flag = ""
            if col != take and n_same != len(ch):
                flag = "  <- UNEXPECTED DIFFERENCE"
                ok = False
            if col == take and n_same == len(ch):
                flag = "  <- identical to champion, nothing to take"
                ok = False
            print(f"  {name:<9}{col:<12}{n_same:>5}/{len(ch)} same  "
                  f"({tag}){flag}")
    if not ok:
        raise SystemExit(
            "\nA donor differs from the champion outside the column being "
            "taken, or\ndoes not differ inside it. Merging would silently ship "
            "an unmeasured change.")

    out = ch.copy()
    out["evidence"] = ev["evidence"].values
    out["factors"] = fa["factors"].values
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    n_ev = int((out["evidence"].astype(str)
                != ch["evidence"].astype(str)).sum())
    n_fa = int((out["factors"].astype(str)
                != ch["factors"].astype(str)).sum())
    n_rk = int((out["risk_level"].astype(str)
                != ch["risk_level"].astype(str)).sum())
    print(f"\nwrote {args.out}")
    print(f"  evidence differs from champion on {n_ev} posts")
    print(f"  factors  differs from champion on {n_fa} posts")
    print(f"  risk     differs from champion on {n_rk} posts  (must be 0)")
    if n_rk:
        raise SystemExit("risk column changed -- this is not the intended merge")

    s1, s2 = 0.7990, 0.6519
    print(f"\nPREDICTED  S1 {s1:.4f}  S2 {s2:.4f}  "
          f"composite {0.7 * s1 + 0.3 * s2:.4f}  "
          f"(champion 0.7536, so {0.7 * s1 + 0.3 * s2 - 0.7536:+.4f})")
    print("Anything else means the evidence and factor columns interact, and")
    print("the additivity conclusion from today's two reads is wrong.")


if __name__ == "__main__":
    main()

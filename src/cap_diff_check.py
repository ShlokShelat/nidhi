"""
cap_diff_check.py

SEPARATE COSMETIC REORDERING FROM A REAL SPAN-SET CHANGE.

`evidence_vote_test.py --cap N` sorts the kept spans by agreement before
truncating, and `sorted()` reorders even when it does not truncate. So a naive
string diff between two submissions overstates the change: most differing rows
may be the same span SET in a different ORDER.

Order does not affect scoring. `scoring.phrase_f1` matches predictions to gold
with maximum bipartite matching (`_max_bipartite_matching`), which is
order-independent. Only a changed SET can move the score.

Usage:
  python src/cap_diff_check.py --base submissions/NIDHI_recond.csv \
                               --new  submissions/NIDHI_cap3.csv
"""
import argparse
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, "src")
from scoring import parse_evidence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submissions/NIDHI_recond.csv")
    ap.add_argument("--new", default="submissions/NIDHI_cap3.csv")
    ap.add_argument("--cap", type=int, default=3)
    args = ap.parse_args()

    a = pd.read_csv(args.base, keep_default_na=False)
    b = pd.read_csv(args.new, keep_default_na=False)
    assert list(a.row_id.astype(str)) == list(b.row_id.astype(str)), "ROW ORDER"

    reorder = real = 0
    lost = []
    gained = 0
    for x, y in zip(a.evidence, b.evidence):
        sa, sb = parse_evidence(x), parse_evidence(y)
        if sa == sb:
            continue
        if sorted(sa) == sorted(sb):
            reorder += 1
        else:
            real += 1
            d = len(sa) - len(sb)
            lost.append(d)
            if d < 0:
                gained += 1

    over = sum(1 for x in a.evidence if len(parse_evidence(x)) > args.cap)
    print(f"rows                            {len(a)}")
    print(f"reorder only (NO score effect)  {reorder}")
    print(f"span SET actually changed       {real}")
    print(f"net spans dropped               {sum(lost)}")
    print(f"per-post delta histogram        {dict(Counter(lost))}")
    print(f"posts with >{args.cap} spans in base    {over}")
    print(f"posts that GAINED a span        {gained}   (should be 0 for a cap)")

    ok = gained == 0 and real <= over
    print("\n" + ("PASS -- consistent with a pure per-post cap; only the set "
                  "changes matter and they are bounded by the >cap posts"
                  if ok else
                  "SUSPECT -- more set-changes than posts above the cap, or a "
                  "post gained a span. A cap can only REMOVE. Do not ship "
                  "until this is explained."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

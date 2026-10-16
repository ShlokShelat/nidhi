"""
canonicalize_evidence.py

Fold submitted evidence spans into the same text space the annotators used.

THE BUG. src/evidence_dump_all.py reads data/test.csv RAW, while the training
data and every OOF fold go through clean_text (src/text_norm.py). So the
submitted spans carry whatever typographic characters the raw post had, while
the model was trained and validated entirely in cleaned space.

WHY IT MATTERS. phrase-F1 matches a PREDICTION against GOLD, not against the
post. Measured on data/labeled.csv: of the spans on posts containing
typographic characters, 953 of 964 gold spans use straight characters, 86 spans
match only in cleaned space, and ZERO match only in raw space. Annotators
demonstrably worked in cleaned text. A prediction carrying a curly apostrophe
therefore cannot match a gold span that spells it straight.

SIZE. Exactly 12 of 378 evidence cells in NIDHI_evidence_vote8.csv contain a
typographic character, and clean_text changes all 12.

RISK PROFILE. If the official scorer normalizes before comparing, this is a
no-op. If it compares raw, it recovers up to 12 cells. It cannot lose anything
unless a gold span itself contains a curly character AND the raw predicted
span matched it -- which is the 1.1% tail (11 of 964). Expected value is small
and positive, downside is near zero, and it fits NOTHING.

This is a single-variable change: the evidence column only.

Usage:
  python src/canonicalize_evidence.py \
      --base submissions/NIDHI_evidence_vote8.csv \
      --out  submissions/NIDHI_evidence_canon.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_norm import clean_text

TYPO = "‘’“”–—… "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.base, keep_default_na=False)
    for c in ("row_id", "risk_level", "evidence", "factors"):
        if c not in base.columns:
            raise SystemExit(f"base missing column {c!r}")

    out = base.copy()
    out["evidence"] = [clean_text(str(x)) for x in base["evidence"]]

    for c in ("risk_level", "factors", "row_id"):
        if not out[c].equals(base[c]):
            raise SystemExit(f"internal error: {c} changed; refusing to write")

    changed = base.index[out["evidence"] != base["evidence"]].tolist()
    n_typo = sum(1 for x in base["evidence"] if any(ch in TYPO for ch in str(x)))
    print(f"base    : {args.base}")
    print(f"rows    : {len(out)}")
    print(f"cells containing typographic characters : {n_typo}")
    print(f"cells changed by canonicalization       : {len(changed)}")
    if len(changed) != n_typo:
        raise SystemExit("mismatch between typographic cells and changed cells "
                         "-- inspect before writing")
    if not changed:
        print("\nnothing to do; the base is already canonical.")
        return

    print("\nevery change, for the record:")
    for i in changed:
        print(f"  {base.at[i, 'row_id']}")
        print(f"    - {base.at[i, 'evidence']!r}")
        print(f"    + {out.at[i, 'evidence']!r}")

    # A canonicalized span must still be findable in the canonicalized post,
    # otherwise it has been folded into something the annotator never wrote.
    tests = "data/test.csv"
    if os.path.exists(tests):
        te = pd.read_csv(tests, keep_default_na=False).set_index("row_id")
        col = "post" if "post" in te.columns else te.columns[0]
        bad = []
        for i in changed:
            rid = base.at[i, "row_id"]
            post = clean_text(str(te.loc[rid, col]))
            for span in [s.strip() for s in str(out.at[i, "evidence"]).split(";")]:
                if span and span not in post:
                    bad.append((rid, span))
        print(f"\ncanonicalized spans not found in the canonicalized post: "
              f"{len(bad)}")
        for rid, s in bad[:10]:
            print(f"    {rid}  {s!r}")
        if bad:
            raise SystemExit("refusing to write -- a folded span no longer "
                             "occurs in its post")
    else:
        print(f"\n(skipped post-membership check: {tests} not found)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("risk_level and factors carried over byte-identically.")


if __name__ == "__main__":
    main()

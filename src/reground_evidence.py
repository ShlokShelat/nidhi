"""
reground_evidence.py

Make every submitted evidence span a VERBATIM substring of its post.

THE PROBLEM. Auditing submissions/NIDHI_evidence_vote8.csv against data/test.csv:

    total spans      435
    verbatim in post 423  (97.2%)
    match only after normalisation  12
    match neither     0

Those 12 are all punctuation mismatches -- the tagger emits a straight
apostrophe where the post has a curly one:

    "why can't it be my last one"      vs   "why can't it be my last one"
    "I'll be another statistic"
    "hope I don't wake up tomorrow"

The LOCAL scorer normalises both sides before checking containment, so it sees
these as matches. But scoring.py's own docstring flags that as an ASSUMPTION TO
VERIFY against the official scorer. If the official implementation does raw
substring matching -- which is the straightforward reading of "the predicted
span CONTAINS the gold span" -- those 12 spans score ZERO, and 2.8% of
predictions are discarded over a character encoding difference.

WHY THIS IS A FREE FIX. Re-grounding replaces each span with the exact
characters found in the post. Under a raw-matching scorer it recovers 12 spans.
Under a normalising scorer it changes nothing, because normalisation maps both
forms to the same string. There is no scenario where it costs anything, and
zero parameters are fitted.

HOW. For each span, if it is already a substring, leave it. Otherwise build a
regex where apostrophe-like characters match any apostrophe variant, quote-like
characters match any quote variant, whitespace matches any run of whitespace,
and dashes match any dash variant -- then search the raw post case-insensitively
and emit whatever text actually matched.

Also reports, but does not touch, any span that cannot be found at all. A span
that appears nowhere in its post is unmatchable by any scorer and is a genuine
bug worth seeing.

Usage:
  python src/reground_evidence.py \
      --sub submissions/NIDHI_evidence_vote8.csv \
      --out submissions/NIDHI_evidence_vote8_grounded.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_span, parse_evidence

APOS = "'’‘´`ʼ"
QUOT = '"“”«»'
DASH = "-‐‑‒–—―−"


def flexible(span: str) -> re.Pattern:
    """A regex matching `span` while tolerating punctuation and whitespace
    variants, so the RAW text in the post can be recovered."""
    out = []
    for ch in span:
        if ch in APOS:
            out.append(f"[{re.escape(APOS)}]")
        elif ch in QUOT:
            out.append(f"[{re.escape(QUOT)}]")
        elif ch in DASH:
            out.append(f"[{re.escape(DASH)}]")
        elif ch.isspace():
            out.append(r"\s+")
        else:
            out.append(re.escape(ch))
    # collapse runs of the whitespace token
    pat = "".join(out)
    pat = re.sub(r"(?:\\s\+)+", r"\\s+", pat)
    return re.compile(pat, re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    te = pd.read_csv(args.test).set_index("row_id")
    sub = pd.read_csv(args.sub, keep_default_na=False)
    if "evidence" not in sub.columns:
        raise SystemExit(f"no evidence column; have {list(sub.columns)}")

    n_tot = n_ok = n_fixed = 0
    unfound = []
    new_col = []
    for _, row in sub.iterrows():
        rid = row["row_id"]
        post = str(te.loc[rid, "post"]) if rid in te.index else ""
        spans = parse_evidence(row["evidence"])
        out = []
        for s in spans:
            n_tot += 1
            if s in post:
                n_ok += 1
                out.append(s)
                continue
            m = flexible(s).search(post)
            if m:
                n_fixed += 1
                out.append(m.group(0))
            else:
                # keep the original; a span found nowhere is a separate bug and
                # silently dropping it would change recall as well as encoding
                unfound.append((rid, s))
                out.append(s)
        new_col.append("; ".join(out))

    sub["evidence"] = new_col
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sub.to_csv(args.out, index=False)

    print(f"spans total        : {n_tot}")
    print(f"already verbatim   : {n_ok}")
    print(f"re-grounded        : {n_fixed}")
    print(f"not found at all   : {len(unfound)}")
    for rid, s in unfound[:10]:
        print(f"    {rid}  {s[:70]!r}")

    # verify
    chk = pd.read_csv(args.out, keep_default_na=False)
    bad = 0
    for _, row in chk.iterrows():
        post = str(te.loc[row["row_id"], "post"])
        for s in parse_evidence(row["evidence"]):
            if s not in post:
                bad += 1
    print(f"\nafter re-grounding, spans not verbatim: {bad}")
    print(f"wrote {args.out}")
    if n_fixed and bad == 0:
        print("\nEvery span is now an exact substring of its post. Under a")
        print("raw-matching official scorer this recovers the affected spans;")
        print("under a normalising one it is a no-op. It cannot lose.")


if __name__ == "__main__":
    main()

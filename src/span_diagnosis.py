"""
Diagnose the ungrounded gold evidence spans found by data_audit.py.

For every gold span that plain matching can't locate, try a ladder of
progressively looser normalizations and report which rung recovers it:

  L1  unicode NFKC + curly quote/dash/ellipsis folding
  L2  L1 + collapse all whitespace (incl. newlines) to single spaces
  L3  L2 + strip ALL punctuation from both span and post
  L4  L3 + fuzzy: best contiguous word-window, report similarity

Output: counts per rung + concrete examples, and unrecovered_spans.csv
for manual inspection. The rung distribution tells us exactly which
normalization to fold into find_span_in_post() so the tagger trains on
clean targets — and how many spans are genuinely unrecoverable (likely
annotator paraphrases), which we then drop from training WITH a count,
not silently.

Usage:
  python src/span_diagnosis.py --train data/labeled.csv
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scoring import parse_evidence

# characters that Reddit/Excel/annotation tools commonly swap
FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ", "\u200b": "",
}


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    for a, b in FOLD.items():
        s = s.replace(a, b)
    return s


def l1(s: str) -> str:
    return fold(s).lower()


def l2(s: str) -> str:
    return re.sub(r"\s+", " ", l1(s)).strip()


def l3(s: str) -> str:
    return re.sub(r"[^\w\s]", "", l2(s))


def fuzzy_best_window(span: str, post: str) -> tuple[float, str]:
    """Best-matching contiguous word window in the post (same word count
    as the span, +/-2), by difflib ratio on L2-normalized text."""
    sw = l2(span).split()
    pw = l2(post).split()
    if not sw or not pw:
        return 0.0, ""
    best, best_txt = 0.0, ""
    for size in range(max(1, len(sw) - 2), len(sw) + 3):
        for i in range(0, max(1, len(pw) - size + 1)):
            cand = " ".join(pw[i:i + size])
            r = difflib.SequenceMatcher(None, " ".join(sw), cand).ratio()
            if r > best:
                best, best_txt = r, cand
    return best, best_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.85)
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"suicide risk": "risk_level",
                            "evidence for suicide risk level": "evidence"})

    rungs = Counter()
    examples = {k: [] for k in ["L1", "L2", "L3", "FUZZY", "UNRECOVERED"]}
    unrecovered = []

    total = 0
    for i, row in df.iterrows():
        post = str(row["post"])
        for span in parse_evidence(row.get("evidence")):
            total += 1
            # rung 0: what data_audit already does (lower + ws-collapse, no folding)
            base_post = re.sub(r"\s+", " ", post).strip().lower()
            base_span = re.sub(r"\s+", " ", str(span)).strip().lower()
            if base_span in base_post:
                rungs["OK_baseline"] += 1
                continue
            if l1(span) in l1(post):
                rungs["L1_unicode_fold"] += 1
                examples["L1"].append((i, span))
            elif l2(span) in l2(post):
                rungs["L2_whitespace"] += 1
                examples["L2"].append((i, span))
            elif l3(span) and l3(span) in l3(post):
                rungs["L3_punct_strip"] += 1
                examples["L3"].append((i, span))
            else:
                sim, window = fuzzy_best_window(span, post)
                if sim >= args.fuzzy_threshold:
                    rungs["L4_fuzzy"] += 1
                    examples["FUZZY"].append((i, span, f"{sim:.2f}", window))
                else:
                    rungs["UNRECOVERED"] += 1
                    examples["UNRECOVERED"].append((i, span, f"{sim:.2f}", window))
                    unrecovered.append({"row": i, "row_id": row.get("row_id"),
                                        "span": span, "best_sim": sim,
                                        "best_window": window})

    print(f"total gold spans: {total}")
    for k, v in rungs.most_common():
        print(f"  {k:20s} {v:5d}  ({100*v/total:.1f}%)")

    for name in ["L1", "L2", "L3", "FUZZY", "UNRECOVERED"]:
        ex = examples[name][:4]
        if ex:
            print(f"\n--- {name} examples ---")
            for e in ex:
                print(f"  row {e[0]}: span={e[1]!r}"
                      + (f"  sim={e[2]}  best_window={e[3]!r}" if len(e) > 2 else ""))

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(unrecovered).to_csv(
        os.path.join(args.outdir, "unrecovered_spans.csv"), index=False)
    print(f"\nwrote {len(unrecovered)} unrecovered spans to "
          f"{args.outdir}/unrecovered_spans.csv")


if __name__ == "__main__":
    main()

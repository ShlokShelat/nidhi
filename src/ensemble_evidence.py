"""
Cross-model evidence ensemble.

Same lever that worked for risk and factors: combine two
independently-trained architectures. Here the spans each model
proposes are unioned, then every span (from both sources) is re-verified
against the actual post text with the grounding checker. A span that doesn't
literally appear in the post is dropped, regardless of which model
proposed it — this guarantees the union can only add real, checkable
recall, never hallucination.

Why union (not blend/vote) for evidence specifically: spans aren't
probabilities, they're discrete text extractions. Two models with
different training (different encoder, different decoding threshold) will
often catch DIFFERENT true spans in the same post. Precision is protected
by the grounding filter + the official scorer's containment/one-to-one
matching (an extra correct span can only help, an extra WRONG span costs
one false positive per post, capped by the containment rule).

Friend's file uses the literal string 'none' for empty evidence (a known
quirk in their pipeline) — handled here, not treated as a span.

Usage:
  python src/ensemble_evidence.py \
      --test data/test.csv \
      --our-evidence-run runs/evidence_v2 \
      --friend-submission artifacts/friend_nidhi.csv \
      --out artifacts/ensembled_evidence.csv

--friend-submission is their final submission CSV (row_id, risk_level,
evidence, factors) — only the evidence column is read, matched by row_id.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import normalize_span, parse_evidence
from train_evidence import find_span_in_post, verify_spans
from predict import predict_evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--our-evidence-run", required=True)
    ap.add_argument("--friend-submission", required=True)
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--out", default="artifacts/ensembled_evidence.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load(args.test)
    ids = df["row_id"].tolist()
    posts = df["post"].astype(str).tolist()

    print("computing evidence spans ...")
    ours = predict_evidence(args.our_evidence_run, df, device)  # list[list[str]]

    fs = pd.read_csv(args.friend_submission, keep_default_na=False)
    fs = fs.set_index("row_id")

    out_rows = []
    n_added, n_rejected_friend, n_posts_gained = 0, 0, 0
    for i, (rid, post) in enumerate(zip(ids, posts)):
        our_spans = ours[i]
        friend_raw = fs.loc[rid, "evidence"] if rid in fs.index else ""
        friend_spans = [] if str(friend_raw).strip().lower() in ("none", "", "nan") \
            else parse_evidence(friend_raw)

        # ground EVERY friend span against THIS post; only literal matches survive
        friend_accepted, friend_rejected = verify_spans(
            friend_spans, post, max_tokens=args.max_tokens)
        n_rejected_friend += len(friend_rejected)

        # union, de-duplicated (normalized), one-to-one style: keep first
        # occurrence's exact casing from whichever source found it
        seen = set()
        union = []
        for sp in our_spans + friend_accepted:
            key = normalize_span(sp)
            if key in seen:
                continue
            seen.add(key)
            union.append(sp)

        added_here = len(union) - len(our_spans)
        if added_here > 0:
            n_added += added_here
            n_posts_gained += 1

        out_rows.append({"row_id": rid, "evidence": "; ".join(union)})

    out = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"friend spans rejected by grounding check: {n_rejected_friend}")
    print(f"net spans ADDED by union: {n_added}, across {n_posts_gained}/{len(ids)} posts")
    print(f"local total spans: {sum(len(s) for s in ours)}, "
          f"union total spans: {sum(len(r['evidence'].split(';')) if r['evidence'] else 0 for r in out_rows)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

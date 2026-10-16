"""
evidence_enum_candidates.py

CANDIDATE GENERATION FOR THE SPAN RERANKER -- from raw text, no model.

WHY THIS FILE EXISTS. `evidence_tagger_rank.py` established that the seven
taggers, used as SPAN SCORERS via mean P(not-O), reach **81.7% of a 0.942
pool oracle against an 82.4% bar** -- far above the 55-62% model-free floor
but short of the incumbent 0.7767, and raising the pool cap made it WORSE
(79.8% of 0.965). That is the middle band, and the pre-registered reading is
that the next step is a TRAINED span scorer, not a bigger pool. It is also
exactly what Culotta & McCallum (2004) measured directly: a mean of token
marginals is the weak estimator (.912 avg precision) against a properly
scored span (.976).

THE LEAKAGE TRAP THIS AVOIDS, WHICH IS WHY GENERATION IS MODEL-FREE. The seven
`_rc` taggers are SINGLE-SPLIT models -- trained on `train_split`, validated on
`val_split`. Running them over all 1635 labelled posts scores their own
training data on ~1263 of them, so a reranker trained on tagger confidence
would be learning from leaked scores. The fold-based `ev5f_*` taggers are
honest but are not what ships, so training on those and applying to RC7
candidates is the artifact-mismatch that cost three submissions.

Enumerating from raw text dodges both: **no tagger participates in training.**
The cross-encoder reads (risk, post, span) and predicts whether the span
matches gold. Author-disjoint folds make its OOF scores honest. The taggers
reappear only in the final combination, where val and test dumps are matched
by construction.

WHY THESE ENUMERATION SETTINGS. Gold spans are short and sub-clausal: mean 4.6
whitespace tokens, median 4, p90 8, max 19; only 5.2% coincide exactly with a
clause and 2.5% with a sentence, so linguistic segmentation is WORSE than dumb
n-grams (70.9% and 46.8% coverage against 96.9% for in-sentence 3-8 grams).
The match rule accepts containment either way, so a candidate need only land
INSIDE a gold span -- boundaries do not have to be found, which is why a
mechanical generator saturates at all.

NEGATIVE SUBSAMPLING. The full pool is ~5% positive, so training on all of it
spends 95% of the GPU on easy negatives. Positives are ALL kept; negatives are
sampled at `--neg-ratio` per positive, per post, so no post drops out entirely.
SpERT uses 100 negatives/sentence and reports performance stagnating well
below that, so 4:1 is not obviously thin -- but it is the one knob here that
trades training signal for wall clock. Inference uses `--no-subsample`, which
keeps everything: the reranker must rank the pool it will actually face.

Usage:
  # training pool, author-disjoint folds happen inside the trainer
  python src/evidence_enum_candidates.py \
      --data artifacts/full_train.csv \
      --out artifacts/evidence_oof/enum_oof_candidates.csv

  # inference pools -- every candidate, nothing dropped
  python src/evidence_enum_candidates.py --data artifacts/val_split.csv \
      --no-subsample --out artifacts/evidence_oof/enum_val_candidates.csv
  python src/evidence_enum_candidates.py --data data/test.csv \
      --no-subsample --no-gold --out artifacts/evidence_oof/enum_test_candidates.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_span, parse_evidence,  # noqa: E402
                     token_len)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def matches(pred: str, gold: str) -> bool:
    p, g = normalize_span(pred), normalize_span(gold)
    if not p or not g:
        return False
    if p not in g and g not in p:
        return False
    return token_len(p) <= 3 * token_len(g)


def enumerate_post(post: str, lo: int, hi: int) -> list[str]:
    """In-sentence whitespace-word n-grams, deduped on the normalised form."""
    out, seen = [], set()
    for s in SENT_SPLIT.split(post):
        w = s.split()
        for nn in range(lo, hi + 1):
            for i in range(len(w) - nn + 1):
                t = " ".join(w[i:i + nn])
                k = normalize_span(t)
                if k and k not in seen:
                    seen.add(k)
                    out.append(t)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--data", default="artifacts/full_train.csv")
    a_.add_argument("--out", required=True)
    a_.add_argument("--lo", type=int, default=2)
    a_.add_argument("--hi", type=int, default=8)
    a_.add_argument("--neg-ratio", type=int, default=4)
    a_.add_argument("--no-subsample", action="store_true")
    a_.add_argument("--no-gold", action="store_true",
                    help="the file has no evidence column (test set)")
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    from data_audit import load
    df = load(args.data)
    df["row_id"] = df["row_id"].astype(str)
    posts = [str(p) for p in df["post"]]
    have_gold = (not args.no_gold) and "evidence" in df.columns
    golds = ([parse_evidence(v) for v in df["evidence"]] if have_gold
             else [[] for _ in posts])
    rng = np.random.default_rng(args.seed)

    rows, npos_t, ntot_t, cov = [], 0, 0, 0
    ngold = 0
    for rid, post, g in zip(df["row_id"], posts, golds):
        cands = enumerate_post(post, args.lo, args.hi)
        ntot_t += len(cands)
        if not have_gold:
            for t in cands:
                rows.append((rid, t, 1.0, -1))
            continue
        y = [int(any(matches(t, s) for s in g)) for t in cands]
        npos_t += sum(y)
        ngold += len(g)
        cov += sum(any(matches(t, s) for t in cands) for s in g)
        pos = [i for i, v in enumerate(y) if v]
        neg = [i for i, v in enumerate(y) if not v]
        if not args.no_subsample and neg:
            keep = min(len(neg), max(args.neg_ratio * max(len(pos), 1), 8))
            neg = list(rng.choice(neg, size=keep, replace=False))
        for i in sorted(pos + list(neg)):
            rows.append((rid, cands[i], 1.0, y[i]))

    out = pd.DataFrame(rows, columns=["row_id", "span", "conf", "y"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    n = len(df)
    print(f"{args.data}: {n} posts")
    print(f"  enumerated {ntot_t / n:.1f} candidates/post before sampling "
          f"(in-sentence {args.lo}-{args.hi} grams)")
    if have_gold:
        print(f"  gold coverage {cov}/{ngold} = {cov / max(ngold, 1):.1%} "
              f"-- this is the ORACLE RECALL the reranker inherits and can "
              f"never exceed")
        print(f"  positives {npos_t} ({npos_t / max(ntot_t, 1):.2%} of the "
              f"full pool)")
    print(f"  wrote {len(out)} rows -> {args.out}"
          + ("  (all candidates kept)" if args.no_subsample
             else f"  (negatives sampled at {args.neg_ratio}:1)"))
    print(f"  {len(out) / n:.1f} rows/post")
    print("\n  `y` is written for convenience; train_span_reranker.py")
    print("  recomputes it from gold, so the two cannot silently disagree.")


if __name__ == "__main__":
    main()

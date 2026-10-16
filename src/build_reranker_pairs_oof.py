"""
build_reranker_pairs_oof.py

Training pairs for the span selector, built from REAL out-of-fold tagger
candidates instead of synthetic negatives.

WHY THIS RE-OPENS A CLOSED LANE. Evidence SELECTION was declared closed after
four failed attempts. But attempt 4 (`src/train_span_reranker.py`) trained on
pairs from `build_training_pairs`, whose negatives are
`perturb(gold, post)` -- boundary shifts of TRUE spans -- plus
`sentence_fragments(post)`. It then scored REAL BIO-tagger candidates at
inference. The model never saw a tagger output during training.

That is a train/serve distribution mismatch, and it is the specific
configuration the stacking literature identifies as broken: a second-stage
model must be trained on FIRST-STAGE OUTPUTS produced out-of-fold. The
symptoms match exactly -- training loss collapsed to 0.048 (memorisation) while
val went nowhere, and a 4.6x smaller model overfit identically, which says the
constraint was the DATA, not capacity.

Real tagger near-misses are semantically plausible but CONVENTIONALLY wrong --
the annotator quoted different words. Perturbed gold and sentence fragments are
not that distribution, so the hard cases were absent from training. The failed
reranker reached recall 0.8575 at precision 0.608: it found the right regions
and could not tell a conventional match from a near-miss. That is exactly the
discrimination the missing negatives would have taught.

This closure was pre-registered as needing "new evidence, not a new intuition".
A verified distribution mismatch in the experiment itself is new evidence.

WHAT THIS BUILDS. Every candidate from every OOF tagger, labelled y=1 if it
matches ANY gold span for its post under the OFFICIAL rule
(`_span_pair_matches`, containment either way + 3x token cap) and y=0
otherwise. Same schema as `build_training_pairs` -- post, span, y -- so
`train_span_reranker.py --pairs` consumes it unchanged.

The labels ARE the target: "would the scorer accept this candidate", not "is
this a sensible quote". That is the annotation convention the selector has to
learn, stated in the metric's own terms.

Usage:
  python src/build_reranker_pairs_oof.py
  python src/build_reranker_pairs_oof.py --out artifacts/reranker_pairs_oof.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_vote_oof import TAGS, dedupe  # noqa: E402
from scoring import (_span_pair_matches, normalize_span,  # noqa: E402
                     parse_evidence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--dir", default="artifacts/evidence_oof")
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--out", default="artifacts/reranker_pairs_oof.csv")
    ap.add_argument("--min-conf", action="store_true",
                    help="apply each tagger's own shipped min_conf. Default is "
                         "OFF: the selector should see the FULL candidate "
                         "surface it will be asked to rank, including the "
                         "low-confidence spans a threshold would have hidden.")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    cand = {}
    for t in [x.strip() for x in args.tags.split(",")]:
        p = os.path.join(args.dir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            print(f"  skip {t}: no {p}")
            continue
        d = pd.read_csv(p)
        d["row_id"] = d["row_id"].astype(str)
        if args.min_conf and "min_conf" in d.columns:
            d = d[d["conf"] >= float(d["min_conf"].iloc[0])]
        n = 0
        for r, s in zip(d["row_id"], d["span"]):
            cand.setdefault(r, set()).add(str(s))
            n += 1
        print(f"  {t}: {n} candidate spans")
    if not cand:
        raise SystemExit(f"no candidate files found in {args.dir}")

    rows, n_pos, n_no_gold, n_all_pos = [], 0, 0, 0
    for r, spans in cand.items():
        if r not in lab.index:
            continue
        post = str(lab.loc[r, "post"])
        gold = parse_evidence(lab.loc[r, "evidence"])
        if not gold:
            # Gold is empty, so EVERY candidate here is a negative. These posts
            # are the suppression cases and they are genuinely informative: the
            # selector must learn that a plausible-looking clause on an
            # `indicator` post is still not evidence.
            n_no_gold += 1
        keep = dedupe(sorted(spans))
        pos_here = 0
        for s in keep:
            y = int(any(_span_pair_matches(s, g) for g in gold))
            pos_here += y
            rows.append({"row_id": r, "post": post, "span": s, "y": y})
        n_pos += pos_here
        if gold and pos_here == len(keep) and keep:
            n_all_pos += 1

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nwrote {args.out}")
    print(f"  {len(out)} pairs over {out['row_id'].nunique()} posts")
    print(f"  positives {n_pos} ({100 * n_pos / max(len(out), 1):.1f}%)")
    print(f"  posts with EMPTY gold (all candidates negative): {n_no_gold}")
    print(f"  posts where every candidate matches (no discrimination to "
          f"learn): {n_all_pos}")

    print("\nCOMPARE WITH THE FAILED ATTEMPT: 7087 synthetic pairs, 1485 "
          "positives (21.0%),")
    print("negatives = perturbed gold + sentence fragments, none of them a "
          "real tagger output.")
    print("\nNEXT:")
    print(f"  python src/train_span_reranker.py --pairs {args.out}")
    print("BAR, unchanged from attempt 4: beat cluster>=2 on FULL val AND on")
    print("BOTH author halves. ABORT if training loss collapses below 0.05 --")
    print("that was the memorisation signature, and distribution-matched data")
    print("is supposed to prevent it. If it collapses again, the data was not")
    print("the constraint and evidence selection is closed for good.")


if __name__ == "__main__":
    main()

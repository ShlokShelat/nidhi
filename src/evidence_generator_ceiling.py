"""
evidence_generator_ceiling.py

CAN A CANDIDATE GENERATOR WITH NO MODEL IN IT SATURATE EVIDENCE RECALL?

THE BINDING CONSTRAINT, RESTATED. The candidate pool holds ~1.1 spans/post
against 1.12 gold spans, and 42% of missed golds are ABSENT from it entirely,
because `_decode_candidates` walks `probs.argmax(-1)` and a span the tagger
scores at 45% is never proposed. The two-stage IE literature runs stage-1
recall at >99% BEFORE a ranker is applied; this pool is nowhere near that, so
ranking work on it has been solving the wrong stage.

THE ASYMMETRY THAT MAKES THIS WORTH ASKING. `_span_pair_matches` accepts a
prediction when either string CONTAINS the other and
`token_len(pred) <= 3 * token_len(gold)`. If pred is a substring of gold the
length test passes automatically. So **any fragment of a gold span matches it**
-- the generator does not have to find the boundaries, only to land inside.
That is a far weaker requirement than the BIO decode is currently meeting, and
it means a purely mechanical enumerator might already contain the answers.

WHY ORACLE F1 IS FULLY DETERMINED BY COVERAGE. Matching is one-to-one, so the
best possible subset is "one matching candidate per covered gold": k
predictions, all matched, against m golds gives P=1, R=k/m, F1 = 2k/(k+m).
Empty gold with an empty prediction scores 1.0. So the oracle macro-F1 of a
generator is a pure function of per-post gold coverage -- no selection modelling
needed, and no model outputs needed either. **This entire script runs on gold
data alone.**

WHAT IT MEASURES

  0. the hard extractive ceiling: how many gold spans are actually present in
     the post verbatim after canonicalisation. A gold span that is not in the
     post cannot be produced by ANY extractive system, ours included.
  1. where gold spans sit: token length, and whether they align to sentence or
     clause boundaries. If rationales are near-always clause substrings, a
     clause enumerator saturates recall with zero model involvement.
  2. oracle macro phrase-F1 and pool size for a ladder of generators, from
     sentences (tiny pool) to all n-grams (huge pool). The point is the
     TRADE-OFF CURVE: coverage against candidates per post, because a pool
     that cannot be selected from is not progress.

HOW TO READ IT. The incumbent is ~1.1 candidates/post. If a generator reaches
>=99% coverage at, say, 15 candidates/post, then the evidence problem stops
being "generation is starved" and becomes "rank 15 candidates" -- which is the
regime the literature says rankers work in, and there are seven trained
taggers available to score with. If NO mechanical generator gets there at a
sane pool size, then
boundaries genuinely need a model and the soft-decode sweep is the only route.

Usage:
  python src/evidence_generator_ceiling.py
  python src/evidence_generator_ceiling.py --data artifacts/val_split.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_span, parse_evidence,  # noqa: E402
                     token_len)
from text_norm import clean_text  # noqa: E402

# Sentence enders, then the finer clause cuts. Kept deliberately dumb: the
# whole point is that this contains no model and no tuning.
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
CLAUSE_SPLIT = re.compile(r"(?<=[.!?,;:])\s+|\n+|\s+(?:and|but|because|so|"
                          r"though|although|while|since|then|or)\s+", re.I)


def matches(pred: str, gold: str) -> bool:
    """The official rule, inlined so the analysis cannot drift from it."""
    p, g = normalize_span(pred), normalize_span(gold)
    if not p or not g:
        return False
    if p not in g and g not in p:
        return False
    return token_len(p) <= 3 * token_len(g)


def pieces(post: str, rx) -> list[str]:
    return [s.strip() for s in rx.split(post) if s and s.strip()]


def runs_of(units: list[str], max_run: int) -> list[str]:
    """Contiguous unions of up to max_run adjacent units."""
    out = []
    for i in range(len(units)):
        for n in range(1, max_run + 1):
            if i + n <= len(units):
                out.append(" ".join(units[i:i + n]))
    return out


def ngrams(post: str, lo: int, hi: int) -> list[str]:
    w = post.split()
    out = []
    for n in range(lo, hi + 1):
        for i in range(len(w) - n + 1):
            out.append(" ".join(w[i:i + n]))
    return out


def coverage(cands: list[str], golds: list[str]) -> int:
    """How many golds have at least one matching candidate. One-to-one
    matching means distinct candidates are needed, but a gold with any match
    almost always has a private one (fragments differ per gold); this is the
    standard oracle-recall approximation and it is an UPPER bound."""
    return sum(any(matches(c, g) for c in cands) for g in golds)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--data", default="artifacts/full_train.csv")
    a_.add_argument("--max-ngram", type=int, default=12)
    a_.add_argument("--show", type=int, default=8)
    args = a_.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    posts = [clean_text(str(p)) for p in df["post"]]
    golds = [parse_evidence(v) for v in df["evidence"]]
    n = len(df)
    ne = sum(1 for g in golds if g)
    allg = [s for g in golds for s in g]
    print(f"{args.data}: {n} posts, {ne} with non-empty gold "
          f"({ne / n:.1%}), {len(allg)} gold spans "
          f"({len(allg) / max(ne, 1):.2f} per non-empty post)")

    # -- 0. the hard extractive ceiling ------------------------------------
    hdr = "=" * 76
    print("\n" + hdr)
    print("0. IS THE GOLD SPAN EVEN IN THE POST?")
    print(hdr)
    present = 0
    for p, g in zip(posts, golds):
        np_ = normalize_span(p)
        present += sum(1 for s in g if normalize_span(s) in np_)
    print(f"  gold spans found verbatim in the canonicalised post: "
          f"{present}/{len(allg)}  ({present / max(len(allg), 1):.2%})")
    print("  NOT a ceiling, and it is important not to read it as one: the")
    print("  match rule accepts containment EITHER way, so a gold span that")
    print("  is not in the post verbatim can still be matched by a shorter")
    print("  fragment of itself that IS. That is why the coverage numbers")
    print("  below legitimately exceed this figure.")

    # -- 1. shape of the gold spans ----------------------------------------
    print("\n" + hdr)
    print("1. WHERE DO GOLD SPANS SIT?")
    print(hdr)
    tl = np.array([token_len(s) for s in allg])
    print(f"  token length: mean {tl.mean():.1f}  median "
          f"{np.median(tl):.0f}  p90 {np.percentile(tl, 90):.0f}  "
          f"max {tl.max()}")
    print(f"  spans over 12 tokens (the hardcoded decode cap): "
          f"{(tl > 12).sum()} ({(tl > 12).mean():.1%})")
    print(f"  spans over 24 tokens: {(tl > 24).sum()} "
          f"({(tl > 24).mean():.1%})")
    print(f"\n  IS THE 12-TOKEN DECODE CAP ACTUALLY COSTING US ANYTHING?")
    print(f"  A prediction may be up to 3x the gold length, so the cap only")
    print(f"  binds where 3*len(gold) > 12, i.e. gold >= 5 tokens.")
    print(f"    golds with 3x allowance <= 12 (cap irrelevant): "
          f"{(3 * tl <= 12).mean():>6.1%}")
    print(f"    golds with 3x allowance  > 12 (cap can bind):   "
          f"{(3 * tl > 12).mean():>6.1%}")
    print(f"  and a prediction only benefits from the extra room if it is")
    print(f"  LONGER than 12 tokens AND still contains the gold -- so the")
    print(f"  upper bound on what raising the cap to 64 can buy is the second")
    print(f"  row, not the whole set. Judge `evidence_redump.py`'s premise")
    print(f"  ('the 3x rule LICENSES long spans') against these two numbers.")

    exact_s = exact_c = run_c = starts = ends = 0
    for p, g in zip(posts, golds):
        if not g:
            continue
        S = {normalize_span(x) for x in pieces(p, SENT_SPLIT)}
        C = pieces(p, CLAUSE_SPLIT)
        Cn = {normalize_span(x) for x in C}
        R = {normalize_span(x) for x in runs_of(C, 3)}
        for s in g:
            ns = normalize_span(s)
            exact_s += ns in S
            exact_c += ns in Cn
            run_c += ns in R
            starts += any(c.startswith(ns[:12]) for c in Cn if ns)
            ends += any(c.endswith(ns[-12:]) for c in Cn if ns)
    m = max(len(allg), 1)
    print(f"\n  gold span is EXACTLY one sentence:            "
          f"{exact_s / m:>6.1%}")
    print(f"  gold span is EXACTLY one clause:              {exact_c / m:>6.1%}")
    print(f"  gold span is EXACTLY a run of <=3 clauses:    {run_c / m:>6.1%}")
    print(f"  gold span STARTS at a clause boundary:        {starts / m:>6.1%}")
    print(f"  gold span ENDS at a clause boundary:          {ends / m:>6.1%}")
    print("\n  Exact equality is the strict test. The metric is far more")
    print("  forgiving -- see the coverage table, which is what decides this.")

    # -- 2. the trade-off curve --------------------------------------------
    print("\n" + hdr)
    print("2. ORACLE CEILING vs POOL SIZE  (gold data only, no model)")
    print(hdr)

    def within(p, rx, lo, hi):
        """n-grams that do not cross a unit boundary. Same coverage idea at a
        fraction of the pool, because a rationale that straddles a sentence
        break is rare -- and pool size is what a ranker has to survive."""
        out = []
        for u in pieces(p, rx):
            out.extend(ngrams(u, lo, hi))
        return out

    gens = [
        ("sentences", lambda p: pieces(p, SENT_SPLIT)),
        ("clauses", lambda p: pieces(p, CLAUSE_SPLIT)),
        ("clause runs <=2", lambda p: runs_of(pieces(p, CLAUSE_SPLIT), 2)),
        ("clause runs <=3", lambda p: runs_of(pieces(p, CLAUSE_SPLIT), 3)),
        ("ngrams 3-8 in-clause", lambda p: within(p, CLAUSE_SPLIT, 3, 8)),
        ("ngrams 3-8 in-sent", lambda p: within(p, SENT_SPLIT, 3, 8)),
        ("ngrams 2-10 in-sent", lambda p: within(p, SENT_SPLIT, 2, 10)),
        ("ngrams 3-8", lambda p: ngrams(p, 3, 8)),
        (f"ngrams 2-{args.max_ngram}", lambda p: ngrams(p, 2, args.max_ngram)),
    ]

    print(f"  {'generator':<20}{'cands/post':>12}{'gold cov':>10}"
          f"{'posts full':>12}{'oracle macroF1':>16}")
    for name, fn in gens:
        tot_c = 0
        cov = 0
        full = 0
        f1s = []
        for p, g in zip(posts, golds):
            if not g:
                f1s.append(1.0)          # empty gold + predict nothing = 1.0
                continue
            c = fn(p)
            tot_c += len(c)
            k = coverage(c, g)
            cov += k
            full += (k == len(g))
            f1s.append(2.0 * k / (k + len(g)))
        print(f"  {name:<20}{tot_c / n:>12.1f}{cov / m:>10.1%}"
              f"{full / max(ne, 1):>12.1%}{float(np.mean(f1s)):>16.4f}")

    print("\n  cands/post counts EVERY generated string, before any dedupe or")
    print("  scoring. 'posts full' is the share of non-empty-gold posts where")
    print("  every gold span is reachable -- the number the two-stage")
    print("  literature reports as stage-1 recall and holds above 99%.")
    print("\n  INCUMBENT for comparison: ~1.1 candidates/post from the argmax")
    print("  decode, with 42% of missed golds absent from the pool.")

    # -- 3. what the misses look like --------------------------------------
    print("\n" + hdr)
    print("3. WHAT DOES THE BEST SANE GENERATOR STILL MISS?")
    print(hdr)
    miss_len, miss_ex = [], []
    for p, g in zip(posts, golds):
        if not g:
            continue
        c = runs_of(pieces(p, CLAUSE_SPLIT), 3)
        for s in g:
            if not any(matches(x, s) for x in c):
                miss_len.append(token_len(s))
                if len(miss_ex) < args.show:
                    inpost = normalize_span(s) in normalize_span(p)
                    miss_ex.append((token_len(s), inpost))
    if miss_len:
        ml = np.array(miss_len)
        print(f"  clause-runs<=3 misses {len(ml)} of {len(allg)} gold spans "
              f"({len(ml) / m:.1%})")
        print(f"  their token length: mean {ml.mean():.1f} median "
              f"{np.median(ml):.0f} max {ml.max()}   "
              f"(all gold mean {tl.mean():.1f})")
        print(f"  of the first {len(miss_ex)}: "
              f"{sum(1 for _, i in miss_ex if not i)} are not verbatim in the "
              f"post at all")
        print(f"  length histogram of misses: "
              f"{dict(sorted(Counter(np.clip(ml, 0, 30)).items()))}")
    else:
        print("  clause-runs<=3 misses NOTHING. The pool is saturated and the")
        print("  entire evidence problem is selection, not generation.")

    print("\n" + hdr)
    print("VERDICT TO DRAW")
    print(hdr)
    print("  If a mechanical generator reaches >=99% coverage at a pool size")
    print("  the seven taggers could plausibly rank (say <=20/post), then")
    print("  generation was never the hard part -- boundaries were -- and the")
    print("  right build is enumerate-then-rank with the taggers as scorers,")
    print("  which is ALSO what makes MBR subset selection well-posed.")
    print("  If coverage stalls well below that, the soft-decode sweep in")
    print("  onejob Part 1 is the only route and this lane is closed.")


if __name__ == "__main__":
    main()

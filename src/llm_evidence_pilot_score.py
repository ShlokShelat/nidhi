#!/usr/bin/env python
"""Score the never-evaluated LLM evidence pilot against gold.

WHY. Evidence is the only component with large headroom: the candidate pool's
exact ceiling is 0.9864 (`evidence_true_oracle.py`) against RC7's ~0.806, and
every selector tried so far -- vote thresholds, per-post caps, and two span
rerankers -- lands at ~0.806. The bottleneck is choosing WHICH clause matters,
which is what an instruction-tuned LLM is actually good at.

A pilot run produced exactly that on 200 posts (Qwen-14B, 5-sample
self-consistency), yielding 403 grounded spans over 157 posts. The command to
judge it against gold was never run, so the pilot has sat unevaluated.

THE NUMBER THAT DECIDES IT. Every RC7 tagger's candidates hit gold at 75-81%
(the `all hit` column of evidence_member_loo). If the LLM's spans hit
materially above that, it is a better proposer than anything in the pool and a
full pass over 1635+378 posts is worth 3-4 GPU-hours. If it hits below, the
lane closes for the cost of this script.

Uses the OFFICIAL matcher (containment either direction, 3x token cap, maximum
bipartite matching) so nothing can drift from the official scorer. Reports on
the posts the pilot actually covered -- comparing its macro against a global
figure computed over a different post set would be a denominator mismatch.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     parse_evidence, phrase_f1)


def matches(pred, gold):
    return _span_pair_matches(pred, gold) or _span_pair_matches(gold, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot",
                    default="artifacts/evidence_oof/llmpilot_oof_candidates.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--rc7-dir", default="artifacts/evidence_runs")
    ap.add_argument("--rc7", default="evidence_ep9_rc,evidence_lr2_rc,"
                                     "evidence_s14_rc,evidence_s15_rc,"
                                     "evidence_s16_rc,evidence_s17_rc,"
                                     "evidence_v1_rc")
    args = ap.parse_args()

    d = pd.read_csv(args.pilot, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    gold_by = {r: parse_evidence(e)
               for r, e in zip(lab["row_id"], lab["evidence"])}

    posts = sorted(d["row_id"].unique())
    posts = [p for p in posts if p in gold_by]
    print(f"pilot covers {len(posts)} labelled posts, {len(d)} spans "
          f"({len(d)/max(len(posts),1):.2f}/post)")

    by = {}
    for r, s, c in zip(d["row_id"], d["span"], d.get("conf", [1.0]*len(d))):
        by.setdefault(r, []).append((float(c), str(s)))

    # ---- THE HEADLINE: span-level hit rate vs the taggers' 75-81% ----------
    hit = tot = 0
    gold_hit = gold_tot = 0
    for r in posts:
        g = gold_by[r]
        sp = [s for _, s in by.get(r, [])]
        tot += len(sp)
        hit += sum(1 for s in sp if any(matches(s, gg) for gg in g))
        if g:
            adj = [[j for j, s in enumerate(sp) if matches(s, gg)] for gg in g]
            gold_hit += _max_bipartite_matching(adj, len(sp))
            gold_tot += len(g)
    print()
    print(f"SPAN HIT RATE   {hit}/{tot} = {100*hit/max(tot,1):.1f}%")
    print(f"  RC7 taggers sit at 75-81% on this measure -- that is the bar")
    print(f"GOLD COVERAGE   {gold_hit}/{gold_tot} = "
          f"{100*gold_hit/max(gold_tot,1):.1f}%  (distinct-candidate matching)")

    # ---- macro on the covered posts, LLM vs RC7, same denominator ----------
    gold = [gold_by[r] for r in posts]

    def macro(pred):
        return phrase_f1(gold, pred, average="macro").f1

    for thr in (0.0, 0.4, 0.6, 0.8, 1.0):
        pred = [[s for c, s in by.get(r, []) if c >= thr] for r in posts]
        spp = np.mean([len(p) for p in pred])
        print(f"  LLM conf>={thr:.1f}: macro {macro(pred):.4f}   "
              f"{spp:.2f} spans/post")

    # RC7 on the SAME posts, if its dumps cover them
    agg = {}
    found = 0
    for run in [x.strip() for x in args.rc7.split(",")]:
        for split in ("val", "test"):
            p = os.path.join(args.rc7_dir, f"{run}_{split}_candidates.csv")
            if not os.path.exists(p):
                continue
            t = pd.read_csv(p, keep_default_na=False)
            t["row_id"] = t["row_id"].astype(str)
            for r, s in zip(t["row_id"], t["span"]):
                agg.setdefault(r, {}).setdefault(str(s), 0)
                agg[r][str(s)] += 1
            found += 1
    overlap = [r for r in posts if r in agg]
    print()
    print(f"RC7 dumps cover {len(overlap)}/{len(posts)} of the pilot's posts "
          f"({found} dump files read)")
    if len(overlap) >= 30:
        go = [gold_by[r] for r in overlap]
        lp = [[s for _, s in by.get(r, [])] for r in overlap]
        rp = [[s for s, c in agg[r].items() if c >= 2] for r in overlap]
        print(f"  on those {len(overlap)} posts:")
        print(f"    LLM (all spans)  macro "
              f"{phrase_f1(go, lp, average='macro').f1:.4f}")
        print(f"    RC7 vote>=2      macro "
              f"{phrase_f1(go, rp, average='macro').f1:.4f}")
        print("  NOTE: no indicator suppression applied to either side here, "
              "so both are below their shipped numbers. The COMPARISON is the "
              "point, not the levels.")
    else:
        print("  too little overlap for a paired comparison -- read the hit "
              "rate above instead.")

    print()
    print("=" * 66)
    print("DECIDE: hit rate clearly above 81% -> the LLM is a better proposer")
    print("than any tagger in the pool, and a full 72B pass over 1635+378 posts is")
    print("worth the GPU. Below ~75% -> it is not, and the lane closes here.")
    print("=" * 66)


if __name__ == "__main__":
    main()

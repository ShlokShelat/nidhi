"""
risk_boundary_errors.py

Why does behavior -> ideation resist a fix? Read the actual posts.

The measurement says it is 102 of 343 out-of-fold errors (30%), behavior recall
0.665, ideation precision 0.684, and together those two classes are 68% of the
total weighted-F1 loss. Three approaches have not resolved it:

  - shifting the operating point toward behavior:  flat to negative over 50
                                                   author splits
  - a dedicated binary plan-or-self-harm model:    AP 0.8640 vs the chain's
                                                   0.8948, correlation 0.886
  - 25 mined lexical rules across 4 families:      zero passed

That pattern says the information is not reachable. This script checks that
claim directly, because there is one thing none of those tests looked at: the
GOLD EVIDENCE SPAN. The annotators quoted the exact clause that justified the
label. If those clauses contain plain, explicit plan language, the signal IS in
the post and the models are simply failing to weigh it -- which would mean the
lane is not really closed. If the clauses are oblique, or if they are absent, or
if they look indistinguishable from what appears in genuine ideation posts, then
the boundary is annotator judgement rather than text, and no model reaches it.

That is the difference between "the fix has not been found" and "there is no
fix in the text", and it is worth knowing before spending more GPU on the
largest error bucket in the risk model.

Prints, for the behavior posts predicted ideation:
  - the gold evidence span (the annotator's stated reason)
  - the model's confidence in its wrong answer
  - how often the same span text appears in posts that ARE gold ideation --
    the direct test of whether the cue is discriminative at all

Usage:
  python src/risk_boundary_errors.py
  python src/risk_boundary_errors.py --show 25
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
from scoring import normalize_label, normalize_span, parse_evidence  # noqa: E402

# The official definition of the boundary: behavior needs a plan or an act.
PLAN_CUES = re.compile(
    r"\b(plan|method|pills|rope|hang|jump|bridge|gun|knife|blade|cut|cutting|"
    r"overdose|od|noose|razor|bleach|tonight|tomorrow|next week|date|"
    r"letter|note|will|goodbye|prepare|preparing|bought|saved up|stockpil)\w*",
    re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="artifacts/champion_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    d = pd.read_csv(args.oof, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    d["g"] = [normalize_label(x) for x in d["gold"]]
    d["p"] = [normalize_label(x) for x in d["pred_risk_level"]]
    miss = d[(d.g == "behavior") & (d.p == "ideation")]
    hit = d[(d.g == "behavior") & (d.p == "behavior")]
    ideat = d[d.g == "ideation"]
    print(f"{len(miss)} posts: gold BEHAVIOR, we said IDEATION")
    print(f"{len(hit)} behavior posts we got right, {len(ideat)} gold ideation")

    def spans_of(ids):
        out = []
        for r in ids:
            if r in lab.index:
                out += parse_evidence(lab.loc[r, "evidence"])
        return out

    s_miss, s_hit = spans_of(miss.row_id), spans_of(hit.row_id)
    s_id = spans_of(ideat.row_id)

    def cue_rate(spans):
        if not spans:
            return float("nan")
        return float(np.mean([bool(PLAN_CUES.search(s)) for s in spans]))

    print("\n" + "=" * 72)
    print("IS THE PLAN LANGUAGE ACTUALLY THERE?")
    print("=" * 72)
    print(f"{'subset':<38}{'spans':>7}{'with plan cue':>15}")
    print("-" * 62)
    print(f"{'behavior we MISSED (said ideation)':<38}{len(s_miss):>7}"
          f"{cue_rate(s_miss):>15.3f}")
    print(f"{'behavior we GOT RIGHT':<38}{len(s_hit):>7}"
          f"{cue_rate(s_hit):>15.3f}")
    print(f"{'gold IDEATION (should NOT have plans)':<38}{len(s_id):>7}"
          f"{cue_rate(s_id):>15.3f}")
    print("\nIf the missed behavior posts carry plan cues at a rate close to")
    print("gold IDEATION, the cue does not separate the classes and no model")
    print("or rule can use it. If they look like the behavior posts we get")
    print("right, the signal is there and we are under-weighting it.")

    # Do the SAME span texts appear on both sides of the boundary?
    n_id = Counter(normalize_span(s) for s in s_id)
    both = [s for s in s_miss if n_id.get(normalize_span(s), 0) > 0]
    print(f"\n{len(both)} of {len(s_miss)} missed-behavior spans appear "
          f"VERBATIM in gold-ideation posts too")
    if both:
        print("  e.g.", [s for s in both[:6]])
        print("  Those are annotator judgement calls: identical text, opposite")
        print("  label. No model can separate them from the post alone.")

    print("\n" + "=" * 72)
    print(f"THE {min(args.show, len(miss))} MOST CONFIDENT MISSES")
    print("=" * 72)
    pc = [f"prob_{c}" for c in ("indicator", "ideation", "behavior", "attempt")]
    m = miss.copy()
    if all(c in m.columns for c in pc):
        m["conf"] = m[pc].to_numpy(float).max(1)
        m["p_beh"] = m["prob_behavior"].astype(float)
        m = m.sort_values("conf", ascending=False)
    for _, r in m.head(args.show).iterrows():
        rid = r["row_id"]
        post = str(lab.loc[rid, "post"]) if rid in lab.index else ""
        ev = parse_evidence(lab.loc[rid, "evidence"]) if rid in lab.index else []
        c = f"{r['conf']:.2f}" if "conf" in r else "?"
        pb = f"{r['p_beh']:.2f}" if "p_beh" in r else "?"
        print(f"\n--- {rid}   said ideation @ {c}   P(behavior)={pb}")
        print(f"    GOLD SPAN: {ev if ev else '(none)'}")
        body = " ".join(post.split())
        print(f"    POST: {body[:300]}{'...' if len(body) > 300 else ''}")

    print("\n" + "=" * 72)
    print("READ THE SPANS. The annotator quoted them as the REASON for")
    print("`behavior`. If you cannot tell from those quotes why the post is")
    print("behavior rather than ideation, neither can a model trained on 1635")
    print("posts -- and the lane is closed on evidence rather than on effort.")


if __name__ == "__main__":
    main()

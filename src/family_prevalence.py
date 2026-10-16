"""
family_prevalence.py

Screens every candidate rule family from the symbolic notebook for TEST-SET
PREVALENCE before any engineering effort is spent on it.

Motivation: a rule family can carry strong symbolic evidence, survive
author-disjoint A/B and leave-one-author-out validation, and show a
substantial val-set score gain -- and still change only a handful of test
predictions, because the phenomenon it captures is concentrated in a couple of
val authors and is nearly absent from test. Validation error density and test
error density are different distributions. A perfect rule for something that
occurs rarely in the test set has no leverage on a submission's score.

So each family is scored on the only quantity that predicts submission-score
movement: how many TEST posts the family could even be eligible to change,
i.e. the probe fires AND the champion's current test prediction is a label the
family's transition is allowed to act on.

IMPORTANT -- these probes are deliberately LOOSE (recall-oriented). They are
prevalence upper bounds, not finished rules. A real rule, with the guards and
exclusions the methodology demands, will always fire on FEWER posts than its
probe. That asymmetry is what makes the screen safe to act on:

  * class C (0 eligible test posts even under a loose probe) is a DEFINITIVE
    kill -- no precise rule built from this family can do better than zero.
  * class A/B is necessary but NOT sufficient: it says the phenomenon is
    present often enough to be worth engineering, not that a precise rule
    will reach that count.

Usage:
  python src/family_prevalence.py
  python src/family_prevalence.py --show-posts
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import match_any
from scoring import normalize_label

# CALIBRATION -- corrected after further analysis of a rule-layer submission.
#
# The first estimate divided the deployed rule layer's composite gain by its
# count of changed test posts, giving too small a per-post value. That was
# wrong: it averaged correct and incorrect changes together, so it measured
# the NET effect per changed post, not the value of a CORRECT one.
#
# A later single-variable submission gives a clean read: it changed only a
# small, known number of test posts and nothing else (evidence and factors
# byte-identical, S2 unchanged), so the whole composite move is attributable
# to those few risk corrections. Dividing that composite gain by the count of
# posts changed, assuming all were corrected, gives the per-post value below
# -- several times the original (wrong) estimate. It also implies the
# original rule layer was mostly, but not entirely, correct -- consistent
# with it over-firing before a later fix.
#
#     value of one CORRECTED test post ~= 0.001 composite
PER_CORRECT_POST = 0.001
CLASS_A_MIN = 3   # worth building a rule for if corrected
CLASS_B_MIN = 1

# Each family: probe patterns, the transition it would encode, and the val
# row_ids that motivated it (from the symbolic notebook's family catalog).
FAMILIES = {
    "1 preparatory_farewell": {
        "from": ["ideation"], "to": "behavior",
        "cases": "P00876 P00878 P00879 P00880 P01949 P01950 P01953 P01955",
        "patterns": [
            r"\bgood\s?bye\b", r"\bfarewell\b",
            r"\bsuicid(?:e|al)\s+(?:note|letter)s?\b",
            r"\bmy (?:last|final) (?:note|letter|words|message)\b",
            r"\bwhen i die\b", r"\bmy grave\b", r"\bforget me\b",
            r"\byou'?ll learn to live without me\b",
        ],
    },
    "2 third_person_death": {
        "from": ["ideation"], "to": "indicator",
        "cases": "P00118 P00119 P00559 P00568",
        "patterns": [
            r"\b(?:everyone|everybody|people|they)\b[^.!?]{0,30}\bwants?\b"
            r"[^.!?]{0,15}\bme dead\b",
            r"\bpeople posting about\b[^.!?]{0,40}\bdie\b",
            r"\bi hope\b[^.!?]{0,25}\b(?:my (?:dad|mom|mother|father)|he|she|they)\b"
            r"[^.!?]{0,15}\bdies?\b",
        ],
    },
    "3 protective_negation": {
        "from": ["ideation"], "to": "indicator",
        "cases": "P00753 P00990",
        "patterns": [
            r"\bi (?:just )?(?:do not|don'?t|dont|not) want to (?:kill myself|die)\b",
            r"\bi (?:just )?don'?t wanna (?:die|kill myself)\b",
            r"\bi'?m not going to kill myself\b",
            r"\bi (?:won'?t|will not) kill myself\b",
        ],
    },
    "4 future_commitment_dated": {
        "from": ["ideation"], "to": "behavior",
        "cases": "P01138 P01556 P01958",
        "patterns": [
            r"\bon my birthday\b", r"\bhappy death day\b",
            r"\bmy deadline is\b",
            r"\b(?:giving|gave) myself\b[^.!?]{0,20}\b(?:week|month|day)s?\b",
            r"\bsoon my time will come\b",
            r"\b(?:going to|gonna) (?:commit suicide|kill myself|end it)\b"
            r"[^.!?]{0,15}\bsoon\b",
            r"\b(?:kill myself|commit suicide|end it all)\b[^.!?]{0,25}"
            r"\bin (?:\d+|a few|two|three) (?:weeks?|months?|days?)\b",
        ],
    },
    "5 behavior_in_progress": {
        "from": ["indicator", "ideation"], "to": "behavior",
        "cases": "P00998 P01960",
        "patterns": [
            r"\bstarvation\b", r"\bstarving myself\b",
            r"\b(?:no|not|haven'?t) (?:eating|eaten|ate|drinking)\b"
            r"[^.!?]{0,40}\b(?:die|month|week|days?)\b",
            r"\bend of the (?:first|second|third|\d+)(?:st|nd|rd|th)? day\b",
            r"\bday \d+\b[^.!?]{0,30}\b(?:no|not) (?:eating|drinking|food)\b",
            r"\bsince i'?ve (?:ate|eaten)\b",
        ],
    },
    "6 prior_attempt_plus_current": {
        "from": ["ideation", "behavior"], "to": "attempt",
        "cases": "P00679 P01152 P01156",
        "patterns": [
            r"\b(?:tried|attempted) to (?:kill myself|commit suicide|end my life)\b",
            r"\bsuicide attempts?\b", r"\battempted suicide\b",
            r"\bi'?ve (?:already )?tried to (?:commit|kill)\b",
            r"\balready had \d+ suicide attempts?\b",
        ],
    },
    "7 conditional_method": {
        "from": ["behavior"], "to": "ideation",
        "cases": "P02251",
        "patterns": [
            r"\bif i (?:just |only )?had\b", r"\bif i had access to\b",
            r"\bwould vanish if\b",
            r"\bi wish i (?:had|could)\b[^.!?]{0,30}"
            r"\b(?:gun|pills?|rope|blade|knife)\b",
        ],
    },
    "8 method_inquiry": {
        "from": ["behavior"], "to": "ideation",
        "cases": "P01623",
        "patterns": [
            r"\bpassive ways to die\b", r"\bways to die\b",
            r"\bhow (?:do|can) i (?:die|kill myself)\b",
            r"\bpainless (?:way|method)s?\b",
            r"\bbest way to (?:die|kill myself)\b",
        ],
    },
    "9 method_seeking_failed_attempts": {
        "from": ["behavior"], "to": "attempt",
        "cases": "P00206",
        "patterns": [
            r"\bsearching for methods\b", r"\bkeep failing attempts\b",
            r"\bi need (?:to find )?a way to do it\b",
            r"\bfailed attempts?\b",
        ],
    },
    "10 post_attempt_evidence": {
        "from": ["ideation", "behavior"], "to": "attempt",
        "cases": "P01226",
        "patterns": [
            r"\bwelfare check\b", r"\bbloody razor\b",
            r"\bwoke up in the hospital\b", r"\bstomach pumped\b",
            r"\bthe cops\b[^.!?]{0,40}\b(?:razor|blood|blade)\b",
        ],
    },
    "11 final_commitment": {
        "from": ["ideation"], "to": "behavior",
        "cases": "P01227 P01230 P01233",
        "patterns": [
            r"\bmade my (?:final )?decision\b",
            r"\bi'?m not going to change my mind\b",
            r"\beverything is ready\b",
            r"\bno point in continuing to live\b",
            r"\bi'?ve decided\b[^.!?]{0,30}\b(?:die|end|go)\b",
        ],
    },
    "12 ongoing_self_harm_boundary": {
        "from": ["ideation", "attempt"], "to": "behavior",
        "cases": "P01389 P01957 P02345",
        "patterns": [
            r"\bi tried cutting\b", r"\bnot going to fight back\b",
            r"\bcarve\b", r"\bself[- ]harm(?:ing|ed)?\b",
            r"\bactually lethal\b",
        ],
    },
    "13 farewell_failed_help": {
        "from": ["ideation"], "to": "behavior",
        "cases": "P01949",
        "patterns": [
            r"\btried to look for help\b[^.!?]{0,30}\bwithout any success\b",
            r"\bno one (?:helped|helps|would help) me\b",
            r"\bi (?:just )?have no otherwise\b",
            r"\bi'?ve tried everything\b",
        ],
    },
}


def classify(n_test_eligible):
    if n_test_eligible >= CLASS_A_MIN:
        return "A pursue"
    if n_test_eligible >= CLASS_B_MIN:
        return "B report-only"
    return "C drop"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--val-baseline", default="artifacts/val_baseline_reconstructed.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--test-preds", default="artifacts/risk_blend_with_rules.csv")
    ap.add_argument("--show-posts", action="store_true")
    args = ap.parse_args()

    val = pd.read_csv(args.val).merge(
        pd.read_csv(args.val_baseline)[["row_id", "champion_final"]], on="row_id")
    val["gold"] = [normalize_label(x) for x in val["risk_level"]]
    val["champ"] = [normalize_label(x) for x in val["champion_final"]]

    test = pd.read_csv(args.test).merge(
        pd.read_csv(args.test_preds)[["row_id", "pred_risk_level"]], on="row_id")
    test["champ"] = [normalize_label(x) for x in test["pred_risk_level"]]

    rows = []
    eligible_test_union = {}
    for name, f in FAMILIES.items():
        vm = val[[match_any(p, f["patterns"]) for p in val["post"]]]
        v_elig = vm[vm["champ"].isin(f["from"])]
        v_corr = v_elig[(v_elig["champ"] != v_elig["gold"]) & (v_elig["gold"] == f["to"])]
        v_break = v_elig[v_elig["champ"] == v_elig["gold"]]

        tm = test[[match_any(p, f["patterns"]) for p in test["post"]]]
        t_elig = tm[tm["champ"].isin(f["from"])]

        rows.append({
            "family": name,
            "transition": f"{'/'.join(f['from'])}->{f['to']}",
            "val_match": len(vm), "val_elig": len(v_elig),
            "val_corr": len(v_corr), "val_break": len(v_break),
            "test_match": len(tm), "test_elig": len(t_elig),
            "class": classify(len(t_elig)),
        })

        for _, r in t_elig.iterrows():
            eligible_test_union.setdefault(r["row_id"], []).append(name.split()[0])

        if args.show_posts and len(t_elig):
            print(f"\n### {name}  eligible TEST posts ({len(t_elig)}):")
            for _, r in t_elig.iterrows():
                print(f"  {r['row_id']}  champion={r['champ']:<9} "
                      f"{str(r['post'])[:150]}")

    out = pd.DataFrame(rows).sort_values(
        ["test_elig", "val_corr"], ascending=False, kind="mergesort")

    print("\n" + "=" * 118)
    print("FAMILY PREVALENCE SCREEN  (loose probes = upper bound on what any "
          "precise rule from the family could reach)")
    print("=" * 118)
    print(f"{'family':<34}{'transition':<22}"
          f"{'val:match':>10}{'elig':>6}{'corr':>6}{'brk':>5}"
          f"{'test:match':>12}{'elig':>6}   class")
    for _, r in out.iterrows():
        print(f"{r['family']:<34}{r['transition']:<22}"
              f"{r['val_match']:>10}{r['val_elig']:>6}{r['val_corr']:>6}{r['val_break']:>5}"
              f"{r['test_match']:>12}{r['test_elig']:>6}   {r['class']}")

    print("\nlegend: elig = probe fires AND champion's current prediction is a "
          "label this\n        transition may act on. corr/brk = val posts the "
          "family could correct / could break.")
    print(f"class A >= {CLASS_A_MIN} eligible test posts, B = {CLASS_B_MIN}-"
          f"{CLASS_A_MIN - 1}, C = 0 (definitive kill).")
    n = {c: int((out['class'] == c).sum()) for c in out['class'].unique()}
    print(f"\nverdict counts: {n}")
    print(f"total eligible test posts across ALL families: "
          f"{int(out['test_elig'].sum())} of {len(test)} "
          f"({len(eligible_test_union)} UNIQUE posts after dedup -- families overlap)")

    print("\n" + "=" * 118)
    print("CEILING ON THE REMAINING RISK-SIDE SYMBOLIC PROGRAM")
    print("=" * 118)
    tp = test.set_index("row_id")
    for rid, fams in sorted(eligible_test_union.items()):
        print(f"  {rid}  champion={tp.loc[rid, 'champ']:<9} "
              f"families={','.join(fams)}")
        print(f"      {str(tp.loc[rid, 'post'])[:190]}")
    ceiling = len(eligible_test_union) * PER_CORRECT_POST
    print(f"\n  {len(eligible_test_union)} unique test posts are eligible for ANY "
          f"rule from ANY remaining family.")
    print(f"  If every single one were corrected -- which no realistic rule set "
          f"achieves -- the\n  upper bound on composite gain is about "
          f"{ceiling:+.4f}, against a gap to first place of ~0.049.")


if __name__ == "__main__":
    main()

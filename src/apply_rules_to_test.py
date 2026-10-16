"""
apply_rules_to_test.py

Builds the final risk prediction file: takes an EXISTING risk_blend CSV of
test-set predictions from the equal-weight risk ensemble and applies the
deployed symbolic rules on top, in the order they are meant to compose --
urgency, then completed_act.

This does NOT recompute any model probabilities. It only overrides the final
predicted label for posts where a rule fires, so the input file fully
determines everything the rules do not touch.

Usage:
  python src/apply_rules_to_test.py \
      --test data/test.csv \
      --risk-blend artifacts/risk_blend_equal.csv \
      --rules urgency,completed_act \
      --out artifacts/risk_blend_with_rules.csv

IMPORTANT -- the deployed input is artifacts/risk_blend_equal.csv. Any other
blend file is a different risk model and will run without error while
producing different labels, because this script overrides labels rather than
recomputing them. Check --risk-blend before running.
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

URGENCY_PHRASES = [
    r"\babout to\b", r"\bgoing to end\b", r"\bgoing to kill\b",
    r"\bget(ting)? up the nerve\b", r"\bbefore .{0,20} (gets?|comes?) back\b",
    r"\btonight\b.{0,30}\b(kill|end|die|suicide)\b",
    r"\bwanna end it\b", r"\bwant to end it\b",
    r"\bgave myself\b.{0,20}(month|week|day|deadline)",
    r"\bsoon i'?ll be gone\b", r"\bi don'?t know if i'?m doing it\b",
]

METHOD_PHRASES = [
    r"\bpills?\b", r"\bgun\b", r"\bloaded gun\b",
    r"\baccess to (a |an )?(loaded )?(gun|weapon|firearm)\b",
    r"\brope\b", r"\bhang(ing|ed)?\b",
    r"\boverdos(e|ed|ing)\b", r"\bcut(ting)? my\b", r"\bcut my (wrist|neck|arm)",
    r"\bbridge\b", r"\bjump(ing|ed)? (off|from)\b", r"\bpoison\b",
    r"\bblade\b", r"\bknife\b", r"\bnoose\b",
    r"\b(took|taking|taken|take)\b.{0,15}\d+\b", r"\bswallow(ed|ing)?\b",
]

COMPLETED_ACT_PHRASES = [
    r"\bi'?ve cut\b", r"\bi cut my\b",
    r"\b(i|just|already)\b.{0,10}\btook\b.{0,15}\d+\b",
    r"\bi'?ve taken\b", r"\bi have taken\b", r"\btaken \d+\b",
    r"\bi swallowed\b", r"\bi overdosed\b",
]

from candidate_patterns import (CANDIDATE_DEFS, PRIOR_ATTEMPT,  # noqa: E402
                                MISSING_METHOD_PHRASES, SCHEDULED_TIME,
                                SCHEDULED_TIME_LOOSE)

# ---------------------------------------------------------------------------
# DEPLOYED rules.
#
# `urgency` caps urgent language at Ideation only when the post names no
# concrete method and no concrete plan, so its absent-list has to enumerate
# both. METHOD_PHRASES plus MISSING_METHOD_PHRASES covers the method half;
# SCHEDULED_TIME covers the plan half. A method or a named schedule puts the
# post outside urgency's premise, where the downgrade would be wrong.
# ---------------------------------------------------------------------------
RULE_DEFS = {
    "urgency": {
        "present": URGENCY_PHRASES,
        # A named hour or date is preparatory planning, so it belongs in the
        # absent-list alongside the named methods.
        "absent": METHOD_PHRASES + MISSING_METHOD_PHRASES + SCHEDULED_TIME,
        "from_labels": ["behavior", "attempt"], "to_label": "ideation",
    },
    "completed_act": {
        "present": COMPLETED_ACT_PHRASES, "absent": None,
        "from_labels": ["behavior"], "to_label": "attempt",
    },
}

# ---------------------------------------------------------------------------
# CANDIDATE rules -- under evaluation, NOT deployed. Promoting one is a move
# into RULE_DEFS above; nothing here changes apply_rules()'s default.
# ---------------------------------------------------------------------------

# urgency without the extended method and schedule absent-lists, retained so
# that prediction files produced by the narrower variant stay reproducible.
CANDIDATE_DEFS["urgency_v1_prefix"] = {
    "present": URGENCY_PHRASES, "absent": METHOD_PHRASES,
    "from_labels": ["behavior", "attempt"], "to_label": "ideation",
}

# Candidate isolating the schedule guard: urgency must not downgrade when the
# post names a SPECIFIC scheduled time. A named schedule is planning, which
# the taxonomy places at Behavior -- distinct from the vague imminence
# ("about to", "soon") urgency is meant to cap.
CANDIDATE_DEFS["urgency_scheduled_fix"] = {
    "present": URGENCY_PHRASES,
    "absent": METHOD_PHRASES + MISSING_METHOD_PHRASES + SCHEDULED_TIME,
    "from_labels": ["behavior", "attempt"], "to_label": "ideation",
}

# Loose variant, kept only so the season/"same plan" patterns stay measurable.
CANDIDATE_DEFS["urgency_scheduled_fix_loose"] = {
    "present": URGENCY_PHRASES,
    "absent": METHOD_PHRASES + MISSING_METHOD_PHRASES + SCHEDULED_TIME_LOOSE,
    "from_labels": ["behavior", "attempt"], "to_label": "ideation",
}

# Built here (not in candidate_patterns) so it always reuses the LIVE deployed
# urgency lists -- the guarded variant differs from `urgency` by exactly the
# extra prior-attempt absent-conditions and nothing else.
CANDIDATE_DEFS["urgency_guarded"] = {
    "present": URGENCY_PHRASES,
    "absent": METHOD_PHRASES + MISSING_METHOD_PHRASES + PRIOR_ATTEMPT,
    "from_labels": ["behavior", "attempt"], "to_label": "ideation",
}

ALL_DEFS = {**RULE_DEFS, **CANDIDATE_DEFS}


def match_any(text, patterns):
    text_l = str(text).lower()
    return any(re.search(p, text_l) for p in patterns)


def trigger_fires(post, rd):
    """Evaluate a rule's trigger. Supports three keys:

      present     -- ANY-of  (fires if at least one pattern matches)
      present_all -- ALL-of-ANY: a list of pattern GROUPS; every group must
                     contribute at least one match. This is what compound
                     rules need (e.g. farewell language AND a corroborating
                     signal), which a present/absent-only schema cannot
                     express.
      absent      -- NONE-of (trigger suppressed if any pattern matches)
      max_words   -- optional length cap; trigger suppressed on posts longer
                     than this. Some patterns are only valid when the matched
                     phrase IS essentially the whole post rather than one
                     clause inside a longer first-person narrative (see the
                     third_person_death candidate), and a cap is the honest
                     way to encode that rather than an endless absent-list.

    A rule may use `present`, `present_all`, or both.
    """
    if rd.get("max_words") is not None:
        if len(str(post).split()) > rd["max_words"]:
            return False
    ok = True
    if rd.get("present"):
        ok = ok and match_any(post, rd["present"])
    if rd.get("present_all"):
        ok = ok and all(match_any(post, grp) for grp in rd["present_all"])
    if not (rd.get("present") or rd.get("present_all")):
        raise ValueError("rule has neither 'present' nor 'present_all'")
    if ok and rd.get("absent"):
        ok = not match_any(post, rd["absent"])
    return ok


def apply_rules(post, label, rule_keys=("urgency", "completed_act"), defs=None):
    """SINGLE SOURCE OF TRUTH for the deployed rule layer. Any script that
    needs to know whether/how the rules affect a given (post, label) pair --
    the test-set submission builder, the failure miner, the ablation harness --
    must import and call THIS function rather than re-implementing the regex
    logic, so they can never silently drift apart.

    Returns (final_label, trace); trace has one dict per rule checked with
    keys rule / fired / applicable / changed. Rules apply in the given order,
    each seeing the previous one's output.

    `defs` defaults to ALL_DEFS so candidate keys can be applied by name;
    the default rule_keys are the deployed pair, so a caller that passes
    neither gets exactly the deployed behaviour.
    """
    d = ALL_DEFS if defs is None else defs
    lbl = label
    trace = []
    for k in rule_keys:
        rd = d[k]
        fires = trigger_fires(post, rd)
        applicable = rd["from_labels"] is None or lbl in rd["from_labels"]
        changed = False
        if fires and applicable and lbl != rd["to_label"]:
            changed = True
            lbl = rd["to_label"]
        trace.append({"rule": k, "fired": fires, "applicable": applicable, "changed": changed})
    return lbl, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--risk-blend", required=True,
                    help="existing risk_blend CSV from the EQUAL-WEIGHT risk "
                         "ensemble — verify this path before running")
    ap.add_argument("--rules", default="urgency,completed_act")
    ap.add_argument("--out", default="artifacts/risk_blend_with_rules.csv")
    args = ap.parse_args()

    # CANONICAL TEXT, matching the space the rule patterns are written for.
    #
    # Every labelled split reaches the patterns through data_audit.load, which
    # applies clean_text (HTML unescape, NFKC, folding of typographic
    # characters). The regexes contain literal straight apostrophes -- i'?ll,
    # don'?t, i'?ve cut -- which cannot match a curly one, so reading the test
    # CSV raw would silently disable those patterns on every post that uses
    # typographic punctuation. Loading through the same function keeps the
    # matching space identical on both sides.
    from data_audit import load as _load_canonical
    test_df = _load_canonical(args.test).set_index("row_id")
    blend_df = pd.read_csv(args.risk_blend)

    rule_keys = [r.strip() for r in args.rules.split(",")]
    for k in rule_keys:
        if k not in ALL_DEFS:
            raise SystemExit(f"unknown rule: {k}. known: {sorted(ALL_DEFS)}")

    preds = blend_df["pred_risk_level"].str.lower().tolist()
    row_ids = blend_df["row_id"].tolist()
    posts = [test_df.loc[rid, "post"] for rid in row_ids]

    # Delegate to apply_rules() rather than re-deriving the match logic here:
    # a second copy of the trigger evaluation is exactly the drift the
    # apply_rules docstring warns about, and a local copy would silently skip
    # any rule using the compound `present_all` schema.
    total_changed = 0
    per_rule_counts = {k: 0 for k in rule_keys}
    for i, post in enumerate(posts):
        new_lbl, trace = apply_rules(post, preds[i], rule_keys)
        for t in trace:
            if t["changed"]:
                per_rule_counts[t["rule"]] += 1
                total_changed += 1
        preds[i] = new_lbl
    for k in rule_keys:
        print(f"  rule '{k}': changed {per_rule_counts[k]} predictions")

    out_df = blend_df.copy()
    out_df["pred_risk_level"] = [p.capitalize() for p in preds]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"\nTotal predictions changed vs input: {total_changed}")
    print(f"Per-rule breakdown: {per_rule_counts}")
    print(f"wrote {args.out}")

    from collections import Counter
    print(f"\noriginal risk distribution:  {dict(Counter(blend_df['pred_risk_level']))}")
    print(f"with-rules risk distribution: {dict(Counter(out_df['pred_risk_level']))}")


if __name__ == "__main__":
    main()

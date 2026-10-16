"""
validate_rule_triggers.py

Before writing ANY post-processing rule, check it against the FULL
training set (not just the 44 hand-picked P1 cases where the ensemble
was already wrong). A trigger that looks perfect on 44 confirmed-wrong
cases could still fire on hundreds of correctly-classified posts
elsewhere — this script is the reality check.

For each candidate trigger, reports:
  - count: how many posts in the full training set match
  - gold label distribution among matches
  - precision: fraction of matches whose gold label is in the
    "expected" direction for that rule
  - a handful of example matches for manual spot-check

Usage:
  python src/validate_rule_triggers.py --data artifacts/train_split.csv
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

import pandas as pd


# ---------------------------------------------------------------------------
# Candidate triggers, derived from the 37-case manual read-through.
# ---------------------------------------------------------------------------

URGENCY_PHRASES = [
    r"\babout to\b", r"\bgoing to end\b", r"\bgoing to kill\b",
    r"\bget(ting)? up the nerve\b", r"\bbefore .{0,20} (gets?|comes?) back\b",
    r"\btonight\b.{0,30}\b(kill|end|die|suicide)\b",
    r"\bwanna end it\b", r"\bwant to end it\b", r"\bgave myself\b.{0,20}(month|week|day|deadline)",
    r"\bsoon i'?ll be gone\b", r"\bi don'?t know if i'?m doing it\b",
]

METHOD_PHRASES = [
    r"\bpills?\b", r"\bgun\b", r"\brope\b", r"\bhang(ing|ed)?\b",
    r"\boverdos(e|ed|ing)\b", r"\bcut(ting)? my\b", r"\bcut my (wrist|neck|arm)",
    r"\bbridge\b", r"\bjump(ing|ed)? (off|from)\b", r"\bpoison\b",
    r"\bblade\b", r"\bknife\b", r"\bnoose\b", r"\btook \d+\b",
    r"\bswallow(ed|ing)?\b",
]

COMPLETED_ACT_PHRASES = [
    r"\bi'?ve cut\b", r"\bi took \d+\b", r"\bjust took\b",
    r"\bi swallowed\b", r"\bi overdosed\b", r"\bi cut my\b",
]

PASSIVE_DEATH_PHRASES = [
    r"\bwait(ing)? for (my )?death\b", r"\blet my body decay\b",
    r"\bi'?ll probably die\b", r"\bnothing to live for\b",
    r"\bi give up\b.{0,30}\bdone\b",
]

SUICIDE_NOTE_TITLE = [r"^suicidal notes?\b"]


def match_any(text, patterns):
    text_l = str(text).lower()
    return any(re.search(p, text_l) for p in patterns)


def evaluate_trigger(df, name, matcher_patterns, expected_labels, example_n=3):
    matches = df[df["post"].apply(lambda t: match_any(t, matcher_patterns))]
    n = len(matches)
    if n == 0:
        print(f"\n{name}: 0 matches in training set")
        return

    label_counts = Counter(matches["risk_level"].str.lower())
    n_expected = sum(label_counts.get(l, 0) for l in expected_labels)
    precision = n_expected / n if n else 0.0

    print(f"\n{'=' * 90}")
    print(f"TRIGGER: {name}")
    print(f"{'=' * 90}")
    print(f"  matches in training set: {n}")
    print(f"  gold label distribution: {dict(label_counts)}")
    print(f"  precision (fraction matching expected {expected_labels}): {precision:.3f}")
    print(f"  example matches:")
    for _, row in matches.head(example_n).iterrows():
        print(f"    [{row['risk_level']}] {str(row['post'])[:120]}")


def evaluate_compound(df, name, present_patterns, absent_patterns, expected_labels, example_n=3):
    """For the Urgency-not-Planning-style rules: present_patterns must match,
    absent_patterns must NOT match, for the trigger to fire."""
    def fires(text):
        t = str(text).lower()
        has_present = any(re.search(p, t) for p in present_patterns)
        has_absent = any(re.search(p, t) for p in absent_patterns)
        return has_present and not has_absent

    matches = df[df["post"].apply(fires)]
    n = len(matches)
    if n == 0:
        print(f"\n{name}: 0 matches in training set")
        return

    label_counts = Counter(matches["risk_level"].str.lower())
    n_expected = sum(label_counts.get(l, 0) for l in expected_labels)
    precision = n_expected / n if n else 0.0

    print(f"\n{'=' * 90}")
    print(f"COMPOUND TRIGGER: {name}")
    print(f"{'=' * 90}")
    print(f"  matches (urgency present AND method absent): {n}")
    print(f"  gold label distribution: {dict(label_counts)}")
    print(f"  precision (fraction matching expected {expected_labels}): {precision:.3f}")
    print(f"  example matches:")
    for _, row in matches.head(example_n).iterrows():
        print(f"    [{row['risk_level']}] {str(row['post'])[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="train_split.csv or full labeled data — needs 'post' and 'risk_level' columns")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} posts. Gold label distribution overall: "
          f"{dict(Counter(df['risk_level'].str.lower()))}")

    # 1. Urgency not Planning -- expect these to be Ideation (not Behavior/Attempt)
    evaluate_compound(
        df, "Urgency present + method/means absent -> expect IDEATION",
        URGENCY_PHRASES, METHOD_PHRASES, expected_labels=["ideation"])

    # 2. Completed act -> expect Attempt (not Behavior)
    evaluate_trigger(
        df, "Completed self-harm act phrasing -> expect ATTEMPT",
        COMPLETED_ACT_PHRASES, expected_labels=["attempt"])

    # 3. Passive death acceptance -> expect Indicator (not Ideation)
    evaluate_trigger(
        df, "Passive/fatalistic death phrasing -> expect INDICATOR",
        PASSIVE_DEATH_PHRASES, expected_labels=["indicator"])

    # 4. "Suicidal notes" literal title -> expect Behavior/Attempt
    evaluate_trigger(
        df, "Post starts with 'Suicidal notes' -> expect BEHAVIOR/ATTEMPT",
        SUICIDE_NOTE_TITLE, expected_labels=["behavior", "attempt"])


if __name__ == "__main__":
    main()

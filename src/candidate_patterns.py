"""
candidate_patterns.py

CANDIDATE symbolic rules under evaluation -- NOT deployed. Imported by
apply_rules_to_test.py into ALL_DEFS so that every consumer (ablation
harness, failure miner, submission builder) sees one consistent set of
definitions and no script re-implements trigger logic locally.

Each candidate documents the mechanism it encodes. Schema is the same as the
deployed RULE_DEFS, plus the compound `present_all` key (list of pattern
GROUPS, every group must contribute at least one match) for triggers that
need co-occurring signals rather than a single keyword OR-list.

Nothing in this file is active until a key is passed explicitly to
apply_rules(rule_keys=...) or promoted into RULE_DEFS.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Preparatory farewell / suicide note  ->  escalate IDEATION to BEHAVIOR
#
# Mechanism: the post is (or quotes) a valediction -- an explicit note/letter
# artefact or a goodbye -- AND carries a corroborating signal addressed to
# survivors (apology, posthumous instruction, grave reference). Writing a
# farewell artefact is preparatory behaviour under the risk taxonomy, which
# places it above ideation.
#
# The conjunction is the point: farewell-flavoured language alone is common in
# posts that are only ideation, so a bare keyword OR-list over-triggers and is
# not acceptable here. The rule deliberately misses valedictions carried only
# by temporal finality or by a long letter with no valediction token, rather
# than adding bespoke patterns that generalise to nothing.
# ---------------------------------------------------------------------------

# Group A -- the valediction / note artefact itself.
FAREWELL_ARTEFACT = [
    r"\bgood\s?bye\b",
    r"\bfarewell\b",
    r"\bsuicid(?:e|al)\s+(?:note|letter)s?\b",
    r"\bmy\s+(?:last|final)\s+(?:note|letter|words|message|post)\b",
    r"\bthis is the end\b",
]

# Group A' -- broader variant: any bare note/letter token. Held as a separate
# candidate because the narrow patterns require the note/letter token to sit
# inside a fixed template, which a misspelling or an unusual modifier defeats;
# the looser trigger buys that recall at some precision cost, and keeping it
# separate means that cost is measured rather than assumed.
FAREWELL_ARTEFACT_BROAD = FAREWELL_ARTEFACT + [
    r"\b(?:note|letter)s?\b",
]

# Group B -- corroborating signal addressed to those left behind.
POSTHUMOUS_ADDRESS = [
    r"\bi'?m sorry\b", r"\bi am sorry\b", r"\bsorry i\b",
    r"\bnone of it was your fault\b", r"\bit wasn'?t your (?:fault|job)\b",
    r"\bnever blame yourself\b",
    r"\b(?:at |to )?my grave\b",
    r"\bwhen i die\b",
    r"\bi'?m gone\b",
    r"\bdon'?t (?:cry|grieve)\b",
    r"\bforget me\b",
    r"\byou'?ll learn to live without me\b",
    r"\bi love you all\b",
]

# ---------------------------------------------------------------------------
# 2. Third-person / externalised death reference -> cap IDEATION at INDICATOR
#
# Mechanism: the death-adjacent vocabulary that drove the model to Ideation
# is not the author stating self-directed intent -- it is someone ELSE
# wanting the author dead, a comment about other people's posts, or a wish
# about a third party. Guarded hard by an explicit first-person-intent
# absent-list so it can never fire on a genuine self-statement.
# ---------------------------------------------------------------------------
THIRD_PERSON_DEATH = [
    r"\b(?:everyone|everybody|every one|people|they|nobody|no one)\b[^.!?]{0,30}"
    r"\bwants?\b[^.!?]{0,15}\bme dead\b",
    r"\b(?:everyone|everybody|people|they)\b[^.!?]{0,30}\bwants? me to die\b",
    r"\bpeople posting about\b[^.!?]{0,40}\bdie\b",
    r"\bi hope\b[^.!?]{0,25}\b(?:my (?:dad|mom|mum|mother|father|brother|sister)|he|she|they)\b"
    r"[^.!?]{0,15}\bdies?\b",
]

FIRST_PERSON_INTENT = [
    r"\bi wan(?:t|na) to die\b", r"\bi wanna die\b", r"\bi want to die\b",
    r"\bkill myself\b", r"\bkms\b", r"\bend my life\b", r"\bend it all\b",
    r"\bcommit suicide\b", r"\bi'?m suicidal\b", r"\bmy suicide\b",
    r"\btake my (?:own )?life\b", r"\bi want to be dead\b",
    r"\bsuicide note\b", r"\bhurt myself\b",
    # Indirect statements of self-directed intent. Without these, a long
    # first-person suicidal narrative that also contains an externalised
    # death clause would slip the guard and be capped at indicator.
    r"\bi deserve to die\b",
    r"\bwish i (?:didn'?t|hadn'?t) wake up\b",
    r"\bwish i (?:was|were) dead\b",
    r"\bi should have taken\b",
]

# ---------------------------------------------------------------------------
# 3. Explicit protective negation -> cap IDEATION at INDICATOR
#
# Mechanism: grammatical negation of self-directed intent, usually with a
# stated reason (family, a specific person). Deliberately narrow -- it matches
# explicit negation only, and not the far fuzzier judgement of whether the
# author is engaged with help, which no surface pattern can decide reliably.
# ---------------------------------------------------------------------------
PROTECTIVE_NEGATION = [
    r"\bi (?:just )?(?:do not|don'?t|dont) want to (?:kill myself|die)\b",
    r"\bi (?:just )?not want to kill myself\b",
    r"\bi (?:just )?don'?t wanna (?:die|kill myself)\b",
    r"\bi'?m not going to kill myself\b",
    r"\bi (?:won'?t|will not) kill myself\b",
    r"\bnot want to kill myself because of\b",
]

# ---------------------------------------------------------------------------
# 4. Prior-attempt guard on the deployed `urgency` rule
#
# Not a new rule: a proposed EXTRA absent-condition for urgency. When a post
# reports a prior suicide attempt, urgency's downgrade to Ideation is
# suspect: a reported prior attempt is itself evidence of past behaviour, so
# the "urgent language without a plan" premise no longer holds. Registered as
# a full variant rule so the harness can score `urgency` and `urgency_guarded`
# head to head on identical predictions.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5. Missing methods in the deployed `urgency` rule's guard
#
# Not a new hypothesis -- a completion of an existing rule, justified by that
# rule's OWN stated premise. urgency exists to stop urgent language being read
# as more severe than Ideation *when no concrete method or plan is present*,
# and its METHOD_PHRASES absent-list is what enforces the "no method" half.
# Any named method missing from that list lets urgency downgrade a post that
# does name a method, i.e. a post outside urgency's own premise.
#
# These entries close that gap: methods that a post can name explicitly but
# that the original list does not cover.
# ---------------------------------------------------------------------------
MISSING_METHOD_PHRASES = [
    r"\bsuicide by cop\b",
    r"\bdrug myself\b",
    r"\bcarbon monoxide\b",
    r"\basphyxiat(?:e|ion|ing)\b",
    r"\bsuffocat(?:e|ion|ing)\b",
    r"\bdrown(?:ing)?\b",
    r"\bslit(?:ting)? my\b",
    r"\brazor\b",
    r"\bbleach\b",
    r"\bantifreeze\b",
    r"\bexhaust fumes?\b",
]

# ---------------------------------------------------------------------------
# 6. Specific scheduled time -- second missing guard on `urgency`
#
# urgency exists to cap VAGUE imminence ("about to", "soon") that carries no
# plan. A NAMED schedule is the opposite: committing to a specific hour or
# date is preparatory planning, which the taxonomy places at Behavior. So a
# concrete schedule belongs in urgency's absent-list for the same reason a
# concrete method does -- both put the post outside urgency's premise.
#
# Deliberately excludes bare relative words ("soon", "tonight") -- those ARE
# the vague imminence urgency is supposed to act on, and putting them here
# would gut the rule rather than guard it.
# ---------------------------------------------------------------------------
SCHEDULED_TIME = [
    r"\bin \d+ (?:hours?|days?|weeks?|months?)\b",
    r"\bin (?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r" (?:hours?|days?|weeks?|months?)\b",
    r"\bafter \d+ (?:hours?|days?|weeks?)\b",
    r"\b(?:at|by|when it is) midnight\b",
    r"\bpostponed the date\b",
    r"\bmy deadline\b",
    r"\bon my birthday\b",
    r"\bon the \d+(?:st|nd|rd|th) of\b",
]

# Looser temporal phrases, held OUT of SCHEDULED_TIME on purpose. A season is
# not a schedule: naming a season is compatible with a conditional intention
# ("if things do not improve"), which the taxonomy places at Ideation, so
# treating it as planning would escalate conditional posts to Behavior. Kept
# here so the choice stays visible and testable rather than silently dropped.
SCHEDULED_TIME_LOOSE = SCHEDULED_TIME + [
    r"\bthis (?:winter|summer|spring|fall|autumn)\b",
    r"\bthe same plan\b",
]

PRIOR_ATTEMPT = [
    r"\b(?:tried|attempted) to (?:kill myself|commit suicide|end my life)\b",
    r"\bi'?ve (?:already )?tried to (?:commit|kill)\b",
    r"\bsuicide attempts?\b",
    r"\b(?:my|the) (?:last|previous|first|failed) attempt\b",
    r"\bfailed attempts?\b",
    r"\battempted suicide\b",
]

# NOTE: the `urgency_guarded` variant is assembled in apply_rules_to_test.py,
# where the deployed URGENCY_PHRASES / METHOD_PHRASES lists live, so that it
# always reuses the live lists and this module stays import-cycle free.

CANDIDATE_DEFS = {
    "farewell_note": {
        "present_all": [FAREWELL_ARTEFACT, POSTHUMOUS_ADDRESS],
        "absent": None,
        "from_labels": ["ideation"], "to_label": "behavior",
    },
    "farewell_note_broad": {
        "present_all": [FAREWELL_ARTEFACT_BROAD, POSTHUMOUS_ADDRESS],
        "absent": None,
        "from_labels": ["ideation"], "to_label": "behavior",
    },
    "third_person_death": {
        "present": THIRD_PERSON_DEATH,
        "absent": FIRST_PERSON_INTENT,
        # The rule is only valid when the externalised death reference IS
        # the post. The same clause buried inside a long first-person
        # suicidal narrative carries no such implication, and downgrading
        # that post to indicator would be badly wrong. The word cap encodes
        # "this remark is the whole post", which is the actual mechanism,
        # and does it without an unbounded absent-list.
        "max_words": 25,
        "from_labels": ["ideation"], "to_label": "indicator",
    },
    "protective_negation": {
        "present": PROTECTIVE_NEGATION,
        "absent": None,
        "from_labels": ["ideation"], "to_label": "indicator",
    },
}

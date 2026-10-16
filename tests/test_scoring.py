"""Sanity tests for scoring.py. Run: python -m pytest tests/ -q"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring import (
    phrase_f1, risk_weighted_f1, factors_macro_f1, composite_score,
    _span_pair_matches, parse_evidence, parse_factors,
)


# ---------- span pair rules ----------

def test_exact_match():
    assert _span_pair_matches("want to kill myself", "want to kill myself")

def test_pred_contained_in_gold():
    assert _span_pair_matches("kill myself", "I want to kill myself tonight")

def test_pred_contains_gold_within_cap():
    # gold = 2 tokens, pred = 5 tokens, 5 <= 3*2 -> ok
    assert _span_pair_matches("I want to kill myself", "kill myself")

def test_three_x_cap_violation():
    # gold = 2 tokens, pred = 7 tokens, 7 > 6 -> rejected
    pred = "honestly I just want to kill myself"
    assert not _span_pair_matches(pred, "kill myself")

def test_no_containment_no_match():
    assert not _span_pair_matches("cut my arm", "kill myself")

def test_case_and_whitespace_insensitive():
    assert _span_pair_matches("Kill  Myself", "kill myself")


# ---------- one-to-one matching ----------

def test_one_to_one_no_double_credit():
    # Two identical predictions vs one gold span: only one may match.
    gold = [["kill myself"]]
    pred = [["kill myself", "kill myself"]]
    r = phrase_f1(gold, pred)
    assert r.n_matched == 1
    assert r.precision == 0.5 and r.recall == 1.0

def test_maximum_matching_prefers_best_assignment():
    # pred1 could match either gold, pred2 only matches gold1.
    # Greedy could waste gold1 on pred1; maximum matching finds both.
    gold = [["kill myself", "cut my arm badly"]]
    pred = [["kill myself", "cut my arm"]]
    r = phrase_f1(gold, pred)
    assert r.n_matched == 2

def test_vacuous_posts_are_neutral():
    gold = [[], ["kill myself"]]
    pred = [[], ["kill myself"]]
    r = phrase_f1(gold, pred)
    assert r.f1 == 1.0

def test_hallucinated_span_costs_precision():
    gold = [["kill myself"]]
    pred = [["kill myself", "the moon is nice"]]
    r = phrase_f1(gold, pred)
    assert r.precision == 0.5


# ---------- risk / factors / composite ----------

def test_risk_f1_handles_casing():
    assert risk_weighted_f1(["Ideation", "attempt"], ["ideation", "Attempt"]) == 1.0

def test_factors_macro_perfect():
    g = [["hopelessness", "dysfunctional family"], []]
    p = [["hopelessness", "dysfunctional family"], []]
    assert factors_macro_f1(g, p) > 0  # only 2 of 24 labels present -> macro < 1

def test_factors_rare_label_weight():
    # Missing one rare label costs the same macro mass as missing a common one.
    g = [["hopelessness"]] * 10 + [["exposure to others' suicide"]]
    p_hit = [["hopelessness"]] * 10 + [["exposure to others' suicide"]]
    p_miss = [["hopelessness"]] * 10 + [[]]
    assert factors_macro_f1(g, p_hit) > factors_macro_f1(g, p_miss)

def test_composite_weights():
    s = composite_score(
        ["ideation"], ["ideation"],
        [["kill myself"]], [["kill myself"]],
        [["hopelessness"]], [["hopelessness"]],
    )
    expected = 0.4 * s["risk_weighted_f1"] + 0.3 * s["evidence_phrase_f1"] + 0.3 * s["factors_macro_f1"]
    assert abs(s["composite"] - expected) < 1e-9


# ---------- parsers ----------

def test_parse_evidence():
    assert parse_evidence("a; b ;c") == ["a", "b", "c"]
    assert parse_evidence("") == []
    assert parse_evidence(None) == []

def test_parse_factors_formats():
    assert parse_factors("hopelessness; low self-esteem") == ["hopelessness", "low self-esteem"]
    assert parse_factors("['hopelessness', 'low self-esteem']") == ["hopelessness", "low self-esteem"]

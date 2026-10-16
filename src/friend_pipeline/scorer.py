# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm can
# be rebuilt from source without the original project tree.
#
# This is the pipeline's own independent implementation of the official metric.
# src/scoring.py is the implementation the rest of this repository uses; the two
# agree on the rules below and are kept separate so neither is tuned to match
# the other.
"""
Local scorer for the IEEE Big Data Cup - Explainable Suicide Risk Detection.
Implements the official rules from the competition page:

  Score = 0.4 * WeightedF1(risk) + 0.3 * PhraseF1(evidence) + 0.3 * MacroF1(factors)

Evidence Phrase F1 rules (per the official description):
  - case-insensitive normalization
  - a predicted span matches a gold span if one CONTAINS the other
  - predicted span is only valid if its token length <= 3x the gold span's token length
  - one-to-one matching: each predicted span matches at most one gold span and vice versa
  - F1 computed per post, then averaged over all posts

Factors: Macro F1 across the 24 fixed categories.
"""
import json
import numpy as np
from sklearn.metrics import f1_score

RISK_LEVELS = ["Indicator", "Ideation", "Behavior", "Attempt"]

FACTORS = [
    "mental health issues", "physical health/characteristic", "substance use",
    "hopelessness", "emotion dysregulation", "low self-esteem",
    "poor school performance", "low socio-economic status", "interpersonal violence",
    "prior self-harm or suicidal thought/attempt", "poor social support",
    "interpersonal difficulty", "dysfunctional family", "exposure to others' suicide",
    "stressful life event", "traumatic experience", "cognitive deficits",
    "suicide means (with access)", "sexual orientation related issues",
    "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
]


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _ntok(s: str) -> int:
    return len(_norm(s).split())


def risk_weighted_f1(gold_levels, pred_levels) -> float:
    return f1_score(gold_levels, pred_levels, labels=RISK_LEVELS,
                    average="weighted", zero_division=0)


def phrase_f1_single(gold_spans, pred_spans) -> float:
    """Phrase F1 for ONE post, with containment matching, 3x length cap,
    and greedy one-to-one matching."""
    gold = [_norm(g) for g in gold_spans if _norm(g)]
    pred = [_norm(p) for p in pred_spans if _norm(p)]
    if not gold and not pred:
        return 1.0          # correctly predicted "none"
    if not gold or not pred:
        return 0.0
    used_gold = set()
    matched = 0
    for p in pred:
        for gi, g in enumerate(gold):
            if gi in used_gold:
                continue
            contains = (g in p) or (p in g)
            if contains and _ntok(p) <= 3 * _ntok(g):
                used_gold.add(gi)
                matched += 1
                break
    prec = matched / len(pred)
    rec = matched / len(gold)
    return 0.0 if matched == 0 else 2 * prec * rec / (prec + rec)


def evidence_phrase_f1(gold_lists, pred_lists) -> float:
    return float(np.mean([phrase_f1_single(g, p)
                          for g, p in zip(gold_lists, pred_lists)]))


def factors_macro_f1(gold_lists, pred_lists) -> float:
    idx = {f: i for i, f in enumerate(FACTORS)}
    Yg = np.zeros((len(gold_lists), len(FACTORS)), dtype=int)
    Yp = np.zeros_like(Yg)
    for r, lst in enumerate(gold_lists):
        for f in lst:
            if f in idx:
                Yg[r, idx[f]] = 1
    for r, lst in enumerate(pred_lists):
        for f in lst:
            if f in idx:
                Yp[r, idx[f]] = 1
    return f1_score(Yg, Yp, average="macro", zero_division=0)


def composite(gold_levels, pred_levels,
              gold_ev, pred_ev,
              gold_fac, pred_fac, verbose=True):
    a = risk_weighted_f1(gold_levels, pred_levels)
    b = evidence_phrase_f1(gold_ev, pred_ev)
    c = factors_macro_f1(gold_fac, pred_fac)
    s = 0.4 * a + 0.3 * b + 0.3 * c
    if verbose:
        print(f"Risk Weighted-F1 : {a:.4f}  (x0.4 = {0.4*a:.4f})")
        print(f"Evidence Phrase-F1: {b:.4f}  (x0.3 = {0.3*b:.4f})")
        print(f"Factors Macro-F1 : {c:.4f}  (x0.3 = {0.3*c:.4f})")
        print(f"COMPOSITE        : {s:.4f}")
    return s


if __name__ == "__main__":
    # sanity checks on the evidence matcher
    assert phrase_f1_single(["I want to kill myself"], ["kill myself"]) == 1.0
    assert phrase_f1_single(["kill myself"], ["the whole post text here is way too long to count as valid " * 3]) == 0.0
    assert phrase_f1_single([], []) == 1.0
    assert phrase_f1_single([], ["something"]) == 0.0
    assert phrase_f1_single(["a b c", "x y z"], ["a b c x y z junk junk"]) < 1.0  # one pred can't cover two golds
    print("scorer sanity checks passed")

"""
Local reimplementation of the IEEE BigData Cup 2026 scoring.

Composite = 0.4 * weighted_F1(risk_level)
          + 0.3 * phrase_F1(evidence)
          + 0.3 * macro_F1(factors)

Phrase F1 matching rule (from the task description):
  - A predicted span matches a ground-truth span if the predicted span
    CONTAINS the gold span, or is CONTAINED WITHIN the gold span
    (substring containment on normalized text).
  - A match only counts if len_tokens(pred) <= 3 * len_tokens(gold).
  - Matching is one-to-one: each predicted span and each gold span can be
    used in at most one match. This implementation takes the MAXIMUM
    one-to-one matching (maximum bipartite matching), which is the most
    favorable valid interpretation and the standard way such rules are scored.

ASSUMPTIONS about details the task description leaves open (each is marked in
code):
  [A1] Both averagings of phrase F1 are implemented and selected with
       `average`: 'micro' pools counts over the whole dataset (total matches /
       total predicted, total matches / total gold), 'macro' averages per-post
       F1. The official metric is MACRO; `composite_score` passes it
       explicitly.
  [A2] Containment is checked case-insensitively with whitespace
       normalization. Set normalize=False for strict byte-level matching.
  [A3] Posts where both gold and predicted evidence are empty contribute
       nothing to micro counts (they neither help nor hurt); under macro they
       count as F1 = 1.0.
  [A4] Token length = whitespace-split token count.

This is a reimplementation, not the official scorer: confirm that it tracks the
official numbers before tuning hard against it.
"""

from __future__ import annotations

import re
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from dataclasses import dataclass, field

from sklearn.metrics import f1_score

from text_norm import clean_text, split_discontinuous

RISK_LEVELS = ["indicator", "ideation", "behavior", "attempt"]

FACTOR_TAXONOMY = [
    "mental health issues",
    "physical health/characteristic",
    "substance use",
    "hopelessness",
    "emotion dysregulation",
    "low self-esteem",
    "poor school performance",
    "low socio-economic status",
    "interpersonal violence",
    "prior self-harm or suicidal thought/attempt",
    "poor social support",
    "interpersonal difficulty",
    "dysfunctional family",
    "exposure to others' suicide",
    "stressful life event",
    "traumatic experience",
    "cognitive deficits",
    "suicide means (with access)",
    "sexual orientation related issues",
    "social support",
    "coping strategy",
    "psychological capital",
    "sense of responsibility",
    "meaning in life",
]


# ----------------------------------------------------------------------
# Normalization helpers
# ----------------------------------------------------------------------

def normalize_label(label: str) -> str:
    """Lowercase/strip risk labels. Fixes the casing noise in the raw data."""
    return str(label).strip().lower()


def normalize_span(text: str) -> str:
    """[A2] Canonicalize (HTML/unicode fold), lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", clean_text(text)).strip().lower()


def token_len(text: str) -> int:
    """[A4] Whitespace token count."""
    t = str(text).strip()
    return len(t.split()) if t else 0


def parse_evidence(cell: str | float | None) -> list[str]:
    """Split a semicolon-separated evidence cell into clean spans.
    Discontinuous spans joined by '...' are split into fragments [A5]."""
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    out = []
    for p in s.split(";"):
        p = p.strip()
        if p:
            out.extend(split_discontinuous(clean_text(p)))
    return out


def parse_factors(cell: str | float | None) -> list[str]:
    """Parse a factors cell: accepts 'a; b', 'a, b', or "['a', 'b']" forms."""
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    s = s.strip("[]")
    parts = re.split(r"[;,]", s)
    out = []
    for p in parts:
        p = p.strip().strip("'\"").strip()
        if p:
            out.append(p.lower())
    return out


# ----------------------------------------------------------------------
# Metric 1: weighted F1 on risk level
# ----------------------------------------------------------------------

def risk_weighted_f1(gold: list[str], pred: list[str]) -> float:
    g = [normalize_label(x) for x in gold]
    p = [normalize_label(x) for x in pred]
    return f1_score(g, p, labels=RISK_LEVELS, average="weighted", zero_division=0)


# ----------------------------------------------------------------------
# Metric 2: phrase F1 on evidence spans
# ----------------------------------------------------------------------

def _span_pair_matches(pred: str, gold: str, normalize: bool = True) -> bool:
    """Containment either way + 3x token-length cap on the prediction."""
    p = normalize_span(pred) if normalize else str(pred)
    g = normalize_span(gold) if normalize else str(gold)
    if not p or not g:
        return False
    if p not in g and g not in p:
        return False
    return token_len(p) <= 3 * token_len(g)


def _max_bipartite_matching(adj: list[list[int]], n_right: int) -> int:
    """Maximum one-to-one matching via augmenting paths (Kuhn's algorithm).
    adj[i] = list of right-node indices eligible for left node i."""
    match_right = [-1] * n_right

    def try_kuhn(u: int, visited: list[bool]) -> bool:
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_right[v] == -1 or try_kuhn(match_right[v], visited):
                    match_right[v] = u
                    return True
        return False

    matches = 0
    for u in range(len(adj)):
        if try_kuhn(u, [False] * n_right):
            matches += 1
    return matches


@dataclass
class PhraseF1Result:
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gold: int
    n_matched: int
    per_post: list[dict] = field(default_factory=list)


_MICRO_DEFAULT_WARNED = False


def phrase_f1(
    gold_spans: list[list[str]],
    pred_spans: list[list[str]],
    normalize: bool = True,
    average: str | None = None,
) -> PhraseF1Result:
    """Evidence phrase F1 over the dataset.

    gold_spans / pred_spans: one list of spans per post, aligned by index.
    average: 'micro' [A1] or 'macro' (mean of per-post F1, posts with no gold
             and no predicted spans count as F1=1.0).

    THE OFFICIAL METRIC IS MACRO. The default here stays 'micro' because many
    existing callers were written against it and changing the default would
    silently redefine their results, so an omitted `average` warns once per
    process instead of passing unnoticed. Always pass it explicitly.

    The two averagings are not interchangeable, and macro is the stricter
    objective in two ways. A post with no gold spans scores 1.0 under macro
    only if nothing is predicted for it, so emitting a span there is directly
    penalised; under micro such a post merely adds to the predicted count.
    And macro weights every post equally, so a post with one gold span counts
    as much as a post with five, while micro lets span-dense posts dominate.
    Optimising a threshold against micro therefore pushes towards emitting
    more spans than macro rewards.
    """
    if average is None:
        global _MICRO_DEFAULT_WARNED
        if not _MICRO_DEFAULT_WARNED:
            _MICRO_DEFAULT_WARNED = True
            print("WARNING: phrase_f1() called without `average`; using "
                  "'micro'. The official board is MACRO. Pass "
                  "average='macro' unless you specifically want micro.",
                  file=_sys.stderr)
        average = "micro"
    assert len(gold_spans) == len(pred_spans), "gold/pred length mismatch"

    total_pred = total_gold = total_matched = 0
    per_post = []

    for i, (gs, ps) in enumerate(zip(gold_spans, pred_spans)):
        if not gs and not ps:
            # [A3] vacuous post
            per_post.append({"idx": i, "pred": 0, "gold": 0, "matched": 0, "f1": 1.0})
            continue
        adj = [
            [j for j, g in enumerate(gs) if _span_pair_matches(p, g, normalize)]
            for p in ps
        ]
        m = _max_bipartite_matching(adj, len(gs)) if ps and gs else 0
        total_pred += len(ps)
        total_gold += len(gs)
        total_matched += m
        pp = m / len(ps) if ps else 0.0
        pr = m / len(gs) if gs else 0.0
        pf1 = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0.0
        per_post.append({"idx": i, "pred": len(ps), "gold": len(gs), "matched": m, "f1": pf1})

    if average == "micro":
        precision = total_matched / total_pred if total_pred else 0.0
        recall = total_matched / total_gold if total_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    elif average == "macro":
        vals = [p["f1"] for p in per_post]
        f1 = sum(vals) / len(vals) if vals else 0.0
        precision = recall = float("nan")
    else:
        raise ValueError(f"unknown average: {average}")

    return PhraseF1Result(precision, recall, f1, total_pred, total_gold, total_matched, per_post)


# ----------------------------------------------------------------------
# Metric 3: macro F1 on factors
# ----------------------------------------------------------------------

def factors_macro_f1(
    gold_factors: list[list[str]],
    pred_factors: list[list[str]],
    taxonomy: list[str] | None = None,
    return_per_label: bool = False,
):
    """Macro F1 across the 24 factor labels (binary F1 per label, averaged).

    The average is unweighted over labels, so each label contributes 1/24 of
    the score regardless of how rare it is: a label for which nothing is ever
    predicted scores F1 0 and costs a full 1/24, while a rare label predicted
    even moderately well is worth as much as the commonest one.

    Labels with zero gold positives AND zero predicted positives get F1=0 from
    sklearn's zero_division=0, and still occupy a slot in the average. If the
    official scorer instead drops such labels from the average, the denominator
    differs and this function reads low; verify before relying on the absolute
    value. [assumption]
    """
    tax = taxonomy or FACTOR_TAXONOMY
    idx = {t: k for k, t in enumerate(tax)}

    def to_binary(rows: list[list[str]]):
        mat = [[0] * len(tax) for _ in rows]
        for r, labels in enumerate(rows):
            for lab in labels:
                lab = lab.strip().lower()
                if lab in idx:
                    mat[r][idx[lab]] = 1
        return mat

    g = to_binary(gold_factors)
    p = to_binary(pred_factors)
    per_label = f1_score(g, p, average=None, zero_division=0)
    macro = float(per_label.mean())
    if return_per_label:
        return macro, dict(zip(tax, per_label.tolist()))
    return macro


# ----------------------------------------------------------------------
# Composite
# ----------------------------------------------------------------------

def composite_score(
    gold_risk, pred_risk,
    gold_evidence, pred_evidence,
    gold_factors, pred_factors,
    weights=(0.4, 0.3, 0.3),
) -> dict:
    w_risk, w_ev, w_fac = weights
    r = risk_weighted_f1(gold_risk, pred_risk)
    # MACRO, explicitly. The official evidence metric is macro; `phrase_f1`'s
    # own default is micro for backwards compatibility with older callers, so
    # this function -- whose job is to report the official composite -- must
    # name the averaging rather than inherit that default.
    e = phrase_f1(gold_evidence, pred_evidence, average="macro")
    f = factors_macro_f1(gold_factors, pred_factors)
    return {
        "risk_weighted_f1": r,
        "evidence_phrase_f1": e.f1,
        "evidence_precision": e.precision,
        "evidence_recall": e.recall,
        "factors_macro_f1": f,
        "composite": w_risk * r + w_ev * e.f1 + w_fac * f,
    }

# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm can
# be rebuilt from source without the original project tree.
"""The symbolic 'sandwich' guards + decoding of model outputs into final answers.

Catches handled here:
  * evidence is decoded from token probabilities into contiguous spans, taken
    VERBATIM from the post via character offsets (never paraphrased)
  * spans are split at sentence punctuation and hard-capped at MAX_SPAN_WORDS
    (protects against the official 3x-length rule; short spans are safe under
    containment matching)
  * one-to-one matching favors several separate short spans over one merged one
  * consistency rule: non-Indicator prediction with zero evidence -> try the
    training lexicon to recover a span; Indicator keeps only high-confidence spans
  * rare-factor keyword rules (macro-F1 insurance for factors the model
    cannot learn from a handful of examples)
"""
import json
import os
import re

import numpy as np

import config as C

_SENT_SPLIT = re.compile(r"[.!?\n;]+")


def decode_spans(post, offsets, token_probs, thr):
    """token probabilities -> list of verbatim short spans."""
    n = len(post)
    cmask = np.zeros(n, dtype=bool)
    for (a, b), p in zip(offsets, token_probs):
        if b > a and p >= thr:
            cmask[a:min(b, n)] = True
    spans, i = [], 0
    while i < n:
        if cmask[i]:
            j = i
            while j < n and (cmask[j] or (j > i and post[j].isspace() and j + 1 < n and cmask[j + 1])):
                j += 1
            spans.append(post[i:j])
            i = j
        else:
            i += 1
    # split on sentence punctuation, trim, cap length
    out = []
    for s in spans:
        for piece in _SENT_SPLIT.split(s):
            piece = piece.strip(" \t\"'`,:-")
            if len(piece) < C.MIN_SPAN_CHARS:
                continue
            words = piece.split()
            while len(words) > C.MAX_SPAN_WORDS:      # hard cap; split long spans
                out.append(" ".join(words[:C.MAX_SPAN_WORDS]))
                words = words[C.MAX_SPAN_WORDS:]
            if words:
                out.append(" ".join(words))
    # dedupe, keep order
    seen, final = set(), []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            final.append(s)
    return final[:6]  # never flood; precision matters


def load_lexicon():
    p = os.path.join(C.OUT_DIR, "lexicon.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def consistency_rules(post, level, spans, lexicon):
    """Guard rules run AFTER the neural model."""
    low = post.lower()
    # non-Indicator with no evidence: try to recover from the lexicon
    if level != "Indicator" and not spans:
        for phrase in lexicon.get(level, [])[:200]:
            i = low.find(phrase)
            if i >= 0:
                spans = [post[i:i + len(phrase)]]
                break
    # verbatim guarantee (spans come from offsets, but be paranoid)
    spans = [s for s in spans if s.lower() in low]
    return level, spans


# --- rare-factor keyword rules (symbolic insurance for macro F1) ---
RARE_FACTOR_RULES = {
    "sexual orientation related issues":
        ["gay", "lesbian", "bisexual", "trans ", "transgender", "queer",
         "coming out", "closeted", "homophob", "my sexuality"],
    "exposure to others' suicide":
        ["his suicide", "her suicide", "friend killed", "brother killed himself",
         "sister killed herself", "died by suicide", "committed suicide last",
         "lost my friend to suicide", "lost my brother", "lost my sister to"],
    "poor school performance":
        ["failing my", "failed my exam", "failed the exam", "flunk", "bad grades",
         "failing school", "failing college", "failing class", "my gpa"],
    "substance use":
        ["alcohol", "drunk", "weed", "cocaine", "heroin", "meth", "drugs",
         "getting high", "smoking", "addiction", "addicted to"],
    "low socio-economic status":
        ["no money", "can't afford", "cant afford", "broke ", "in debt",
         "poverty", "homeless", "unemployed", "lost my job", "can't pay",
         "financial"],
}


def apply_factor_rules(post, factor_set):
    low = " " + post.lower() + " "
    for factor, kws in RARE_FACTOR_RULES.items():
        if factor in factor_set:
            continue
        if any(k in low for k in kws):
            factor_set = sorted(set(factor_set) | {factor})
    return list(factor_set)

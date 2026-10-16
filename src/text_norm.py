"""
Shared text canonicalization. Applied ONCE to posts and evidence at load
time; every downstream component (training, prediction, span extraction,
local scoring, submission) then operates in this single canonical space.

Why this is the right space to submit in: the evidence scorer compares a
predicted span against the gold span as text, and the gold annotations live
in a cleaned version of the posts (straight apostrophes where the posts
carry typographic ones, '&' where the posts carry '&amp;'). Spans extracted
from cleaned text therefore land in the annotators' text-space, while
raw-post substrings can differ from the gold string by characters the
annotator never saw.

Measured over the 1833 gold spans of the official data, gold-vs-post
agreement decomposes as: 90.7% match the raw text as-is, 4.7% require
quote folding, 1.5% punctuation drift, 1.6% HTML entities / annotator
typos / spacing, 1.4% unrecoverable by any normalization. The folds below
are exactly the classes this canonical space can recover.
"""

from __future__ import annotations

import html
import re
import unicodedata

FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ", "\u200b": "", "\ufeff": "",
}


def clean_text(s: str) -> str:
    """HTML-unescape + NFKC + fold typographic characters.
    Does NOT lowercase and does NOT collapse whitespace: this is the
    canonical text itself, not a comparison key."""
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKC", s)
    for a, b in FOLD.items():
        s = s.replace(a, b)
    return s


_ELLIPSIS_SPLIT = re.compile(r"\s*\.{3,}\s*")


def split_discontinuous(span: str) -> list[str]:
    """Split a gold span that joins two non-adjacent fragments of the post
    with an ellipsis ('<phrase A> ... <phrase B>') into its fragments, so
    each fragment can be located in the post on its own.

    The ellipsis is only treated as a join when it sits strictly BETWEEN
    words and each side has >= 2 words. A trailing '...' or a one-word side
    is far more likely to be the author's own punctuation or an elision
    inside a single quotation than an annotator joining two fragments, so
    those spans are kept whole."""
    s = str(span).strip()
    if "..." not in s:
        return [s]
    parts = [p.strip() for p in _ELLIPSIS_SPLIT.split(s) if p.strip()]
    if len(parts) >= 2 and all(len(p.split()) >= 2 for p in parts):
        return parts
    return [s]

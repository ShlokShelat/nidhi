"""
apply_span_rule.py

Promotes a risk label from behavior to attempt when a completed-act pattern
appears in the EVIDENCE CLAUSE. Zero parameters, and no probabilities needed.

WHY THE SPAN AND NOT THE POST. Scoping the same completed-act regex to the
evidence span rather than the whole post raises its precision sharply, because
the evidence tagger has already located the clause an annotator would
highlight. "tried" inside that clause asserts a completed act; "tried" in
paragraph three of the post is far more often an unrelated use.

The span-scoped form needs no probability margin to be precise, so this applier
reads only the risk and evidence columns already present in a submission file
and needs no probability artefact.

WHY IT IS SAFE TO STACK ON AN EXISTING SUBMISSION. The promotion is
behavior -> attempt. It cannot create or destroy an `indicator`, so evidence
suppression is untouched, the evidence column is copied verbatim, and the
evidence term of the score must come back bit-identical. Only the risk term
can move.

`attempt` is the thinnest-supervised risk class and the weakest on recall, so
each correct promotion is worth an unusual amount of its F1.

Usage:
  python src/apply_span_rule.py --base submissions/NIDHI_factfit.csv
  python src/apply_span_rule.py --base submissions/NIDHI_factfit.csv \\
      --out submissions/NIDHI_spanrule.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re  # noqa: E402

from risk_symbolic_fences import DONE  # noqa: E402
from scoring import normalize_label, parse_evidence  # noqa: E402

# The weak tokens of the full DONE pattern (e.g. `again`, `last time`) do not
# assert a completed act by themselves -- they read as a completed attempt only
# with surrounding post context, and they attach just as readily to an ongoing
# behavior or a plan. Span scoping removes that context, so only tokens that
# assert a completed act on their own belong in the strict pattern.
STRICT_DONE = re.compile(
    r"\b(tried|attempted|survived|woke up|hospital|emergency|ambulance|icu|"
    r"stomach pump|overdosed|od'?d|swallowed|resuscitat\w*|failed attempt)\b",
    re.I)
# METHOD: the `behavior` criterion is "explicit expression AND a plan or act",
# so a named method inside the evidence clause is direct evidence of the "plan
# or act" half. Method words appear in behavior evidence spans and essentially
# never in ideation ones. This list is written from the taxonomy DEFINITION
# rather than mined from the gold spans it is scored against, so it carries no
# fitted parameters.
METHOD = re.compile(
    r"\b(jump(?:ing|ed)?|bridge|pills?|knife|knives|gun|shoot|hang(?:ing)?|"
    r"rope|noose|overdose|od|blade|razor|bleach|poison|train|traffic|"
    r"wrists?|veins?|cliff|roof|balcony)\b", re.I)
PATTERNS = {"strict": STRICT_DONE, "full": DONE, "method": METHOD}


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--base", default="submissions/NIDHI_factfit.csv")
    a_.add_argument("--out", default=None)
    a_.add_argument("--from-class", default="behavior")
    a_.add_argument("--to-class", default="attempt")
    a_.add_argument("--pattern", default="strict",
                    choices=["strict", "full", "method"],
                    help="`strict` = completed act, for behavior->attempt. "
                         "`method` = plan vocabulary, for ideation->behavior. "
                         "`full` includes weak tokens (again, last time) that "
                         "do not survive span scoping -- avoid.")
    a_.add_argument("--scope", default="span", choices=["span", "post"])
    a_.add_argument("--posts", default="data/test.csv",
                    help="csv with row_id and post, needed for --scope post")
    args = a_.parse_args()
    RX = PATTERNS[args.pattern]

    d = pd.read_csv(args.base, keep_default_na=False)
    rc = "risk_level" if "risk_level" in d.columns else "pred_risk_level"
    if "evidence" not in d.columns:
        raise SystemExit(f"{args.base} has no evidence column")

    risk = [normalize_label(x) for x in d[rc]]
    spans = [" ... ".join(parse_evidence(str(x))) for x in d["evidence"]]
    if args.scope == "post":
        # POST scoping is measured on exactly the text it is applied to: the
        # post is the same object offline and at deployment, whereas the span
        # source can differ (an unconditioned tagger vote offline vs
        # risk-conditioned spans in a submission, which are fewer and shorter).
        if args.posts and os.path.exists(args.posts):
            pm = pd.read_csv(args.posts, keep_default_na=False)
            pm["row_id"] = pm["row_id"].astype(str)
            m = dict(zip(pm["row_id"], pm["post"].astype(str)))
            txt = [m.get(str(r), "") for r in d["row_id"].astype(str)]
            if not any(txt):
                raise SystemExit(f"no post text matched from {args.posts}")
        else:
            raise SystemExit("--scope post needs --posts pointing at a csv "
                             "with row_id and post (e.g. data/test.csv)")
    else:
        txt = spans
    hit = [bool(RX.search(t)) for t in txt]
    fire = [r == args.from_class and h for r, h in zip(risk, hit)]

    n = int(sum(fire))
    print(f"{args.base}   {len(d)} rows")
    print(f"predicted {args.from_class}: {sum(1 for r in risk if r == args.from_class)}")
    where = "EVIDENCE SPAN" if args.scope == "span" else "POST"
    print(f"of those, {args.pattern.upper()} matches the {where}: {n}")
    print(f"  -> promoted to {args.to_class}\n")
    for i in range(len(d)):
        if fire[i]:
            m = RX.search(txt[i])
            hit_word = m.group(0) if m else "?"
            print(f"   {d.iloc[i]['row_id']}  [{hit_word}]  "
                  f"span={spans[i][:70]!r}")

    if not args.out:
        print("\n(diagnose only; pass --out to write)")
        return

    seen = {normalize_label(x): x for x in d[rc]}
    tgt = seen.get(args.to_class, args.to_class)
    out = d.copy()
    out[rc] = [tgt if f else v for f, v in zip(fire, d[rc])]

    ev_same = all(str(a) == str(b) for a, b in zip(d["evidence"],
                                                   out["evidence"]))
    fa_same = ("factors" not in d.columns) or all(
        str(a) == str(b) for a, b in zip(d["factors"], out["factors"]))
    if not (ev_same and fa_same):
        raise SystemExit("REFUSING: evidence or factors changed")
    ind_b = sum(1 for r in risk if r == "indicator")
    ind_a = sum(1 for x in out[rc] if normalize_label(x) == "indicator")
    if ind_b != ind_a:
        raise SystemExit(
            f"REFUSING: indicator count moved {ind_b} -> {ind_a}, so evidence "
            f"suppression is no longer consistent and S2 would shift")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}   {n} risk labels changed")
    print("  evidence and factors BYTE-IDENTICAL, indicator count unchanged")
    print("  -> the evidence term must return exactly the base submission's")
    print("     value; only the risk term moves.")


if __name__ == "__main__":
    main()

"""
risk_indicator_boundary.py

Is the INDICATOR boundary lexical, and is that boundary being missed anyway?

WHY THIS BOUNDARY MATTERS. risk_evidence_coupling.py showed the evidence
score is suppressed to empty whenever the model predicts `indicator`, and
gold evidence is empty on (nearly) every gold-indicator post. So the 147
errors that cross this boundary cost the risk term AND the evidence term,
while the 196 errors among ideation/behavior/attempt cost the risk term only:

    crosses indicator   ~0.00041 composite each   147 errors   pool ~0.060
    inside ide/beh/att  ~0.00022 composite each   196 errors   pool ~0.043

Most attention so far has gone to the second, cheaper, harder boundary; this
examines the first.

WHY IT SHOULD BE EASIER. behavior-vs-attempt turns on intent to die at the time
of the act -- unreadable from text. indicator-vs-the-rest turns on a SURFACE
property: whether the post explicitly mentions suicide, self-harm, or wanting to
die. Gaur et al. treat it that way and so does the organizers' own wording. If
that is really the operative rule, the cue is in the string and both a
definition rewrite and a hard rule can reach it.

WHAT THIS PRINTS.
  1. Explicit-mention rate on gold indicator vs each other class. If indicator
     is near zero and the others near one, the boundary is lexical and the 147
     errors are a decoding failure, not a hard limit.
  2. The two error directions separately, because they need opposite fixes:
       FALSE NEGATIVE (gold=indicator, predicted higher) -- the model read a
         mention that is not there, or that is quoted/hypothetical/about
         someone else
       FALSE POSITIVE (gold=higher, predicted indicator) -- the model missed a
         mention, probably an oblique one
  3. A ceiling: how much weighted-F1 a PERFECT explicit-mention rule would buy,
     applied as a hard constraint on the current predictions. That is the number
     that decides whether this lane is worth GPU.

If the rate table separates cleanly but the errors persist, the fix is cheap.
If gold indicator posts DO carry explicit mentions at a high rate, then the
annotators used something other than the stated rule and this lane is as dead as
the last one.

Usage:
  python src/risk_indicator_boundary.py
  python src/risk_indicator_boundary.py --show 20
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

# "explicitly mentions suicide, self-harm, or wanting to die"
EXPLICIT = re.compile(
    r"\b(suicid\w*|kill (?:my ?self|me)|killing my ?self|end (?:my|it all|my "
    r"life)|ending (?:it|my life)|take my (?:own )?life|taking my (?:own )?"
    r"life|want to die|wanna die|wish (?:i (?:was|were) dead|i could die|to "
    r"die)|rather (?:be dead|die)|better off dead|don'?t want to (?:be alive|"
    r"live|exist)|not want to (?:be alive|live)|self.?harm|hurt(?:ing)? my ?"
    r"self|harm(?:ing)? my ?self|cut(?:ting)? my ?self|off my ?self|unalive|"
    r"kms|ctb|overdose|od'?ing|hang my ?self|shoot my ?self)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="artifacts/champion_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    d = pd.read_csv(args.oof, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d[d["row_id"].isin(lab.index)].copy()
    d["g"] = [normalize_label(x) for x in d["gold"]]
    d["p"] = [normalize_label(x) for x in d["pred_risk_level"]]
    d["post"] = [" ".join(str(lab.loc[r, "post"]).split()) for r in d["row_id"]]
    d["exp"] = [bool(EXPLICIT.search(t)) for t in d["post"]]

    print(f"{len(d)} OOF posts")
    print("\n" + "=" * 66)
    print("1. IS THE BOUNDARY LEXICAL?")
    print("=" * 66)
    print(f"{'gold class':<14}{'posts':>7}{'has explicit mention':>24}")
    print("-" * 46)
    for c in RISK_LEVELS:
        m = (d["g"] == c).to_numpy()
        if m.sum():
            print(f"{c:<14}{int(m.sum()):>7}{d['exp'].to_numpy()[m].mean():>24.3f}")
    print("\nThe stated rule says the indicator row should be ~0.00 and every")
    print("other row ~1.00. Whatever gap you see there is the annotators using")
    print("a different rule than the one we feed the model.")

    # a perfect lexical rule as a hard constraint -- the ceiling
    print("\n" + "=" * 66)
    print("2. CEILING -- what a PERFECT explicit-mention rule would buy")
    print("=" * 66)
    y = d["g"].to_numpy()
    p = d["p"].to_numpy()
    base = risk_weighted_f1(list(y), list(p))

    # rule: no mention -> must be indicator; mention -> must not be indicator.
    # when forced off indicator, fall back to the model's best non-indicator
    # class, so this measures the CONSTRAINT and not a new classifier.
    pc = [f"prob_{c}" for c in RISK_LEVELS]
    have_probs = all(c in d.columns for c in pc)
    if have_probs:
        P = d[pc].to_numpy(float)
        alt = np.array([RISK_LEVELS[1:][i] for i in P[:, 1:].argmax(1)])
    else:
        alt = np.array(["ideation"] * len(d))
        print("  (no prob columns; forced posts default to `ideation`)")

    forced = p.copy()
    no_mention = ~d["exp"].to_numpy()
    forced[no_mention] = "indicator"
    forced[~no_mention & (p == "indicator")] = alt[~no_mention & (p == "indicator")]
    rule = risk_weighted_f1(list(y), list(forced))
    n_moved = int((forced != p).sum())
    print(f"  weighted-F1  {base:.4f} -> {rule:.4f}   ({rule - base:+.4f}) "
          f"moving {n_moved} posts")
    print(f"  composite via the 0.4 risk weight            "
          f"{0.4 * (rule - base):+.4f}")
    print("  A NEGATIVE number here kills the hard-rule form outright and")
    print("  says the regex is cruder than the model. It does NOT kill the")
    print("  definition rewrite, which lets the model keep its judgement.")

    # the two error directions
    fn = d[(d.g == "indicator") & (d.p != "indicator")]
    fp = d[(d.g != "indicator") & (d.p == "indicator")]
    print("\n" + "=" * 66)
    print("3. THE TWO DIRECTIONS NEED OPPOSITE FIXES")
    print("=" * 66)
    print(f"  gold INDICATOR, we said higher : {len(fn):>4}   "
          f"of these {int(fn['exp'].sum())} DO contain an explicit mention")
    print(f"  gold higher, we said INDICATOR : {len(fp):>4}   "
          f"of these {int(fp['exp'].sum())} DO contain an explicit mention")
    print("\n  Row 1 with a high mention count = annotators saw the mention and")
    print("  still said indicator (quoted, hypothetical, about someone else).")
    print("  Row 2 with a high mention count = the cue was right there and the")
    print("  model ignored it. That one is straightforwardly ours to fix.")

    for nm, sub in (("GOLD INDICATOR -> we said higher", fn),
                    ("GOLD HIGHER -> we said INDICATOR", fp)):
        print("\n" + "=" * 66)
        print(f"{nm}  (showing {min(args.show, len(sub))} of {len(sub)})")
        print("=" * 66)
        for _, r in sub.head(args.show).iterrows():
            hit = EXPLICIT.search(r["post"])
            print(f"\n--- {r['row_id']}   gold {r['g']}  pred {r['p']}   "
                  f"mention: {hit.group(0) if hit else '(none)'}")
            print(f"    POST: {r['post'][:240]}"
                  f"{'...' if len(r['post']) > 240 else ''}")

    print("\n" + "=" * 66)
    print("Read direction 2 first -- those are the posts where the rule we")
    print("already believe in was visible in the text and we still missed it.")


if __name__ == "__main__":
    main()

"""
risk_attempt_boundary.py

How did the annotators separate BEHAVIOR from ATTEMPT?

WHY ASK. The per-class probability table on 1590 OOF posts shows the chain is
nearly guessing on this boundary:

    gold        indicator  ideation  behavior  attempt
    behavior        0.055     0.247     0.618    0.080
    attempt         0.048     0.126     0.401    0.426

On gold ATTEMPT posts the model puts 0.401 on behavior against 0.426 on
attempt. Attempt recall is 0.591 -- the worst of the four classes -- and 29 of
110 attempt posts are called behavior.

THE DEFINITIONS SAY the line is INTENT TO DIE AT THE TIME OF THE ACT:
  behavior = self-harm without a wish to die, OR a named/available means, OR a
             stated time, OR preparation -- anything beyond the thought
  attempt  = an act already carried out WITH at least some wish to die
Self-harm therefore lands in EITHER class depending on intent, which is the
hardest possible cue to read from text, and the confusion above suggests the
model is not reading it.

WHAT THIS PRINTS. The annotators' own quoted spans on both sides, so the
operational rule they used is visible rather than assumed:
  - how often self-harm words appear in behavior vs attempt spans
  - how often past-tense/completed-act markers appear in each
  - the attempt posts the model calls behavior, with their spans

If completed-act markers separate the classes cleanly, the cue is learnable and
the `attempt` definition should name it explicitly -- the same fix now being
tested for behavior vs ideation. If the spans look alike, the boundary rests on
inferred intent and no definition rewrite reaches it.

Usage:
  python src/risk_attempt_boundary.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, parse_evidence  # noqa: E402

SELF_HARM = re.compile(
    r"\b(cut|cutting|cuts|burn|burned|scar|scars|hurt myself|hurting myself|"
    r"harm myself|harming myself|self.?harm|blade|razor|slit)\w*", re.I)

# markers that an act was CARRIED OUT rather than contemplated
DONE = re.compile(
    r"\b(tried|attempted|attempt|survived|failed|woke up|hospital|"
    r"emergency|ambulance|icu|stomach pump|overdosed|od'?d|took \d+|"
    r"swallowed|last (night|week|month|year)|ago|didn'?t work|"
    r"came back|second time|again)\w*", re.I)

FUTURE = re.compile(
    r"\b(going to|gonna|will|plan|planning|tonight|tomorrow|soon|"
    r"about to|thinking of|want to|wish)\w*", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="artifacts/champion_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    lab["g"] = [normalize_label(x) for x in lab["risk_level"]]

    def spans(cls):
        out = []
        for r in lab.index[lab.g == cls]:
            out += parse_evidence(lab.loc[r, "evidence"])
        return out

    s_beh, s_att = spans("behavior"), spans("attempt")
    s_ide = spans("ideation")

    def rate(ss, pat):
        return float(np.mean([bool(pat.search(s)) for s in ss])) if ss else float("nan")

    print(f"gold spans:  behavior {len(s_beh)}   attempt {len(s_att)}   "
          f"ideation {len(s_ide)}")
    print(f"\n{'cue in the annotator-quoted span':<34}"
          f"{'ideation':>10}{'behavior':>10}{'attempt':>10}")
    print("-" * 64)
    for nm, pat in (("self-harm words", SELF_HARM),
                    ("completed-act markers", DONE),
                    ("future/intent markers", FUTURE)):
        print(f"{nm:<34}{rate(s_ide, pat):>10.3f}{rate(s_beh, pat):>10.3f}"
              f"{rate(s_att, pat):>10.3f}")

    print("\nIf 'completed-act markers' separates attempt from behavior")
    print("sharply, the cue is in the text and naming it in the `attempt`")
    print("definition is the same fix now being tested for behavior/ideation.")
    print("If self-harm words appear at similar rates in BOTH, then self-harm")
    print("alone does not decide the class -- intent does -- and that is far")
    print("harder to read.")

    # where does self-harm actually land?
    sh_beh = [s for s in s_beh if SELF_HARM.search(s)]
    sh_att = [s for s in s_att if SELF_HARM.search(s)]
    print(f"\nspans mentioning self-harm: {len(sh_beh)} on BEHAVIOR posts, "
          f"{len(sh_att)} on ATTEMPT posts")
    if sh_beh:
        print(f"  behavior e.g. {[s[:52] for s in sh_beh[:4]]}")
    if sh_att:
        print(f"  attempt  e.g. {[s[:52] for s in sh_att[:4]]}")

    # the actual confusions
    if os.path.exists(args.oof):
        d = pd.read_csv(args.oof, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        d["g"] = [normalize_label(x) for x in d["gold"]]
        d["p"] = [normalize_label(x) for x in d["pred_risk_level"]]
        miss = d[(d.g == "attempt") & (d.p == "behavior")]
        print(f"\n{'=' * 64}")
        print(f"{len(miss)} ATTEMPT posts we called BEHAVIOR")
        print("=" * 64)
        for _, r in miss.head(args.show).iterrows():
            rid = r["row_id"]
            if rid not in lab.index:
                continue
            ev = parse_evidence(lab.loc[rid, "evidence"])
            body = " ".join(str(lab.loc[rid, "post"]).split())
            print(f"\n--- {rid}")
            print(f"    SPAN: {ev if ev else '(none)'}")
            print(f"    POST: {body[:220]}{'...' if len(body) > 220 else ''}")
        print("\nRead these: is the completed act stated plainly, or does it")
        print("have to be inferred? That is the difference between a fixable")
        print("definition problem and an unreachable one.")


if __name__ == "__main__":
    main()

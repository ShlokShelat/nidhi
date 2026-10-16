"""
risk_method_coverage.py

IF METHOD WORDS ARE EXCLUSIVE TO behavior, WHY CAN WE ONLY FIX 14 OF 89?

`risk_boundary_examples.py` counted method words inside GOLD EVIDENCE SPANS:
jump 21 vs 0, bridge 20 vs 0, pills 18 vs 0, knife 18 vs 0, hang 14 vs 0.
Exclusive. Yet the rule built on them reaches only 21 of 592 predicted-ideation
posts and fixes 14 of 89 errors.

Three different things could explain the gap, and they need different fixes:

  COVERAGE   most behavior posts simply contain no method word anywhere. Then
             the criterion is met by self-harm or by an oblique plan, and no
             method vocabulary can ever reach them. Nothing to fix.

  DILUTION   ideation POSTS contain method words too (just not in their gold
             spans), so the word is only discriminative INSIDE the clause.
             Then post-level rules are hopeless and span-level is the only
             option -- which matches POST precision 0.559 vs SPAN 0.667.

  EXTRACTION the method is in the post AND in the gold span, but OUR tagger
             extracted a different clause, so the rule never sees it. Then the
             information exists and the evidence pipeline is throwing it away
             -- a fixable plumbing problem.

This measures all three by asking where the method word appears, for each
class, at three levels: the whole POST, the GOLD span, and OUR PREDICTED span.

Usage:
  python src/risk_method_coverage.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span, parse_evidence,  # noqa
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]
METHOD = re.compile(
    r"\b(jump(?:ing|ed)?|bridge|pills?|knife|knives|gun|shoot|hang(?:ing)?|"
    r"rope|noose|overdose|od|blade|razor|bleach|poison|train|traffic|"
    r"wrists?|veins?|cliff|roof|balcony)\b", re.I)
SELFHARM = re.compile(
    r"\b(cut(?:ting|s)?\s+my\s?self|hurt(?:ing|s)?\s+my\s?self|"
    r"harm(?:ing|s)?\s+my\s?self|self[-\s]?harm\w*|scars?)\b", re.I)


def dedupe(sp):
    out = []
    for s in sorted(sp, key=lambda x: (-token_len(x),
                                       -len(normalize_span(x)))):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--tags", default=",".join(TAGS))
    args = a_.parse_args()

    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in oof["row_id"] if r in lab.index]
    ix = oof.set_index("row_id")
    P = np.stack([ix.loc[rows, f"prob_{c}"].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    pred = np.array(LEVELS)[P.argmax(1)]
    post = [str(lab.loc[r, "post"]) for r in rows]
    gev = [" ... ".join(parse_evidence(lab.loc[r, "evidence"])) for r in rows]

    per = {}
    for t in [x.strip() for x in args.tags.split(",")]:
        p = os.path.join(args.dir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"].astype(str), d["span"]):
            m.setdefault(r, []).append(str(s))
        per[t] = m
    pspan = []
    for r in rows:
        c = {}
        for t in per:
            for s in per[t].get(r, []):
                c.setdefault(normalize_span(s), [set(), s])[0].add(t)
        keep = dedupe([v[1] for v in c.values() if len(v[0]) >= 2])
        if not keep:
            pool = [s for t in per for s in per[t].get(r, [])]
            keep = sorted(pool, key=token_len)[:1]
        pspan.append(" ... ".join(keep))

    def hit(rx, txts):
        return np.array([bool(rx.search(t)) for t in txts])

    groups = {
        "gold ideation": gold == "ideation",
        "gold behavior": gold == "behavior",
        "  of which pred behavior (ok)":
            (gold == "behavior") & (pred == "behavior"),
        "  of which pred ideation (ERR)":
            (gold == "behavior") & (pred == "ideation"),
        "gold attempt": gold == "attempt",
    }

    for rxname, rx in (("METHOD", METHOD), ("SELF-HARM", SELFHARM)):
        print("\n" + "=" * 88)
        print(f"WHERE DOES A {rxname} WORD APPEAR?  "
              f"(share of posts in each group)")
        print("=" * 88)
        hp, hg, hs = hit(rx, post), hit(rx, gev), hit(rx, pspan)
        print(f"{'group':<34}{'n':>6}{'in POST':>10}{'in GOLD span':>14}"
              f"{'in OUR span':>13}")
        for nm, m in groups.items():
            if not m.sum():
                continue
            print(f"{nm:<34}{int(m.sum()):>6}{hp[m].mean():>10.1%}"
                  f"{hg[m].mean():>14.1%}{hs[m].mean():>13.1%}")

        mb = gold == "behavior"
        mi = gold == "ideation"
        print(f"\n  COVERAGE : {hp[mb].mean():.1%} of behavior posts contain a "
              f"{rxname} word at all.")
        print(f"             {1-hp[mb].mean():.1%} do NOT -- unreachable by "
              f"this vocabulary, whatever we do.")
        print(f"  DILUTION : {hp[mi].mean():.1%} of IDEATION posts also "
              f"contain one, vs {hg[mi].mean():.1%} in their gold span.")
        if hp[mi].mean() > 0:
            print(f"             the word is {hp[mb].mean()/max(hp[mi].mean(),1e-9):.1f}x "
                  f"more common in behavior POSTS, but "
                  f"{hg[mb].mean()/max(hg[mi].mean(),1e-9):.1f}x in gold SPANS.")
        me = (gold == "behavior") & (pred == "ideation")
        both = hg[me] & ~hs[me]
        print(f"  EXTRACTION: of the {int(me.sum())} errors, "
              f"{int((hg[me]).sum())} have the word in the GOLD span but only "
              f"{int((hs[me]).sum())} in OUR span.")
        print(f"             -> {int(both.sum())} posts where the evidence "
              f"pipeline LOST a signal that was there.")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    mb = gold == "behavior"
    anyw = hit(METHOD, post) | hit(SELFHARM, post)
    print(f"  behavior posts with NEITHER a method NOR a self-harm word: "
          f"{(~anyw[mb]).mean():.1%}")
    print("  Those are behavior for reasons no keyword can express -- an")
    print("  oblique timeline, a question about options, an implied act.")
    print("  That share is the hard ceiling on any lexical approach to this")
    print("  boundary, and it is why the rule reaches 14 of 89 rather than most.")


if __name__ == "__main__":
    main()

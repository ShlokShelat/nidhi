"""
risk_boundary_examples.py

WHAT ACTUALLY SEPARATES ideation FROM behavior? ASK THE ANNOTATOR.

    Ideation  explicit suicidal expressions but NO suicidal plan
    Behavior  explicit suicidal expressions AND self-harm / suicidal plan

behavior = ideation + (a plan OR an act). The 89-post `behavior -> ideation`
cell is the largest error source in the risk model. Feature-based attacks on
it have not resolved it: the factor `suicide means` separates it only 27%,
`prior self-harm` not at all, and a self-harm regex gets precision 0.32.

So instead of guessing what the criterion is, READ WHAT THE ANNOTATOR
HIGHLIGHTED. Gold evidence is defined as the phrase justifying the risk level,
so for a `behavior` post it should contain the plan or the act. Comparing the
gold spans of clean ideation, clean behavior, and the confused cell shows the
criterion directly rather than by inference.

Prints, each with the annotator's own evidence:
  A. gold ideation, predicted ideation  -- what a clean ideation looks like
  B. gold behavior, predicted behavior  -- what a clean behavior looks like
  C. gold behavior, predicted ideation  -- the errors
  D. the vocabulary that actually distinguishes A from B, measured on the
     gold evidence spans rather than on the whole post

Usage:
  python src/risk_boundary_examples.py
  python src/risk_boundary_examples.py --show 10
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, parse_evidence  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
STOP = set("i my me a an the and or but to of in it is am was be been so "
           "that this if for on with just my myself not no do dont don't "
           "have has had will would can cant can't im i'm you your he she "
           "they we at as all any what when how why".split())


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--show", type=int, default=6)
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
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    pred = np.array(LEVELS)[P.argmax(1)]
    ev = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    post = [" ".join(str(lab.loc[r, "post"]).split()) for r in rows]
    top = P.max(1)

    def show(title, mask, note):
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(-top[idx])][:args.show]
        print("\n" + "=" * 92)
        print(title)
        print(note)
        print("=" * 92)
        for i in idx:
            print(f"\n  [{rows[i]}]  gold={gold[i]}  pred={pred[i]} "
                  f"(p={top[i]:.2f})")
            print(f"    ANNOTATOR HIGHLIGHTED: "
                  f"{ev[i] if ev[i] else '(none)'}")
            print(f"    post: {post[i][:260]}")

    show("A. CLEAN IDEATION  (gold ideation, predicted ideation)",
         (gold == "ideation") & (pred == "ideation"),
         "   The wish to die is stated. Look for the ABSENCE of a method,\n"
         "   a time, or an act already performed.")
    show("B. CLEAN BEHAVIOR  (gold behavior, predicted behavior)",
         (gold == "behavior") & (pred == "behavior"),
         "   Same wish, PLUS something concrete. That extra thing is the\n"
         "   whole difference between the two labels.")
    show("C. THE ERRORS  (gold behavior, predicted ideation)",
         (gold == "behavior") & (pred == "ideation"),
         "   The annotator saw a plan or act here and the model did not.\n"
         "   Read the highlighted span -- that is what it keyed on.")

    print("\n" + "=" * 92)
    print("D. WHICH WORDS LIVE IN behavior EVIDENCE BUT NOT ideation EVIDENCE?")
    print("=" * 92)
    cb, ci = Counter(), Counter()
    for i in range(len(rows)):
        if not ev[i]:
            continue
        toks = set(w for s in ev[i]
                   for w in re.findall(r"[a-z']+", s.lower())
                   if w not in STOP and len(w) > 2)
        if gold[i] == "behavior":
            cb.update(toks)
        elif gold[i] == "ideation":
            ci.update(toks)
    nb = sum(1 for i in range(len(rows)) if gold[i] == "behavior" and ev[i])
    ni = sum(1 for i in range(len(rows)) if gold[i] == "ideation" and ev[i])
    print(f"  {nb} behavior and {ni} ideation posts have gold evidence\n")
    sc = []
    for w, c in cb.items():
        if c < 5:
            continue
        rb, ri = c / nb, ci.get(w, 0) / max(ni, 1)
        sc.append((rb - ri, w, c, ci.get(w, 0)))
    sc.sort(reverse=True)
    print(f"  {'word':<18}{'in behavior ev':>16}{'in ideation ev':>16}"
          f"{'lift':>8}")
    for d, w, c, c2 in sc[:18]:
        print(f"  {w:<18}{c:>10} ({c/nb:>4.1%}){c2:>10} ({c2/max(ni,1):>4.1%})"
              f"{(c/nb)/max(c2/max(ni,1), 1e-9):>8.1f}x")
    print("\n  These are the words the ANNOTATOR put inside a behavior span")
    print("  and not inside an ideation span. If they are concrete acts and")
    print("  methods, the criterion is lexical and our models should reach it.")
    print("  If they are the same words in both, the criterion is contextual")
    print("  and no vocabulary will separate the classes.")


if __name__ == "__main__":
    main()

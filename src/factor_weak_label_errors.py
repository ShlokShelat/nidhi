"""
factor_weak_label_errors.py

WHAT is the model confusing the weak factor labels WITH?

WHY THIS COMES BEFORE ANY NEW ARCHITECTURE. Our diagnostics say HOW OFTEN the
weak labels fail; none of them say WHAT the failure looks like. Those imply
different fixes and the difference is expensive to get wrong:

  A. CONFUSION -- `meaning in life` is being absorbed by `hopelessness` /
     `low self-esteem`, i.e. the model has the concept but cannot separate it
     from a neighbour. -> label-specific attention, or contrastive separation
     of those specific pairs, is the right fix.

  B. BLINDNESS -- the positives share no signal the model responds to at all.
     -> attention will not help; the concept is not in the representation, and
     the honest answer may be an annotation ceiling (`meaning in life` has an
     oracle F1 of 0.235 no matter what reads it).

  C. ANNOTATOR IDIOSYNCRASY -- the gold positives have no shared vocabulary and
     look unlike each other. -> no architecture reaches it.

THE SIX WEAK LABELS ACCOUNT FOR MOST OF THE S2 DEFICIT. Held-out per-label F1:
meaning in life 0.231, sense of responsibility 0.411, social support 0.533,
psychological capital 0.577, plus the rare `sexual orientation related issues`
and `exposure to others' suicide`.

WHAT IT PRINTS, per weak label:
  1. FALSE NEGATIVES -- posts that HAVE it and we rank low. Which other labels
     do those posts carry? A label that dominates the FN set is the absorber
     (case A). A flat spread is case B.
  2. FALSE POSITIVES -- posts we rank high that do NOT have it, and what they
     actually carry. The mirror image of the same confusion.
  3. LEXICAL COHERENCE of the gold positives -- distinctive terms vs the corpus
     baseline. No distinctive vocabulary at all points at case C.
  4. CO-OCCURRENCE LIFT against every other label, so a structural absorber
     shows up even when the model ranks it correctly.
  5. The actual FN post text, so the concept can be eyeballed.

Ranking comes from the shipped 5-arm mean at the incumbent's own F1-optimal fire
count, so "false negative" means below the shipped cutoff, not below 0.5.

Usage:
  python src/factor_weak_label_errors.py
  python src/factor_weak_label_errors.py --labels "meaning in life" --show 12
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

D = "artifacts/factor_runs"
WEAK = ["meaning in life", "sexual orientation related issues",
        "exposure to others' suicide", "sense of responsibility",
        "stressful life event", "psychological capital"]
STOP = set("""a an the and or but if of to in on at for with without from by as
is are was were be been being do does did have has had i me my myself we our you
your he she it they them his her its their this that these those there here what
which who whom when where why how all any both each few more most other some
such no nor not only own same so than too very can will just dont don't im i'm
about into over after before again once because while up down out off then
""".split())


def load_arm(arm, rows):
    fs = sorted(glob.glob(f"{D}/{arm}_cv_oof_f*.npz"))
    if not fs:
        return None
    chunks, ids = [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        if "row_id" not in z.files:
            return None
        chunks.append(np.asarray(z["probs"], float))
        ids += [str(x) for x in z["row_id"]]
    P = np.concatenate(chunks, axis=0)
    idx = {}
    for i, r in enumerate(ids):
        idx.setdefault(r, i)
    if any(r not in idx for r in rows):
        return None
    return P[[idx[r] for r in rows]]


def f1_at_count(score, y, k):
    if k <= 0:
        return 0.0 if y.sum() else 1.0
    k = min(k, len(score))
    o = np.argsort(-score, kind="stable")
    p = np.zeros(len(score), bool)
    p[o[:k]] = True
    tp = int((p & (y == 1)).sum())
    if tp == 0:
        return 0.0
    pr, rc = tp / k, tp / max(int(y.sum()), 1)
    return 2 * pr * rc / (pr + rc)


def toks(t):
    return [w for w in re.findall(r"[a-z']+", str(t).lower())
            if w not in STOP and len(w) > 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="mnli,dmnli,cmine,prot,d4")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--labels", default=",".join(WEAK))
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = list(lab.index)
    labels = list(FACTOR_TAXONOMY)
    Y = np.array([[f in set(parse_factors(lab.loc[r, "factors"]))
                   for f in labels] for r in rows], int)
    posts = [" ".join(str(lab.loc[r, "post"]).split()) for r in rows]

    mats = []
    for a in [x.strip() for x in args.arms.split(",")]:
        P = load_arm(a, rows)
        if P is not None and P.shape[1] == len(labels):
            mats.append(P)
    if not mats:
        raise SystemExit(f"no arm OOF dumps in {D}/")
    INC = np.mean(mats, axis=0)
    print(f"{len(rows)} posts, {len(mats)} arms pooled, {len(labels)} labels")

    # corpus-wide token frequency, for the distinctiveness test
    corpus = Counter()
    for p in posts:
        corpus.update(set(toks(p)))
    n_doc = len(posts)

    want = [x.strip() for x in args.labels.split(",")]
    for f in want:
        if f not in labels:
            print(f"\n!! unknown label {f!r}")
            continue
        j = labels.index(f)
        y = Y[:, j]
        s = INC[:, j]
        # the incumbent's own optimal count
        step = max(1, len(rows) // 300)
        k, bf = 0, -1.0
        for kk in range(0, len(rows) + 1, step):
            v = f1_at_count(s, y, kk)
            if v > bf:
                bf, k = v, kk
        order = np.argsort(-s, kind="stable")
        pred = np.zeros(len(rows), bool)
        pred[order[:min(k, len(rows))]] = True
        fn = np.where(y.astype(bool) & ~pred)[0]
        fp = np.where(~y.astype(bool) & pred)[0]
        tp = np.where(y.astype(bool) & pred)[0]

        print("\n" + "=" * 78)
        print(f"{f}   n={int(y.sum())}  fires k={k}  F1={bf:.3f}  "
              f"TP={len(tp)} FN={len(fn)} FP={len(fp)}")
        print("=" * 78)

        def cooc(idx, title):
            if len(idx) == 0:
                print(f"  {title}: none")
                return
            c = Counter()
            for i in idx:
                for jj, g in enumerate(labels):
                    if jj != j and Y[i, jj]:
                        c[g] += 1
            tot = len(idx)
            print(f"  {title} ({tot} posts) carry:")
            for g, n in c.most_common(6):
                base = Y[:, labels.index(g)].mean()
                lift = (n / tot) / max(base, 1e-9)
                print(f"    {g[:38]:<40}{n:>4}/{tot} = {n/tot:>5.2f}   "
                      f"corpus {base:>5.2f}   lift {lift:>4.1f}x")

        cooc(fn, "FALSE NEGATIVES")
        cooc(fp, "FALSE POSITIVES")
        print("  A single label dominating BOTH lists with high lift is the")
        print("  ABSORBER -- case A, attention/contrastive can target it.")
        print("  A flat spread is case B: the concept is not represented.")

        # lexical distinctiveness of the gold positives
        pos = np.where(y.astype(bool))[0]
        pc = Counter()
        for i in pos:
            pc.update(set(toks(posts[i])))
        scored = []
        for w, n in pc.items():
            if n < max(2, len(pos) // 10):
                continue
            p_in = n / len(pos)
            p_all = corpus[w] / n_doc
            scored.append((p_in / max(p_all, 1e-9), w, n))
        scored.sort(reverse=True)
        print(f"  distinctive terms in the {len(pos)} gold positives "
              f"(lift vs corpus):")
        if not scored:
            print("    NONE above the frequency floor -- case C, the positives")
            print("    share no vocabulary. No architecture reaches that.")
        else:
            print("    " + "  ".join(f"{w}({l:.1f}x)" for l, w, _ in scored[:12]))

        print(f"\n  {min(args.show, len(fn))} FALSE NEGATIVES -- does the post "
              f"visibly contain the concept?")
        for i in fn[:args.show]:
            others = [g for jj, g in enumerate(labels) if jj != j and Y[i, jj]]
            print(f"\n    --- {rows[i]}   rank {int(np.where(order == i)[0][0])}"
                  f"/{len(rows)}   also: {others[:4]}")
            print(f"        {posts[i][:230]}")

    print("\n" + "=" * 78)
    print("READ THE CO-OCCURRENCE LIFT FIRST. Case A (one dominant absorber)")
    print("justifies label-specific attention or a targeted contrastive pair.")
    print("Case B/C do not, and `meaning in life` having an oracle F1 of 0.235")
    print("regardless of the reader is the standing warning that some of these")
    print("are annotation-bounded rather than architecture-bounded.")


if __name__ == "__main__":
    main()

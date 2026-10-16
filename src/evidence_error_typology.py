"""
evidence_error_typology.py

WHAT KIND of span error is responsible for the composite gap?

`evidence_suppression_rule.py` proved all remaining evidence headroom sits in
span quality on non-empty-gold posts (the largest single component, ~5x
everything else).
It did NOT say what is wrong with the spans, and the fix differs completely:

  LOCALIZATION  the tagger points at the wrong text entirely
                -> needs a better tagger. This is what evidence_mnli.sh buys.
  BOUNDARY      right text, wrong extent, fails containment
                -> a decoding change. No retraining.
  3x CAP        the predicted span CONTAINS the gold but is more than 3x its
                token
                length, so `_span_pair_matches` rejects it
                -> FREE to fix by trimming. Costs nothing but code.
  SPURIOUS      extra spans on posts that had fewer
                -> a count/threshold change. Free.

The scorer matches pred to gold by CONTAINMENT EITHER WAY plus a 3x token cap,
one-to-one (Kuhn's algorithm). So a prediction can be semantically perfect and
score zero purely for being too long. That failure mode is invisible in an F1
number and is the cheapest thing in the project to fix IF it is common.

CEILINGS REPORTED. Each "oracle" relaxes exactly one constraint and rescores,
which bounds what perfecting that one thing could ever pay. A ceiling is not a
gain -- it is the most the lane could hold.

CAVEAT, stated up front: this runs on the FRIEND's single OOF tagger (mean span
F1 0.496), the only gold-paired evidence predictions available; the shipped
vote runs ~0.698. So absolute counts are pessimistic. The QUESTION here
is the error MIX, which is a property of how span taggers fail, not of how good
one is. Treat the ranking as informative and the magnitudes as an upper bound.

Usage:
  python src/evidence_error_typology.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_vote_oof import author_halves, dedupe  # noqa: E402
from scoring import (normalize_span, parse_evidence, phrase_f1,  # noqa: E402
                     token_len)


def toks(s):
    return set(normalize_span(s).split())


def contains(p, g):
    p, g = normalize_span(p), normalize_span(g)
    return bool(p) and bool(g) and (p in g or g in p)


def match(p, g):
    return contains(p, g) and token_len(p) <= 3 * token_len(g)


def score(G, P):
    return phrase_f1(G, P, average="macro").f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--cands",
                    default="artifacts/evidence_oof/friend_oof_candidates.csv")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    cd = pd.read_csv(args.cands)
    cd["row_id"] = cd["row_id"].astype(str)
    if "min_conf" in cd.columns:
        cd = cd[cd["conf"] >= float(cd["min_conf"].iloc[0])]
    cd = cd.sort_values("conf", ascending=False)
    sp, cf = {}, {}
    for r, s, c in zip(cd["row_id"], cd["span"], cd["conf"]):
        sp.setdefault(r, []).append(str(s))
        cf.setdefault(r, []).append(float(c))
    sp = {r: dedupe(v) for r, v in sp.items()}

    # Only posts whose gold has spans -- the empty half is a solved, separate
    # problem (see evidence_suppression_rule.py).
    rows = [r for r in sorted(set(lab.index))
            if parse_evidence(lab.loc[r, "evidence"])]
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    P = [sp.get(r, []) for r in rows]
    auth = np.array([lab.loc[r, "anon_user_id"] for r in rows])

    ng = sum(len(g) for g in G)
    npd = sum(len(p) for p in P)
    print(f"{len(rows)} posts with non-empty gold, {pd.unique(auth).size} authors")
    print(f"gold spans {ng} ({ng / len(rows):.2f}/post)   "
          f"pred spans {npd} ({npd / len(rows):.2f}/post)   "
          f"mean span F1 {score(G, P):.4f}")
    print(f"silent on {sum(1 for p in P if not p)} of these posts "
          f"-- guaranteed 0.0 each under macro")

    # ---------------- classify every gold span and every prediction --------
    cat_g = {"matched": 0, "cap only": 0, "boundary": 0, "localization": 0}
    cat_p = {"matched": 0, "cap only": 0, "boundary": 0, "spurious": 0}
    cap_ratio = []
    for g, p in zip(G, P):
        gm = [False] * len(g)
        pm = [False] * len(p)
        # greedy one-to-one, mirroring the scorer closely enough for counting
        for i, gs in enumerate(g):
            for j, ps in enumerate(p):
                if not pm[j] and match(ps, gs):
                    gm[i], pm[j] = True, True
                    break
        for i, gs in enumerate(g):
            if gm[i]:
                cat_g["matched"] += 1
                continue
            if any(contains(ps, gs) for ps in p):
                cat_g["cap only"] += 1
                r = [token_len(ps) / max(token_len(gs), 1)
                     for ps in p if contains(ps, gs)]
                cap_ratio.append(max(r))
            elif any(toks(ps) & toks(gs) for ps in p):
                cat_g["boundary"] += 1
            else:
                cat_g["localization"] += 1
        for j, ps in enumerate(p):
            if pm[j]:
                cat_p["matched"] += 1
            elif any(contains(ps, gs) for gs in g):
                cat_p["cap only"] += 1
            elif any(toks(ps) & toks(gs) for gs in g):
                cat_p["boundary"] += 1
            else:
                cat_p["spurious"] += 1

    print(f"\n{'GOLD spans (recall side)':<28}{'n':>7}{'share':>9}")
    print("-" * 44)
    for k, v in cat_g.items():
        print(f"  {k:<26}{v:>7}{v / max(ng, 1):>9.3f}")
    print(f"\n{'PREDICTED spans (prec side)':<28}{'n':>7}{'share':>9}")
    print("-" * 44)
    for k, v in cat_p.items():
        print(f"  {k:<26}{v:>7}{v / max(npd, 1):>9.3f}")
    if cap_ratio:
        q = np.percentile(cap_ratio, [50, 75, 90])
        print(f"\n  cap-only overshoot ratio  median {q[0]:.1f}x  "
              f"p75 {q[1]:.1f}x  p90 {q[2]:.1f}x   (limit is 3.0x)")

    # ---------------- ceilings: relax exactly one constraint ---------------
    def sc_relaxed(cap):
        """Rescore with the 3x cap raised to `cap` -- bounds the trim lever."""
        tot = 0.0
        for g, p in zip(G, P):
            pm = [False] * len(p)
            hit = 0
            for gs in g:
                for j, ps in enumerate(p):
                    if not pm[j] and contains(ps, gs) and \
                            token_len(ps) <= cap * token_len(gs):
                        pm[j], hit = True, hit + 1
                        break
            pr = hit / len(p) if p else 0.0
            rc = hit / len(g) if g else 0.0
            tot += 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)
        return tot / len(rows)

    base = score(G, P)
    print("\n" + "=" * 68)
    print("CEILINGS -- each relaxes ONE constraint. Not gains; upper bounds.")
    print("=" * 68)
    print(f"{'change':<38}{'span F1':>9}{'delta':>9}{'compos':>10}")
    print("-" * 66)
    print(f"{'shipped (friend tagger)':<38}{base:>9.4f}{0.0:>+9.4f}{0.0:>+10.4f}")
    for cap, nm in ((float('inf'), "3x cap removed entirely"),):
        v = sc_relaxed(cap)
        print(f"{nm:<38}{v:>9.4f}{v - base:>+9.4f}"
              f"{0.3 * 0.642 * (v - base):>+10.4f}")

    # oracle count truncation: keep the top-k by confidence, k = |gold|
    Ptr = []
    for r, g, p in zip(rows, G, P):
        c = cf.get(r, [])
        order = sorted(range(len(p)), key=lambda i: -(c[i] if i < len(c) else 0))
        Ptr.append([p[i] for i in sorted(order[:len(g)])])
    v = score(G, Ptr)
    print(f"{'oracle count (keep top-|gold|)':<38}{v:>9.4f}{v - base:>+9.4f}"
          f"{0.3 * 0.642 * (v - base):>+10.4f}")

    # oracle localization: perfect spans wherever the sentence is already touched
    Pl = []
    for g, p in zip(G, P):
        fixed = [gs for gs in g if any(toks(ps) & toks(gs) for ps in p)]
        Pl.append(fixed if fixed else p)
    v = score(G, Pl)
    print(f"{'oracle boundary (right text->exact)':<38}{v:>9.4f}"
          f"{v - base:>+9.4f}{0.3 * 0.642 * (v - base):>+10.4f}")

    print("\nThe composite column already scales by 0.30 (evidence weight) and")
    print("0.642 (share of posts with non-empty gold).")
    print("\nDECISION RULE. If 'localization' dominates the recall side, the")
    print("spans are wrong TEXT and only a better tagger helps -- that is")
    print("evidence_mnli.sh, and it is the right lane. If 'cap only' or")
    print("'boundary' dominate, the right text is being found and lost on")
    print("extent, which is a free decoding fix and should ship FIRST.")


if __name__ == "__main__":
    main()

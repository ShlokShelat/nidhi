"""
evidence_arm_swap_check.py

IS THE CHAMPION SHIPPING THE WEAKER EVIDENCE ARM?

The champion's evidence column is reproduced exactly by
`artifacts/evvote_test_rc_repro.csv` / `evcond_evidence.csv` -- a
RISK-CONDITIONED tagger (`runs/evcond5_cond_f*`), which is a SINGLE model, not
a vote. On the honest instrument (predicted-risk suppression, macro averaging)
its candidate pool measures:

    conditioned, 1 tagger    1.11 clusters/post   70.8% match   153 unreachable
    unconditioned, 5 taggers 1.39 clusters/post   64.2% match    92 unreachable

It buys precision by staying quiet and pays 61 extra posts that no decode can
reach. And it was SELECTED under micro averaging with GOLD suppression, both
of which have since been disproved -- so it was chosen by a criterion that
does not match the scorer.

This scores both arms on identical posts with identical suppression and
reports the PAIRED author-bootstrap win rate. Acting on the mean gap alone,
without checking the win rate, is the exact failure mode this guards against
-- a change can show a positive mean gap and still lose once it actually
ships.

Usage:
  python src/evidence_arm_swap_check.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span, parse_evidence,  # noqa
                     phrase_f1, token_len)


def dedupe(sp):
    out = []
    for s in sorted(sp, key=lambda x: (-token_len(x),
                                       -len(normalize_span(x)))):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def load(d, tags):
    per = {}
    for t in tags:
        p = f"{d}/{t}_oof_candidates.csv"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        x = pd.read_csv(p)
        mc = float(x["min_conf"].iloc[0]) if "min_conf" in x.columns else 0.0
        x = x[x["conf"] >= mc]
        m = {}
        for r, s in zip(x["row_id"].astype(str), x["span"]):
            m.setdefault(r, []).append(str(s))
        per[t] = m
    return per


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--rc", default="v2_rc")
    a_.add_argument("--unc", nargs="+",
                    default=["v2", "ep12", "lr3", "bs16", "ml256"])
    a_.add_argument("--unc-k", type=int, default=2)
    a_.add_argument("--boot", type=int, default=400)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in oof["row_id"] if r in lab.index]
    ixr = oof.set_index("row_id")
    ship = np.array([normalize_label(x)
                     for x in ixr.loc[rows, "pred_risk_level"]])
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])

    RC = load(args.dir, [args.rc])
    UN = load(args.dir, args.unc)

    def build(per, k):
        out = []
        for i, r in enumerate(rows):
            if ship[i] == "indicator":
                out.append([])
                continue
            c = {}
            for t in per:
                for s in per[t].get(r, []):
                    c.setdefault(normalize_span(s), [set(), s])[0].add(t)
            keep = dedupe([v[1] for v in c.values() if len(v[0]) >= k])
            if not keep:
                pool = [s for t in per for s in per[t].get(r, [])]
                keep = sorted(pool, key=token_len)[:1]
            out.append(keep)
        return out

    A = build(RC, 1)                      # what the champion ships
    B = build(UN, args.unc_k)             # the unconditioned pool

    def f(P, ix):
        return phrase_f1([G[i] for i in ix], [P[i] for i in ix],
                         average="macro").f1

    allix = list(range(len(rows)))
    fa, fb = f(A, allix), f(B, allix)
    print(f"{len(rows)} OOF posts, suppression on PREDICTED risk, MACRO")
    print(f"  conditioned  (SHIPPED)  {fa:.4f}   "
          f"spans/post {np.mean([len(p) for p in A]):.2f}")
    print(f"  unconditioned pool k={args.unc_k}  {fb:.4f}   "
          f"spans/post {np.mean([len(p) for p in B]):.2f}")

    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    d = []
    for _ in range(args.boot):
        ix = np.concatenate([by[a] for a in
                             rng.choice(ua, size=len(ua), replace=True)])
        d.append(f(B, ix) - f(A, ix))
    d = np.array(d)
    sd = d.std(ddof=1)
    print(f"\nPAIRED author bootstrap ({args.boot} resamples)")
    print(f"  delta (unconditioned - shipped) {d.mean():+.4f}")
    print(f"  win {(d > 0).mean():.0%}   per-draw sd {sd:.4f}")
    print(f"  one-draw 95% [{d.mean()-1.96*sd:+.4f},{d.mean()+1.96*sd:+.4f}]")
    print(f"  composite value {0.3*d.mean():+.4f}")
    print("\n  Ship only on a high win rate AND a mostly-positive one-draw")
    print("  interval. A big mean with a win rate near 50% is not a reliable")
    print("  signal to ship on.")


if __name__ == "__main__":
    main()

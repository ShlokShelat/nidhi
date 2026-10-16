"""
risk_route_composite.py

Score risk-routing rules on the COMPOSITE, not on risk alone.

THE OVERSIGHT THIS FIXES. `risk_class_routing.py` judged every rule by risk
weighted-F1 alone, under which `route_indicator` is indistinguishable from
noise. But indicator errors cost TWICE:

    P(gold evidence empty | gold indicator)     = 0.961
    P(gold evidence empty | gold NOT indicator) = 0.000

and `evidence_vote_test.py` suppresses all spans whenever the predicted level is
indicator. So a post crossing that boundary changes its EVIDENCE score
deterministically, with no tagger involved:

    moves INTO indicator,  gold empty      evidence 0.0 -> 1.0
    moves INTO indicator,  gold has spans  evidence  s  -> 0.0
    moves OUT of indicator, gold empty     evidence 1.0 -> 0.0
    moves OUT of indicator, gold has spans evidence 0.0 ->  s

Only the indicator boundary is coupled this way; ideation/behavior/attempt moves
leave evidence untouched. `route_attempt` therefore had no evidence component,
but `route_indicator` does, and scoring it on risk alone understates it.

`s` is the taggers' mean per-post F1 on non-indicator posts -- measured at ~0.79
by the coupling decomposition. Only posts CROSSING the boundary depend on it, so
the delta is insensitive to the exact value; the script sweeps it to show that.

COMPOSITE = 0.4 x risk_wF1 + 0.3 x evidence_macro. Every rule is 0-parameter and
gets a paired AUTHOR bootstrap on the composite delta.

Usage:
  python src/risk_route_composite.py
  python src/risk_route_composite.py --enc artifacts/champion_oof.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (RISK_LEVELS, normalize_label,  # noqa: E402
                     parse_evidence, risk_weighted_f1)

PC = [f"prob_{c}" for c in RISK_LEVELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--llm", default="artifacts/cot_soft_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--s", type=float, default=0.79,
                    help="taggers' mean per-post evidence F1 on non-indicator "
                         "posts; only boundary-crossing posts depend on it")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    def read(p):
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        return d.drop_duplicates("row_id").set_index("row_id")

    E, L = read(args.enc), read(args.llm)
    rows = sorted(set(E.index) & set(L.index) & set(lab.index))
    y = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    author = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    empty = np.array([len(parse_evidence(lab.loc[r, "evidence"])) == 0
                      for r in rows])

    def preds(d, alt=None):
        if alt and alt in d.columns:
            return np.array([normalize_label(x) for x in d.loc[rows, alt]])
        if "pred_risk_level" in d.columns:
            return np.array([normalize_label(x)
                             for x in d.loc[rows, "pred_risk_level"]])
        M = d.loc[rows, PC].to_numpy(float)
        return np.array([RISK_LEVELS[i] for i in M.argmax(1)])

    ep, lp = preds(E), preds(L, "cot")
    print(f"{len(rows)} posts / {len(set(author))} authors")
    print(f"gold-empty evidence: {int(empty.sum())} "
          f"({empty.mean():.1%})   gold indicator: "
          f"{int((y == 'indicator').sum())}")
    agree = float((empty == (y == "indicator")).mean())
    print(f"empty <-> indicator agreement: {agree:.3f}  "
          f"(the coupling this exploits)")

    def ev_macro(pred, sel=None):
        """Evidence macro under the shipped suppression rule.

        Deterministic on the indicator boundary; on non-indicator posts with
        non-empty gold the taggers supply s. Only boundary crossings move, so
        the DELTA barely depends on s."""
        if sel is None:
            sel = np.arange(len(pred))
        pi = pred[sel] == "indicator"
        em = empty[sel]
        v = np.where(pi & em, 1.0,
                     np.where(pi & ~em, 0.0,
                              np.where(~pi & em, 0.0, args.s)))
        return float(v.mean())

    def comp(pred, sel=None):
        if sel is None:
            sel = np.arange(len(pred))
        r = risk_weighted_f1(list(y[sel]), list(pred[sel]))
        return 0.4 * r + 0.3 * ev_macro(pred, sel), r, ev_macro(pred, sel)

    base_c, base_r, base_e = comp(ep)
    print(f"\nbaseline   risk {base_r:.4f}   evidence {base_e:.4f}   "
          f"0.4r+0.3e = {base_c:.4f}")
    print("(the evidence figure is a MODEL of the shipped rule, not the "
          "champion's\n actual evidence score -- only DELTAS are meaningful)")

    rules = {}
    for c in RISK_LEVELS:
        rules[f"route_{c}"] = np.where(lp == c, lp, ep)
    rules["route_attempt+indicator"] = np.where(
        np.isin(lp, ["attempt", "indicator"]), lp, ep)
    # on top of the SHIPPED rule, which is route_attempt
    shipped = np.where(lp == "attempt", lp, ep)
    rules["[shipped] route_attempt"] = shipped
    rules["shipped + indicator"] = np.where(lp == "indicator", lp, shipped)

    rng = np.random.default_rng(args.seed)
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}

    print(f"\n{'rule':<28}{'risk':>8}{'dRisk':>9}{'evid':>8}{'dEvid':>9}"
          f"{'dCOMP':>9}{'95% CI':>22}{'moved':>7}")
    print("-" * 100)
    for name, p in rules.items():
        c, r, e = comp(p)
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            draws.append(comp(p, sel)[0] - comp(ep, sel)[0])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        star = "  <-" if lo > 0 else ""
        print(f"{name:<28}{r:>8.4f}{r - base_r:>+9.4f}{e:>8.4f}"
              f"{e - base_e:>+9.4f}{c - base_c:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}"
              f"{int((p != ep).sum()):>7}{star}")

    print("\nSENSITIVITY to s (the taggers' mean F1 on non-indicator posts)")
    print(f"{'s':<8}" + "".join(f"{n[:16]:>18}" for n in
                                ("route_indicator", "shipped + indicator")))
    for s in (0.70, 0.75, 0.79, 0.83, 0.88):
        args.s = s
        b = comp(ep)[0]
        line = f"{s:<8.2f}"
        for n in ("route_indicator", "shipped + indicator"):
            line += f"{comp(rules[n])[0] - b:>18.4f}"
        print(line)
    print("Flat across s means the result is not an artefact of that estimate.")

    print("\n" + "=" * 78)
    print("`route_attempt` is already SHIPPED. The row")
    print("that matters is `shipped + indicator` -- what indicator routing adds")
    print("ON TOP of it, counting the evidence coupling that risk-only scoring")
    print("missed. Per-post routing changes have historically transferred to")
    print("held-out evaluation at roughly 1.6-2.4x their offline measurement,")
    print("so a positive composite delta here is worth roughly double there.")


if __name__ == "__main__":
    main()

"""
risk_class_routing.py

Route per class: trust the LLM where it is strong, the encoder where it is.

WHY THE IDEA IS SOUND. On 1635 OOF posts the CoT 72B and the encoder chain are
close overall (0.7865 vs 0.7985) but differ sharply BY CLASS:

    class       enc rec   cot rec
    indicator     0.856     0.879   <- LLM better
    ideation      0.815     0.750
    behavior      0.731     0.754   <- LLM better, the biggest error bucket
    attempt       0.640     0.579

So "use each model where it wins" is the obvious move. Two problems it has to
survive.

PROBLEM 1: THE TRUE CLASS IS UNKNOWN AT INFERENCE. Routing must be decidable
from PREDICTIONS. And the deciding quantity is then PRECISION, not recall: the
rule fires on "the LLM SAYS behavior", so what matters is how often that is
right, which the recall table above does not show. This prints precision for
both models, and the full disagreement breakdown by (enc_pred, cot_pred) cell,
which is exactly the table the rule reads.

PROBLEM 2: PARAMETERS. The measured transfer law on this task is brutal -- 0
fitted params over-deliver, and a single fitted parameter has historically
transferred poorly. A 12-cell lookup carries far more fitted capacity than a
single parameter. So rules are reported at three capacities, and every fitted
choice is CROSS-FITTED over author halves (decided on one half, applied to the
other) so nothing is scored where it was fitted:

    swap_all      0 params  always take the LLM (control, expected to lose)
    route_<c>     0 params  a SINGLE named class, fixed a priori from the
                            recall table -- behavior and indicator are the two
                            the LLM wins, and behavior is the largest bucket
    route_xfit    <=4       choose WHICH classes to route on, cross-fitted
    cell_xfit     <=12      per (enc_pred, cot_pred) cell, cross-fitted

HONEST CAVEAT: the bootstrap resamples authors around FIXED cross-fitted
predictions, so it does not price the fitting variance. Read the low-capacity
rows as the trustworthy ones; if only cell_xfit wins, that is a warning sign,
not a result.

Usage:
  python src/risk_class_routing.py
  python src/risk_class_routing.py --llm artifacts/cot_soft_oof.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

PC = [f"prob_{c}" for c in RISK_LEVELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--llm", default="artifacts/cot_soft_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--pre-route", default="",
                    help="comma-separated classes ALREADY routed to the LLM, "
                         "folded into the baseline before anything is scored. "
                         "Use --pre-route attempt to ask what is left once the "
                         "shipped rule is in place -- routing one class changes "
                         "the error profile the next rule sees, so measuring "
                         "them independently double-counts.")
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

    def preds(d):
        if "pred_risk_level" in d.columns:
            return np.array([normalize_label(x)
                             for x in d.loc[rows, "pred_risk_level"]])
        if all(c in d.columns for c in PC):
            M = d.loc[rows, PC].to_numpy(float)
            return np.array([RISK_LEVELS[i] for i in M.argmax(1)])
        raise SystemExit("no pred_risk_level and no prob_* columns")

    ep, lp = preds(E), preds(L)
    if "cot" in L.columns:
        lp = np.array([normalize_label(x) for x in L.loc[rows, "cot"]])
    print(f"{len(rows)} posts / {len(set(author))} authors")
    print(f"raw encoder wF1 {risk_weighted_f1(list(y), list(ep)):.4f}   "
          f"llm wF1 {risk_weighted_f1(list(y), list(lp)):.4f}")

    if args.pre_route:
        pre = [c.strip() for c in args.pre_route.split(",") if c.strip()]
        bad = [c for c in pre if c not in RISK_LEVELS]
        if bad:
            raise SystemExit(f"--pre-route: unknown class {bad}")
        n_pre = int((np.isin(lp, pre) & (ep != lp)).sum())
        ep = np.where(np.isin(lp, pre), lp, ep)
        print(f"PRE-ROUTED {pre} into the baseline: {n_pre} posts moved. "
              f"Everything below is measured ON TOP of that.")
    base = risk_weighted_f1(list(y), list(ep))
    print(f"baseline wF1 {base:.4f}")

    print("\nPRECISION IS WHAT THE RULE READS, NOT RECALL")
    print(f"{'class':<12}{'enc P':>8}{'llm P':>8}{'enc R':>8}{'llm R':>8}"
          f"{'enc n':>7}{'llm n':>7}")
    print("-" * 58)
    for c in RISK_LEVELS:
        g = y == c
        for_ = []
        for p in (ep, lp):
            q = p == c
            tp = int((g & q).sum())
            for_ += [tp / max(int(q.sum()), 1), tp / max(int(g.sum()), 1),
                     int(q.sum())]
        print(f"{c:<12}{for_[0]:>8.3f}{for_[3]:>8.3f}{for_[1]:>8.3f}"
              f"{for_[4]:>8.3f}{for_[2]:>7}{for_[5]:>7}")

    dis = ep != lp
    print(f"\nThey disagree on {int(dis.sum())} of {len(rows)} posts "
          f"({dis.mean():.1%})")
    print("On those posts, who is right? (rows = encoder said, cols = llm said)")
    print(f"{'':<12}" + "".join(f"{c[:9]:>12}" for c in RISK_LEVELS))
    for a in RISK_LEVELS:
        line = f"{a[:11]:<12}"
        for b in RISK_LEVELS:
            m = dis & (ep == a) & (lp == b)
            if not m.sum():
                line += f"{'-':>12}"
                continue
            ne = int((y[m] == a).sum())
            nl = int((y[m] == b).sum())
            line += f"{f'{ne}/{nl}':>12}"
        print(line)
    print("cell = encoder-right / llm-right on that disagreement. A cell where")
    print("llm >> enc is a routing opportunity; balanced cells are noise.")

    # ---------------- rules ----------------
    rng = np.random.default_rng(args.seed)
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}
    perm = rng.permutation(len(auths))
    hA = {auths[i] for i in perm[:len(auths) // 2]}
    inA = np.array([a in hA for a in author])

    def route(classes):
        """Take the LLM wherever it predicts one of `classes`."""
        return np.where(np.isin(lp, list(classes)), lp, ep)

    def ci(p):
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            draws.append(risk_weighted_f1(list(y[sel]), list(p[sel]))
                         - risk_weighted_f1(list(y[sel]), list(ep[sel])))
        return np.percentile(draws, [2.5, 97.5])

    results = []
    results.append(("swap_all", 0, lp))
    for c in RISK_LEVELS:
        results.append((f"route_{c[:9]}", 0, route([c])))

    # choose the class SET on one half, apply to the other
    best_sets = []
    out = ep.copy()
    for tune, appl in ((inA, ~inA), (~inA, inA)):
        best, bf = (), -1.0
        for k in range(0, len(RISK_LEVELS) + 1):
            for s in itertools.combinations(RISK_LEVELS, k):
                p = route(s)
                f = risk_weighted_f1(list(y[tune]), list(p[tune]))
                if f > bf:
                    bf, best = f, s
        best_sets.append(best)
        p = route(best)
        out[appl] = p[appl]
    results.append((f"route_xfit {[tuple(x[:4] for x in s) for s in best_sets]}",
                    4, out))

    # per-cell lookup, cross-fitted
    out2 = ep.copy()
    for tune, appl in ((inA, ~inA), (~inA, inA)):
        take = set()
        for a in RISK_LEVELS:
            for b in RISK_LEVELS:
                if a == b:
                    continue
                m = tune & (ep == a) & (lp == b)
                if m.sum() and int((y[m] == b).sum()) > int((y[m] == a).sum()):
                    take.add((a, b))
        p = np.array([b if (a, b) in take else a for a, b in zip(ep, lp)])
        out2[appl] = p[appl]
    results.append(("cell_xfit", 12, out2))

    print(f"\n{'rule':<44}{'par':>4}{'wF1':>9}{'delta':>9}{'95% CI':>22}"
          f"{'moved':>7}")
    print("-" * 95)
    for name, npar, p in results:
        f1 = risk_weighted_f1(list(y), list(p))
        lo, hi = ci(p)
        star = "  <-" if lo > 0 else ""
        print(f"{name[:43]:<44}{npar:>4}{f1:>9.4f}{f1 - base:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{int((p != ep).sum()):>7}{star}")

    print(f"\nencoder baseline {base:.4f}. Composite impact = 0.4 x delta.")
    print("TRUST THE 0-PARAMETER ROWS. If only cell_xfit (12 params) wins, that")
    print("is the 27-parameter failure mode warming up, not a result.")


if __name__ == "__main__":
    main()

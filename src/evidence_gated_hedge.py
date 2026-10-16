"""
evidence_gated_hedge.py

HEDGE THE EVIDENCE CONDITIONING ONLY WHERE THE RISK LABEL IS LIKELY WRONG.

`evidence_conditioning_oracle.py` confirmed the mechanism cleanly:

    decode        ALL     risk RIGHT  risk WRONG      (372 val posts, 63 wrong)
    predicted   0.7864     0.8515       0.5789        <- shipped
    gold        0.8261     0.8515       0.7386        <- oracle
    weighted    0.7786     0.8428       0.5668        <- lost
    best-conf   0.7790     0.8399       0.5820        <- lost

Identical on risk-RIGHT posts to four decimals, which is the control working:
conditioning cannot matter where the label is already correct. On risk-WRONG
posts the wrong question costs **0.16 evidence F1**. Converted through the
0.30 evidence weight, that gap is the prize this hedge is chasing.

WHY BLANKET HEDGING LOST, AND WHAT THE ARITHMETIC ACTUALLY SAYS. phrase-F1 is
macro per post with bipartite matching, so adding one extra span to a post that
is already right takes it from 1.0 to ~0.67, while adding the right clause to a
post scoring 0 takes it to ~0.67. A hedge is therefore worth it exactly when

    P(risk wrong) * 0.67  >  P(risk right) * 0.33      ->  P(wrong) > ~1/3

Overall P(wrong) is 0.169, so hedging EVERY post must lose -- and it did, by
almost exactly the predicted margin. The question is not whether to hedge but
whether a findable SLICE clears 1/3.

THE GATE. Risk errors are confidently wrong, but
"confidently" is not "uniformly": `risk_error_anatomy.py` measured the
behavior->ideation cell at 44% error above p=0.7 and 16% below p=0.5, so the
gradient exists even if it is shallow. This sweeps three gates and reports the
ERROR RATE inside each slice, which is the number that decides the lane before
any F1 is computed:

    maxprob   hedge when the top risk probability is below t
    margin    hedge when top minus runner-up is below t
    entropy   hedge when the risk distribution is flat

HEDGE FORM. Union the predicted conditioning's spans with the RUNNER-UP
conditioning's, deduped. Not all four -- that is what `weighted` did and the
extra two conditionings are almost always noise.

Everything is cross-fitted: t is chosen on FIT authors and scored on HELD-OUT
authors, so the threshold never sees the posts it is judged on. Indicator posts
keep empty evidence throughout.

Usage:
  python src/evidence_conditioning_oracle.py     # writes the span cache first
  python src/evidence_gated_hedge.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def pf1(gold, pred):
    return phrase_f1(gold, pred).f1


def dedupe(spans):
    out = []
    for s in sorted(spans, key=lambda x: (-token_len(x), -len(x))):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    if not os.path.exists(args.cache):
        raise SystemExit(f"{args.cache} missing -- run "
                         f"evidence_conditioning_oracle.py first")
    S = json.load(open(args.cache))

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    va = va[va["row_id"].isin(S)].reset_index(drop=True)
    rows = va["row_id"].tolist()
    gold_risk = np.array([normalize_label(v) for v in va["risk_level"]])
    gold_ev = [parse_evidence(v) for v in va["evidence"]]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])

    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    P = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    pred_risk = np.array([normalize_label(v)
                          for v in pv.loc[rows, "pred_risk_level"]])
    wrong = pred_risk != gold_risk

    srt = np.sort(P, axis=1)
    gates = {
        "maxprob": srt[:, -1],
        "margin": srt[:, -1] - srt[:, -2],
        "entropy": -(P * np.log(np.clip(P, 1e-9, None))).sum(1) * -1.0,
    }

    def spans(r, lv):
        return [s for s, c in S[r][lv] if c >= args.min_conf]

    base, hedged = [], []
    for i, r in enumerate(rows):
        if pred_risk[i] == "indicator":
            base.append([])
            hedged.append([])
            continue
        b = dedupe(spans(r, pred_risk[i]))
        second = LEVELS[int(np.argsort(-P[i])[1])]
        base.append(b)
        hedged.append(dedupe(b + spans(r, second)))

    print(f"{len(rows)} val posts, risk wrong on {wrong.mean():.1%} "
          f"({int(wrong.sum())} posts)")
    print(f"  ship {pf1(gold_ev, base):.4f}   hedge-everything "
          f"{pf1(gold_ev, hedged):.4f}\n")

    print("=" * 84)
    print("1. DOES ANY SLICE CLEAR THE 1/3 ERROR RATE THE HEDGE NEEDS?")
    print("=" * 84)
    for gname, g in gates.items():
        print(f"  {gname}")
        for q in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
            t = np.quantile(g, q)
            m = g <= t
            if m.sum() < 5:
                continue
            flag = "  <== clears 1/3" if wrong[m].mean() > 1 / 3 else ""
            print(f"    bottom {q:>4.0%}  n {int(m.sum()):>4}  "
                  f"error rate {wrong[m].mean():>6.1%}{flag}")
    print("\n  Overall error is 16.9%. A gate that cannot lift a slice above")
    print("  ~33% cannot make hedging pay, whatever the F1 sweep says --")
    print("  any apparent gain would be fitted to this sample.")

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    QGRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0]
    print("\n" + "=" * 84)
    print("2. CROSS-FITTED: gate quantile chosen on FIT authors, scored on")
    print("   HELD-OUT authors")
    print("=" * 84)
    print(f"  {'gate':<10}{'d evidence':>12}{'d comp':>9}{'win':>6}{'LOSS':>6}"
          f"{'sd':>9}{'q chosen':>12}{'hedged/split':>14}")
    for gname, g in gates.items():
        ds, qs, nh = [], [], []
        for _ in range(args.splits):
            o = list(ua)
            rng.shuffle(o)
            k = max(int(len(o) * 0.25), 2)
            ho = set(o[:k])
            hm = np.array([a in ho for a in auth])
            fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
            if len(fi) < 40 or len(hi) < 15:
                continue

            def build(q, idx):
                if q <= 0:
                    return [base[i] for i in idx]
                t = np.quantile(g[fi], q)
                return [hedged[i] if g[i] <= t else base[i] for i in idx]

            bq = max(QGRID, key=lambda q: pf1([gold_ev[i] for i in fi],
                                              build(q, fi)))
            gh = [gold_ev[i] for i in hi]
            ds.append(pf1(gh, build(bq, hi)) - pf1(gh, [base[i] for i in hi]))
            qs.append(bq)
            nh.append(sum(1 for i in hi
                          if bq > 0 and g[i] <= np.quantile(g[fi], bq)))
        d = np.array(ds)
        print(f"  {gname:<10}{d.mean():>+12.4f}{0.3*d.mean():>+9.4f}"
              f"{np.mean(d > 0):>5.0%}{np.mean(d < 0):>6.0%}"
              f"{0.3*d.std():>9.4f}"
              f"{f'{np.mean(qs):.2f}+-{np.std(qs):.2f}':>12}"
              f"{np.mean(nh):>14.1f}")
    print("\n  A wandering q means the gate is being fitted per sample and")
    print("  will not transfer -- the same signature that killed the factor")
    print("  threshold shrinkage today (k = 65 +- 100).")
    print("  Oracle for reference: converting the evidence gap through the")
    print("  0.30 evidence weight gives the composite ceiling.")


if __name__ == "__main__":
    main()

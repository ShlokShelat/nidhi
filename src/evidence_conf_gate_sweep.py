"""
evidence_conf_gate_sweep.py

Sweep the candidate confidence gate jointly with the vote threshold k.

WHY THE TWO KNOBS BELONG TOGETHER. Each deployed tagger contributes spans at
ITS OWN `min_conf`, fitted when that tagger was trained alone. A threshold that
maximises one tagger's solo F1 is not the threshold that maximises a 7-way
vote: in a vote, a sub-threshold span that two taggers independently propose is
strong evidence, but solo tuning discards it before the vote can see it.

The gate also controls whether the candidate pool is large enough to contain
the answer at all. When the gate leaves fewer candidates per post than there
are gold spans per post, no selection rule can recover full recall, because it
is choosing from a pool with fewer candidates than there are correct answers.
The length cap does not bind at the decoded span lengths, so the confidence
gate is the only active constraint.

THE TRADE. Lowering the gate raises coverage and lowers per-candidate
precision. Whether that helps depends on the vote: k>=2 filters noise, so a
wider pool with a stricter k may beat a narrow pool with a loose k. That is a
2-D surface (gate x k), and the two axes have to be swept against each other
rather than one at a time.

Cross-fitted by author: both the gate and k are chosen on FIT authors and
scored on HELD-OUT. Judge by WIN RATE and the share of resamples that LOSE
rather than by the mean, since the mean of a small-sample delta is noisy while
the win rate is not.

Usage:
  python src/evidence_conf_gate_sweep.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def f1_of(gold, pred):
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    adj = [[j for j, g in enumerate(gold) if _span_pair_matches(p, g)]
           for p in pred]
    m = _max_bipartite_matching(adj, len(gold))
    pr, rc = m / len(pred), m / len(gold)
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def dedupe_nested(spans):
    out = []
    for s in sorted(spans, key=lambda x: -token_len(x)):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--dir", default="artifacts/evidence_runs")
    a_.add_argument("--runs", default=",".join(RC7))
    a_.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    rk = pd.read_csv(args.risk, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    col = "pred_risk_level" if "pred_risk_level" in rk.columns else "risk_level"
    supp = {r: normalize_label(v) == "indicator"
            for r, v in zip(rk["row_id"], rk[col])}

    # keep EVERY candidate with its confidence; the gate is applied later
    per, shipped = {}, {}
    for run in [x.strip() for x in args.runs.split(",")]:
        p = os.path.join(args.dir, f"{run}_val_candidates.csv")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        shipped[run] = (float(d["min_conf"].iloc[0])
                        if "min_conf" in d.columns else 0.0)
        cf = (d["conf"].astype(float) if "conf" in d.columns
              else pd.Series(np.ones(len(d))))
        m = {}
        for r, s, c in zip(d["row_id"], d["span"], cf):
            m.setdefault(r, []).append((str(s), float(c)))
        per[run] = m
    if len(per) < 3:
        raise SystemExit("need the val candidate dumps")
    print(f"{len(rows)} val posts, {len(per)} taggers")
    print("  per-tagger gates: " + ", ".join(
        f"{k.replace('evidence_','')}={v:g}" for k, v in shipped.items()))

    def vote(gate, k):
        """gate=None uses each tagger's own deployed min_conf."""
        out = []
        for r in rows:
            if supp.get(r, False):
                out.append([])
                continue
            cnt = {}
            for run in per:
                g = shipped[run] if gate is None else gate
                for s, c in per[run].get(r, ()):
                    if c < g:
                        continue
                    n = normalize_span(s)
                    if n:
                        e = cnt.setdefault(n, [0, s])
                        e[0] += 1
            out.append(dedupe_nested([v[1] for v in cnt.values()
                                      if v[0] >= k]))
        return out

    def macro(pred, ix=None):
        ix = range(len(rows)) if ix is None else ix
        return float(np.mean([f1_of(gold[i], pred[i]) for i in ix]))

    nz = [i for i in range(len(rows)) if gold[i]]
    base = vote(None, 2)
    print(f"\n  DEPLOYED (own gates, k=2): macro {macro(base):.4f}   "
          f"spans/post {np.mean([len(base[i]) for i in nz]):.2f}   "
          f"gold {np.mean([len(gold[i]) for i in nz]):.2f}")

    GATES = [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    KS = [2, 3, 4, 5]
    print("\n" + "=" * 74)
    print("1. THE 2-D SURFACE -- gate x k, full sample")
    print("=" * 74)
    print(f"  {'gate':>7}" + "".join(f"{'k=' + str(k):>10}" for k in KS)
          + "   spans/post at best k")
    cache = {}
    for g in GATES:
        cells = []
        for k in KS:
            v = vote(g, k)
            cache[(g, k)] = v
            cells.append(macro(v))
        bk = KS[int(np.argmax(cells))]
        sp = np.mean([len(cache[(g, bk)][i]) for i in nz])
        tag = "  <- deployed" if g is None else ""
        print(f"  {'own' if g is None else f'{g:.1f}':>7}"
              + "".join(f"{c:>10.4f}" for c in cells)
              + f"   {sp:>5.2f} (k={bk}){tag}")
    print("\n  A wider pool needs a stricter k. A best cell at a lower gate")
    print("  AND a higher k than the deployed corner means the two knobs")
    print("  interact and cannot be set independently.")

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        n = max(int(len(o) * 0.25), 2)
        ho = set(o[:n])
        hm = np.array([a in ho for a in auth])
        subs.append((np.flatnonzero(~hm), np.flatnonzero(hm)))

    print("\n" + "=" * 74)
    print("2. CROSS-FITTED -- gate and k chosen on FIT, scored on HELD-OUT")
    print("=" * 74)
    ds, picks = [], []
    for fi, hi in subs:
        best, bp = -1.0, (None, 2)
        for g in GATES:
            for k in KS:
                v = macro(cache[(g, k)], fi)
                if v > best:
                    best, bp = v, (g, k)
        ds.append(macro(cache[bp], hi) - macro(base, hi))
        picks.append(bp)
    d = np.array(ds)
    from collections import Counter
    print(f"  d evidence {d.mean():+.4f}   d comp {0.3*d.mean():+.4f}   "
          f"win {np.mean(d > 0):.0%}   LOSS {np.mean(d < 0):.0%}   "
          f"sd {0.3*d.std():.4f}")
    print("  chosen (gate,k): " + ", ".join(
        f"{('own' if a is None else a)}/{b} x{c}"
        for (a, b), c in Counter(picks).most_common(5)))
    print("\n  A STABLE (gate,k) across splits is a property of the problem; a")
    print("  wandering one is the search fitting each resample and will not")
    print("  transfer to unseen authors.")


if __name__ == "__main__":
    main()

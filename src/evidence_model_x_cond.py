"""
evidence_model_x_cond.py

TWO INDEPENDENT AGREEMENT AXES: ACROSS MODELS, AND ACROSS QUESTIONS.

The deployed decode keeps a span when >=2 of 7 taggers propose it, all
conditioned on the SAME predicted risk level. That measures agreement across
MODELS only.

Agreement across CONDITIONINGS is a second, independent axis: a span that
survives being asked for under several different risk labels is robust to the
risk model being wrong, and unlike an oracle-style fix it does not require
knowing which conditioning was correct.

Combining the two axes needs every tagger dumped under every conditioning. The
`cond4` job dumps all SEVEN deployed taggers under all FOUR conditionings, so a
span carries:

    n_models  how many of 7 proposed it under the PREDICTED conditioning
    n_cond    how many of 4 conditionings produced it (any model)
    n_cells   how many of the 28 (model, conditioning) cells produced it
    w_cond    conditioning count weighted by the risk model's probability

Model-agreement and question-agreement are different quantities: a span every
tagger finds under one label is not the same as a span one tagger finds under
every label.

THE BASELINE MUST BE THE REAL ONE. A rule compared against the votes=1 UNION
rather than the shipped vote scores a gain that is only the baseline being
weaker, so this script reconstructs the shipped decode from the
predicted-conditioning dumps and refuses to report anything if it does not land
on the independently measured 7-tagger baseline (EXPECTED_BASE).

Usage (after the cond4 dump):
  python src/evidence_model_x_cond.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]
EXPECTED_BASE = 0.8045


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
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    a_.add_argument("--dirfmt", default="artifacts/evidence_cond4_{lv}")
    a_.add_argument("--runs", default=",".join(RC7))
    a_.add_argument("--splits", type=int, default=250)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    gold_risk = np.array([normalize_label(v) for v in va["risk_level"]])
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    rk = pd.read_csv(args.risk, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    col = "pred_risk_level" if "pred_risk_level" in rk.columns else "risk_level"
    pred_risk = np.array([normalize_label(v) for v in
                          rk.set_index("row_id").loc[rows, col]])
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    P = pv.set_index("row_id").loc[rows,
                                   [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)

    runs = [x.strip() for x in args.runs.split(",")]
    # cand[i][norm_span] = {"s":text, "cells":set((run,lv)), "pred":set(run)}
    cand = [dict() for _ in rows]
    ix = {r: i for i, r in enumerate(rows)}
    got = 0
    for lv in LEVELS:
        d = args.dirfmt.format(lv=lv)
        for run in runs:
            p = os.path.join(d, f"{run}_val_candidates.csv")
            if not os.path.exists(p):
                continue
            got += 1
            t = pd.read_csv(p, keep_default_na=False)
            t["row_id"] = t["row_id"].astype(str)
            for r, s in zip(t["row_id"], t["span"]):
                if r not in ix:
                    continue
                i = ix[r]
                n = normalize_span(str(s))
                if not n:
                    continue
                e = cand[i].setdefault(n, {"s": str(s), "cells": set(),
                                           "pred": set()})
                e["cells"].add((run, lv))
                if lv == pred_risk[i]:
                    e["pred"].add(run)
    if got < len(runs) * len(LEVELS):
        print(f"  WARNING only {got} of {len(runs)*len(LEVELS)} "
              f"(run, conditioning) dumps found")
    if got == 0:
        raise SystemExit("no cond4 dumps -- has the job finished?")

    for i in range(len(rows)):
        for e in cand[i].values():
            e["n_models"] = len(e["pred"])
            e["n_cond"] = len({lv for _, lv in e["cells"]})
            e["n_cells"] = len(e["cells"])
            e["w_cond"] = sum(P[i, LEVELS.index(lv)]
                              for lv in {lv for _, lv in e["cells"]})

    def build(fn):
        out = []
        for i in range(len(rows)):
            if pred_risk[i] == "indicator":
                out.append([])
                continue
            out.append(dedupe_nested([e["s"] for e in cand[i].values()
                                      if fn(e)]))
        return out

    def macro(pred, sub=None):
        sub = range(len(rows)) if sub is None else sub
        return float(np.mean([f1_of(gold[i], pred[i]) for i in sub]))

    base = build(lambda e: e["n_models"] >= 2)
    b = macro(base)
    nz = [i for i in range(len(rows)) if gold[i]]
    print(f"{len(rows)} val posts   reconstructed shipped decode "
          f"macro {b:.4f}   spans/post "
          f"{np.mean([len(base[i]) for i in nz]):.2f}")
    if abs(b - EXPECTED_BASE) > 0.01:
        print(f"  !! WARNING baseline {b:.4f} != {EXPECTED_BASE} measured by "
              f"evidence_member_loo.py")
        print("     The decode is not reproduced; every delta below is against")
        print("     the WRONG reference. Fix the reconstruction first.")

    print("\n" + "=" * 76)
    print("1. ARE THE TWO AXES INDEPENDENT?")
    print("=" * 76)
    allc = [e for i in range(len(rows)) if gold[i] for e in cand[i].values()]
    ys = [any(_span_pair_matches(e["s"], g) for g in gold[i])
          for i in range(len(rows)) if gold[i] for e in cand[i].values()]
    ys = np.array(ys)
    nm = np.array([e["n_models"] for e in allc])
    nc = np.array([e["n_cond"] for e in allc])
    print(f"  corr(n_models, n_cond) = {np.corrcoef(nm, nc)[0,1]:.3f}"
          f"   <- near 0 means genuinely different information")
    hdr = "n_models v / n_cond >"   # py3.10: no backslash inside an f-string
    print("\n  " + f"{hdr:<22}" + "".join(f"{c:>10}" for c in (1, 2, 3, 4)))
    for m in (1, 2, 3, 4, 5, 6, 7):
        cells = []
        for c in (1, 2, 3, 4):
            s = (nm == m) & (nc == c)
            cells.append(f"{ys[s].mean():.2f}({int(s.sum())})"
                         if s.sum() >= 8 else "  -")
        print(f"  {m:<22}" + "".join(f"{x:>10}" for x in cells))
    print("\n  Each cell is the share matching a gold span. If precision rises")
    print("  along BOTH axes the signals compose; if it rises along only one,")
    print("  the other is redundant.")

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        subs.append((np.flatnonzero(~hm), np.flatnonzero(hm)))

    rules = {}
    for r in (2, 3, 4):
        rules[f"cond>={r}"] = (lambda e, r=r: e["n_cond"] >= r)
        rules[f"models>=2 AND cond>={r}"] = (
            lambda e, r=r: e["n_models"] >= 2 and e["n_cond"] >= r)
        rules[f"models>=2 OR cond>={r}"] = (
            lambda e, r=r: e["n_models"] >= 2 or e["n_cond"] >= r)
    for c in (3, 5, 8, 12):
        rules[f"cells>={c}"] = (lambda e, c=c: e["n_cells"] >= c)
    for w in (0.9, 1.2, 1.6):
        rules[f"w_cond>={w}"] = (lambda e, w=w: e["w_cond"] >= w)

    print("\n" + "=" * 76)
    print("2. CROSS-FITTED -- rule chosen on FIT authors, scored on HELD-OUT")
    print("=" * 76)
    built = {k: build(v) for k, v in rules.items()}
    print(f"  {'rule':<26}{'macro':>9}{'d':>9}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'sd':>9}")
    res = []
    for k, v in built.items():
        d = np.array([macro(v, hi) - macro(base, hi) for _, hi in subs])
        if np.mean(d > 0) >= 0.55 or d.mean() > 0:
            print(f"  {k:<26}{macro(v):>9.4f}{d.mean():>+9.4f}"
                  f"{0.3*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
                  f"{np.mean(d < 0):>6.0%}{0.3*d.std():>9.4f}")
        res.append((float(np.mean(d > 0)), float(d.mean()), k))
    print("  (only rules at >=55% win or a positive mean are listed)")

    # best rule chosen per split, the honest version
    ds, picks = [], []
    keys = list(built)
    for fi, hi in subs:
        bk = max(keys, key=lambda k: macro(built[k], fi))
        ds.append(macro(built[bk], hi) - macro(base, hi))
        picks.append(bk)
    d = np.array(ds)
    print(f"\n  nested rule selection: d {d.mean():+.4f}   "
          f"comp {0.3*d.mean():+.4f}   win {np.mean(d > 0):.0%}   "
          f"LOSS {np.mean(d < 0):.0%}")
    print("  chosen: " + ", ".join(f"{k} x{v}"
                                   for k, v in Counter(picks).most_common(4)))

    if res:
        res.sort(reverse=True)
        w, dm, k = res[0]
        v = built[k]
        ok = np.flatnonzero(pred_risk == gold_risk)
        bad = np.flatnonzero(pred_risk != gold_risk)
        print("\n" + "=" * 76)
        print(f"3. WHERE '{k}' ACTS")
        print("=" * 76)
        for nm_, sub in (("all", None), ("risk RIGHT", ok),
                         ("risk WRONG", bad)):
            print(f"  {nm_:<14}{macro(base, sub):>10.4f}{macro(v, sub):>10.4f}"
                  f"{macro(v, sub)-macro(base, sub):>+9.4f}")
        print("\n  The mechanism predicts the gain concentrates on risk-WRONG")
        print("  posts. If it does not, the rule works for another reason and")
        print("  the explanation is wrong even if the number is real.")
        print(f"\n  BEST: {k}  {0.3*dm:+.4f} composite at {w:.0%} win")
        print(f"  {'CLEARS' if w >= 0.90 and dm > 0 else 'does NOT clear'} "
              f"the bar.  Divide by ~2.8 for val->test attenuation.")


if __name__ == "__main__":
    main()

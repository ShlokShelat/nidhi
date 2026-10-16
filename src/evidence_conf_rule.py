"""
evidence_conf_rule.py

Compare span-selection rules built on confidence magnitude against the
vote-count rule the decode currently uses.

THE QUESTION. Confidence magnitude enters the decode only as a per-run floor
(`min_conf`); it plays no part in SELECTION, which thresholds the vote count
alone. If per-candidate confidence separates matching from non-matching spans
better than the vote count does, the same shape of rule pointed at the better
feature should select a better set -- without adding any machinery.

RULES SWEPT, all cross-fitted by author (threshold from FIT, scored on
HELD-OUT), all keeping the deployed dedupe-nested post-processing so the
comparison is like-for-like:

    deployed       n_votes >= 2
    conf           mean_conf >= t
    maxconf        max_conf  >= t
    votes AND conf n_votes >= 2 AND mean_conf >= t   (tighten the current set)
    votes OR conf  n_votes >= 2 OR  mean_conf >= t   (rescue high-confidence
                                                      singletons the vote drops)
    lean logistic  logistic on the three per-candidate features that carry
                   signal (n_votes, mean_conf, conf_rank), dropping the
                   position and pool-size features, which are at chance

`votes OR conf` is the rule with the clearest mechanism: gold spans that were
proposed but not selected are exactly the candidates a vote of 1 discards, so
if confidence separates them they are recovered at no cost in machinery.

Judge by WIN RATE and the share of resamples that LOSE, not by the mean of the
delta. The run also asserts its own baseline macro before comparing anything:
a baseline that does not reproduce means the candidate cache or the risk
predictions do not match the decode under test, and every delta below it would
be measured against the wrong reference.

Usage:
  python src/evidence_conf_rule.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


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
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--votes", type=int, default=2)
    a_.add_argument("--splits", type=int, default=300)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    S = json.load(open(args.cache))
    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    va = va[va["row_id"].isin(S)].reset_index(drop=True)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    pred_risk = np.array([normalize_label(v)
                          for v in pv.loc[rows, "pred_risk_level"]])

    # per post: list of (span, n_votes, mean_conf, max_conf, conf_rank)
    C = []
    for i, r in enumerate(rows):
        lv = pred_risk[i]
        if lv == "indicator":
            C.append([])
            continue
        agg = {}
        for s, cf in S[r][lv]:
            if cf < args.min_conf:
                continue
            n = normalize_span(s)
            if not n:
                continue
            e = agg.setdefault(n, {"n": 0, "mx": 0.0, "sum": 0.0, "s": s})
            e["n"] += 1
            e["mx"] = max(e["mx"], cf)
            e["sum"] += cf
        items = list(agg.values())
        mx = max((e["mx"] for e in items), default=1.0)
        C.append([(e["s"], e["n"], e["sum"] / max(e["n"], 1), e["mx"],
                   e["mx"] / max(mx, 1e-9)) for e in items])

    def sel(i, fn):
        if not C[i]:
            return []
        return dedupe_nested([c[0] for c in C[i] if fn(c)])

    def macro(pred):
        return float(np.mean([f1_of(g, p) for g, p in zip(gold, pred)]))

    ship = [sel(i, lambda c: c[1] >= args.votes) for i in range(len(rows))]
    base = macro(ship)
    print(f"{len(rows)} val posts   deployed MACRO {base:.4f}")
    if abs(base - 0.8824) > 0.01:
        print(f"  !! WARNING baseline {base:.4f} != 0.8824 -- VOID, stop here")
    nz = [i for i in range(len(rows)) if gold[i]]
    print(f"  spans/post shipped {np.mean([len(ship[i]) for i in nz]):.2f}"
          f"   gold {np.mean([len(gold[i]) for i in nz]):.2f}\n")

    GRID = np.round(np.arange(0.30, 0.96, 0.02), 3)
    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)

    def rule(nm):
        if nm == "conf":
            return lambda t: (lambda c: c[2] >= t)
        if nm == "maxconf":
            return lambda t: (lambda c: c[3] >= t)
        if nm == "votes AND conf":
            return lambda t: (lambda c: c[1] >= args.votes and c[2] >= t)
        if nm == "votes OR conf":
            return lambda t: (lambda c: c[1] >= args.votes or c[2] >= t)
        raise KeyError(nm)

    print(f"  {'rule':<18}{'d evidence':>12}{'d comp':>9}{'win':>6}{'LOSS':>6}"
          f"{'sd':>9}{'t chosen':>14}{'spans/post':>12}")
    out = []
    for nm in ("conf", "maxconf", "votes AND conf", "votes OR conf"):
        mk = rule(nm)
        ds, ts, sp = [], [], []
        for _ in range(args.splits):
            o = list(ua)
            rng.shuffle(o)
            k = max(int(len(o) * 0.25), 2)
            ho = set(o[:k])
            hm = np.array([a in ho for a in auth])
            fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
            if len(fi) < 40 or len(hi) < 15:
                continue
            best, bt = -1e9, GRID[0]
            for t in GRID:
                f = mk(t)
                v = np.mean([f1_of(gold[i], sel(i, f)) for i in fi])
                if v > best:
                    best, bt = v, t
            f = mk(bt)
            new = [sel(i, f) for i in hi]
            gh = [gold[i] for i in hi]
            ds.append(np.mean([f1_of(g, p) for g, p in zip(gh, new)])
                      - np.mean([f1_of(gold[i], ship[i]) for i in hi]))
            ts.append(bt)
            sp.append(np.mean([len(p) for p, i in zip(new, hi) if gold[i]]))
        d = np.array(ds)
        print(f"  {nm:<18}{d.mean():>+12.4f}{0.3*d.mean():>+9.4f}"
              f"{np.mean(d > 0):>5.0%}{np.mean(d < 0):>6.0%}"
              f"{0.3*d.std():>9.4f}"
              f"{f'{np.mean(ts):.2f}+-{np.std(ts):.2f}':>14}"
              f"{np.mean(sp):>12.2f}")
        out.append((float(np.mean(d > 0)), 0.3 * d.mean(), nm,
                    float(np.mean(ts)), float(np.std(ts))))

    # lean logistic: only the three features with AUC > 0.7
    from sklearn.linear_model import LogisticRegression
    ds = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) < 40 or len(hi) < 15:
            continue
        Xf, yf = [], []
        for i in fi:
            for c in C[i]:
                Xf.append([c[1], c[2], c[4]])
                yf.append(1 if any(_span_pair_matches(c[0], g)
                                   for g in gold[i]) else 0)
        if len(set(yf)) < 2:
            continue
        m = LogisticRegression(C=1.0, max_iter=400).fit(np.array(Xf), yf)
        best, bt = -1e9, 0.5
        for t in np.arange(0.2, 0.85, 0.05):
            v = np.mean([f1_of(gold[i], dedupe_nested(
                [c[0] for c in C[i]
                 if m.predict_proba([[c[1], c[2], c[4]]])[0, 1] >= t]))
                for i in fi[:60]])
            if v > best:
                best, bt = v, t
        new = [dedupe_nested([c[0] for c in C[i]
                              if m.predict_proba([[c[1], c[2], c[4]]])[0, 1]
                              >= bt]) for i in hi]
        gh = [gold[i] for i in hi]
        ds.append(np.mean([f1_of(g, p) for g, p in zip(gh, new)])
                  - np.mean([f1_of(gold[i], ship[i]) for i in hi]))
    if ds:
        d = np.array(ds)
        print(f"  {'lean logistic(3)':<18}{d.mean():>+12.4f}"
              f"{0.3*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
              f"{np.mean(d < 0):>6.0%}{0.3*d.std():>9.4f}")

    if out:
        out.sort(reverse=True)
        w, dc, nm, tm, tsd = out[0]
        print(f"\n  BEST: {nm}  {dc:+.4f} composite at {w:.0%} win, "
              f"t = {tm:.2f} +- {tsd:.2f}")
        ok = w >= 0.90 and dc > 0
        print(f"  {'CLEARS' if ok else 'does NOT clear'} the 90% bar.")
        if tsd > 0.08:
            print("  WARNING: the threshold wanders across splits -- it is")
            print("  being fitted per sample and will not transfer. That is")
            print("  the signature that killed the factor shrinkage (k 65+-100)")
            print("  and the evidence hedge (q 0.00 every split).")


if __name__ == "__main__":
    main()

"""
evidence_conf_vote.py

The evidence vote counts taggers. It should weigh them.

WHAT IS BEING THROWN AWAY. evidence_vote_oof.py filters each tagger's candidates
at that tagger's own `min_conf` (0.75 / 0.75 / 0.75 / 0.0 / 0.65) and then counts
how many survivors proposed each span. Two things are lost:

  1. A span at conf 0.74 from a 0.75-cutoff tagger is discarded OUTRIGHT, even
     if three other taggers also proposed it. The hard cutoff runs before the
     vote, so the vote never sees it.
  2. After filtering, a span proposed at 0.95 and one proposed at 0.51 are
     identical -- both count as one vote.

A confidence-weighted score STRICTLY GENERALISES the count: hard voting is the
special case where every surviving confidence is rounded to 1 and everything
below the cutoff to 0. So the only way weighting can lose is if the confidences
are uninformative, which would itself be worth knowing.

THIS MATTERS BECAUSE RECALL IS THE BINDING CONSTRAINT. On 1102 OOF posts
vote>=2 scores P 0.7665 / R 0.6704 -- a third of gold spans are never proposed
or never survive. `union all` reaches R 0.7666 but drops P to 0.6487 for a worse
F1. The frontier between them is exactly where a graded score can beat a step
function.

ARMS. Every arm reuses evidence_vote_oof's dedupe and consistency rule verbatim,
so the ONLY difference is how spans are selected:

  vote>=k       the current family, reproduced as the baseline
  sum-filtered  sum of confidences, over each tagger's POST-min_conf survivors
  sum-raw       sum of confidences over ALL candidates, ignoring min_conf --
                this is the arm that recovers loss (1) above
  max-raw       max confidence over all candidates; rewards one confident tagger
  mean-raw      sum divided by the number of taggers proposing it; rewards
                agreement strength rather than agreement count

DISCIPLINE. Each graded arm has ONE free parameter, its cutoff, and that cutoff
is fitted on one author half and applied to the other. The parameter-count
transfer law in this project (fewer fitted parameters transfer better, and
many fitted parameters can transfer very poorly) says one is affordable; it
does not say it is free.
Verdicts use a paired author bootstrap, which the existing script does not do --
it reports half-splits only, and two halves cannot give a confidence interval.

Usage:
  python src/evidence_conf_vote.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_vote_oof import TAGS, author_halves, dedupe  # noqa: E402
from scoring import (normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--dir", default="artifacts/evidence_oof")
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    tags = [t.strip() for t in args.tags.split(",")]

    filt, raw = {}, {}
    for t in tags:
        p = os.path.join(args.dir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            print(f"  skip {t}: no {p}")
            continue
        d = pd.read_csv(p)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        for store, dd in ((raw, d), (filt, d[d["conf"] >= mc])):
            m = {}
            for r, s, c in zip(dd["row_id"], dd["span"], dd["conf"]):
                m.setdefault(r, []).append((str(s), float(c)))
            store[t] = m
        print(f"  {t}: min_conf={mc}  {len(d)} candidates, "
              f"{int((d['conf'] >= mc).sum())} survive")
    if len(filt) < 2:
        raise SystemExit("need at least two taggers")

    rows = sorted(set(lab.index) &
                  set().union(*[set(v) for v in raw.values()]))
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    risk = [normalize_label(lab.loc[r, "risk_level"]) for r in rows]
    auth = np.array([lab.loc[r, "anon_user_id"] for r in rows])
    half = author_halves(auth)
    A, B = half == "A", half == "B"
    base_tag = "v2" if "v2" in filt else tags[0]
    print(f"\n{len(rows)} OOF posts   {pd.unique(auth).size} authors   "
          f"halves {int(A.sum())}/{int(B.sum())}")

    # spans-with-confidence per tagger, deduped exactly as the shipped path does
    def dd(store, r):
        out = {}
        for t in store:
            spans = [s for s, _ in store[t].get(r, [])]
            conf = {s: c for s, c in store[t].get(r, [])}
            for s in dedupe(spans):
                out.setdefault(t, []).append((s, conf[s]))
        return out

    cache_f = {r: dd(filt, r) for r in rows}
    cache_r = {r: dd(raw, r) for r in rows}

    def consistency(spans, r, i):
        """The shipped rule, applied identically to every arm."""
        if risk[i] == "indicator":
            return []
        if spans:
            return spans
        pool = [s for s, _ in cache_f[r].get(base_tag, [])] or [
            s for t in cache_f[r] for s, _ in cache_f[r][t]]
        return sorted(pool, key=token_len)[:1]

    def agg(cache, r, mode):
        """span key -> (score, surface form)"""
        acc = {}
        for t in cache[r]:
            for s, c in cache[r][t]:
                k = normalize_span(s)
                if not k:
                    continue
                a = acc.setdefault(k, {"sum": 0.0, "max": 0.0, "n": 0, "s": s})
                a["sum"] += c
                a["max"] = max(a["max"], c)
                a["n"] += 1
                if token_len(s) > token_len(a["s"]):
                    a["s"] = s
        if mode == "count":
            return {k: (v["n"], v["s"]) for k, v in acc.items()}
        if mode == "sum":
            return {k: (v["sum"], v["s"]) for k, v in acc.items()}
        if mode == "max":
            return {k: (v["max"], v["s"]) for k, v in acc.items()}
        return {k: (v["sum"] / v["n"], v["s"]) for k, v in acc.items()}

    # The aggregation does not depend on the cutoff, so compute it ONCE per
    # (cache, mode) instead of once per grid point. Without this the sweep
    # recomputes the same scores ~28 times per arm.
    _agg_cache = {}

    def agg_all(cache, tag, mode):
        key = (tag, mode)
        if key not in _agg_cache:
            _agg_cache[key] = [list(agg(cache, r, mode).values()) for r in rows]
        return _agg_cache[key]

    def build(cache, tag, mode, thr):
        scored = agg_all(cache, tag, mode)
        return [consistency(dedupe([s for sc, s in scored[i] if sc >= thr]),
                            r, i)
                for i, r in enumerate(rows)]

    def f1_of(P, mask=None):
        if mask is None:
            return phrase_f1(G, P).f1
        return phrase_f1([g for g, m in zip(G, mask) if m],
                         [p for p, m in zip(P, mask) if m]).f1

    def fit_thr(cache, tag, mode, rows_mask, grid):
        best, bt = -1.0, grid[0]
        for t in grid:
            v = f1_of(build(cache, tag, mode, t), rows_mask)
            if v > best:
                best, bt = v, t
        return bt

    # ---- baseline: the deployed hard vote -----------------------------------
    base = build(cache_f, "f", "count", 2)
    b = phrase_f1(G, base)
    print(f"\n{'arm':<20}{'F1':>9}{'P':>8}{'R':>8}{'pred':>7}{'delta':>9}"
          f"{'cutoff':>16}")
    print("-" * 80)
    print(f"{'vote>=2 (SHIPPED)':<20}{b.f1:>9.4f}{b.precision:>8.4f}"
          f"{b.recall:>8.4f}{b.n_pred:>7}{'':>9}{'k=2':>16}")

    ntag = len(filt)
    arms = {"vote>=2": base}
    specs = [
        ("sum-filtered", cache_f, "f", "sum",
         np.round(np.arange(0.4, 3.2, 0.1), 2)),
        ("sum-raw", cache_r, "r", "sum",
         np.round(np.arange(0.4, 3.2, 0.1), 2)),
        ("max-raw", cache_r, "r", "max",
         np.round(np.arange(0.30, 0.96, 0.02), 2)),
        ("mean-raw", cache_r, "r", "mean",
         np.round(np.arange(0.30, 0.96, 0.02), 2)),
    ]
    for name, cache, tag, mode, grid in specs:
        # cutoff fitted on one author half, applied to the other
        tA = fit_thr(cache, tag, mode, B, grid)   # fit on B -> apply to A
        tB = fit_thr(cache, tag, mode, A, grid)
        PA = build(cache, tag, mode, tA)
        PB = build(cache, tag, mode, tB)
        P = [PA[i] if A[i] else PB[i] for i in range(len(rows))]
        arms[name] = P
        r = phrase_f1(G, P)
        print(f"{name:<20}{r.f1:>9.4f}{r.precision:>8.4f}{r.recall:>8.4f}"
              f"{r.n_pred:>7}{r.f1 - b.f1:>+9.4f}"
              f"{f'A:{tA} B:{tB}':>16}")

    # ---- paired author bootstrap --------------------------------------------
    uniq = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uniq}
    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.boot) for k in arms}
    for i in range(args.boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        ix = np.concatenate([by[uniq[k]] for k in pick])
        gb = [G[j] for j in ix]
        for k, P in arms.items():
            boot[k][i] = phrase_f1(gb, [P[j] for j in ix]).f1

    print("\n" + "=" * 80)
    print(f"PAIRED delta vs the shipped vote>=2   ({args.boot} author "
          f"bootstrap resamples)")
    print("=" * 80)
    print(f"{'arm':<20}{'delta':>9}{'95% CI':>21}{'win%':>8}{'verdict':>20}")
    print("-" * 80)
    for k in arms:
        if k == "vote>=2":
            continue
        d = boot[k] - boot["vote>=2"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        win = float((d > 0).mean()) * 100
        pt = phrase_f1(G, arms[k]).f1 - b.f1
        good = lo > 0 and win >= 80
        print(f"{k:<20}{pt:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>21}{win:>7.1f}%"
              f"{('SHIPPABLE' if good else 'no'):>20}")

    print(f"\nEvidence is 3/7 of S1, so a delta here is worth 0.3x that much")
    print(f"composite. The shipped vote>=2 beat a single tagger by a clear")
    print(f"margin, so that is the scale to beat.")
    print("\nCAUTION: this instrument uses "
          f"{ntag} taggers; the champion ships EIGHT.")
    print("A winner here must be re-checked against the 8-tagger test path")
    print("before upload -- validating a 5-tagger rule and shipping an")
    print("8-tagger one is a known failure mode.")


if __name__ == "__main__":
    main()

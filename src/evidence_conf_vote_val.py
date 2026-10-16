"""
evidence_conf_vote_val.py

Test the confidence-weighted vote on the EIGHT TAGGERS THAT ACTUALLY SHIP.

WHY THIS EXISTS RATHER THAN REUSING THE OOF RESULT. src/evidence_conf_vote.py
found `sum-raw` outperforming the baseline -- but it ran on the OOF taggers
`v2, ep12, lr3, bs16, ml256`, while the champion votes over `v2, v1, s14, s15,
s16, s17, lr2, ep9`. **Only `v2` is in both sets.** That is a worse mismatch
than the risk-bag case where at least the architecture was shared. A number
measured on five other models is not a prediction about these eight.

THE HONEST VALIDATION SET. The deployed eight trained on `train_split`, so their
`*_val_candidates.csv` predictions on the 372 `val_split` posts are genuinely
held out FROM THEM. Small and noisy, but it is the exact eight-model object that
would be uploaded, which is the property that has mattered every time.

THE MECHANISM BEING TESTED. Each tagger's candidates are cut at its own
`evidence_min_conf` BEFORE the vote counts them, so a span at 0.64 from a
0.65-cutoff tagger is invisible to the vote even if four other taggers proposed
it. Every saved candidate sits at conf >= ~0.50 on both the val and test dumps,
so the rescued band is real and identical on both sides. Weighting also lets a
0.95 outweigh a 0.51, which counting cannot.

Arms:
  vote>=k       the shipped rule, reproduced exactly from evidence_vote_test.py
  sum-filtered  sum of confidences over each tagger's post-min_conf survivors
  sum-raw       sum of confidences over ALL candidates, ignoring min_conf
  max-raw       max confidence over all candidates

The cutoff for each graded arm is fitted on one author half and applied to the
other. One free parameter; this project's record says one transfers at ~17%.

ONE COMPROMISE, STATED. The consistency rule needs a risk label, and on val this
script uses the GOLD label rather than a predicted one, so the rule fires more
accurately here than it will at test time. It is applied identically to every
arm, so it inflates the absolute level but not the comparison. Read deltas.

Usage:
  python src/evidence_conf_vote_val.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_vote_oof import author_halves, dedupe  # noqa: E402
from scoring import (normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)

# the eight from scripts/evidence_more_models.sh -- NOT evidence_vote_test.py's
# four-run default, which silently built a 4-model vote once before
SHIPPED = ["evidence_v2", "evidence_v1", "evidence_s14", "evidence_s15",
           "evidence_s16", "evidence_s17", "evidence_lr2", "evidence_ep9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--runs", default=",".join(SHIPPED))
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    rows = va["row_id"].tolist()
    G = [parse_evidence(x) for x in va["evidence"]]
    risk = {r: normalize_label(x)
            for r, x in zip(va["row_id"], va["risk_level"])}
    auth = va["anon_user_id"].to_numpy()
    half = author_halves(auth)
    A, B = half == "A", half == "B"

    asked = [r.strip() for r in args.runs.split(",")]
    filt, raw = {}, {}
    for run in asked:
        cp = os.path.join(args.dir, f"{run}_val_candidates.csv")
        cfgp = os.path.join("runs", run, "run_config.json")
        if not os.path.exists(cp):
            print(f"  skip {run}: no {cp}")
            continue
        mc = float(json.load(open(cfgp)).get("evidence_min_conf", 0.0)) \
            if os.path.exists(cfgp) else 0.0
        d = pd.read_csv(cp)
        for store, dd in ((raw, d), (filt, d[d["conf"] >= mc])):
            m = {}
            for r, s, c in zip(dd["row_id"], dd["span"], dd["conf"]):
                m.setdefault(r, []).append((str(s), float(c)))
            store[run] = m
        print(f"  {run}: min_conf={mc}  {len(d)} candidates, "
              f"{int((d['conf'] >= mc).sum())} survive  "
              f"conf min {d['conf'].min():.3f}")

    lost = [r for r in asked if r not in filt]
    if lost:
        raise SystemExit(
            f"{len(lost)} of {len(asked)} shipped runs have no val "
            f"candidates: {lost}\nRefusing to validate a smaller ensemble "
            f"than the one that ships -- that is the exact failure this "
            f"script exists to avoid.")
    print(f"\n{len(rows)} val posts   {pd.unique(auth).size} authors   "
          f"halves {int(A.sum())}/{int(B.sum())}   {len(filt)} taggers")

    def deduped(store, r):
        out = {}
        for t in store:
            got = store[t].get(r, [])
            conf = {s: c for s, c in got}
            for s in dedupe([s for s, _ in got]):
                out.setdefault(t, []).append((s, conf[s]))
        return out

    cache_f = {r: deduped(filt, r) for r in rows}
    cache_r = {r: deduped(raw, r) for r in rows}

    def consistency(kept, r):
        """Verbatim from evidence_vote_test.py."""
        if risk[r] == "indicator":
            return []
        if kept:
            return kept
        pool = [s for t in cache_f[r] for s, _ in cache_f[r][t]]
        return [min(pool, key=token_len)] if pool else []

    def agg(cache, r, mode):
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
        key = {"count": "n", "sum": "sum", "max": "max"}[mode]
        return [(v[key], v["s"]) for v in acc.values()]

    _cache = {}

    def scored(cache, tag, mode):
        if (tag, mode) not in _cache:
            _cache[(tag, mode)] = [agg(cache, r, mode) for r in rows]
        return _cache[(tag, mode)]

    def build(cache, tag, mode, thr):
        sc = scored(cache, tag, mode)
        return [consistency(dedupe([s for v, s in sc[i] if v >= thr]), r)
                for i, r in enumerate(rows)]

    def f1_of(P, mask=None):
        if mask is None:
            return phrase_f1(G, P).f1
        return phrase_f1([g for g, m in zip(G, mask) if m],
                         [p for p, m in zip(P, mask) if m]).f1

    base = build(cache_f, "f", "count", args.k)
    b = phrase_f1(G, base)
    print(f"\n{'arm':<16}{'F1':>9}{'P':>8}{'R':>8}{'pred':>7}{'delta':>9}"
          f"{'cutoff':>16}")
    print("-" * 74)
    print(f"{f'vote>={args.k} SHIPPED':<16}{b.f1:>9.4f}{b.precision:>8.4f}"
          f"{b.recall:>8.4f}{b.n_pred:>7}{'':>9}{f'k={args.k}':>16}")

    arms = {"shipped": base}
    nt = len(filt)
    specs = [
        ("sum-filtered", cache_f, "f", "sum",
         np.round(np.arange(0.5, min(5.0, nt) + 0.01, 0.1), 2)),
        ("sum-raw", cache_r, "r", "sum",
         np.round(np.arange(0.5, min(5.0, nt) + 0.01, 0.1), 2)),
        ("max-raw", cache_r, "r", "max",
         np.round(np.arange(0.50, 0.99, 0.02), 2)),
    ]
    for name, cache, tag, mode, grid in specs:
        tA = max(grid, key=lambda t: f1_of(build(cache, tag, mode, t), B))
        tB = max(grid, key=lambda t: f1_of(build(cache, tag, mode, t), A))
        PA, PB = build(cache, tag, mode, tA), build(cache, tag, mode, tB)
        P = [PA[i] if A[i] else PB[i] for i in range(len(rows))]
        arms[name] = P
        r = phrase_f1(G, P)
        print(f"{name:<16}{r.f1:>9.4f}{r.precision:>8.4f}{r.recall:>8.4f}"
              f"{r.n_pred:>7}{r.f1 - b.f1:>+9.4f}{f'A:{tA} B:{tB}':>16}")

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

    print("\n" + "=" * 74)
    print(f"PAIRED delta vs the shipped vote   ({args.boot} author bootstrap)")
    print("=" * 74)
    print(f"{'arm':<16}{'delta':>9}{'95% CI':>21}{'win%':>8}{'verdict':>18}")
    print("-" * 74)
    winner = None
    for k in arms:
        if k == "shipped":
            continue
        d = boot[k] - boot["shipped"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        win = float((d > 0).mean()) * 100
        pt = phrase_f1(G, arms[k]).f1 - b.f1
        good = lo > 0 and win >= 80
        if good and (winner is None or pt > winner[1]):
            winner = (k, pt)
        print(f"{k:<16}{pt:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>21}{win:>7.1f}%"
              f"{('SHIPPABLE' if good else 'no'):>18}")

    print(f"\nOnly {len(rows)} posts and {pd.unique(auth).size} authors, so this")
    print("instrument is the noisy one -- but it is measuring the EXACT eight")
    print("taggers that ship, which the 1102-post OOF version was not.")
    if winner:
        print(f"\n>>> {winner[0]} wins by {winner[1]:+.4f} evidence F1 "
              f"= {0.3 * winner[1]:+.4f} composite.")
        print("    Build the test column with the same cutoff refitted on ALL")
        print("    372 val posts, then patch evidence onto the champion.")
    else:
        print("\nNothing clears the bar on the shipped eight. The earlier OOF")
        print("gain belonged to those five other taggers, not this ensemble.")
        print("Move to the threshold `shrink` arm, which transfers zero")
        print("fitted parameters.")


if __name__ == "__main__":
    main()

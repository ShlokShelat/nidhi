"""
evidence_member_loo.py

Leave-one-out over the evidence tagger pool: does every member earn its place?

The deployed decode votes k>=2 over eight `evidence_*_rc` runs.

Membership and threshold are entangled, because `k` is a FRACTION of the pool.
At 8 members k=2 is 25% agreement; at 7 it is 29%. So dropping a member
tightens the vote even if the member itself was neutral, and the two effects
have to be separated -- this reports both fixed k and proportionally-scaled k.

The mechanism a membership change can reach, which no fixed-membership
selection rule can: part of the unmatched gold is proposed by exactly one
tagger and discarded by the vote. If one member is a noise source, removing it
lets k=2 fire on genuine agreement rather than on agreement-with-noise.

Also reports, per member:
  * how many candidates it uniquely contributes (nobody else proposes them)
  * what fraction of those unique candidates match a gold span
A member with many unique candidates and poor precision on them is exactly what
a k>=2 vote cannot filter, because its proposals never reach two votes anyway --
it only adds noise to the cluster step.

Cross-fitted by author for the deltas; the raw table is full-sample.

Usage:
  python src/evidence_member_loo.py
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

RC8 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc", "evidence_v2_rc"]


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
    a_.add_argument("--runs", default=",".join(RC8))
    a_.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    a_.add_argument("--k", type=int, default=2)
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

    runs = [x.strip() for x in args.runs.split(",") if x.strip()]
    per = {}
    for run in runs:
        p = os.path.join(args.dir, f"{run}_val_candidates.csv")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        if "conf" in d.columns:
            d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"], d["span"]):
            m.setdefault(r, set()).add(str(s))
        per[run] = m
    if len(per) < 3:
        raise SystemExit("need at least 3 runs with val candidate dumps")
    print(f"{len(rows)} val posts   {len(per)} taggers with val dumps\n")

    def vote(members, k):
        out = []
        for r in rows:
            if supp.get(r, False):
                out.append([])
                continue
            cnt = {}
            for run in members:
                for s in per[run].get(r, ()):
                    n = normalize_span(s)
                    if n:
                        e = cnt.setdefault(n, [0, s])
                        e[0] += 1
            out.append(dedupe_nested([v[1] for v in cnt.values()
                                      if v[0] >= k]))
        return out

    def macro(pred, idx=None):
        idx = range(len(rows)) if idx is None else idx
        return float(np.mean([f1_of(gold[i], pred[i]) for i in idx]))

    full = vote(list(per), args.k)
    base = macro(full)
    print(f"  all {len(per)} taggers, k={args.k}:  MACRO {base:.4f}"
          f"   spans/post "
          f"{np.mean([len(full[i]) for i in range(len(rows)) if gold[i]]):.2f}")

    print("\n" + "=" * 78)
    print("1. UNIQUE CONTRIBUTION PER MEMBER")
    print("=" * 78)
    print(f"  {'tagger':<22}{'cands':>8}{'unique':>8}{'uniq hit':>10}"
          f"{'all hit':>9}")
    for run in per:
        tot = uniq = uhit = hit = 0
        for i, r in enumerate(rows):
            if supp.get(r, False) or not gold[i]:
                continue
            mine = per[run].get(r, set())
            others = set()
            for o in per:
                if o != run:
                    others |= {normalize_span(x) for x in per[o].get(r, ())}
            for s in mine:
                tot += 1
                ok = any(_span_pair_matches(s, g) for g in gold[i])
                hit += ok
                if normalize_span(s) not in others:
                    uniq += 1
                    uhit += ok
        print(f"  {run:<22}{tot:>8}{uniq:>8}"
              f"{(uhit/max(uniq,1)):>10.1%}{(hit/max(tot,1)):>9.1%}")
    print("\n  `unique` = candidates nobody else proposes. A k>=2 vote can")
    print("  never select them, so a member with many low-precision uniques is")
    print("  contributing noise to the cluster step and nothing else.")

    print("\n" + "=" * 78)
    print("2. LEAVE ONE OUT -- cross-fitted by author")
    print("=" * 78)
    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        n = max(int(len(o) * 0.25), 2)
        ho = set(o[:n])
        subs.append(np.flatnonzero([a in ho for a in auth]))
    print(f"  {'drop':<22}{'k':>3}{'MACRO':>9}{'d':>9}{'d comp':>9}"
          f"{'win':>6}{'LOSS':>6}{'spans':>8}")
    out = []
    for run in list(per):
        keep = [x for x in per if x != run]
        for k in sorted({args.k, max(2, round(args.k * len(keep) / len(per)))}):
            v = vote(keep, k)
            d = np.array([macro(v, s) - macro(full, s) for s in subs])
            sp = np.mean([len(v[i]) for i in range(len(rows)) if gold[i]])
            print(f"  {run:<22}{k:>3}{macro(v):>9.4f}{d.mean():>+9.4f}"
                  f"{0.3*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
                  f"{np.mean(d < 0):>6.0%}{sp:>8.2f}")
            out.append((float(np.mean(d > 0)), 0.3 * d.mean(), run, k))

    print("\n" + "=" * 78)
    print("3. VOTE LEVEL ON THE FULL POOL")
    print("=" * 78)
    print(f"  {'k':>3}{'MACRO':>9}{'d':>9}{'spans/post':>12}")
    for k in range(1, len(per) + 1):
        v = vote(list(per), k)
        print(f"  {k:>3}{macro(v):>9.4f}{macro(v)-base:>+9.4f}"
              f"{np.mean([len(v[i]) for i in range(len(rows)) if gold[i]]):>12.2f}")

    # ------------------------------------------------------------------ 4
    # NESTED MEMBERSHIP SELECTION. Dropping ONE member is a single decision and
    # can be read off the table above. Dropping the BEST SET is a search over
    # 2^8 subsets on a few hundred posts, which overfits if the subset is
    # chosen on the same authors it is scored on: the search will find whatever
    # membership happens to suit that sample. So the membership is selected
    # greedily on FIT authors and scored on HELD-OUT authors,
    # every split. The reported delta is therefore what the SELECTION PROCEDURE
    # buys, not what the best-in-hindsight subset scores.
    print("\n" + "=" * 78)
    print("4. NESTED GREEDY ELIMINATION -- membership chosen on FIT authors,")
    print("   scored on HELD-OUT authors")
    print("=" * 78)
    names = list(per)
    cache = {}

    def vote_cached(members, k):
        key = (tuple(sorted(members)), k)
        if key not in cache:
            cache[key] = vote(list(members), k)
        return cache[key]

    ds, sizes, dropped = [], [], []
    for s in subs:
        fit = np.setdiff1d(np.arange(len(rows)), s)
        cur = list(names)
        cur_v = macro(vote_cached(cur, args.k), fit)
        while len(cur) > 4:
            best, bm = cur_v, None
            for m in cur:
                cand = [x for x in cur if x != m]
                v = macro(vote_cached(cand, args.k), fit)
                if v > best:
                    best, bm = v, m
            if bm is None:
                break
            cur = [x for x in cur if x != bm]
            cur_v = best
            dropped.append(bm)
        ds.append(macro(vote_cached(cur, args.k), s)
                  - macro(full, s))
        sizes.append(len(cur))
    d = np.array(ds)
    print(f"  d evidence {d.mean():+.4f}   d comp {0.3*d.mean():+.4f}"
          f"   win {np.mean(d > 0):.0%}   LOSS {np.mean(d < 0):.0%}"
          f"   sd {0.3*d.std():.4f}")
    print(f"  members kept {np.mean(sizes):.1f} +- {np.std(sizes):.1f} of "
          f"{len(names)}")
    from collections import Counter as _C
    freq = _C(dropped)
    print("  dropped how often across splits: "
          + ", ".join(f"{k.replace('evidence_','')} {v}"
                      for k, v in freq.most_common(5)))
    print("\n  A member dropped in almost EVERY split is a stable finding; one")
    print("  dropped in half is the search fitting that sample. Compare this")
    print("  delta with the single-drop table -- if the greedy search does not")
    print("  beat the best single drop, the extra freedom is only overfitting.")

    if out:
        out.sort(reverse=True)
        w, dc, run, k = out[0]
        print(f"\n  BEST DROP: {run} at k={k}  {dc:+.4f} composite, "
              f"{w:.0%} win")
        print(f"  {'CLEARS' if w >= 0.90 and dc > 0 else 'does NOT clear'} "
              f"the 90% bar.")
        print("  Note: dropping a member also tightens the vote FRACTION, so")
        print("  compare against the k sweep in section 3 before crediting the")
        print("  member -- a gain that k alone reproduces is not about the")
        print("  member at all.")


if __name__ == "__main__":
    main()

"""
evidence_arms_honest.py

RE-RANK EVERY EVIDENCE ARM UNDER THE METRIC AND THE REGIME ACTUALLY DEPLOYED.

Every evidence arm in this project was chosen under TWO defects at once:

  1. MICRO averaging. evidence_vote_oof.py:76 defaults to micro. Its own
     docstring states the official scorer is MACRO ("an all-empty column
     scores ~0.357 ... every evidence decision in this project was tuned
     against micro").

  2. GOLD suppression. evidence_vote_oof.py:114 builds `risk` from
     lab["risk_level"] -- the GOLD label -- and line 121 suppresses on it.
     Measured: evidence is 0.8086 under gold suppression and 0.7605 under
     the shipped prediction -- a real leak, in a regime that never deploys.

So the modal-vs-short-vs-capsafe decision, the k threshold, and the
cluster-vs-exact choice were all made on the wrong metric in a regime where
indicator classification is perfect. This re-runs the whole comparison with
macro averaging and PREDICTED-risk suppression.

WHY `short` IS THE SPECIFIC SUSPECT. The scorer's rule
(`_span_pair_matches`, scoring.py:141-149) is asymmetric:

    a prediction CONTAINED IN gold passes the 3x cap automatically
    a prediction CONTAINING gold fails once it exceeds 3x gold's length

so erring short is one-sided-safe. A FORCE-rule ablation measured the
consequence directly: the shortest available candidate, fired blind on 109
posts where the tagger had gone silent, averages ~0.79 F1 -- i.e. it usually
matches outright. Removing the force rule measurably costs accuracy at a low
win rate.

If short spans match that readily when guessed, `rep=short` may beat `modal`
once the metric is macro and the leak is gone. That is a 0-parameter change to
an existing switch.

JUDGED BY PAIRED AUTHOR BOOTSTRAP against the currently shipped arm, because
the absolute level moves with the sample and only paired deltas survive that.

Usage (cluster, CPU, a few minutes):
  python src/evidence_arms_honest.py
  python src/evidence_arms_honest.py --ship "cluster>=2 modal"
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_span_pair_matches, normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)

TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]


def dedupe(spans):
    out = []
    for s in sorted(spans, key=lambda x: (-token_len(x),
                                          -len(normalize_span(x)))):
        n = normalize_span(s)
        if not n or any(n in normalize_span(k) for k in out):
            continue
        out.append(s)
    return out


def cluster_votes(items, k, rep):
    items = sorted(items, key=lambda x: (-token_len(x[1]),
                                         normalize_span(x[1])))
    groups = []
    for t, s in items:
        for g in groups:
            if any(_span_pair_matches(s, s2) or _span_pair_matches(s2, s)
                   for _, s2 in g):
                g.append((t, s))
                break
        else:
            groups.append([(t, s)])
    out = []
    for g in groups:
        if len({t for t, _ in g}) < k:
            continue
        if rep == "short":
            out.append(min((s for _, s in g), key=token_len))
        elif rep == "long":
            out.append(max((s for _, s in g), key=token_len))
        elif rep == "median":
            ss = sorted((s for _, s in g), key=token_len)
            out.append(ss[len(ss) // 2])
        elif rep == "capsafe":
            cnt = {}
            for _, s in g:
                cnt[s] = cnt.get(s, 0) + 1
            modal = max(cnt.items(),
                        key=lambda kv: (kv[1], -token_len(kv[0])))[0]
            shortest = min((s for _, s in g), key=token_len)
            out.append(shortest
                       if token_len(modal) > 3 * max(token_len(shortest), 1)
                       else modal)
        else:
            cnt = {}
            for _, s in g:
                cnt[s] = cnt.get(s, 0) + 1
            out.append(max(cnt.items(),
                           key=lambda kv: (kv[1], -token_len(kv[0])))[0])
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--tags", default=",".join(TAGS))
    a_.add_argument("--boot", type=int, default=300)
    a_.add_argument("--seed", type=int, default=13)
    a_.add_argument("--ship", default="cluster>=2 modal",
                    help="the arm currently deployed; everything is paired "
                         "against it")
    args = a_.parse_args()

    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    per_tag = {}
    for t in [x.strip() for x in args.tags.split(",")]:
        p = os.path.join(args.dir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            print(f"  skip {t}: no {p}")
            continue
        d = pd.read_csv(p)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"].astype(str), d["span"]):
            m.setdefault(r, []).append(str(s))
        per_tag[t] = {r: dedupe(v) for r, v in m.items()}
    if len(per_tag) < 2:
        raise SystemExit("need at least two taggers")
    base_tag = "v2" if "v2" in per_tag else list(per_tag)[0]

    rows = [r for r in oof["row_id"] if r in lab.index]
    n = len(rows)
    ix = oof.set_index("row_id")
    yship = np.array([normalize_label(x)
                      for x in ix.loc[rows, "pred_risk_level"]])
    ygold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}

    print(f"{n} OOF posts / {len(ua)} authors   taggers {list(per_tag)}")
    print("suppression uses PREDICTED risk; averaging is MACRO. Both differ")
    print("from how every previous arm comparison was run.\n")

    def build(pick, risk):
        out = []
        for i, r in enumerate(rows):
            if risk[i] == "indicator":
                out.append([])
                continue
            s = dedupe(pick(r))
            if not s:
                pool = per_tag[base_tag].get(r, []) or [
                    x for t in per_tag for x in per_tag[t].get(r, [])]
                s = sorted(pool, key=token_len)[:1]
            out.append(s)
        return out

    arms = {}
    arms[f"{base_tag} alone"] = (lambda r: per_tag[base_tag].get(r, []))
    arms["union all"] = (lambda r: [x for t in per_tag
                                    for x in per_tag[t].get(r, [])])
    for k in range(2, len(per_tag) + 1):
        for rp in ("modal", "capsafe", "median", "short", "long"):
            arms[f"cluster>={k} {rp}"] = (
                lambda r, k=k, rp=rp: cluster_votes(
                    [(t, x) for t in per_tag for x in per_tag[t].get(r, [])],
                    k, rp))

    built = {nm: build(f, yship) for nm, f in arms.items()}
    built_gold = {nm: build(f, ygold) for nm, f in arms.items()}
    if args.ship not in built:
        raise SystemExit(f"--ship {args.ship!r} is not one of "
                         f"{sorted(built)}")

    def mac(P, idx=None):
        if idx is None:
            return phrase_f1(G, P, average="macro").f1
        return phrase_f1([G[i] for i in idx], [P[i] for i in idx],
                         average="macro").f1

    def mic(P):
        return phrase_f1(G, P, average="micro").f1

    ship = built[args.ship]
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in
                             rng.choice(ua, size=len(ua), replace=True)])
             for _ in range(args.boot)]
    ship_b = np.array([mac(ship, ix_) for ix_ in boots])

    print("=" * 104)
    print(f"EVERY ARM, MACRO + PREDICTED SUPPRESSION, paired against "
          f"{args.ship!r}")
    print("=" * 104)
    print(f"{'arm':<20}{'macro':>9}{'delta':>9}{'95% CI':>21}{'win':>6}"
          f"{'micro':>9}{'GOLD-sup':>10}{'leak':>8}{'toks':>7}")
    res = []
    for nm, P in built.items():
        v = mac(P)
        d = np.array([mac(P, ix_) for ix_ in boots]) - ship_b
        g = mac(built_gold[nm])
        tk = np.mean([token_len(s) for p in P for s in p]) if any(P) else 0.0
        res.append((nm, v, d, mic(P), g, tk))
    res.sort(key=lambda x: -x[1])
    for nm, v, d, mi, g, tk in res:
        mark = "  <- SHIPPED" if nm == args.ship else ""
        print(f"{nm:<20}{v:>9.4f}{d.mean():>+9.4f}"
              f"{f'[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]':>21}"
              f"{(d > 0).mean():>6.0%}{mi:>9.4f}{g:>10.4f}"
              f"{g - v:>+8.4f}{tk:>7.1f}{mark}")

    best = res[0]
    print(f"\n  best under the honest regime: {best[0]!r} at {best[1]:.4f}, "
          f"{best[2].mean():+.4f} vs shipped, win {(best[2] > 0).mean():.0%}")
    print(f"  composite value {0.3 * best[2].mean():+.4f}")

    mi_rank = [nm for nm, _, _, _, _, _ in sorted(res, key=lambda x: -x[3])]
    ma_rank = [nm for nm, _, _, _, _, _ in res]
    print(f"\n  MICRO would have picked {mi_rank[0]!r}")
    print(f"  MACRO picks             {ma_rank[0]!r}")
    if mi_rank[0] != ma_rank[0]:
        print("  -> the metric alone changes the winner. Every arm chosen "
              "under micro\n     was chosen by the wrong criterion.")
    gl = [nm for nm, _, _, _, _, _ in sorted(res, key=lambda x: -x[4])]
    print(f"  GOLD-SUPPRESSION would have picked {gl[0]!r}")
    if gl[0] != ma_rank[0]:
        print("  -> the leak alone changes the winner too.")
    print("\n  'leak' is GOLD-sup minus honest macro: how much each arm was")
    print("  flattered. 'toks' is mean predicted span length -- if the winners")
    print("  are short, the 3x cap asymmetry is what is driving this.")
    print("\n  ~12 arms compared: require win >= 65% AND a CI excluding zero,")
    print("  and re-run with --boot 1000 before shipping anything.")


if __name__ == "__main__":
    main()

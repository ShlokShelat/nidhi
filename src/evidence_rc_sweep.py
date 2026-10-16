"""
evidence_rc_sweep.py

SWEEP THE VOTE ON THE EIGHT TAGGERS ACTUALLY SHIPPED.

The shipped evidence config is `k=2 --cluster --rep modal`. It was chosen on a
372-post val split scored with MICRO averaging. The scorer is MACRO. So the
deployed config was never optimised against the real objective.

The alternative instrument cannot substitute for the real one: `capsafe`
measured a small POSITIVE gain on the 5-tagger OOF set but a NEGATIVE result
on the shipped 8 -- a SIGN FLIP, because `capsafe` falls back to the shortest
span when the modal exceeds 3x it, and bigger clusters have a wider length
range so the trigger fires more often. The rule's behaviour depends on
ensemble size.

THE JUDGEMENT BEHIND THIS SCRIPT. That failure was BIAS (wrong ensemble), not
variance (small split). Bias cannot be averaged away; noise can be bounded.

    7-tagger, 1635 posts   right instrument, WRONG ensemble   -> sign-flipped
    8-tagger,  372 posts   noisy instrument, RIGHT ensemble   -> this

`artifacts/evidence_runs/evidence_*_rc_val_candidates.csv` already exists, so
this is free. `evidence_vote_test.py` documents `--split val` as "SCORE a vote
configuration against gold before shipping it" -- the capability was built and
never used for a sweep.

WHAT IT SWEEPS, scored as MACRO phrase-F1 (the scorer's own metric) with a
paired AUTHOR bootstrap over the ~32 val authors:
    k        1 2 3 4
    cluster  on / off
    rep      modal capsafe median short long        (cluster only)
and the suppression rule is applied exactly as deployed: empty on predicted-
indicator posts.

HOW TO READ IT, given ~32 authors. Every CI will be wide. Do NOT ship on a point
estimate. Ship only a config that beats `k=2 cluster modal` on BOTH author
halves AND has a win rate above ~80%, and prefer one whose neighbours in the
sweep also look good -- an isolated peak on 372 posts is noise, a plateau is
signal. That plateau test is what `capsafe` would have failed: it won at every
cluster level on the WRONG ensemble, which is a consistency check for variance,
not for bias.

Usage:
  python src/evidence_rc_sweep.py
  python src/evidence_rc_sweep.py --boot 4000
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, parse_evidence, phrase_f1,  # noqa: E402
                     token_len)

RC8 = ["evidence_v2_rc", "evidence_v1_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_lr2_rc", "evidence_ep9_rc"]


def norm(s):
    return " ".join(str(s).lower().split())


def contains(a, b):
    na, nb = norm(a), norm(b)
    return na in nb or nb in na


def dedupe(spans):
    out = []
    for s in spans:
        if not any(norm(s) == norm(t) for t in out):
            out.append(s)
    return out


def cluster_vote(per_run, r, k, rep):
    """Group candidates by the scorer's containment+3x equivalence, count
    DISTINCT taggers per group, emit one representative per surviving group."""
    items = []
    for run in per_run:
        for s in per_run[run].get(r, []):
            items.append((run, s))
    if not items:
        return []
    items.sort(key=lambda x: (-token_len(x[1]), x[1]))
    groups = []
    for run, s in items:
        placed = False
        for g in groups:
            if any(contains(s, t) and
                   max(token_len(s), token_len(t)) <=
                   3 * max(min(token_len(s), token_len(t)), 1)
                   for _, t in g):
                g.append((run, s))
                placed = True
                break
        if not placed:
            groups.append([(run, s)])
    out = []
    for g in groups:
        if len({run for run, _ in g}) < k:
            continue
        cnt = Counter(s for _, s in g)
        modal = max(cnt.items(), key=lambda kv: (kv[1], -token_len(kv[0])))[0]
        lens = sorted(token_len(s) for _, s in g)
        shortest = min((s for _, s in g), key=token_len)
        longest = max((s for _, s in g), key=token_len)
        med = sorted((s for _, s in g), key=token_len)[len(g) // 2]
        if rep == "modal":
            out.append(modal)
        elif rep == "short":
            out.append(shortest)
        elif rep == "long":
            out.append(longest)
        elif rep == "median":
            out.append(med)
        else:  # capsafe
            out.append(shortest
                       if token_len(modal) > 3 * max(lens[0], 1) else modal)
    return out


def flat_vote(per_run, r, k):
    cnt = Counter()
    keep = {}
    for run in per_run:
        for s in set(norm(x) for x in per_run[run].get(r, [])):
            cnt[s] += 1
        for x in per_run[run].get(r, []):
            keep.setdefault(norm(x), x)
    return [keep[s] for s, c in cnt.items() if c >= k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--runs", default=",".join(RC8))
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    per_run, rows = {}, None
    for run in runs:
        p = f"{args.dir}/{run}_val_candidates.csv"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        if "conf" in d.columns and "min_conf" in d.columns and len(d):
            d = d[d["conf"] >= float(d["min_conf"].iloc[0])]
        g = {}
        for rid, s in zip(d["row_id"], d["span"]):
            g.setdefault(rid, []).append(str(s))
        per_run[run] = g
        ids = set(g)
        rows = ids if rows is None else (rows | ids)
    rows = sorted(rows)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in rows if r in lab.index]

    rk = pd.read_csv(args.risk, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    rk = rk.drop_duplicates("row_id").set_index("row_id")
    rcol = next((c for c in ("pred_risk_level", "risk_level")
                 if c in rk.columns), None)
    pc = [f"prob_{c}" for c in
          ("indicator", "ideation", "behavior", "attempt")]
    if rcol is not None:
        risk = {r: normalize_label(rk.loc[r, rcol]) for r in rk.index}
    else:
        RL = ["indicator", "ideation", "behavior", "attempt"]
        M = rk[pc].to_numpy(float)
        risk = {r: RL[i] for r, i in zip(rk.index, M.argmax(1))}
    rows = [r for r in rows if r in risk]

    gold = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    print(f"{len(rows)} val posts / {len(uauth)} authors   "
          f"{len(runs)} `_rc` taggers -- THE SHIPPED ENSEMBLE")
    print(f"gold: {sum(len(g) for g in gold)} spans, "
          f"{sum(1 for g in gold if not g)} empty posts\n")

    def build(k, cl, rep):
        out = []
        for r in rows:
            kept = (cluster_vote(per_run, r, k, rep) if cl
                    else flat_vote(per_run, r, k))
            kept = dedupe(kept)
            if risk[r] == "indicator":
                kept = []
            out.append(kept)
        return out

    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([by[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]
    half = set(uauth[: len(uauth) // 2])
    HA = np.concatenate([by[a] for a in uauth if a in half])
    HB = np.concatenate([by[a] for a in uauth if a not in half])

    def sc(pred, idx=None):
        if idx is None:
            return phrase_f1(gold, pred, average="macro").f1
        g = [gold[i] for i in idx]
        p = [pred[i] for i in idx]
        return phrase_f1(g, p, average="macro").f1

    ship = build(2, True, "modal")
    base = sc(ship)
    print("=" * 104)
    print(f"SHIPPED  k=2 cluster modal  macro phrase-F1 {base:.4f}")
    print("=" * 104)
    print(f"{'config':<26}{'spans':>7}{'F1':>9}{'d vs ship':>11}"
          f"{'95% CI':>20}{'win':>6}{'halfA':>9}{'halfB':>9}")
    bA, bB = sc(ship, HA), sc(ship, HB)

    cfgs = [(k, False, "modal") for k in (1, 2, 3, 4)]
    cfgs += [(k, True, rep) for k in (1, 2, 3, 4)
             for rep in ("modal", "capsafe", "median", "short", "long")]
    res = []
    for k, cl, rep in cfgs:
        pred = build(k, cl, rep)
        v = sc(pred)
        name = f"k={k} {'cluster ' + rep if cl else 'flat'}"
        if (k, cl, rep) == (2, True, "modal"):
            print(f"{name + '  <- SHIPPED':<26}{sum(len(p) for p in pred):>7}"
                  f"{v:>9.4f}{'--':>11}{'--':>20}{'--':>6}"
                  f"{bA:>9.4f}{bB:>9.4f}")
            continue
        d = np.array([sc(pred, ix) - sc(ship, ix) for ix in IXS])
        lo, hi = np.percentile(d, [2.5, 97.5])
        dA = sc(pred, HA) - bA
        dB = sc(pred, HB) - bB
        both = "  <- both halves" if dA > 0 and dB > 0 else ""
        print(f"{name:<26}{sum(len(p) for p in pred):>7}{v:>9.4f}"
              f"{v-base:>+11.4f}{f'[{lo:+.4f},{hi:+.4f}]':>20}"
              f"{(d > 0).mean():>6.0%}{dA:>+9.4f}{dB:>+9.4f}{both}")
        res.append((v - base, (d > 0).mean(), dA, dB, name))

    res.sort(reverse=True)
    print("\n" + "=" * 104)
    print("DECISION")
    print("=" * 104)
    print(f"~{len(uauth)} authors, so every CI is wide. Do NOT ship a point "
          f"estimate.")
    print("Require ALL THREE: beats the shipped config, wins >= 80% of author")
    print("resamples, AND is positive on BOTH halves.")
    print("")
    print("THEN apply the PLATEAU TEST. An isolated peak on 372 posts is noise;")
    print("a config whose NEIGHBOURS in the sweep also look good is signal.")
    print("`capsafe` won at every cluster level on the WRONG ensemble -- that")
    print("consistency guarded against variance, not against bias, which is why")
    print("it still sign-flipped. Here the ensemble is right, so consistency")
    print("across neighbouring k and rep is the check that actually applies.")
    print("")
    if res:
        d, w, dA, dB, nm = res[0]
        print(f"best: {nm}   {d:+.4f}   win {w:.0%}   "
              f"halves {dA:+.4f}/{dB:+.4f}   composite {0.3*d:+.4f}")
        if d <= 0 or w < 0.80 or dA <= 0 or dB <= 0:
            print("  -> fails at least one gate. The shipped config stands and")
            print("     the evidence decode is closed on the RIGHT ensemble.")


if __name__ == "__main__":
    main()

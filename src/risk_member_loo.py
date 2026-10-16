"""
risk_member_loo.py

Audits the risk ensemble's membership. Risk carries a weight of 0.4 in the
composite score, more than any other component.

A prior membership audit found a harmful member in the evidence ensemble.
The factor pool came back clean (`factor_member_loo.py`: nested greedy
selection never favored dropping a member, and drop frequencies were flat).
Risk has not yet been audited this way.

THE CHAIN, verified to reproduce the deployed champion:

    ens = 0.5 * mean(risk_v1, risk_v1_s14, risk_v1_s15) + 0.5 * friend
    M   = 0.70 * ens + 0.30 * mean(NLI seeds)
        -> apply_rules(urgency, completed_act) -> route_attempt

Five members and two weights. The weights were fixed long ago
(`scripts/rebuild_and_apply_rules.sh` passes `--weight-ours 0.5` and
`--force-weight 0.30`) and nothing since has revisited them against an
instrument that reproduces the champion.

WHY THIS IS NOT THE `risk_member_search.py` LANE. That one asks whether a NEW
model (LoRA'd LLMs, CoT members) should be ADDED. This asks the opposite
question: is one of the members already IN the blend dragging it down? A pool
of several good members plus one bad one still beats each member alone, so
the two questions are independent and one's result does not answer the other.

MEASURED ON THE TWIN (`artifacts/champion_val_probs.csv` reproduces the
deployed weighted-F1 at high precision), with rules + route_attempt applied on
top of every variant -- a gain the symbolic layer would have produced anyway
is not a real gain, and omitting that layer when measuring risk-only changes
misrepresents what the deployed pipeline will actually do.

The friend member covers only 354 of 372 val rows, so every variant is scored
on the intersection; the baseline is recomputed there too.

BAR, from the evidence audit: worst in the solo table BEFORE the leave-one-out,
>=90% nested selection frequency, >=90% win with real losses -- and divide the
predicted delta by ~2.8, the measured validation-to-test attenuation.

Usage:
  python src/risk_member_loo.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import apply_rules  # noqa: E402
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def wf1(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    tot = 0.0
    for c in LEVELS:
        g, p = (gold == c), (pred == c)
        tp = float((g & p).sum())
        pr = tp / max(p.sum(), 1e-9)
        rc = tp / max(g.sum(), 1e-9)
        tot += 2 * pr * rc / max(pr + rc, 1e-9) * g.sum()
    return tot / max(len(gold), 1)


def probs_of(path, rows):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, keep_default_na=False)
    low = {c.lower(): c for c in d.columns}
    rid = next((low[c] for c in ("row_id", "id") if c in low), None)
    if rid is None:
        return None
    cols = None
    for pat in ("p_{}", "prob_{}", "{}"):
        c = [low.get(pat.format(x)) for x in LEVELS]
        if all(c):
            cols = c
            break
    if cols is None:
        return None
    d[rid] = d[rid].astype(str)
    d = d.drop_duplicates(subset=[rid]).set_index(rid)
    have = [r for r in rows if r in d.index]
    if not have:
        return None
    X = np.full((len(rows), 4), np.nan)
    ix = {r: i for i, r in enumerate(rows)}
    V = d.loc[have, cols].to_numpy(float)
    V = V / np.clip(V.sum(1, keepdims=True), 1e-9, None)
    for r, v in zip(have, V):
        X[ix[r]] = v
    return X


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--splits", type=int, default=300)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold_all = np.array([normalize_label(v) for v in va["risk_level"]])
    posts_all = list(va["post"].astype(str))
    auth_all = np.asarray([str(x) for x in va["anon_user_id"]])

    SRC = {
        "v1_s13": "runs/risk_v1/val_predictions.csv",
        "v1_s14": "runs/risk_v1_s14/val_predictions.csv",
        "v1_s15": "runs/risk_v1_s15/val_predictions.csv",
        "friend": "artifacts/friend_risk_probs_val.csv",
        "nli_a": "artifacts/nli_risk_val_probs.csv",
        "nli_b": "artifacts/nli_risk_val_probs2.csv",
    }
    P = {}
    for k, p in SRC.items():
        X = probs_of(p, rows)
        if X is None:
            print(f"  MISS {k:<8}{p}")
            continue
        P[k] = X
        print(f"  OK   {k:<8}{p}  covers "
              f"{int((~np.isnan(X[:, 0])).sum())}/{len(rows)}")
    need = ["v1_s13", "v1_s14", "v1_s15", "nli_a"]
    if any(k not in P for k in need):
        raise SystemExit("missing a required member")

    ok = np.ones(len(rows), bool)
    for X in P.values():
        ok &= ~np.isnan(X[:, 0])
    idx = np.flatnonzero(ok)
    gold, auth = gold_all[idx], auth_all[idx]
    posts = [posts_all[i] for i in idx]
    rws = [rows[i] for i in idx]
    P = {k: v[idx] for k, v in P.items()}
    print(f"\nscored on {len(idx)} of {len(rows)} val posts "
          f"(intersection of all members), {len(pd.unique(auth))} authors")

    route = {}
    if os.path.exists(args.route):
        d = pd.read_csv(args.route, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        c = "cot" if "cot" in d.columns else "pred_risk_level"
        route = {r: normalize_label(v) for r, v in zip(d["row_id"], d[c])}

    def finalize(M):
        lab = [LEVELS[i] for i in M.argmax(1)]
        lab = [normalize_label(apply_rules(p, l)[0]) for p, l in zip(posts, lab)]
        return np.array([("attempt" if (route.get(r) == "attempt"
                                        and x != "indicator") else x)
                         for r, x in zip(rws, lab)])

    ENC = ["v1_s13", "v1_s14", "v1_s15"]
    NLI = [k for k in ("nli_a", "nli_b") if k in P]

    def build(members, w_ours=0.5, w_nli=0.30):
        enc = [m for m in ENC if m in members]
        nli = [m for m in NLI if m in members]
        parts = []
        if enc:
            e = np.mean([P[m] for m in enc], axis=0)
            if "friend" in members and "friend" in P:
                e = w_ours * e + (1 - w_ours) * P["friend"]
            parts.append((1 - w_nli if nli else 1.0, e))
        if nli:
            parts.append((w_nli if enc else 1.0,
                          np.mean([P[m] for m in nli], axis=0)))
        M = sum(w * x for w, x in parts)
        return M / np.clip(M.sum(1, keepdims=True), 1e-9, None)

    allm = ENC + (["friend"] if "friend" in P else []) + NLI
    base_lab = finalize(build(allm))
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.splits)]
    print(f"  full chain (rules + route) wF1 {wf1(gold, base_lab):.4f}\n")

    print("=" * 72)
    print("1. EACH MEMBER ALONE (rules + route on top)")
    print("=" * 72)
    for m in allm:
        print(f"  {m:<10}{wf1(gold, finalize(P[m])):>9.4f}")

    print("\n" + "=" * 72)
    print("2. LEAVE ONE OUT -- author-resampled")
    print("=" * 72)
    print(f"  {'drop':<10}{'wF1':>9}{'d':>9}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'sd':>9}")
    res = []
    for m in allm:
        keep = [x for x in allm if x != m]
        if not [x for x in keep if x in ENC] and not [x for x in keep
                                                      if x in NLI]:
            continue
        lab = finalize(build(keep))
        d = np.array([wf1(gold[b], lab[b]) - wf1(gold[b], base_lab[b])
                      for b in boots])
        print(f"  {m:<10}{wf1(gold, lab):>9.4f}{d.mean():>+9.4f}"
              f"{0.4*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
              f"{np.mean(d < 0):>6.0%}{0.4*d.std():>9.4f}")
        res.append((float(np.mean(d > 0)), 0.4 * float(d.mean()), m))

    print("\n" + "=" * 72)
    print("3. THE TWO BLEND WEIGHTS -- never revisited since they were set")
    print("=" * 72)
    print(f"  {'w_ours':>7}{'w_nli':>7}{'wF1':>9}{'d':>9}{'d comp':>9}"
          f"{'win':>6}{'LOSS':>6}")
    for wo in (0.3, 0.4, 0.5, 0.6, 0.7):
        for wn in (0.20, 0.30, 0.40):
            lab = finalize(build(allm, wo, wn))
            d = np.array([wf1(gold[b], lab[b]) - wf1(gold[b], base_lab[b])
                          for b in boots])
            if wo == 0.5 and wn == 0.30:
                tag = "  <- shipped"
            elif np.mean(d > 0) >= 0.75:
                tag = ""
            else:
                continue
            print(f"  {wo:>7.1f}{wn:>7.2f}{wf1(gold, lab):>9.4f}"
                  f"{d.mean():>+9.4f}{0.4*d.mean():>+9.4f}"
                  f"{np.mean(d > 0):>5.0%}{np.mean(d < 0):>6.0%}{tag}")
    print("\n  Only rows at >=75% win are shown, plus the shipped setting.")
    print("  A weight sweep is a 2-parameter fit on 354 posts -- treat a")
    print("  winner as a candidate for a NESTED test, not as a result.")

    if res:
        res.sort(reverse=True)
        w, dc, m = res[0]
        print(f"\n  BEST DROP: {m}   {dc:+.4f} composite at {w:.0%} win")
        ok_ = w >= 0.90 and dc > 0
        print(f"  {'CLEARS' if ok_ else 'does NOT clear'} the bar."
              + (f"  Expect ~{dc/2.8:+.4f} on held-out evaluation." if ok_ else ""))


if __name__ == "__main__":
    main()

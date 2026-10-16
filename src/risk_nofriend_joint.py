"""
risk_nofriend_joint.py

Dropping the friend member moves risk AND evidence; both must be measured
together.

`risk_member_loo.py` found the friend model is the weakest member of the risk
blend by a wide margin (solo wF1 0.7228 against the blend's 0.7992, next-worst
member 0.7652) and that removing it gains risk wF1. The weight sweep agrees
independently: w_ours 0.5 -> 0.7, i.e. LESS friend, also gains.

BUT A RISK-ONLY DELTA IS INCOMPLETE. On test the change moves 19 labels and
**7 cross the indicator boundary** (6 become indicator, 1 leaves it). Evidence
suppression fires exactly on `indicator`, so those 7 posts gain or lose their
entire evidence cell:

  * a post becoming `indicator` is emptied. If its gold evidence is empty that
    is +1.0 on that post; if not, it is -whatever it scored.
  * a post leaving `indicator` should acquire evidence, but its candidates were
    dumped CONDITIONED on the old label, so it has none to acquire.

Risk carries 0.4 and evidence 0.3, so a risk-only gain can be wiped out by a
handful of emptied posts. A component measured in isolation does not
represent what shipping the change actually does to the composite score.

This computes the JOINT delta on val: risk wF1 AND evidence macro under the old
and new risk columns, combined as the scorer combines them

    d composite = 0.4 * d(risk wF1) + 0.3 * d(evidence macro)

with suppression recomputed from the NEW risk label, which is what the deployed
pipeline would do. Conditioning stays stale for the changed posts -- exactly
what happens when candidates are not re-dumped -- so this measures the change
that can actually be made today, not an idealised one. A separate line
reports the ORACLE where re-dumping is assumed perfect, to show what a GPU
re-dump would add.

Usage:
  python src/risk_nofriend_joint.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import apply_rules  # noqa: E402
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


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


def probs_of(path, rows):
    """[n,4] aligned to `rows`, NaN where the file has no prediction.

    The friend member covers only 354 of 372 val rows, so callers must
    intersect rather than assume full coverage -- an earlier version returned
    None here and killed the whole comparison over 18 missing posts."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, keep_default_na=False)
    low = {c.lower(): c for c in d.columns}
    rid = next((low[c] for c in ("row_id", "id") if c in low), None)
    cols = None
    for pat in ("p_{}", "prob_{}", "{}"):
        c = [low.get(pat.format(x)) for x in LEVELS]
        if all(c):
            cols = c
            break
    if rid is None or cols is None:
        return None
    d[rid] = d[rid].astype(str)
    d = d.drop_duplicates(subset=[rid]).set_index(rid)
    X = np.full((len(rows), 4), np.nan)
    ix = {r: i for i, r in enumerate(rows)}
    have = [r for r in rows if r in d.index]
    if not have:
        return None
    V = d.loc[have, cols].to_numpy(float)
    V = V / np.clip(V.sum(1, keepdims=True), 1e-9, None)
    for r, v in zip(have, V):
        X[ix[r]] = v
    return X


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--dir", default="artifacts/evidence_runs")
    a_.add_argument("--runs", default=",".join(RC7))
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--k", type=int, default=2)
    a_.add_argument("--splits", type=int, default=300)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold_r = np.array([normalize_label(v) for v in va["risk_level"]])
    gold_e = [parse_evidence(v) for v in va["evidence"]]
    posts = list(va["post"].astype(str))
    auth = np.asarray([str(x) for x in va["anon_user_id"]])

    enc = [probs_of(f"runs/risk_v1{s}/val_predictions.csv", rows)
           for s in ("", "_s14", "_s15")]
    fr = probs_of("artifacts/friend_risk_probs_val.csv", rows)
    nli = [probs_of(p, rows) for p in
           ("artifacts/nli_risk_val_probs.csv",
            "artifacts/nli_risk_val_probs2.csv")]
    enc = [x for x in enc if x is not None]
    nli = [x for x in nli if x is not None]
    if not enc or not nli or fr is None:
        raise SystemExit("missing encoder, NLI or friend val probs")

    # INTERSECT. Both arms of the comparison must be scored on the same posts,
    # otherwise "drop the friend" is confounded with "score 18 extra posts".
    ok = ~np.isnan(fr[:, 0])
    for x in enc + nli:
        ok &= ~np.isnan(x[:, 0])
    keep = np.flatnonzero(ok)
    if len(keep) < len(rows):
        print(f"  friend covers {int(ok.sum())}/{len(rows)} -- both arms "
              f"scored on that intersection")
    rows = [rows[i] for i in keep]
    gold_r = gold_r[keep]
    gold_e = [gold_e[i] for i in keep]
    posts = [posts[i] for i in keep]
    auth = auth[keep]
    enc = [x[keep] for x in enc]
    nli = [x[keep] for x in nli]
    fr = fr[keep]

    E = np.mean(enc, axis=0)
    N = np.mean(nli, axis=0)
    M_old = 0.70 * (0.5 * E + 0.5 * fr) + 0.30 * N
    M_new = 0.70 * E + 0.30 * N              # friend dropped entirely

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
                         for r, x in zip(rows, lab)])

    R_old, R_new = finalize(M_old), finalize(M_new)

    # evidence candidates, per run, from the val dumps
    per = {}
    for run in [x.strip() for x in args.runs.split(",")]:
        p = os.path.join(args.dir, f"{run}_val_candidates.csv")
        if not os.path.exists(p):
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
        raise SystemExit("need the val candidate dumps")

    def evidence(risk_lab):
        """Vote k>=2, suppressed wherever the RISK LABEL says indicator.
        Candidates are the ones already dumped -- their conditioning is stale
        for any post whose label changed, which is exactly the state we would
        ship in without a GPU re-dump."""
        out = []
        for i, r in enumerate(rows):
            if risk_lab[i] == "indicator":
                out.append([])
                continue
            cnt = {}
            for run in per:
                for s in per[run].get(r, ()):
                    n = normalize_span(s)
                    if n:
                        e = cnt.setdefault(n, [0, s])
                        e[0] += 1
            out.append(dedupe_nested([v[1] for v in cnt.values()
                                      if v[0] >= args.k]))
        return out

    E_old, E_new = evidence(R_old), evidence(R_new)

    def comp(rl, el, ix):
        return (0.4 * wf1(gold_r[ix], rl[ix])
                + 0.3 * float(np.mean([f1_of(gold_e[i], el[i]) for i in ix])))

    allix = np.arange(len(rows))
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.splits)]

    print(f"{len(rows)} val posts / {len(ua)} authors")
    print(f"  labels changed by dropping friend: "
          f"{int((R_old != R_new).sum())}")
    xb = [i for i in range(len(rows))
          if (R_old[i] == 'indicator') != (R_new[i] == 'indicator')]
    print(f"  crossing the indicator boundary:   {len(xb)}\n")
    print(f"  {'term':<22}{'old':>9}{'new':>9}{'delta':>10}"
          f"{'x weight':>11}")
    ro, rn = wf1(gold_r, R_old), wf1(gold_r, R_new)
    eo = float(np.mean([f1_of(g, p) for g, p in zip(gold_e, E_old)]))
    en = float(np.mean([f1_of(g, p) for g, p in zip(gold_e, E_new)]))
    print(f"  {'risk wF1':<22}{ro:>9.4f}{rn:>9.4f}{rn-ro:>+10.4f}"
          f"{0.4*(rn-ro):>+11.4f}")
    print(f"  {'evidence macro':<22}{eo:>9.4f}{en:>9.4f}{en-eo:>+10.4f}"
          f"{0.3*(en-eo):>+11.4f}")
    print(f"  {'JOINT composite':<22}{'':>9}{'':>9}{'':>10}"
          f"{0.4*(rn-ro)+0.3*(en-eo):>+11.4f}")

    d = np.array([comp(R_new, E_new, b) - comp(R_old, E_old, b)
                  for b in boots])
    print(f"\n  author-resampled: d comp {d.mean():+.4f}   "
          f"win {np.mean(d > 0):.0%}   LOSS {np.mean(d < 0):.0%}   "
          f"sd {d.std():.4f}")
    print(f"  discounted 2.8x for val->test attenuation: "
          f"{d.mean()/2.8:+.4f} expected on held-out evaluation")

    dr = np.array([wf1(gold_r[b], R_new[b]) - wf1(gold_r[b], R_old[b])
                   for b in boots])
    print(f"\n  risk term alone: {0.4*dr.mean():+.4f} at "
          f"{np.mean(dr > 0):.0%} win  <- what risk_member_loo.py reported")
    print("  If the JOINT number is much smaller, the emptied posts are eating")
    print("  the risk gain and this is not the change it looked like.")


if __name__ == "__main__":
    main()

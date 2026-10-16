"""
risk_wnli_joint.py

Measure a change in the risk NLI blend weight on RISK AND EVIDENCE JOINTLY.

The NLI weight `w_nli` is applied by the risk blend, and `rebuild_and_apply_
rules.sh` passes `--force-weight`, overriding the blend script's own fitting
path. This script re-measures the operating point against the full chain rather
than against risk alone.

WHY THE JOINT MEASUREMENT IS MANDATORY. Changing the weight moves posts across
the indicator boundary, and evidence suppression fires exactly there: the
gold-empty evidence cell IS the indicator label. Each crossing post therefore
gains or loses its entire evidence cell. Risk carries 0.4 of the composite and
evidence 0.3, so a handful of crossing posts can swing the composite in either
direction, and the joint delta can be larger or smaller than the risk-only
delta. A component measured in isolation does not predict the composite.

REGIME NOTE. A val measurement transfers to test only when the val and test
artefacts come from the SAME checkpoints. The NLI members (`risk_nli`,
`risk_nli_s2`) are single-split runs on `train_split`, and
`nli_risk_test_probs.csv` comes from those same checkpoints, so val and test
are the same object here. A member whose val artefact is OOF (fold models)
while its test artefact is full-data does not satisfy that condition.

Reports risk wF1, evidence macro, and the joint composite delta with
suppression recomputed from the NEW risk column, which is the state the
pipeline would actually be in, stale conditioning included.

Usage:
  python src/risk_wnli_joint.py --w-new 0.775
  python src/risk_wnli_joint.py --w-new 0.725 --w-new2 0.85
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
    for i, r in enumerate(rows):
        if r in d.index:
            v = pd.to_numeric(d.loc[r, cols], errors="coerce").to_numpy(float)
            if np.isfinite(v).all() and v.sum() > 0:
                X[i] = v / v.sum()
    return X


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--dir", default="artifacts/evidence_runs")
    a_.add_argument("--runs", default=",".join(RC7))
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--w-ours", type=float, default=0.5)
    a_.add_argument("--w-old", type=float, default=0.30)
    a_.add_argument("--w-new", type=float, default=0.775)
    a_.add_argument("--w-new2", type=float, default=None)
    a_.add_argument("--k", type=int, default=2)
    a_.add_argument("--splits", type=int, default=400)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows_all = va["row_id"].tolist()
    enc = [probs_of(f"runs/risk_v1{s}/val_predictions.csv", rows_all)
           for s in ("", "_s14", "_s15")]
    fr = probs_of("artifacts/friend_risk_probs_val.csv", rows_all)
    nli = [probs_of(p, rows_all) for p in
           ("artifacts/nli_risk_val_probs.csv",
            "artifacts/nli_risk_val_probs2.csv")]
    enc = [x for x in enc if x is not None]
    nli = [x for x in nli if x is not None]
    if not enc or not nli or fr is None:
        raise SystemExit("missing a member")
    ok = np.ones(len(rows_all), bool)
    for X in enc + nli + [fr]:
        ok &= np.isfinite(X[:, 0])
    ix = np.flatnonzero(ok)
    rows = [rows_all[i] for i in ix]
    gold_r = np.array([normalize_label(v) for v in va["risk_level"]])[ix]
    gold_e = [parse_evidence(v) for v in va["evidence"]]
    gold_e = [gold_e[i] for i in ix]
    posts = [str(p) for p in va["post"].astype(str)]
    posts = [posts[i] for i in ix]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])[ix]
    E = np.mean([x[ix] for x in enc], axis=0)
    F = fr[ix]
    N = np.mean([x[ix] for x in nli], axis=0)
    print(f"{len(rows)} val posts / {len(pd.unique(auth))} authors")

    route = {}
    if os.path.exists(args.route):
        d = pd.read_csv(args.route, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        c = "cot" if "cot" in d.columns else "pred_risk_level"
        route = {r: normalize_label(v) for r, v in zip(d["row_id"], d[c])}

    def risk_of(wn):
        M = (1 - wn) * (args.w_ours * E + (1 - args.w_ours) * F) + wn * N
        lab = [LEVELS[i] for i in M.argmax(1)]
        lab = [normalize_label(apply_rules(p, l)[0]) for p, l in zip(posts, lab)]
        return np.array([("attempt" if (route.get(r) == "attempt"
                                        and x != "indicator") else x)
                         for r, x in zip(rows, lab)])

    per = {}
    for run in [x.strip() for x in args.runs.split(",")]:
        p = os.path.join(args.dir, f"{run}_val_candidates.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        m = {}
        for r, s in zip(d["row_id"], d["span"]):
            m.setdefault(r, set()).add(str(s))
        per[run] = m
    if len(per) < 3:
        raise SystemExit("need the val candidate dumps")

    def evidence_of(risk_lab):
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

    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.splits)]

    R0 = risk_of(args.w_old)
    E0 = evidence_of(R0)
    r0 = wf1(gold_r, R0)
    e0 = float(np.mean([f1_of(g, p) for g, p in zip(gold_e, E0)]))
    print(f"\n  shipped w_nli={args.w_old}: risk {r0:.4f}   evidence {e0:.4f}")
    print(f"\n  {'w_nli':>7}{'risk':>9}{'d risk':>9}{'evidence':>10}"
          f"{'d evid':>9}{'JOINT':>9}{'win':>6}{'LOSS':>6}{'ind':>5}")
    for wn in [w for w in (args.w_new, args.w_new2) if w is not None]:
        R1 = risk_of(wn)
        E1 = evidence_of(R1)
        r1 = wf1(gold_r, R1)
        e1 = float(np.mean([f1_of(g, p) for g, p in zip(gold_e, E1)]))
        d = np.array([
            (0.4 * wf1(gold_r[b], R1[b])
             + 0.3 * np.mean([f1_of(gold_e[i], E1[i]) for i in b]))
            - (0.4 * wf1(gold_r[b], R0[b])
               + 0.3 * np.mean([f1_of(gold_e[i], E0[i]) for i in b]))
            for b in boots])
        ic = int(sum(1 for i in range(len(rows))
                     if (R0[i] == "indicator") != (R1[i] == "indicator")))
        print(f"  {wn:>7.3f}{r1:>9.4f}{r1-r0:>+9.4f}{e1:>10.4f}"
              f"{e1-e0:>+9.4f}{d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
              f"{np.mean(d < 0):>6.0%}{ic:>5}")
        print(f"          risk term {0.4*(r1-r0):+.4f}   evidence term "
              f"{0.3*(e1-e0):+.4f}   discounted 2.8x {d.mean()/2.8:+.4f}")
    print("\n  If the evidence term is negative and cancels the risk term, the")
    print("  crossing posts are being emptied wrongly and this is not the")
    print("  change the risk-only sweep suggested.")


if __name__ == "__main__":
    main()

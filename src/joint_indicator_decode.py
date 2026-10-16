"""
joint_indicator_decode.py

The risk prediction is scored twice by the composite objective, but is
optimized as if it were scored once.

Gold evidence is empty exactly when gold risk is `indicator` (0.986). Evidence
is macro-per-post and an empty prediction against empty gold scores 1.0. A
probe submission (NIDHI_probe_noev.csv) measured the empty share on held-out
test data directly: **0.357 of the 378 test posts, ~135 posts.**

So `pred_risk_level` is a SHARED PARAMETER of two scored components:

    0.4 * weighted_F1(risk)   <- what every risk model was tuned on
  + 0.3 * phrase_F1(evidence) <- also determined by the same decision, via the
                                 indicator suppression rule, on 135 posts

Per-post expected value on a post torn between indicator (p) and some other
class (q), where s is the span quality on posts where a span is emitted:

    call indicator   0.4p + 0.3p            = 0.7p
    call class c     0.4q + 0.3qs           = q(0.4 + 0.3s)  ~ 0.58q at s=0.6

    => call indicator when p > 0.83q, NOT when p > q.

Nothing in this repo implements that. Every risk arm was selected on risk
weighted-F1 alone, which treats indicator as one class of four and is blind to
the 0.3 riding on it.

AND THE OOF EVIDENCE NUMBERS ARE MEASURED WITH GOLD SUPPRESSION.
evidence_vote_oof.py line 114 builds `risk` from `lab["risk_level"]` -- the
GOLD label -- and line 121 suppresses on it. Every evidence figure this project
has produced (v2 alone = 0.7540, every vote and cluster arm) was scored while
being handed the empty-gold posts for free. Deployment uses PREDICTED risk.
That gap is not a nuisance here, it is the quantity of interest.

WHAT THIS PRINTS

  1. evidence under GOLD suppression vs PREDICTED suppression. The difference
     is how much perfect indicator classification is worth -- Lever A's ceiling
     and the size of the optimism in every past evidence measurement.

  2. the confusion that matters: on posts predicted indicator, how often is
     gold empty, and vice versa. Both error directions cost twice.

  3. a sweep of the indicator bias lambda -- call indicator when
     p_ind > lambda * max(other probs). lambda = 1.0 is argmax, i.e. today.
     Scored on the REAL joint objective 0.4*risk_wF1 + 0.3*evidence_macro,
     not on either component alone.

  4. the same sweep CROSS-FITTED on author partitions: lambda chosen on fit
     authors, scored on held-out. A lambda that only wins in-sample is the
     228-post val sweep all over again.

CAVEAT STATED UP FRONT. The expected-value algebra above is per-post, but risk
is scored by WEIGHTED F1, which is not per-post decomposable -- moving a
threshold shifts precision and recall for two classes at once. That is exactly
why lambda is measured on the composite rather than trusted from the algebra,
and why 0.83 is a hypothesis, not a setting.

Usage (cluster, CPU):
  python src/joint_indicator_decode.py
  python src/joint_indicator_decode.py --arm cluster --k 3 --rep modal
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

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
PCOL = {"indicator": "prob_indicator", "ideation": "prob_ideation",
        "behavior": "prob_behavior", "attempt": "prob_attempt"}
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


def cluster_votes(items, k, rep="modal"):
    """Copied from evidence_vote_oof.cluster_votes -- it is nested in main()
    there so it cannot be imported. items = [(tag, span), ...]."""
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
        else:
            cnt = {}
            for _, s in g:
                cnt[s] = cnt.get(s, 0) + 1
            out.append(max(cnt.items(),
                           key=lambda kv: (kv[1], -token_len(kv[0])))[0])
    return out


def weighted_f1(gold, pred):
    """Support-weighted macro F1, the risk metric."""
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    tot = 0.0
    for c in LEVELS:
        g, p = gold == c, pred == c
        tp = int((g & p).sum())
        den = 2 * tp + int((~g & p).sum()) + int((g & ~p).sum())
        f1 = 0.0 if den == 0 else 2.0 * tp / den
        tot += f1 * g.sum()
    return float(tot / max(len(gold), 1))


def decide(P, lam):
    """Call indicator when p_ind > lam * max(other). lam=1 is plain argmax."""
    ind = P[:, 0]
    other = P[:, 1:]
    best = other.max(axis=1)
    who = np.array(LEVELS[1:])[other.argmax(axis=1)]
    return np.where(ind > lam * best, "indicator", who)


def ev_macro(G, S):
    return phrase_f1(G, S, average="macro").f1


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--tags", default=",".join(TAGS))
    a_.add_argument("--arm", default="base",
                    choices=["base", "cluster", "union"])
    a_.add_argument("--k", type=int, default=3)
    a_.add_argument("--rep", default="modal", choices=["modal", "short"])
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    tags = [t.strip() for t in args.tags.split(",")]
    per_tag = {}
    for t in tags:
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
    if not per_tag:
        raise SystemExit("no tagger candidates found")
    base_tag = "v2" if "v2" in per_tag else list(per_tag)[0]

    rows = [r for r in oof["row_id"] if r in lab.index]
    n = len(rows)
    print(f"champion_oof {len(oof)} rows, labeled {len(lab)}, "
          f"usable intersection {n}")
    print(f"evidence arm: {args.arm}"
          + (f" k={args.k} rep={args.rep}" if args.arm == "cluster" else "")
          + f"   taggers {list(per_tag)}\n")

    ix = oof.set_index("row_id")
    P = np.stack([ix.loc[rows, PCOL[c]].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-9, None)
    ygold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    yship = np.array([normalize_label(x)
                      for x in ix.loc[rows, "pred_risk_level"]])
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])

    # unsuppressed spans, exactly as the shipped decode would build them
    U, UNF = [], []
    for r in rows:
        if args.arm == "base":
            s = per_tag[base_tag].get(r, [])
        elif args.arm == "union":
            s = [x for t in per_tag for x in per_tag[t].get(r, [])]
        else:
            s = cluster_votes([(t, x) for t in per_tag
                               for x in per_tag[t].get(r, [])],
                              args.k, args.rep)
        s = dedupe(s)
        UNF.append(list(s))          # NO force -- stays empty if empty
        if not s:                    # the shipped back-off
            pool = per_tag[base_tag].get(r, []) or [
                x for t in per_tag for x in per_tag[t].get(r, [])]
            s = sorted(pool, key=token_len)[:1]
        U.append(s)
    ua_all = pd.unique(auth)
    by_all = {a: np.flatnonzero(auth == a) for a in ua_all}

    def suppress(pred):
        return [[] if p == "indicator" else u for p, u in zip(pred, U)]

    empty_gold = np.array([len(g) == 0 for g in G])
    print(f"gold-empty posts: {empty_gold.sum()} of {n} "
          f"({empty_gold.mean():.1%})   probe submission measured 35.7% on test")
    print(f"gold indicator:   {(ygold == 'indicator').sum()} "
          f"({(ygold == 'indicator').mean():.1%})\n")

    # ---- 1. how much does gold suppression inflate the measured score? ----
    print("=" * 88)
    print("1. THE LEAK IN EVERY PAST EVIDENCE NUMBER")
    print("=" * 88)
    e_gold = ev_macro(G, suppress(ygold))
    e_ship = ev_macro(G, suppress(yship))
    e_none = ev_macro(G, U)
    print(f"  suppress on GOLD risk      {e_gold:.4f}   <- what "
          f"evidence_vote_oof.py reports")
    print(f"  suppress on SHIPPED pred   {e_ship:.4f}   <- what actually "
          f"deploys")
    print(f"  no suppression at all      {e_none:.4f}")
    print(f"\n  the leak is {e_gold - e_ship:+.4f} evidence "
          f"({0.3*(e_gold-e_ship):+.4f} composite). Every evidence decode")
    print("  decision in this project was tuned in the GOLD regime.")

    # ---- 1b. the FORCE rule, never ablated, zero parameters ---------------
    print("\n" + "=" * 88)
    print("1b. THE FORCE RULE -- manufacturing a span when the tagger is silent")
    print("=" * 88)
    print("  evidence_vote_oof.py:127-129 emits the shortest available candidate")
    print("  when a non-indicator post has no surviving spans. On a post whose")
    print("  gold is EMPTY that turns a free 1.0 into a 0.0. It has never been")
    print("  ablated, and it costs ZERO parameters to remove.")
    ev_f = ev_macro(G, suppress(yship))
    ev_n = ev_macro(G, [[] if p == "indicator" else u
                        for p, u in zip(yship, UNF)])
    forced = np.array([len(u) == 0 and len(v) > 0
                       for u, v in zip(UNF, U)])
    fe = forced & empty_gold
    print(f"\n  posts where FORCE fires:            {int(forced.sum()):>5}")
    print(f"    of those, gold is EMPTY:          {int(fe.sum()):>5}"
          f"   (we hand back a guaranteed 1.0)")
    print(f"    of those, gold is NON-empty:      {int((forced & ~empty_gold).sum()):>5}"
          f"   (a free shot at a match)")
    print(f"\n  evidence WITH force    {ev_f:.4f}")
    print(f"  evidence WITHOUT force {ev_n:.4f}   "
          f"delta {ev_n - ev_f:+.4f}  ({0.3*(ev_n-ev_f):+.4f} composite)")
    rngf = np.random.default_rng(args.seed)
    dd = []
    for _ in range(200):
        pick = rngf.choice(ua_all, size=len(ua_all), replace=True)
        idx = np.concatenate([by_all[a] for a in pick])
        gg = [G[i] for i in idx]
        a1 = ev_macro(gg, [[] if yship[i] == "indicator" else U[i]
                           for i in idx])
        a2 = ev_macro(gg, [[] if yship[i] == "indicator" else UNF[i]
                           for i in idx])
        dd.append(a2 - a1)
    dd = np.array(dd)
    print(f"  author bootstrap: {dd.mean():+.4f}  "
          f"CI [{np.percentile(dd,2.5):+.4f},{np.percentile(dd,97.5):+.4f}]  "
          f"win {(dd > 0).mean():.0%}")

    # ---- 2. the confusion that costs twice --------------------------------
    print("\n" + "=" * 88)
    print("2. INDICATOR CONFUSION -- both directions cost risk AND evidence")
    print("=" * 88)
    pi = yship == "indicator"
    gi = empty_gold
    print(f"  predicted indicator, gold empty      {int((pi & gi).sum()):>5}"
          f"   (we win 1.0 evidence)")
    print(f"  predicted indicator, gold NON-empty  {int((pi & ~gi).sum()):>5}"
          f"   (we score 0.0 -- suppressed real spans)")
    print(f"  predicted OTHER, gold empty          {int((~pi & gi).sum()):>5}"
          f"   (we score 0.0 -- emitted spans on empty gold)")
    print(f"  predicted OTHER, gold NON-empty      {int((~pi & ~gi).sum()):>5}"
          f"   (scored on span quality)")
    rec = (pi & gi).sum() / max(gi.sum(), 1)
    pre = (pi & gi).sum() / max(pi.sum(), 1)
    print(f"\n  indicator recall {rec:.3f}   precision {pre:.3f}")
    print(f"  each of the {int((pi & ~gi).sum() + (~pi & gi).sum())} "
          f"mistakes above costs 1/{n} of evidence AND a risk error.")

    # ---- 3. the lambda sweep on the JOINT objective -----------------------
    print("\n" + "=" * 88)
    print("3. INDICATOR BIAS lambda -- scored on 0.4*risk + 0.3*evidence")
    print("=" * 88)
    lams = [0.60, 0.70, 0.83, 0.90, 1.00, 1.10, 1.25]
    print(f"{'lambda':>8}{'risk wF1':>10}{'evidence':>10}{'joint':>10}"
          f"{'d joint':>10}{'n_ind':>7}")
    ref = None
    for lam in lams:
        pr = decide(P, lam)
        rk = weighted_f1(ygold, pr)
        ev = ev_macro(G, suppress(pr))
        j = 0.4 * rk + 0.3 * ev
        if lam == 1.00:
            ref = j
        print(f"{lam:>8.2f}{rk:>10.4f}{ev:>10.4f}{j:>10.4f}"
              f"{(j-ref) if ref is not None else 0:>+10.4f}"
              f"{int((pr=='indicator').sum()):>7}")
    js = 0.4 * weighted_f1(ygold, yship) + 0.3 * e_ship
    print(f"{'shipped':>8}{weighted_f1(ygold, yship):>10.4f}{e_ship:>10.4f}"
          f"{js:>10.4f}{js-ref:>+10.4f}"
          f"{int((yship=='indicator').sum()):>7}")
    print("\n  'shipped' includes the symbolic rules, the lambda rows do not,")
    print("  so compare lambda to lambda=1.00 and treat shipped as context.")

    # ---- 3b. DECOUPLED: keep the risk label, suppress on its own threshold -
    print("\n" + "=" * 88)
    print("3b. DECOUPLED -- risk_level and evidence are SEPARATE COLUMNS")
    print("=" * 88)
    print("  Nothing forces the consistency rule. Keeping the shipped risk")
    print("  label (optimal for risk wF1) while suppressing evidence on its")
    print("  own threshold dominates changing the label: same risk score, and")
    print("  the evidence decision is free to be right on its own terms.")
    print(f"\n{'t':>8}{'evidence':>10}{'joint':>10}{'d joint':>10}"
          f"{'n_suppr':>9}")
    ts = [0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
    rk_ship = weighted_f1(ygold, yship)
    ref_b = None
    for t in ts:
        sup = P[:, 0] >= t
        ev = ev_macro(G, [[] if s else u for s, u in zip(sup, U)])
        j = 0.4 * rk_ship + 0.3 * ev
        if ref_b is None:
            ref_b = js
        print(f"{t:>8.2f}{ev:>10.4f}{j:>10.4f}{j-js:>+10.4f}"
              f"{int(sup.sum()):>9}")
    print(f"{'shipped':>8}{e_ship:>10.4f}{js:>10.4f}{0.0:>+10.4f}"
          f"{int((yship=='indicator').sum()):>9}")
    print("\n  t=0.00 suppresses everything, t=1.01 suppresses nothing.")
    print("  Compare every row against 'shipped', which is argmax suppression.")

    # ---- 4. cross-fit ------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"4. CROSS-FIT -- parameter chosen on FIT authors, scored on "
          f"held-out ({args.splits} splits)")
    print("=" * 88)
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    frac = 378.0 / (378.0 + n)
    rng = np.random.default_rng(args.seed)
    d_fit, d_best, picks = [], [], []
    for _ in range(args.splits):
        order = list(ua)
        rng.shuffle(order)
        ev_, got = [], 0
        for a in order:
            if got >= frac * n:
                break
            ev_.append(a)
            got += len(by[a])
        E = np.concatenate([by[a] for a in ev_])
        F = np.setdiff1d(np.arange(n), E)

        def jsc(idx, lam):
            pr = decide(P[idx], lam)
            return (0.4 * weighted_f1(ygold[idx], pr)
                    + 0.3 * ev_macro([G[i] for i in idx],
                                     [[] if p == "indicator" else U[i]
                                      for p, i in zip(pr, idx)]))
        def tsc(idx, t):
            sup = P[idx, 0] >= t
            return (0.4 * weighted_f1(ygold[idx], yship[idx])
                    + 0.3 * ev_macro([G[i] for i in idx],
                                     [[] if s else U[i]
                                      for s, i in zip(sup, idx)]))

        lam_f = max(lams, key=lambda L: jsc(F, L))
        t_f = max(ts, key=lambda T: tsc(F, T))
        picks.append((lam_f, t_f))
        b1 = jsc(E, 1.00)
        bship = (0.4 * weighted_f1(ygold[E], yship[E])
                 + 0.3 * ev_macro([G[i] for i in E],
                                  [[] if yship[i] == "indicator" else U[i]
                                   for i in E]))
        d_fit.append(jsc(E, lam_f) - b1)
        d_best.append(tsc(E, t_f) - bship)

    d = np.array(d_fit)
    db = np.array(d_best)
    se = d.std(ddof=1) / np.sqrt(len(d))
    seb = db.std(ddof=1) / np.sqrt(len(db))
    print(f"  COUPLED   (lambda changes the risk label, vs lambda=1.00):")
    print(f"    {d.mean():+.4f}  "
          f"CI [{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]  "
          f"win {(d > 0).mean():.0%}")
    print(f"  DECOUPLED (t suppresses only, risk label untouched, vs shipped):")
    print(f"    {db.mean():+.4f}  "
          f"CI [{db.mean()-1.96*seb:+.4f},{db.mean()+1.96*seb:+.4f}]  "
          f"win {(db > 0).mean():.0%}")
    lv, lc = np.unique([p[0] for p in picks], return_counts=True)
    tv, tc = np.unique([p[1] for p in picks], return_counts=True)
    print("  lambda chosen: " + "  ".join(f"{v:.2f}x{c}"
                                          for v, c in zip(lv, lc)))
    print("  t chosen:      " + "  ".join(f"{v:.2f}x{c}"
                                          for v, c in zip(tv, tc)))
    print("\n  DECOUPLED is the one that can ship -- it changes only the")
    print("  evidence column and leaves risk_level alone, so S1's risk term")
    print("  is untouched. Judge it on mean AND win rate together.")
    print("  If the chosen t is unstable across splits, the parameter is")
    print("  being fitted to noise no matter how good the mean looks.")


if __name__ == "__main__":
    main()

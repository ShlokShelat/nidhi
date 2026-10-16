"""
risk_rule_ablation.py

Checks whether the symbolic risk rules help on held-out posts, or whether
they were fitted to the public evaluation slice.

`champion_oof.csv` carries both columns:

    pre_rule_pred     argmax of the blended risk probabilities
    pred_risk_level   after the symbolic fences and the attempt routing

joint_indicator_decode.py measured, in passing, on 1590 OOF posts:

    argmax    risk wF1  0.7858
    shipped   risk wF1  0.7841

**The rule layer is NEGATIVE out of fold.** This can appear to conflict with
a positive result on the public evaluation slice, because:

  - the public official-scorer slice is ONE draw of 378 posts with a ~+-0.016
    per-draw noise band,
  - the organizers re-run the submitted code on a SEPARATE held-out set for
    final evaluation,

so a rule tuned to gain on a small public slice while losing on 1590
held-out posts is fitted noise, and shipping it risks losing on the final
evaluation set.

THE RULES ALSO MOVE EVIDENCE. Suppression fires on predicted `indicator`, so
any rule that changes an indicator call flips that post's evidence between
"empty" and "the predicted spans" -- worth 1.0 or 0.0 on a gold-empty post. Scoring the
rule layer on risk alone misses 0.3 of its effect, which is why this reports
the joint objective too.

WHAT IS REPORTED
  1. risk wF1 with and without the rules, plus the PER-CLASS breakdown -- the
     rules were justified by attempt recall, so that class is the test.
  2. how many posts the rules actually change, and how many of those they get
     right vs wrong. A rule that fires 12 times and is wrong 7 of them is not
     a rule, it is noise with a mechanism attached.
  3. the joint 0.4*risk + 0.3*evidence, since indicator flips move both.
  4. an AUTHOR BOOTSTRAP with the per-draw spread, not the CI of the mean.

Usage (cluster, CPU, ~1 min):
  python src/risk_rule_ablation.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span, parse_evidence,  # noqa: E402
                     phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]


def wf1(gold, pred, per_class=False):
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    tot, rows = 0.0, []
    for c in LEVELS:
        g, p = gold == c, pred == c
        tp = int((g & p).sum())
        fp = int((~g & p).sum())
        fn = int((g & ~p).sum())
        den = 2 * tp + fp + fn
        f1 = 0.0 if den == 0 else 2.0 * tp / den
        rec = tp / max(int(g.sum()), 1)
        pre = tp / max(tp + fp, 1)
        rows.append((c, int(g.sum()), pre, rec, f1))
        tot += f1 * g.sum()
    v = float(tot / max(len(gold), 1))
    return (v, rows) if per_class else v


def dedupe(spans):
    out = []
    for s in sorted(spans, key=lambda x: (-token_len(x),
                                          -len(normalize_span(x)))):
        n = normalize_span(s)
        if not n or any(n in normalize_span(k) for k in out):
            continue
        out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--tags", default=",".join(TAGS))
    a_.add_argument("--boot", type=int, default=400)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in oof["row_id"] if r in lab.index]
    n = len(rows)
    ix = oof.set_index("row_id")

    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    withr = np.array([normalize_label(x)
                      for x in ix.loc[rows, "pred_risk_level"]])
    nor = np.array([normalize_label(x) for x in ix.loc[rows, "pre_rule_pred"]])
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]

    print(f"{n} OOF posts   rules change {int((withr != nor).sum())} "
          f"predictions ({(withr != nor).mean():.1%})\n")

    print("=" * 78)
    print("1. RISK wF1, PER CLASS")
    print("=" * 78)
    a, ra = wf1(gold, nor, True)
    b, rb = wf1(gold, withr, True)
    print(f"{'class':<12}{'n':>6}"
          f"{'P no-rule':>11}{'R no-rule':>11}{'F1':>8}"
          f"{'P rules':>10}{'R rules':>10}{'F1':>8}{'dF1':>9}")
    for (c, ng, p0, r0, f0), (_, _, p1, r1, f1) in zip(ra, rb):
        print(f"{c:<12}{ng:>6}{p0:>11.3f}{r0:>11.3f}{f0:>8.3f}"
              f"{p1:>10.3f}{r1:>10.3f}{f1:>8.3f}{f1-f0:>+9.3f}")
    print(f"\n{'weighted F1':<12}{'':>6}{'':>11}{'':>11}{a:>8.4f}"
          f"{'':>10}{'':>10}{b:>8.4f}{b-a:>+9.4f}")

    print("\n" + "=" * 78)
    print("2. WHAT THE RULES ACTUALLY DO -- every changed post")
    print("=" * 78)
    ch = np.flatnonzero(withr != nor)
    good = int(sum(1 for i in ch if withr[i] == gold[i]))
    bad = int(sum(1 for i in ch if nor[i] == gold[i]))
    print(f"  posts changed          {len(ch)}")
    print(f"  rule FIXED a mistake   {good}")
    print(f"  rule BROKE a correct   {bad}")
    print(f"  both wrong anyway      {len(ch) - good - bad}")
    if good + bad:
        print(f"  ratio {good}:{bad}  -- a rule worth keeping should be "
              f"clearly above 1:1")
    print(f"\n  {'from':<12}{'to':<12}{'n':>5}{'fixed':>7}{'broke':>7}")
    seen = {}
    for i in ch:
        k = (nor[i], withr[i])
        d = seen.setdefault(k, [0, 0, 0])
        d[0] += 1
        d[1] += withr[i] == gold[i]
        d[2] += nor[i] == gold[i]
    for (f_, t_), (c0, c1, c2) in sorted(seen.items(), key=lambda x: -x[1][0]):
        print(f"  {f_:<12}{t_:<12}{c0:>5}{c1:>7}{c2:>7}")

    # ---- 3. the joint objective, since indicator flips move evidence -------
    per_tag = {}
    for t in [x.strip() for x in args.tags.split(",")]:
        p = os.path.join(args.dir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"].astype(str), d["span"]):
            m.setdefault(r, []).append(str(s))
        per_tag[t] = m

    joint = None
    if len(per_tag) >= 2:
        base_tag = "v2" if "v2" in per_tag else list(per_tag)[0]
        cnt = {}
        for i, r in enumerate(rows):
            c = {}
            for t in per_tag:
                for s in per_tag[t].get(r, []):
                    c.setdefault(normalize_span(s), [0, s])
                    c[normalize_span(s)][0] += 1
            cnt[i] = [s for k, (v, s) in c.items() if v >= 2]
        pool = [sorted([x for t in per_tag for x in per_tag[t].get(r, [])],
                       key=token_len)[:1] for r in rows]

        def ev(pred):
            P = []
            for i in range(n):
                if pred[i] == "indicator":
                    P.append([])
                    continue
                s = dedupe(cnt[i])
                P.append(s if s else pool[i])
            return P

        Pa, Pb = ev(nor), ev(withr)
        ea = phrase_f1(G, Pa, average="macro").f1
        eb = phrase_f1(G, Pb, average="macro").f1
        print("\n" + "=" * 78)
        print("3. THE JOINT OBJECTIVE -- rules move evidence via suppression")
        print("=" * 78)
        print(f"  {'':<10}{'risk':>10}{'evidence':>11}"
              f"{'0.4r+0.3e':>12}")
        print(f"  {'no rules':<10}{a:>10.4f}{ea:>11.4f}"
              f"{0.4*a+0.3*ea:>12.4f}")
        print(f"  {'rules':<10}{b:>10.4f}{eb:>11.4f}"
              f"{0.4*b+0.3*eb:>12.4f}")
        print(f"  {'delta':<10}{b-a:>+10.4f}{eb-ea:>+11.4f}"
              f"{0.4*(b-a)+0.3*(eb-ea):>+12.4f}")
        joint = (Pa, Pb)

    # ---- 4. author bootstrap, per-draw spread ------------------------------
    print("\n" + "=" * 78)
    print(f"4. AUTHOR BOOTSTRAP ({args.boot} resamples) -- per-draw spread, "
          f"not the CI of the mean")
    print("=" * 78)
    ua = pd.unique(auth)
    by = {x: np.flatnonzero(auth == x) for x in ua}
    rng = np.random.default_rng(args.seed)
    dr, dj = [], []
    for _ in range(args.boot):
        pick = rng.choice(ua, size=len(ua), replace=True)
        idx = np.concatenate([by[x] for x in pick])
        dr.append(wf1(gold[idx], withr[idx]) - wf1(gold[idx], nor[idx]))
        if joint is not None:
            gg = [G[i] for i in idx]
            e1 = phrase_f1(gg, [joint[1][i] for i in idx], average="macro").f1
            e0 = phrase_f1(gg, [joint[0][i] for i in idx], average="macro").f1
            dj.append(0.4 * dr[-1] + 0.3 * (e1 - e0))
    dr = np.array(dr)
    print(f"  risk wF1 delta (rules - none)  {dr.mean():+.4f}   "
          f"win {(dr > 0).mean():.0%}   sd {dr.std(ddof=1):.4f}")
    print(f"    one-draw 95% "
          f"[{dr.mean()-1.96*dr.std(ddof=1):+.4f},"
          f"{dr.mean()+1.96*dr.std(ddof=1):+.4f}]")
    if dj:
        dj = np.array(dj)
        print(f"  JOINT delta                    {dj.mean():+.4f}   "
              f"win {(dj > 0).mean():.0%}   sd {dj.std(ddof=1):.4f}")
        print(f"    one-draw 95% "
              f"[{dj.mean()-1.96*dj.std(ddof=1):+.4f},"
              f"{dj.mean()+1.96*dj.std(ddof=1):+.4f}]")
    print("\n  A win rate BELOW 50% means the rule layer is a liability on")
    print("  held-out authors, whatever it did on the public 378. The")
    print("  organizers re-run the submitted code on a set never seen before.")


if __name__ == "__main__":
    main()

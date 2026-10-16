"""
symbolic_on_spans.py

THE RULES SEARCH THE WHOLE POST INSTEAD OF THE EVIDENCE SPAN ALREADY LOCATED.

Every symbolic rule in `risk_symbolic_fences.py` fires on the WHOLE POST:

    DONE     tried|attempted|survived|hospital|ambulance|overdosed|...
    PLAN     pills|rope|gun|jump|hang|tonight|tomorrow|suicide note|...
    EXPLICIT suicid\\w*|kill myself|want to die|...

`DONE` matches "hospital" in paragraph three of a 400-word post. `PLAN`
matches "tonight" wherever it appears. An evidence tagger runs alongside these
rules whose ENTIRE JOB is to locate the clause an annotator would highlight as
the justification for the risk level -- and the rules ignore it.

THE HYPOTHESIS. A regex evaluated on the EVIDENCE SPAN is far more precise
than the same regex on the post, because the span is by construction the
risk-bearing clause. "tried" inside the evidence means a completed act;
"tried" elsewhere usually means "I tried therapy".

WHY IT MATTERS. `risk_rule_ablation.py` decomposes the rule layer's effect by
transition:

    behavior -> attempt    (route_attempt)
    behavior -> ideation
    attempt  -> ideation

These transitions do not all move the score the same direction out of fold. If
span-scoped patterns are sharper, the transition that nets positive fires on
more of the right posts and the ones that net negative stop firing on the
wrong ones.

WHAT IS MEASURED, on the 1590-post OOF with PREDICTED-risk-derived spans:
  1. precision/recall of each pattern for its target class, POST vs SPAN.
     `DONE` should separate attempt; `PLAN` should separate behavior;
     `EXPLICIT` should separate non-indicator.
  2. the lift: how much sharper is the span-scoped version?
  3. the population each would fire on inside the risk model's uncertainty
     band, which is where a rule can actually change a decision.

NO RULE IS SHIPPED FROM THIS. It measures whether the premise holds. The rule
layer itself cannot currently be rebuilt for test anyway -- the champion's risk
probabilities do not exist on disk, and OOF scores do not rank test -- so this is
about whether the IDEA is worth building into the submitted pipeline.

Usage:
  python src/symbolic_on_spans.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_symbolic_fences import DONE, EXPLICIT, PLAN  # noqa: E402
from scoring import normalize_label, normalize_span, token_len  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]


def prf(pred, tgt, name, where):
    tp = int((pred & tgt).sum())
    fp = int((pred & ~tgt).sum())
    fn = int((~pred & tgt).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    print(f"  {name:<10}{where:<8}fires {int(pred.sum()):>5}  "
          f"P {p:.3f}  R {r:.3f}  F1 {f:.3f}")
    return p, r, f


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--tags", default=",".join(TAGS))
    a_.add_argument("--vote-k", type=int, default=2)
    args = a_.parse_args()

    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in oof["row_id"] if r in lab.index]
    ix = oof.set_index("row_id")
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    post = [str(lab.loc[r, "post"]) for r in rows]
    P = np.stack([ix.loc[rows, f"prob_{c}"].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)

    per = {}
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
        per[t] = m
    if not per:
        raise SystemExit("no evidence candidates found")

    # the voted spans, concatenated -- the text a rule would see if it were
    # scoped to the evidence instead of the post
    span_txt = []
    for r in rows:
        c = {}
        for t in per:
            for s in per[t].get(r, []):
                c.setdefault(normalize_span(s), [set(), s])[0].add(t)
        keep = [v[1] for v in c.values() if len(v[0]) >= args.vote_k]
        if not keep:
            pool = [s for t in per for s in per[t].get(r, [])]
            keep = sorted(pool, key=token_len)[:1]
        span_txt.append(" ... ".join(keep))

    nw = np.array([len(p.split()) for p in post], float)
    sw = np.array([len(s.split()) for s in span_txt], float)
    print(f"{len(rows)} OOF posts")
    print(f"post length median {np.median(nw):.0f} words; "
          f"evidence span text median {np.median(sw):.0f} words "
          f"({np.median(sw)/max(np.median(nw),1):.0%} of the post)\n")

    tests = [("DONE", DONE, gold == "attempt", "attempt"),
             ("PLAN", PLAN, np.isin(gold, ["behavior", "attempt"]),
              "behavior+attempt"),
             ("EXPLICIT", EXPLICIT, gold != "indicator", "non-indicator")]

    print("=" * 78)
    print("1. THE SAME PATTERN, SCOPED TO THE POST vs TO THE EVIDENCE SPAN")
    print("=" * 78)
    for nm, rx, tgt, desc in tests:
        print(f"\n  target = {desc}  (n={int(tgt.sum())})")
        pp = np.array([bool(rx.search(t)) for t in post])
        ps = np.array([bool(rx.search(t)) for t in span_txt])
        _, _, fp_ = prf(pp, tgt, nm, "POST")
        _, _, fs_ = prf(ps, tgt, nm, "SPAN")
        print(f"  {'':<10}{'delta':<8}F1 {fs_ - fp_:+.3f}"
              + ("   <- span is sharper" if fs_ > fp_ else ""))

    print("\n" + "=" * 78)
    print("2. WHERE A RULE CAN ACTUALLY CHANGE A DECISION")
    print("=" * 78)
    top = P.max(1)
    band = top < 0.60
    print(f"  posts where the model's top probability < 0.60: "
          f"{int(band.sum())} ({band.mean():.1%})")
    print("  A rule only matters inside that band -- elsewhere the model is")
    print("  confident and the risk architecture audit showed overriding it")
    print("  loses. Precision INSIDE the band is what decides a rule.\n")
    for nm, rx, tgt, desc in tests:
        pp = np.array([bool(rx.search(t)) for t in post]) & band
        ps = np.array([bool(rx.search(t)) for t in span_txt]) & band
        print(f"  target = {desc}")
        prf(pp, tgt, nm, "POST")
        prf(ps, tgt, nm, "SPAN")
        print()

    print("=" * 78)
    print("3. route_attempt's OWN CRITERION: DONE among predicted behavior")
    print("=" * 78)
    pred = np.array(LEVELS)[P.argmax(1)]
    m = pred == "behavior"
    for nm, rx in (("DONE", DONE),):
        pp = np.array([bool(rx.search(t)) for t in post])[m]
        ps = np.array([bool(rx.search(t)) for t in span_txt])[m]
        g = (gold[m] == "attempt")
        print(f"  {int(m.sum())} posts predicted behavior, "
              f"{int(g.sum())} of them gold attempt")
        prf(pp, g, nm, "POST")
        prf(ps, g, nm, "SPAN")
    print("\n  This is exactly the population route_attempt fires on -- the")
    print("  one rule the ablation found clearly net positive.")
    print("  If SPAN precision beats POST precision here, the rule should be")
    print("  re-scoped and it will fix more posts per firing.")


if __name__ == "__main__":
    main()

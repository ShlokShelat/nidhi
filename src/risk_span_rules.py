"""
risk_span_rules.py

SCOPE THE COMPLETED-ACT RULE TO THE EVIDENCE CLAUSE, NOT THE WHOLE POST.

`symbolic_on_spans.py` measured, on 1590 OOF posts, the SAME `DONE` regex
evaluated two ways:

    target = gold attempt (n=110)
    DONE on POST   fires 268   P 0.291  R 0.709   F1 0.413
    DONE on SPAN   fires  68   P 0.824  R 0.509   F1 0.629

and inside route_attempt's own population (358 posts predicted behavior, 36 of
them gold attempt):

    DONE on POST   fires 67   P 0.284
    DONE on SPAN   fires 12   P 0.667

Precision nearly TRIPLES. The reason is mechanical: the evidence tagger
already located the clause an annotator would highlight as the justification
for the risk level, so "tried" inside it means a completed act, while "tried"
in paragraph three means "I tried therapy". `PLAN` and `EXPLICIT` get WORSE on
spans (median span is 3 words vs 37 for the post, so recall collapses) -- only
the completed-act criterion belongs in the evidence clause.

WHY THIS IS THE RULE WORTH FIXING. `risk_rule_ablation.py` found the rule layer
net negative overall but decomposable:

    behavior -> attempt    8 posts,  6 fixed,  2 broke   NET +4  <- route_attempt
    behavior -> ideation  20 posts,  6 fixed, 12 broke   NET -6
    attempt  -> ideation   2 posts,  0 fixed,  1 broke   NET -1

route_attempt is the one that works, and it is throttled to 8 firings because
post-scoped DONE is only 28% precise in that population. At 67% the gate can
open. `attempt` has recall 0.545 -- the weakest number in the risk model -- and
114 gold examples total, so each correct promotion is worth a lot of its F1.

THE RULE: predicted behavior -> attempt when DONE fires AND the model already
puts attempt within `margin` of behavior. Two scopings compared at every
margin, so the ONLY variable is where the regex looks.

Judged on the joint 0.4*risk + 0.3*evidence (a promotion out of behavior
cannot change indicator, so evidence moves only via the suppression rule),
cross-fitted on AUTHOR partitions, on WIN RATE and per-draw sd rather than
the confidence interval of the mean.

Usage:
  python src/risk_span_rules.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re  # noqa: E402

from risk_symbolic_fences import DONE  # noqa: E402

# DONE's WEAK TOKENS DO NOT SURVIVE SPAN SCOPING.
# On a 400-word post, `again` / `last time` / `didn't work` co-occur with real
# signal and the regex as a whole is only 29% precise. Scoped to a 3-word
# evidence clause the weak token IS the entire signal, and it is ambiguous:
# applying the full pattern to the champion's test evidence fired on exactly
# three posts, ALL via `again` --
#     'buy rope again'                        (a plan)
#     'Started self harming again'            (ongoing behavior)
#     'planned to kill myself ... cutting myself again'
# none of which is a completed attempt. STRICT keeps only tokens that assert a
# completed act on their own.
STRICT_DONE = re.compile(
    r"\b(tried|attempted|survived|woke up|hospital|emergency|ambulance|icu|"
    r"stomach pump|overdosed|od'?d|swallowed|resuscitat\w*|failed attempt)\b",
    re.I)
from scoring import (normalize_label, normalize_span, parse_evidence,  # noqa
                     phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]


def wf1(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    tot = 0.0
    for c in LEVELS:
        g, p = gold == c, pred == c
        tp = int((g & p).sum())
        den = 2 * tp + int((~g & p).sum()) + int((g & ~p).sum())
        tot += (0.0 if den == 0 else 2.0 * tp / den) * g.sum()
    return float(tot / max(len(gold), 1))


def dedupe(sp):
    out = []
    for s in sorted(sp, key=lambda x: (-token_len(x),
                                       -len(normalize_span(x)))):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--tags", default=",".join(TAGS))
    a_.add_argument("--vote-k", type=int, default=2)
    a_.add_argument("--boot", type=int, default=400)
    a_.add_argument("--seed", type=int, default=13)
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
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    P = np.stack([ix.loc[rows, f"prob_{c}"].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    base = np.array(LEVELS)[P.argmax(1)]

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

    U, span_txt = [], []
    for r in rows:
        c = {}
        for t in per:
            for s in per[t].get(r, []):
                c.setdefault(normalize_span(s), [set(), s])[0].add(t)
        keep = dedupe([v[1] for v in c.values() if len(v[0]) >= args.vote_k])
        if not keep:
            pool = [s for t in per for s in per[t].get(r, [])]
            keep = sorted(pool, key=token_len)[:1]
        U.append(keep)
        span_txt.append(" ... ".join(keep))

    d_post = np.array([bool(DONE.search(t)) for t in post])
    d_span = np.array([bool(DONE.search(t)) for t in span_txt])
    s_span = np.array([bool(STRICT_DONE.search(t)) for t in span_txt])
    s_post = np.array([bool(STRICT_DONE.search(t)) for t in post])
    pb = P[:, LEVELS.index("behavior")]
    pa = P[:, LEVELS.index("attempt")]

    def apply_rule(hit, margin):
        pr = base.copy()
        fire = (base == "behavior") & hit & ((pb - pa) < margin)
        pr[fire] = "attempt"
        return pr, fire

    def joint(pred, ix_):
        e = phrase_f1([G[i] for i in ix_],
                      [[] if pred[i] == "indicator" else U[i] for i in ix_],
                      average="macro").f1
        return 0.4 * wf1(gold[ix_], pred[ix_]) + 0.3 * e

    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in
                             rng.choice(ua, size=len(ua), replace=True)])
             for _ in range(args.boot)]
    allix = np.arange(len(rows))
    b0 = np.array([joint(base, ixb) for ixb in boots])

    print(f"{len(rows)} OOF posts   "
          f"gold attempt {int((gold=='attempt').sum())}   "
          f"predicted behavior {int((base=='behavior').sum())}")
    print(f"argmax risk wF1 {wf1(gold, base):.4f}   "
          f"joint {joint(base, allix):.4f}\n")
    print("=" * 96)
    print("route_attempt: behavior -> attempt when DONE fires and "
          "(p_behavior - p_attempt) < margin")
    print("=" * 96)
    print(f"{'scope':<7}{'margin':>8}{'fires':>7}{'right':>7}{'wrong':>7}"
          f"{'risk wF1':>10}{'d joint':>10}{'win':>6}{'one-draw 95%':>22}")
    for scope, hit in (("POST", d_post), ("SPAN", d_span),
                       ("SPANstr", s_span), ("POSTstr", s_post)):
        for mg in (0.0, 0.10, 0.20, 0.35, 0.50, 0.80, 1.01):
            pr, fire = apply_rule(hit, mg)
            ok = int(((gold == "attempt") & fire).sum())
            bad = int(fire.sum()) - ok
            dd = np.array([joint(pr, ixb) for ixb in boots]) - b0
            sd = dd.std(ddof=1)
            print(f"{scope:<7}{mg:>8.2f}{int(fire.sum()):>7}{ok:>7}{bad:>7}"
                  f"{wf1(gold, pr):>10.4f}{dd.mean():>+10.4f}"
                  f"{(dd > 0).mean():>6.0%}"
                  f"{f'[{dd.mean()-1.96*sd:+.4f},{dd.mean()+1.96*sd:+.4f}]':>22}")
        print()

    print("  margin 1.01 fires on EVERY DONE hit among predicted behavior;")
    print("  0.0 fires only where attempt already leads. The SPAN rows should")
    print("  tolerate a much wider margin than POST at the same accuracy,")
    print("  because they start from 67% precision instead of 28%.")
    print("  Ship only on win >= 65% AND a mostly-positive one-draw interval.")


if __name__ == "__main__":
    main()

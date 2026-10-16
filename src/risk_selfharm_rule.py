"""
risk_selfharm_rule.py

ONGOING SELF-HARM IS THE BEHAVIOR CRITERION, AND NO EXISTING SIGNAL DETECTS IT.

`risk_error_anatomy.py`: the dominant confusion cell is
**behavior -> ideation, 89 posts**, and 88% of all risk errors are one
ordinal step, so this is calibration, not comprehension.

The C-SSRS definitions say what separates them:

    Ideation  explicit suicidal expression, NO plan
    Behavior  explicit expression AND a plan **OR A SELF-HARM ACT**

`risk_crossmodal_cells.py` then read the confidently-wrong posts:

    "I hurt myself more"                  "I can't stop hurting myself"
    "rip my skin off the urges hit strong"  "the scars on my hips"

These state ONGOING SELF-HARM in plain language. They are not ambiguous -- the
MODEL is wrong, not the annotation.

WHY NO EXISTING SIGNAL CATCHES THEM. The factor pool separates the cell barely
at all:

    suicide means    correct behavior 0.787 | missed 0.293 | correct ideation 0.114
    prior self-harm  correct behavior 0.227 | missed 0.217 | correct ideation 0.167

`suicide means (with access)` is METHOD ACCESS. `PRIOR self-harm or suicidal
thought/attempt` is PRIOR. Present-tense self-harm falls between the two
factor labels, so the factor model is blind to exactly what the risk model
misses -- which is why adding it to the cross-modal stack barely moved the
result.

THE RULE: predicted ideation -> behavior when ongoing self-harm is stated.
Tested POST-scoped and SPAN-scoped (the evidence clause), because scoping the
completed-act pattern to the clause the tagger extracted was shown to triple
its precision -- while WEAK tokens got worse.
Two vocabularies are compared so that finding is not assumed:

    SH_CORE    unambiguous acts: cutting/hurting/harming myself, self-harm,
               burned myself, slit
    SH_WIDE    core + scars, blade, razor, bleeding, relapse -- weaker, and
               `scars` in particular can be historical

Judged on the joint 0.4*risk + 0.3*evidence, author bootstrap, WIN RATE and
per-draw sd, never the CI of the mean. A promotion out of ideation cannot
create or destroy an `indicator`, so evidence suppression is untouched.

Usage:
  python src/risk_selfharm_rule.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span, parse_evidence,  # noqa
                     phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]

SH_CORE = re.compile(
    r"\b(cut(?:ting|s)?\s+my\s?self|hurt(?:ing|s)?\s+my\s?self|"
    r"harm(?:ing|s)?\s+my\s?self|self[-\s]?harm\w*|burn(?:ing|ed|t)?\s+my\s?self|"
    r"slit(?:ting)?\s+my|carv(?:ing|ed)\s+into)\b", re.I)
SH_WIDE = re.compile(
    r"\b(cut(?:ting|s)?\s+my\s?self|hurt(?:ing|s)?\s+my\s?self|"
    r"harm(?:ing|s)?\s+my\s?self|self[-\s]?harm\w*|burn(?:ing|ed|t)?\s+my\s?self|"
    r"slit(?:ting)?\s+my|carv(?:ing|ed)\s+into|scars?|blade|razor|"
    r"bleed(?:ing)?|relaps\w+|fresh\s+cuts?)\b", re.I)


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
    P = np.stack([ix.loc[rows, f"prob_{c}"].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    base = np.array(LEVELS)[P.argmax(1)]
    post = [str(lab.loc[r, "post"]) for r in rows]
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])

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
    U, span = [], []
    for r in rows:
        c = {}
        for t in per:
            for s in per[t].get(r, []):
                c.setdefault(normalize_span(s), [set(), s])[0].add(t)
        keep = dedupe([v[1] for v in c.values() if len(v[0]) >= 2])
        if not keep:
            pool = [s for t in per for s in per[t].get(r, [])]
            keep = sorted(pool, key=token_len)[:1]
        U.append(keep)
        span.append(" ... ".join(keep))

    pi = base == "ideation"
    gb = gold == "behavior"
    print(f"{len(rows)} OOF posts   predicted ideation {int(pi.sum())}   "
          f"of those gold behavior {int((pi & gb).sum())}\n")
    print("=" * 88)
    print("1. DOES A SELF-HARM PATTERN FIND THEM? (within predicted ideation)")
    print("=" * 88)
    print(f"{'pattern':<10}{'scope':<7}{'fires':>7}{'gold beh':>10}"
          f"{'P':>8}{'R':>8}{'F1':>8}")
    hits = {}
    for nm, rx in (("CORE", SH_CORE), ("WIDE", SH_WIDE)):
        for sc, txt in (("POST", post), ("SPAN", span)):
            h = np.array([bool(rx.search(t)) for t in txt]) & pi
            hits[(nm, sc)] = h
            tp = int((h & gb).sum())
            p_ = tp / max(int(h.sum()), 1)
            r_ = tp / max(int((pi & gb).sum()), 1)
            f_ = 2 * p_ * r_ / max(p_ + r_, 1e-9)
            print(f"{nm:<10}{sc:<7}{int(h.sum()):>7}{tp:>10}"
                  f"{p_:>8.3f}{r_:>8.3f}{f_:>8.3f}")
    print("\n  P is what matters: promoting a correct ideation to behavior")
    print("  costs as much as fixing a missed one gains.")

    print("\n" + "=" * 88)
    print("2. THE RULE -- ideation -> behavior, joint 0.4*risk + 0.3*evidence")
    print("=" * 88)
    pb = P[:, 2]
    pid = P[:, 1]

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
    b0 = np.array([joint(base, b) for b in boots])
    allix = np.arange(len(rows))
    print(f"  baseline wF1 {wf1(gold, base):.4f}  joint {joint(base, allix):.4f}\n")
    print(f"{'pattern':<9}{'scope':<7}{'margin':>7}{'fires':>7}{'right':>7}"
          f"{'wrong':>7}{'wF1':>9}{'d joint':>10}{'win':>6}"
          f"{'one-draw 95%':>22}")
    for nm in ("CORE", "WIDE"):
        for sc in ("POST", "SPAN"):
            h = hits[(nm, sc)]
            for mg in (0.20, 0.40, 0.70, 1.01):
                fire = h & ((pid - pb) < mg)
                pr = base.copy()
                pr[fire] = "behavior"
                ok = int((fire & gb).sum())
                dd = np.array([joint(pr, b) for b in boots]) - b0
                sd = dd.std(ddof=1)
                print(f"{nm:<9}{sc:<7}{mg:>7.2f}{int(fire.sum()):>7}{ok:>7}"
                      f"{int(fire.sum())-ok:>7}{wf1(gold, pr):>9.4f}"
                      f"{dd.mean():>+10.4f}{(dd > 0).mean():>6.0%}"
                      f"{f'[{dd.mean()-1.96*sd:+.4f},{dd.mean()+1.96*sd:+.4f}]':>22}")
            print()
    print("  margin 1.01 = no probability gate at all (0 fitted parameters).")
    print("  Ship only on win >= 65% AND a mostly-positive one-draw interval.")


if __name__ == "__main__":
    main()

"""
risk_plan_rule.py

THE behavior CRITERION IS METHOD-AND-TIMELINE, NOT SELF-HARM.

`risk_boundary_examples.py` counted the words annotators put inside gold
evidence spans, by class:

    word      in behavior ev   in ideation ev
    jump          21 (5.4%)         0 (0.0%)
    bridge        20 (5.2%)         0 (0.0%)
    pills         18 (4.6%)         0 (0.0%)
    knife         18 (4.6%)         0 (0.0%)
    hang          14 (3.6%)         0 (0.0%)
    gun           11 (2.8%)         1 (0.2%)
    cut           20 (5.2%)         1 (0.2%)
    note          22 (5.7%)         3 (0.6%)
    plan          17 (4.4%)         2 (0.4%)
    tonight       16 (4.1%)         2 (0.4%)
    tomorrow      16 (4.1%)         3 (0.6%)

Methods, timelines and preparation appear in `behavior` evidence and almost
never in `ideation` evidence. The criterion is LEXICAL, and the clean examples
confirm it: "several bottles of antipsychotics ... Tonight's the night",
"drive off a bridge? Use a knife?", "jump off this building".

`risk_selfharm_rule.py` tested a SELF-HARM vocabulary on the same cell and got
precision 0.32. That was the wrong branch: self-harm is the smaller half of
"a plan OR a self-harm act", and it is also the half people mention while
being genuinely ideation-only.

`PLAN` in risk_symbolic_fences.py already encodes the right branch -- pills,
rope, noose, gun, knife, blade, razor, jump, hang, overdose, tonight,
tomorrow, suicide note, method -- and was written from the definitions long
before this analysis, so using it fits nothing. It measures P 0.725 for
behavior+attempt over whole posts and has never been aimed at this cell.

ALSO TESTED, HONESTLY CROSS-FITTED: a vocabulary DERIVED from gold evidence
word counts. Deriving it on the same posts it is scored on would be fitting,
so the derivation happens on FIT authors and the rule is scored on held-out
authors, every split.

Scoped POST and SPAN, since span-scoped rules found the completed-act
pattern tripled precision inside the evidence clause -- and section D's counts
come from gold spans, so the span is where these words are expected to live.

Usage:
  python src/risk_plan_rule.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_symbolic_fences import PLAN  # noqa: E402
from scoring import (normalize_label, normalize_span, parse_evidence,  # noqa
                     phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]

# methods only -- the branch section D says dominates
METHOD = re.compile(
    r"\b(jump(?:ing|ed)?|bridge|pills?|knife|knives|gun|shoot|hang(?:ing)?|"
    r"rope|noose|overdose|od|blade|razor|bleach|poison|train|traffic|"
    r"wrists?|veins?|cliff|roof|balcony)\b", re.I)
# timeline / preparation
WHEN = re.compile(
    r"\b(tonight|tomorrow|next week|this weekend|by (?:friday|monday|"
    r"christmas|new year)|after (?:graduation|exams?|my birthday)|"
    r"suicide note|goodbye letter|wrote a note|my will|giving away)\b", re.I)
STOP = set("i my me a an the and or but to of in it is am was be been so that "
           "this if for on with just myself not no do dont don't have has had "
           "will would can cant can't im i'm you your he she they we at as all "
           "any what when how why".split())


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
    a_.add_argument("--splits", type=int, default=150)
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
    nb = int((pi & gb).sum())
    print(f"{len(rows)} OOF posts   predicted ideation {int(pi.sum())}   "
          f"of those gold behavior {nb}\n")

    PATS = {"PLAN(existing)": PLAN, "METHOD": METHOD, "WHEN": WHEN}
    print("=" * 90)
    print("1. PRECISION WITHIN PREDICTED IDEATION -- which vocabulary works?")
    print("=" * 90)
    print(f"{'pattern':<16}{'scope':<7}{'fires':>7}{'gold beh':>10}"
          f"{'P':>8}{'R':>8}{'F1':>8}")
    hits = {}
    for nm, rx in PATS.items():
        for sc, txt in (("POST", post), ("SPAN", span)):
            h = np.array([bool(rx.search(t)) for t in txt]) & pi
            hits[(nm, sc)] = h
            tp = int((h & gb).sum())
            p_ = tp / max(int(h.sum()), 1)
            r_ = tp / max(nb, 1)
            print(f"{nm:<16}{sc:<7}{int(h.sum()):>7}{tp:>10}{p_:>8.3f}"
                  f"{r_:>8.3f}{2*p_*r_/max(p_+r_,1e-9):>8.3f}")
    for sc, txt in (("POST", post), ("SPAN", span)):
        h = (np.array([bool(METHOD.search(t)) for t in txt])
             | np.array([bool(WHEN.search(t)) for t in txt])) & pi
        hits[("METHOD|WHEN", sc)] = h
        tp = int((h & gb).sum())
        p_ = tp / max(int(h.sum()), 1)
        r_ = tp / max(nb, 1)
        print(f"{'METHOD|WHEN':<16}{sc:<7}{int(h.sum()):>7}{tp:>10}{p_:>8.3f}"
              f"{r_:>8.3f}{2*p_*r_/max(p_+r_,1e-9):>8.3f}")
    print(f"\n  base rate of gold behavior inside predicted ideation: "
          f"{nb/max(int(pi.sum()),1):.3f}")
    print("  A pattern is only useful if P is well ABOVE that base rate.")

    print("\n" + "=" * 90)
    print("2. AS A RULE -- ideation -> behavior, joint 0.4*risk + 0.3*evidence")
    print("=" * 90)
    pb, pid = P[:, 2], P[:, 1]

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
    print(f"{'pattern':<16}{'scope':<7}{'margin':>7}{'fires':>7}{'right':>7}"
          f"{'wrong':>7}{'wF1':>9}{'d joint':>10}{'win':>6}"
          f"{'one-draw 95%':>22}")
    for key in [("PLAN(existing)", "POST"), ("PLAN(existing)", "SPAN"),
                ("METHOD", "POST"), ("METHOD", "SPAN"),
                ("METHOD|WHEN", "POST"), ("METHOD|WHEN", "SPAN")]:
        h = hits[key]
        for mg in (0.30, 0.60, 1.01):
            fire = h & ((pid - pb) < mg)
            pr = base.copy()
            pr[fire] = "behavior"
            ok = int((fire & gb).sum())
            dd = np.array([joint(pr, b) for b in boots]) - b0
            sd = dd.std(ddof=1)
            print(f"{key[0]:<16}{key[1]:<7}{mg:>7.2f}{int(fire.sum()):>7}"
                  f"{ok:>7}{int(fire.sum())-ok:>7}{wf1(gold, pr):>9.4f}"
                  f"{dd.mean():>+10.4f}{(dd > 0).mean():>6.0%}"
                  f"{f'[{dd.mean()-1.96*sd:+.4f},{dd.mean()+1.96*sd:+.4f}]':>22}")
        print()

    # ---- data-derived vocabulary, cross-fitted so it fits nothing ---------
    print("=" * 90)
    print("3. A VOCABULARY LEARNED FROM GOLD EVIDENCE -- derived on FIT")
    print("   authors, scored on held-out authors, every split")
    print("=" * 90)
    frac = 378.0 / (378.0 + len(rows))
    res, resb, sizes = [], [], []
    rng2 = np.random.default_rng(args.seed + 7)
    for _ in range(args.splits):
        order = list(ua)
        rng2.shuffle(order)
        ev_, got = [], 0
        for a in order:
            if got >= frac * len(rows):
                break
            ev_.append(a)
            got += len(by[a])
        E = np.concatenate([by[a] for a in ev_])
        M = np.setdiff1d(np.arange(len(rows)), E)
        cb, ci, nbh, nih = Counter(), Counter(), 0, 0
        for i in M:
            if not G[i]:
                continue
            toks = set(w for s in G[i]
                       for w in re.findall(r"[a-z']+", s.lower())
                       if w not in STOP and len(w) > 2)
            if gold[i] == "behavior":
                cb.update(toks)
                nbh += 1
            elif gold[i] == "ideation":
                ci.update(toks)
                nih += 1
        voc = {w for w, c in cb.items()
               if c >= 5 and (c / max(nbh, 1)) > 6 * (ci.get(w, 0) /
                                                      max(nih, 1))}
        sizes.append(len(voc))
        if not voc:
            continue
        rx = re.compile(r"\b(" + "|".join(re.escape(w) for w in voc) + r")\b",
                        re.I)
        h = np.array([bool(rx.search(span[i])) for i in range(len(rows))]) & pi
        pr = base.copy()
        pr[h] = "behavior"
        res.append(joint(pr, E))
        resb.append(joint(base, E))
    if res:
        d = np.array(res) - np.array(resb)
        sd = d.std(ddof=1)
        print(f"  vocabulary size: median {int(np.median(sizes))} words")
        print(f"  SPAN-scoped, no margin: {d.mean():+.4f}  "
              f"win {(d > 0).mean():.0%}  "
              f"one-draw [{d.mean()-1.96*sd:+.4f},{d.mean()+1.96*sd:+.4f}]")
    print("\n  Ship only on win >= 65% AND a mostly-positive one-draw interval.")


if __name__ == "__main__":
    main()

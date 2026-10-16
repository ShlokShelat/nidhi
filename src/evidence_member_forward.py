"""
evidence_member_forward.py

Test whether adding any unused trained tagger to the evidence vote helps.

The deployed vote uses the risk-conditioned taggers, but
`artifacts/evidence_runs/` holds many more with BOTH val and test dumps:

    in the vote  ep9_rc lr2_rc s14_rc s15_rc s16_rc s17_rc v1_rc
    unused       bs16 bs4 ep12 ep4 ep9 full lr2 lr2b lr3 ml256 ml384
                 s14 s15 s16 s17 v1 v2

The unconditioned family scores worse alone, but scoring worse alone does not
settle whether it helps as EXTRA VOTES: those taggers see no risk label, so
they propose different spans, and a substantial fraction of unmatched gold
spans are ABSENT from the conditioned pool entirely. Diversity is what that
bucket needs, and reusing existing dumps costs no GPU.

Adding a member is the mirror of dropping one, and uses the same machinery and
the same four checks.

THREE TRAPS THIS GUARDS AGAINST:

  LEAK      a run whose name says full-data is trained on all data, so its val
            predictions are in-sample. A member reading implausibly high on val
            is refused both by name and by score, because an in-sample member
            looks best on val and is worst on unseen authors.
  k IS A FRACTION   at 7 members k=2 is 29% agreement; at 12 it is 17%. Adding
            members LOOSENS the vote, so a gain at fixed k may be the threshold
            moving, not the member helping. k is swept for every subset size.
  SEARCH OVERFIT    picking the best of 17 additions on a few hundred posts
            overfits. Selection is nested: the addition is chosen on FIT
            authors and scored on HELD-OUT authors, and a member added in only
            ~half the splits is the search fitting its sample, not a finding.

Usage:
  python src/evidence_member_forward.py
  python src/evidence_member_forward.py --rounds 2
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

CURRENT = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
           "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
           "evidence_v1_rc"]
LEAK_MACRO = 0.93     # above the annotation ceiling -- trained on these posts


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


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--dir", default="artifacts/evidence_runs")
    a_.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    a_.add_argument("--splits", type=int, default=150)
    a_.add_argument("--rounds", type=int, default=1)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    rk = pd.read_csv(args.risk, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    col = "pred_risk_level" if "pred_risk_level" in rk.columns else "risk_level"
    supp = {r: normalize_label(v) == "indicator"
            for r, v in zip(rk["row_id"], rk[col])}

    per = {}
    for p in sorted(os.listdir(args.dir)):
        if not p.endswith("_val_candidates.csv"):
            continue
        run = p[: -len("_val_candidates.csv")]
        tp = os.path.join(args.dir, f"{run}_test_candidates.csv")
        if not os.path.exists(tp):
            tp2 = os.path.join(args.dir + "_rc", f"{run}_test_candidates.csv")
            if not os.path.exists(tp2):
                print(f"  skip {run}: no TEST dump, no test column possible")
                continue
        d = pd.read_csv(os.path.join(args.dir, p), keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        if "conf" in d.columns:
            d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"], d["span"]):
            m.setdefault(r, set()).add(str(s))
        per[run] = m

    def vote(members, k):
        out = []
        for r in rows:
            if supp.get(r, False):
                out.append([])
                continue
            cnt = {}
            for run in members:
                for s in per[run].get(r, ()):
                    n = normalize_span(s)
                    if n:
                        e = cnt.setdefault(n, [0, s])
                        e[0] += 1
            out.append(dedupe_nested([v[1] for v in cnt.values()
                                      if v[0] >= k]))
        return out

    def macro(pred, ix=None):
        ix = range(len(rows)) if ix is None else ix
        return float(np.mean([f1_of(gold[i], pred[i]) for i in ix]))

    print(f"{len(rows)} val posts, {len(per)} taggers with val+test dumps\n")
    print("=" * 74)
    print("1. SOLO SCORE AND LEAK GATE")
    print("=" * 74)
    usable, solo = [], {}
    for run in sorted(per):
        s = macro(vote([run], 1))
        solo[run] = s
        tag = ""
        if "full" in run:
            tag = "  LEAK-REFUSED (name says full-data: trained on val)"
        elif s > LEAK_MACRO:
            tag = f"  LEAK-REFUSED (solo {s:.4f} above the ceiling)"
        else:
            usable.append(run)
        mark = "  <- in the vote" if run in CURRENT else ""
        print(f"  {run:<22}{s:>9.4f}{tag}{mark}")
    cur = [c for c in CURRENT if c in per]
    cands = [r for r in usable if r not in cur]
    print(f"\n  {len(cur)} in the vote, {len(cands)} usable candidates to add")

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        n = max(int(len(o) * 0.25), 2)
        ho = set(o[:n])
        hm = np.array([a in ho for a in auth])
        subs.append((np.flatnonzero(~hm), np.flatnonzero(hm)))

    cache = {}

    def best_k(members, ix):
        """k is a FRACTION -- re-tune it for every subset size or the
        comparison credits the member for a threshold change."""
        bv, bk = -1.0, 2
        for k in (2, 3, 4):
            if k > len(members):
                break
            key = (tuple(sorted(members)), k)
            if key not in cache:
                cache[key] = vote(list(members), k)
            v = macro(cache[key], ix)
            if v > bv:
                bv, bk = v, k
        return bk, bv

    print("\n" + "=" * 74)
    print("2. ADD ONE -- k re-tuned on FIT, scored on HELD-OUT")
    print("=" * 74)
    base = []
    for fi, hi in subs:
        k, _ = best_k(cur, fi)
        base.append(macro(cache[(tuple(sorted(cur)), k)], hi))
    base = np.array(base)
    print(f"  current {len(cur)} taggers, held-out macro {base.mean():.4f}\n")
    print(f"  {'add':<22}{'macro':>9}{'d':>9}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'k':>4}")
    res = []
    for c in cands:
        mem = cur + [c]
        vals, ks = [], []
        for fi, hi in subs:
            k, _ = best_k(mem, fi)
            vals.append(macro(cache[(tuple(sorted(mem)), k)], hi))
            ks.append(k)
        d = np.array(vals) - base
        res.append((float(np.mean(d > 0)), float(d.mean()), c,
                    float(np.mean(ks))))
        if np.mean(d > 0) >= 0.55 or d.mean() > 0:
            print(f"  {c:<22}{np.mean(vals):>9.4f}{d.mean():>+9.4f}"
                  f"{0.3*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
                  f"{np.mean(d < 0):>6.0%}{np.mean(ks):>4.1f}")
    print("  (only candidates at >=55% win or a positive mean are listed)")

    print("\n" + "=" * 74)
    print("3. NESTED FORWARD SELECTION -- chosen on FIT, scored on HELD-OUT")
    print("=" * 74)
    ds, added = [], []
    for fi, hi in subs:
        inner = fi[: max(len(fi) // 4, 10)]
        outer = fi[max(len(fi) // 4, 10):]
        mem = list(cur)
        _, cv = best_k(mem, inner)
        for _ in range(args.rounds):
            bestc, bestv = None, cv
            for c in cands:
                _, v = best_k(mem + [c], inner)
                if v > bestv:
                    bestv, bestc = v, c
            if bestc is None:
                break
            mem.append(bestc)
            added.append(bestc)
            cv = bestv
        k, _ = best_k(mem, fi)
        kb, _ = best_k(cur, fi)
        ds.append(macro(cache[(tuple(sorted(mem)), k)], hi)
                  - macro(cache[(tuple(sorted(cur)), kb)], hi))
    d = np.array(ds)
    print(f"  d macro {d.mean():+.4f}   d comp {0.3*d.mean():+.4f}"
          f"   win {np.mean(d > 0):.0%}   LOSS {np.mean(d < 0):.0%}")
    print("  added how often: " + ", ".join(
        f"{k.replace('evidence_','')} {v}" for k, v in
        Counter(added).most_common(6)))

    if res:
        res.sort(reverse=True)
        w, dm, c, k = res[0]
        print(f"\n  BEST ADD: {c}  {0.3*dm:+.4f} composite at {w:.0%} win, "
              f"k~{k:.1f}")
        print(f"  solo {solo[c]:.4f} vs the pool's {base.mean():.4f}")
        ok = w >= 0.90 and dm > 0
        print(f"  {'CLEARS' if ok else 'does NOT clear'} the bar.")
        print("  Cross-check against section 3: a member added in most splits")
        print("  is a finding; one added in half is the search fitting noise.")


if __name__ == "__main__":
    main()

"""
risk_member_forward.py

SIXTY-FIVE TRAINED RISK MODELS. THE BLEND USES THREE OF THEM.

The champion's encoder term is `mean(risk_v1, risk_v1_s14, risk_v1_s15)`.
`runs/` holds 65 risk models with val predictions -- class-weighted (`cw`),
focal at nine gammas, hard-example mined (`hard`), concept-bottleneck,
boundary-sampled, LoRA'd LLMs. Most were trained during earlier sweeps, judged
as REPLACEMENTS for the recipe, and shelved when they did not beat it.

**None was ever tried as an ADDITIONAL member.** Those are different questions:
a model that is worse alone can still reduce ensemble variance, and risk
carries 0.4 -- more than evidence or factors.

WHAT HAS AND HAS NOT BEEN TESTED ON RISK
  * removing a member -- `risk_member_loo.py`. The friend looked worst but the
    result did NOT transfer, because its val artefact is OOF (fold models)
    while its test artefact is full-data.
  * adding an LLM -- `risk_member_search.py`. Every LoRA family lost and no
    complementarity ratio reached 1.0.
  * adding an ENCODER VARIANT -- never. That is this script.

The candidates here are the same regime as the deployed members: trained on
`artifacts/train_split.csv`, val held out. That is exactly the property the
friend lacked, and the reason its audit was misleading.

THREE GUARDS, EACH PAID FOR TODAY
  LEAK      any run whose name contains `full` trains on val
            (`risk_ql70full_full` read 0.9415 standalone).
            Refused by name, and by an implausible solo score.
  FOLD RUNS `*_f0..f4` cover disjoint slices of val; a single fold is not
            comparable to a full-coverage model. Stitched if the five together
            cover val exactly, skipped otherwise.
  SEARCH    best-of-N on 32 authors will overfit. Selection is nested --
            chosen on FIT authors, scored on HELD-OUT -- and a member added in
            only half the splits is the search fitting its sample.

The candidate joins the ENCODER MEAN, which is where an extra encoder belongs:

    ens = 0.5 * mean(v1_s13, v1_s14, v1_s15, CANDIDATE) + 0.5 * friend
    M   = 0.70 * ens + 0.30 * mean(NLI seeds)
        -> apply_rules(urgency, completed_act) -> route_attempt

Usage:
  python src/risk_member_forward.py
  python src/risk_member_forward.py --rounds 2
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import apply_rules  # noqa: E402
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
CORE = ["runs/risk_v1", "runs/risk_v1_s14", "runs/risk_v1_s15"]
LEAK_WF1 = 0.88


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


def load_partial(path, rows):
    """{row_id: probs} for whatever subset of `rows` this file covers."""
    if not os.path.exists(path):
        return {}
    try:
        d = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return {}
    low = {c.lower(): c for c in d.columns}
    rid = next((low[c] for c in ("row_id", "id") if c in low), None)
    if rid is None:
        return {}
    cols = None
    for pat in ("prob_{}", "p_{}", "{}"):
        c = [low.get(pat.format(x)) for x in LEVELS]
        if all(c):
            cols = c
            break
    if cols is None:
        return {}
    d[rid] = d[rid].astype(str)
    d = d.drop_duplicates(subset=[rid])
    # some val_predictions.csv files carry blank cells; coerce and drop those
    # rows rather than crashing the whole sweep on one bad file
    V = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    want = set(rows)
    out = {}
    for r, v in zip(d[rid], V):
        if r in want and np.isfinite(v).all():
            s = v.sum()
            out[r] = v / s if s > 0 else v
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--nli", default="artifacts/nli_risk_val_probs.csv,"
                                     "artifacts/nli_risk_val_probs2.csv")
    a_.add_argument("--friend", default="artifacts/friend_risk_probs_val.csv")
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--rounds", type=int, default=1)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = np.array([normalize_label(v) for v in va["risk_level"]])
    posts = list(va["post"].astype(str))
    auth = np.asarray([str(x) for x in va["anon_user_id"]])

    def full(path):
        m = load_partial(path, rows)
        if len(m) < len(rows):
            return None
        return np.stack([m[r] for r in rows])

    # THE FRIEND MUST BE IN THE CHAIN. It covers 354 of 372 val rows, and an
    # earlier version silently dropped it when coverage was partial -- which
    # would have measured additions to a DIFFERENT chain than the deployed one
    # (0.5*enc + 0.5*friend). Restrict every row to the friend's coverage
    # instead, so the object being improved is the object that ships.
    fm = load_partial(args.friend, rows)
    if len(fm) < len(rows):
        print(f"  friend covers {len(fm)}/{len(rows)} -- restricting all rows "
              f"to that intersection so the chain matches deployment")
        keep = [i for i, r in enumerate(rows) if r in fm]
        rows = [rows[i] for i in keep]
        gold = gold[keep]
        posts = [posts[i] for i in keep]
        auth = auth[keep]
    fr = np.stack([fm[r] for r in rows]) if fm else None

    core = [full(f"{d}/val_predictions.csv") for d in CORE]
    if any(x is None for x in core):
        raise SystemExit("core v1 seeds missing val coverage")
    nli = [full(p) for p in args.nli.split(",")]
    nli = [x for x in nli if x is not None]
    if not nli:
        raise SystemExit("no NLI val probs")
    N = np.mean(nli, axis=0)

    route = {}
    if os.path.exists(args.route):
        d = pd.read_csv(args.route, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        c = "cot" if "cot" in d.columns else "pred_risk_level"
        route = {r: normalize_label(v) for r, v in zip(d["row_id"], d[c])}

    def chain(enc_list):
        E = np.mean(enc_list, axis=0)
        ens = 0.5 * E + 0.5 * fr if fr is not None else E
        M = 0.70 * ens + 0.30 * N
        lab = [LEVELS[i] for i in M.argmax(1)]
        lab = [normalize_label(apply_rules(p, l)[0]) for p, l in zip(posts, lab)]
        return np.array([("attempt" if (route.get(r) == "attempt"
                                        and x != "indicator") else x)
                         for r, x in zip(rows, lab)])

    base_lab = chain(core)
    print(f"{len(rows)} val posts / {len(pd.unique(auth))} authors")
    print(f"  champion chain wF1 {wf1(gold, base_lab):.4f}"
          + ("" if fr is not None else "   (no friend val coverage)"))

    # ---- candidates -------------------------------------------------------
    print("\n" + "=" * 74)
    print("1. CANDIDATES AND THE LEAK GATE")
    print("=" * 74)
    fams = {}
    for p in sorted(glob.glob("runs/risk_*/val_predictions.csv")):
        run = os.path.basename(os.path.dirname(p))
        if run in {os.path.basename(c) for c in CORE}:
            continue
        m = re.match(r"^(.*)_f\d+$", run)
        fams.setdefault(m.group(1) if m else run, []).append(p)

    cand = {}
    for fam, paths in sorted(fams.items()):
        if "full" in fam.lower():
            print(f"  LEAK  {fam:<32} name says full-data -- refused")
            continue
        if len(paths) > 1:                      # fold family: stitch
            acc = {}
            for p in paths:
                acc.update(load_partial(p, rows))
            if len(acc) < len(rows):
                print(f"  skip  {fam:<32} {len(paths)} folds cover "
                      f"{len(acc)}/{len(rows)}")
                continue
            X = np.stack([acc[r] for r in rows])
            nm = fam + "_xfit"
        else:
            X = full(paths[0])
            if X is None:
                continue
            nm = fam
        s = wf1(gold, chain(core + [X]))
        solo = wf1(gold, np.array(LEVELS)[X.argmax(1)])
        if solo > LEAK_WF1:
            print(f"  LEAK  {nm:<32} solo {solo:.4f} above the ceiling")
            continue
        cand[nm] = X
        print(f"  ok    {nm:<32} solo {solo:.4f}   +chain {s:.4f}")
    print(f"\n  {len(cand)} usable candidates")
    if not cand:
        return

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        subs.append((np.flatnonzero(~hm), np.flatnonzero(hm)))

    lab_cache = {}

    def labels(names):
        key = tuple(sorted(names))
        if key not in lab_cache:
            lab_cache[key] = chain(core + [cand[n] for n in names])
        return lab_cache[key]

    print("\n" + "=" * 74)
    print("2. ADD ONE -- author-resampled, rules + route on top")
    print("=" * 74)
    base = np.array([wf1(gold[hi], base_lab[hi]) for _, hi in subs])
    print(f"  {'add':<32}{'wF1':>9}{'d':>9}{'d comp':>9}{'win':>6}{'LOSS':>6}")
    res = []
    for nm in cand:
        lab = labels([nm])
        v = np.array([wf1(gold[hi], lab[hi]) for _, hi in subs])
        d = v - base
        res.append((float(np.mean(d > 0)), float(d.mean()), nm))
        if np.mean(d > 0) >= 0.60:
            print(f"  {nm:<32}{wf1(gold, lab):>9.4f}{d.mean():>+9.4f}"
                  f"{0.4*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
                  f"{np.mean(d < 0):>6.0%}")
    print("  (only candidates at >=60% win are listed)")

    print("\n" + "=" * 74)
    print("3. NESTED FORWARD SELECTION -- chosen on FIT, scored on HELD-OUT")
    print("=" * 74)
    ds, added = [], []
    for fi, hi in subs:
        inner = fi[: max(len(fi) // 4, 8)]
        chosen = []
        cur = wf1(gold[inner], base_lab[inner])
        for _ in range(args.rounds):
            best, bn = cur, None
            for nm in cand:
                v = wf1(gold[inner], labels(chosen + [nm])[inner])
                if v > best:
                    best, bn = v, nm
            if bn is None:
                break
            chosen.append(bn)
            added.append(bn)
            cur = best
        lab = labels(chosen) if chosen else base_lab
        ds.append(wf1(gold[hi], lab[hi]) - wf1(gold[hi], base_lab[hi]))
    d = np.array(ds)
    print(f"  d risk {d.mean():+.4f}   d comp {0.4*d.mean():+.4f}"
          f"   win {np.mean(d > 0):.0%}   LOSS {np.mean(d < 0):.0%}")
    print("  added how often: " + ", ".join(
        f"{k.replace('risk_','')} {v}" for k, v in
        Counter(added).most_common(6)))

    if res:
        res.sort(reverse=True)
        w, dm, nm = res[0]
        print(f"\n  BEST ADD: {nm}   {0.4*dm:+.4f} composite at {w:.0%} win")
        ok = w >= 0.90 and dm > 0
        print(f"  {'CLEARS' if ok else 'does NOT clear'} the bar.")
        if ok:
            print("  Before shipping: this run needs TEST predictions from the")
            print("  SAME checkpoint, or it repeats the friend's mistake.")


if __name__ == "__main__":
    main()

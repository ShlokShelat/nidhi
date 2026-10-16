"""
risk_nli_weight_sweep.py

Sweeps the NLI blend weight, a single parameter set once and otherwise never
revisited:

    M = 0.70 * [0.5 * mean(3 risk_v1 seeds) + 0.5 * friend] + 0.30 * mean(NLI)

`scripts/rebuild_and_apply_rules.sh` passes `--force-weight 0.30`. That value
predates the current reproduction check against the deployed champion, so it
has not been checked against an instrument that measures the deployed object.

A coarse sweep found that every high-win cell involved REDUCING the friend
member's weight, but that result does not transfer here: the friend's
validation artefact is out-of-fold (fold models) while its test artefact is
full-data, so validation systematically understates the friend's true
contribution, and a weight fitted on that understatement misleads.

**That mismatch does not apply to the NLI weight.** The NLI members
(`risk_nli`, `risk_nli_s2`) are single-split runs trained on
`artifacts/train_split.csv`, and `artifacts/nli_risk_test_probs.csv` comes
from those same checkpoints. Validation and test are the SAME REGIME. The
friend is the only member in the chain with a regime mismatch, and this
sweep leaves its weight untouched at 0.5.

WHAT IT CHECKS
  * the SHAPE of the curve. A broad plateau is a real preference; a single
    spike indicates the grid is fitting to a small number of authors rather
    than a real signal.
  * whether the chosen weight is STABLE across author splits. A wandering
    optimum will not transfer to unseen authors.
  * how many labels actually move, and whether any cross the indicator
    boundary. Crossing it drags the evidence column too, since evidence is
    suppressed whenever risk predicts indicator, so a risk-only patch would
    be invalid.

Judge the result by win rate and loss rate, then divide by the measured
~2.8x validation-to-test attenuation factor before deciding whether a change
earns a slot.

Usage:
  python src/risk_nli_weight_sweep.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import apply_rules  # noqa: E402
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


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
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--w-ours", type=float, default=0.5,
                    help="friend weight, held at the shipped value -- the "
                         "friend is the one member whose val artefact is a "
                         "different regime, so do not move it here")
    a_.add_argument("--splits", type=int, default=400)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold_all = np.array([normalize_label(v) for v in va["risk_level"]])
    posts_all = list(va["post"].astype(str))
    auth_all = np.asarray([str(x) for x in va["anon_user_id"]])

    enc = [probs_of(f"runs/risk_v1{s}/val_predictions.csv", rows)
           for s in ("", "_s14", "_s15")]
    fr = probs_of("artifacts/friend_risk_probs_val.csv", rows)
    nli = [probs_of(p, rows) for p in
           ("artifacts/nli_risk_val_probs.csv",
            "artifacts/nli_risk_val_probs2.csv")]
    enc = [x for x in enc if x is not None]
    nli = [x for x in nli if x is not None]
    if not enc or not nli or fr is None:
        raise SystemExit("missing a member")

    ok = np.ones(len(rows), bool)
    for X in enc + nli + [fr]:
        ok &= np.isfinite(X[:, 0])
    ix = np.flatnonzero(ok)
    rows = [rows[i] for i in ix]
    gold, posts, auth = gold_all[ix], [posts_all[i] for i in ix], auth_all[ix]
    E = np.mean([x[ix] for x in enc], axis=0)
    F = fr[ix]
    N = np.mean([x[ix] for x in nli], axis=0)
    print(f"{len(rows)} val posts / {len(pd.unique(auth))} authors "
          f"(intersection of all members)")

    route = {}
    if os.path.exists(args.route):
        d = pd.read_csv(args.route, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        c = "cot" if "cot" in d.columns else "pred_risk_level"
        route = {r: normalize_label(v) for r, v in zip(d["row_id"], d[c])}

    def chain(wn):
        M = (1 - wn) * (args.w_ours * E + (1 - args.w_ours) * F) + wn * N
        lab = [LEVELS[i] for i in M.argmax(1)]
        lab = [normalize_label(apply_rules(p, l)[0]) for p, l in zip(posts, lab)]
        return np.array([("attempt" if (route.get(r) == "attempt"
                                        and x != "indicator") else x)
                         for r, x in zip(rows, lab)])

    # THE GRID MUST REACH 1.0. A grid that stops short of the edge while the
    # curve is still rising reports its boundary as the optimum, not the true
    # one. w_nli = 1.0 is NLI-ONLY, and `risk_member_loo.py` measured the NLI
    # member solo at 0.8160 against the full chain's 0.7992, so a blend can
    # dilute its strongest member if the grid is not checked all the way to 1.0.
    GRID = np.round(np.arange(0.15, 1.001, 0.025), 3)
    labs = {w: chain(w) for w in GRID}
    base = labs[0.30] if 0.30 in labs else chain(0.30)

    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.splits)]

    print(f"\n  friend weight held at {args.w_ours} (shipped)")
    print(f"  {'w_nli':>7}{'wF1':>9}{'d':>9}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'moved':>7}{'ind cross':>11}")
    for w in GRID:
        lab = labs[w]
        d = np.array([wf1(gold[b], lab[b]) - wf1(gold[b], base[b])
                      for b in boots])
        ch = int((lab != base).sum())
        ic = int(sum(1 for i in range(len(rows))
                     if (base[i] == "indicator") != (lab[i] == "indicator")))
        tag = ("  <- shipped" if abs(w - 0.30) < 1e-9
               else "  <- NLI only" if abs(w - 1.0) < 1e-9 else "")
        if w < 0.30 and w not in (0.15, 0.20, 0.25):
            continue
        print(f"  {w:>7.3f}{wf1(gold, lab):>9.4f}{d.mean():>+9.4f}"
              f"{0.4*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
              f"{np.mean(d < 0):>6.0%}{ch:>7}{ic:>11}{tag}")

    print("\n  Read the SHAPE. A broad plateau of positive win rates is a real")
    print("  preference; a single spike is the grid fitting 32 authors.")

    # nested: weight chosen on FIT authors, scored on HELD-OUT
    ds, picks = [], []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) < 40 or len(hi) < 15:
            continue
        bw = max(GRID, key=lambda w: wf1(gold[fi], labs[w][fi]))
        ds.append(wf1(gold[hi], labs[bw][hi]) - wf1(gold[hi], base[hi]))
        picks.append(bw)
    d = np.array(ds)
    print(f"\n  NESTED (weight chosen on FIT, scored on HELD-OUT):")
    print(f"    d risk {d.mean():+.4f}   d comp {0.4*d.mean():+.4f}   "
          f"win {np.mean(d > 0):.0%}   LOSS {np.mean(d < 0):.0%}")
    print(f"    chosen w {np.mean(picks):.3f} +- {np.std(picks):.3f}   "
          + ", ".join(f"{k}x{v}" for k, v in Counter(picks).most_common(4)))
    print(f"    discounted 2.8x: {0.4*d.mean()/2.8:+.4f} expected on held-out evaluation")
    print("\n  A stable chosen w means the optimum is a property of the blend.")
    print("  `ind cross` > 0 means the evidence column moves too and a")
    print("  risk-only patch would be invalid.")


if __name__ == "__main__":
    main()

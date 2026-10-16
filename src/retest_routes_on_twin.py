"""
retest_routes_on_twin.py

Re-judges LLM routing variants on a matched base that includes the full
symbolic layer, rather than on a rules-free proxy.

`route_attempt` is part of the deployed champion. Its siblings were
originally rejected using baselines that are now known to be unsound:
`build_route_attempt.py` measured all three against two baselines, and only
`route_attempt` was positive and consistent in direction on both;
`route_behavior` flips sign between them, and `route_indicator` is
indistinguishable from noise on both.

Neither baseline was the champion. `oof_base` and `champ_oof` both score ~0.786
and, critically, **neither carries the rules+route layer**, which
`champion_oof_twin.py` shows is a substantial contribution (0.7896 -> 0.8294).
A rule measured against a base that is missing the symbolic layer can look
valuable simply by duplicating what the layer already does -- and a rule
whose sign flips between two mismatched bases was never measured at all.

THE MATCHED BASE (from champion_oof_twin.py, 372 val posts / 32 authors):

    M_val = 0.70*(mean risk_v1 seeds) + 0.30*(mean NLI seeds)
            -> apply_rules(urgency, completed_act) -> route_attempt = 0.8294

This script rebuilds that, then adds each routing variant ON TOP and measures
with author resamples. Reading order:

  1. route_attempt re-measured -- it is already IN the base, so this reports
     what REMOVING it costs. If that is not clearly positive, the whole
     premise is wrong and nothing else here matters.
  2. the three siblings, each added to the full base rather than to a
     rules-free one.
  3. unanimous variants, where two LLMs must agree -- a stricter variant that
     `build_unanimous_route.py` found improves further over route_attempt
     alone.

WHAT COUNTS AS A PASS:
  * fix/break ratio, not the delta -- a rule that moves many posts at a poor
    ratio can carry the same raw delta as one that moves few posts cleanly,
    but only the latter is likely to generalize.
  * win rate over author resamples, not the confidence interval of the mean
    delta, which understates the per-draw variance a single held-out
    evaluation draw samples from.
  * routing INTO or OUT OF indicator also moves S2, because evidence is
    suppressed exactly on indicator posts. Those rows are flagged; a rule that
    touches them is not a risk-only change and its risk delta is not its
    composite delta.

Usage:
  python src/champion_oof_twin.py        # writes artifacts/champion_val_probs.csv
  python src/retest_routes_on_twin.py
"""

from __future__ import annotations

import argparse
import os
import sys

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


def read_calls(path, rows):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    c = "cot" if "cot" in d.columns else (
        "pred_risk_level" if "pred_risk_level" in d.columns else None)
    if c is None:
        return None
    m = {r: normalize_label(v) for r, v in zip(d["row_id"], d[c])}
    if sum(1 for r in rows if r in m) < len(rows):
        print(f"    {os.path.basename(path)}: covers "
              f"{sum(1 for r in rows if r in m)}/{len(rows)} -- skipped")
        return None
    return m


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--qwen72", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--llama70", default="artifacts/cot_soft_llama_oof.csv")
    a_.add_argument("--splits", type=int, default=400)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = np.array([normalize_label(v) for v in va["risk_level"]])
    posts = list(va["post"].astype(str))
    auth = np.array([str(x) for x in va["anon_user_id"]])

    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    M = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)

    q = read_calls(args.qwen72, rows)
    la = read_calls(args.llama70, rows)
    print(f"val {len(rows)} posts / {len(pd.unique(auth))} authors   "
          f"qwen72 {'OK' if q else 'MISSING'}   "
          f"llama70 {'OK' if la else 'MISSING'}")
    if q is None:
        raise SystemExit("no qwen72 calls covering val -- nothing to re-judge")

    # pre-route: argmax + the two symbolic rules. route variants stack on this.
    pre = [LEVELS[i] for i in M.argmax(1)]
    pre = np.array([normalize_label(apply_rules(p, l)[0])
                    for p, l in zip(posts, pre)])

    def route(lab, calls, to, skip_indicator=True):
        return np.array([(to if (calls.get(r) == to
                                 and not (skip_indicator and l == "indicator"))
                          else l) for r, l in zip(rows, lab)])

    base = route(pre, q, "attempt")           # the DEPLOYED base
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.splits)]

    def judge(nm, new, ref=None):
        ref = base if ref is None else ref
        ch = new != ref
        if not ch.sum():
            print(f"  {nm:<30}{0:>7}")
            return
        fix = int(((new == gold) & (ref != gold) & ch).sum())
        brk = int(((ref == gold) & (new != gold) & ch).sum())
        d = np.array([wf1(gold[b], new[b]) - wf1(gold[b], ref[b])
                      for b in boots])
        ind = int(sum(1 for i in np.flatnonzero(ch)
                      if ref[i] == "indicator" or new[i] == "indicator"))
        print(f"  {nm:<30}{int(ch.sum()):>7}{fix:>6}{brk:>6}"
              f"{fix/max(brk,1):>7.2f}{wf1(gold, new):>9.4f}"
              f"{0.4*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
              f"{0.4*d.std():>8.4f}{ind:>6}")

    print(f"\n  pre-route (argmax + rules)   {wf1(gold, pre):.4f}")
    print(f"  DEPLOYED base (+route_attempt) {wf1(gold, base):.4f}\n")
    print(f"  {'variant':<30}{'moved':>7}{'fix':>6}{'brk':>6}{'f/b':>7}"
          f"{'wF1':>9}{'d comp':>9}{'win':>6}{'sd':>8}{'ind':>6}")

    print("  -- 1. is the deployed route still right? (vs NO routing) --")
    judge("route_attempt (the base)", base, ref=pre)

    print("  -- 2. siblings, added to the DEPLOYED base --")
    for c in ("behavior", "ideation", "indicator"):
        judge(f"+ route_{c}", route(base, q, c))

    if la is not None:
        print("  -- 3. unanimous: qwen72 AND llama70 agree --")
        una = {r: (q[r] if q.get(r) == la.get(r) else None) for r in rows}
        for c in ("attempt", "behavior", "ideation", "indicator"):
            calls = {r: v for r, v in una.items() if v == c}
            judge(f"+ unanimous_{c}", route(base, calls, c))
        print("  -- 3b. unanimous route_attempt INSTEAD of the solo one --")
        judge("unanimous_attempt (replaces)",
              route(pre, {r: v for r, v in una.items() if v == "attempt"},
                    "attempt"), ref=base)

    print("\n  f/b is the fix/break ratio -- route_attempt won at ~3:1 and")
    print("  bag3 lost at ~1:1 on the same number of posts. Read it before")
    print("  the delta. `ind` counts rows crossing the indicator boundary:")
    print("  those move S2 as well, so their risk delta is NOT the composite.")
    print("  Only 32 authors here, so resampling is coarse -- treat a win rate")
    print("  under ~90% with a f/b under 2:1 as untested, not as a result.")


if __name__ == "__main__":
    main()

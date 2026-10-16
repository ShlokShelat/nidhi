"""
score_evidence_variants.py

SCORE EVIDENCE VARIANT FILES AGAINST GOLD, IN THE SCORER'S OWN AGGREGATION.

`evidence_vote_test.py --split val` writes an evidence column per cluster
representative (`modal` ships; `capsafe`, `short`, `median` are the untried
alternatives). This scores them side by side.

WHY A SEPARATE SCORER. Two aggregation traps matter here:
  * phrase_f1() pools MICRO. The scorer MACRO-averages over POSTS, and on val
    the two differ by ~0.07 -- a delta quoted in micro is not the delta that
    determines the official score.
  * an earlier harness suppressed on GOLD risk and defaulted to micro;
    correcting that moved evidence by 0.03. Every representative comparison
    made before that fix needs re-running.

WHY THE REPRESENTATIVE MATTERS AT ALL. The scorer matches when either span
contains the other AND len(pred) <= 3*len(gold). A prediction that is a
SUBSTRING of gold satisfies the cap automatically, so **shorter is strictly
safer against boundary failure** -- it only costs you when the short form falls
outside the gold clause entirely. `evidence_failure_anatomy.py` measured the
shipped decode at 1.855 spans/post against gold's ~1.66, with PARTIAL boundary
errors outnumbering TOO_LONG 14:1.

Reported per variant: MACRO (the scorer), micro (for continuity with older
logs), spans/post, and the author-resampled delta against `modal`. Judge by
WIN RATE and the share of resamples that LOSE, never the CI of the mean.

Usage:
  python src/score_evidence_variants.py
  python src/score_evidence_variants.py --files artifacts/tmp_ev_*.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     parse_evidence, phrase_f1)


def post_f1(gold, pred):
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    adj = [[j for j, g in enumerate(gold) if _span_pair_matches(p, g)]
           for p in pred]
    m = _max_bipartite_matching(adj, len(gold))
    pr, rc = m / len(pred), m / len(gold)
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--files", default="artifacts/tmp_ev_*.csv")
    a_.add_argument("--ref", default="modal")
    a_.add_argument("--boot", type=int, default=400)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])

    variants = {}
    for p in sorted(glob.glob(args.files)):
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        col = next((c for c in ("evidence", "pred_evidence") if c in d.columns),
                   None)
        if col is None:
            continue
        m = dict(zip(d["row_id"], d[col]))
        miss = sum(1 for r in rows if r not in m)
        nm = os.path.basename(p).replace("tmp_ev_", "").replace(".csv", "")
        variants[nm] = [parse_evidence(m.get(r, "")) for r in rows]
        if miss:
            print(f"  note: {nm} missing {miss}/{len(rows)} rows "
                  f"(treated as empty)")
    if not variants:
        raise SystemExit(f"no evidence files matched {args.files}")

    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.boot)]

    per = {k: np.array([post_f1(g, p) for g, p in zip(gold, v)])
           for k, v in variants.items()}
    ref = args.ref if args.ref in per else sorted(per)[0]
    nz = [i for i, g in enumerate(gold) if g]

    print(f"\n{len(rows)} val posts / {len(ua)} authors   "
          f"reference = {ref}\n")
    print(f"  {'variant':<12}{'MACRO':>9}{'micro':>8}{'spans/post':>12}"
          f"{'non-empty F1':>14}{'d MACRO':>10}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}")
    for k in sorted(per):
        v = variants[k]
        sp = np.mean([len(v[i]) for i in nz])
        mic = phrase_f1(gold, v).f1
        d = np.array([per[k][b].mean() - per[ref][b].mean() for b in boots])
        tag = "  <- ships" if k == ref else ""
        print(f"  {k:<12}{per[k].mean():>9.4f}{mic:>8.4f}{sp:>12.2f}"
              f"{per[k][nz].mean():>14.4f}{d.mean():>+10.4f}"
              f"{0.3*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
              f"{np.mean(d < 0):>6.0%}{tag}")

    print("\n  gold spans/post on non-empty posts: "
          f"{np.mean([len(gold[i]) for i in nz]):.2f}")
    print("  A variant that moves spans/post toward gold AND raises MACRO is")
    print("  the real thing. One that raises MACRO while drifting further away")
    print("  is winning for some other reason -- find out what before shipping.")

    best = max((k for k in per if k != ref), key=lambda k: per[k].mean(),
               default=None)
    if best:
        d = np.array([per[best][b].mean() - per[ref][b].mean() for b in boots])
        ok = np.mean(d > 0) >= 0.90 and d.mean() > 0
        print(f"\n  BEST vs {ref}: {best}  {0.3*d.mean():+.4f} composite at "
              f"{np.mean(d > 0):.0%} win")
        print(f"  {'CLEARS' if ok else 'does NOT clear'} the 90% bar.")
        print("  NOTE: suppression here is driven by oof_base's risk column,")
        print("  not the champion's, so the LEVELS are not the champion's.")
        print("  All variants share that suppression, so the DELTAS between")
        print("  them are clean -- read those, not the absolute numbers.")


if __name__ == "__main__":
    main()

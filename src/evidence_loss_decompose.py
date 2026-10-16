"""
evidence_loss_decompose.py

Decompose the missing evidence macro-F1 into disjoint buckets.

Macro phrase-F1 gives every post equal weight and scores an empty-vs-empty
post 1.0, so the per-post loss (1 - F1) partitions exactly four ways:

  A  gold EMPTY, pred EMPTY        -> 1.0, no loss
  B  gold EMPTY, pred NON-EMPTY    -> 0.0, loss 1.0 each   (under-suppression)
  C  gold NON-EMPTY, pred EMPTY    -> 0.0, loss 1.0 each   (over-suppression)
  D  both NON-EMPTY                -> partial, loss 1-F1   (span quality)

Which bucket holds the mass identifies the mechanism, and the three answers
point at unrelated fixes:

  * B dominates -> the suppression rule fires too rarely. Gold-empty tracks
    the indicator label, so this is a risk-classification problem expressed
    in evidence terms.
  * C dominates -> the same rule fires too often, suppressing posts that
    carry gold spans.
  * D dominates -> the spans themselves are wrong, which only a change to
    candidate generation or selection can move.

Inside D the loss is split further into PRECISION (an emitted span matched
nothing) and RECALL (a gold span went unmatched), because those point at
opposite fixes.

SECURITY: local files only, no network, prints aggregates and counts -- no
post text.

Usage (CPU, seconds):
  python src/evidence_loss_decompose.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1  # noqa: E402

RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--min-conf", type=float, default=0.65)
    args = ap.parse_args()

    out = "artifacts/_decomp_val.csv"
    cmd = [sys.executable, "src/evidence_vote_test.py",
           "--runs", ",".join(RC7), "--dir", args.dir,
           "--split", "val", "--test", args.val, "--risk-from", args.risk,
           "--k", str(args.k), "--cluster", "--rep", "modal",
           "--cap", str(args.cap),
           "--min-conf-override", str(args.min_conf), "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-800:], r.stderr[-800:])
        raise SystemExit("vote failed")

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    pv = pd.read_csv(out, keep_default_na=False)
    pv["row_id"] = pv["row_id"].astype(str)
    pm = dict(zip(pv["row_id"], pv[pv.columns[-1]]))

    rows = []
    for r_ in va.itertuples():
        g = parse_evidence(r_.evidence)
        p = parse_evidence(pm.get(str(r_.row_id), ""))
        res = phrase_f1([g], [p], average="macro")
        rows.append({"row_id": str(r_.row_id), "ng": len(g), "np": len(p),
                     "f1": float(res.f1), "match": int(res.n_matched)})
    d = pd.DataFrame(rows)
    n = len(d)
    tot_loss = float((1.0 - d.f1).sum())
    print(f"{n} val posts   macro {d.f1.mean():.4f}   "
          f"total loss {tot_loss:.1f} posts-worth\n")

    A = d[(d.ng == 0) & (d.np == 0)]
    B = d[(d.ng == 0) & (d.np > 0)]
    C = d[(d.ng > 0) & (d.np == 0)]
    D = d[(d.ng > 0) & (d.np > 0)]
    print(f"  {'bucket':<44}{'posts':>7}{'loss':>9}{'share':>8}")
    for name, sub in (("A  gold empty, pred empty      (free 1.0)", A),
                      ("B  gold EMPTY, pred non-empty  (under-suppress)", B),
                      ("C  gold non-empty, pred EMPTY  (over-suppress)", C),
                      ("D  both non-empty              (span quality)", D)):
        L = float((1.0 - sub.f1).sum())
        print(f"  {name:<44}{len(sub):>7}{L:>9.1f}"
              f"{100*L/max(tot_loss,1e-9):>7.0f}%")

    if len(D):
        gp = int(D.ng.sum())
        pp = int(D.np.sum())
        mm = int(D.match.sum())
        print(f"\n  inside D: {gp} gold spans, {pp} predicted, {mm} matched")
        print(f"    precision {mm/max(pp,1):.3f}   recall {mm/max(gp,1):.3f}")
        print(f"    unmatched PREDICTIONS (precision loss): {pp-mm}")
        print(f"    unmatched GOLD        (recall loss):    {gp-mm}")
        perfect = int((D.f1 >= 0.999).sum())
        zero = int((D.f1 <= 0.001).sum())
        print(f"    posts scoring 1.0: {perfect}   scoring 0.0: {zero}   "
              f"partial: {len(D)-perfect-zero}")

    print("\n  Whichever bucket holds the loss names the mechanism.")
    print("  B or C large -> the SUPPRESSION rule, which keys off the")
    print("  predicted risk label. D large -> span quality.")


if __name__ == "__main__":
    main()

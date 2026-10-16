"""
verify_component_artifacts.py

IS THIS COMPONENT'S VAL ARTEFACT THE SAME OBJECT AS ITS TEST ARTEFACT?

Run this BEFORE any blend-weight change. Skipping it risks concrete failure
modes such as:

  * a component fitted on one artefact but applied through a different,
    better model at inference time
  * a val artefact built from OOF folds paired against a test artefact built
    from a full-data model, so the two are not the same object
  * a val file that is one model duplicated into two "seed" slots while the
    test file is genuinely two different models averaged

The 378/378 champion gate does not catch any of these. It validates the chain
at the SHIPPED weights, where a low-weighted term barely moves the argmax and
many different vectors reproduce the same labels. Move the weight and the
discrepancy becomes the deciding signal.

WHAT IT COMPARES, per component, between its val and test probability files:

  duplication      are two supposedly-independent "seed" files byte-identical?
                   `nli_risk_val_probs.csv` and `..._probs2.csv` are, so the
                   val NLI is one model counted twice while test may be an
                   average of two.
  confidence       mean max-prob and entropy. A component that is systematically
                   more confident on one split than the other is not the same
                   object, or is not seeing comparable data.
  argmax mix       the predicted class distribution. Val and test are both
                   author-disjoint samples of the same corpus, so a large shift
                   is a property of the MODEL, not the posts. The NLI files
                   differ by 5.7 points on `behavior`, which is large enough
                   to change which blend weight is safe to ship.
  coverage         does each file cover every row of its split? The friend
                   covers 354 of 372 val rows, which silently drops members.

A component failing any of these can still be SHIPPED at its current weight --
the gate covers that. It must not have its weight CHANGED until the pairing is
understood.

Usage:
  python src/verify_component_artifacts.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS  # noqa: E402

# component -> (val files, test files)
COMPONENTS = {
    "encoder v1 seeds": (
        ["runs/risk_v1/val_predictions.csv",
         "runs/risk_v1_s14/val_predictions.csv",
         "runs/risk_v1_s15/val_predictions.csv"],
        ["artifacts/ensembled_risk.csv"]),
    "friend": (["artifacts/friend_risk_probs_val.csv"],
               ["artifacts/friend_risk_probs.csv"]),
    "NLI": (["artifacts/nli_risk_val_probs.csv",
             "artifacts/nli_risk_val_probs2.csv"],
            ["artifacts/nli_risk_test_probs.csv"]),
}


def load(path):
    if not os.path.exists(path):
        return None, None
    d = pd.read_csv(path, keep_default_na=False)
    low = {c.lower(): c for c in d.columns}
    rid = next((low[c] for c in ("row_id", "id") if c in low), None)
    cols = None
    for pat in ("p_{}", "prob_{}", "{}"):
        c = [low.get(pat.format(x)) for x in RISK_LEVELS]
        if all(c):
            cols = c
            break
    if rid is None or cols is None:
        return None, None
    X = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    keep = np.isfinite(X).all(1) & (np.nansum(X, 1) > 0)
    X = X[keep] / np.nansum(X[keep], 1, keepdims=True)
    return d[rid].astype(str).to_numpy()[keep], X


def stats(X):
    return (X.max(1).mean(),
            float((-(X * np.log(X + 1e-9)).sum(1)).mean()),
            np.bincount(X.argmax(1), minlength=4) / len(X))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--test", default="data/test.csv")
    args = a_.parse_args()
    nval = len(pd.read_csv(args.val))
    ntest = len(pd.read_csv(args.test))
    print(f"val {nval} posts, test {ntest} posts\n")

    for name, (vps, tps) in COMPONENTS.items():
        print("=" * 74)
        print(name)
        print("=" * 74)
        loaded = {}
        for p in vps + tps:
            r, X = load(p)
            if X is None:
                print(f"  MISSING/unparsed {p}")
                continue
            loaded[p] = (r, X)
            split = "val" if p in vps else "test"
            n = nval if split == "val" else ntest
            mp, en, mix = stats(X)
            cov = f"{len(r)}/{n}"
            flag = "  <- INCOMPLETE" if len(r) < n else ""
            print(f"  [{split:<4}] {os.path.basename(p):<34} cov {cov:<9}"
                  f" maxp {mp:.3f}  H {en:.3f}  mix "
                  f"[{' '.join(f'{x:.2f}' for x in mix)}]{flag}")

        # duplication within a split
        for split, paths in (("val", vps), ("test", tps)):
            got = [p for p in paths if p in loaded]
            for a, b in itertools.combinations(got, 2):
                ra, Xa = loaded[a]
                rb, Xb = loaded[b]
                if len(ra) == len(rb) and np.allclose(Xa, Xb):
                    print(f"  !! {os.path.basename(a)} and "
                          f"{os.path.basename(b)} are IDENTICAL -- this "
                          f"{split} 'ensemble' is one model counted twice")

        vg = [loaded[p][1] for p in vps if p in loaded]
        tg = [loaded[p][1] for p in tps if p in loaded]
        if vg and tg:
            V = np.mean(vg, axis=0)
            T = np.mean(tg, axis=0)
            mv, hv, xv = stats(V)
            mt, ht, xt = stats(T)
            dmix = float(np.abs(xv - xt).max())
            print(f"\n  pooled val  maxp {mv:.3f}  H {hv:.3f}  "
                  f"mix [{' '.join(f'{x:.2f}' for x in xv)}]")
            print(f"  pooled test maxp {mt:.3f}  H {ht:.3f}  "
                  f"mix [{' '.join(f'{x:.2f}' for x in xt)}]")
            verdict = []
            if abs(mv - mt) > 0.02:
                verdict.append(f"confidence differs by {abs(mv-mt):.3f}")
            if dmix > 0.04:
                verdict.append(f"argmax mix differs by {dmix:.3f}")
            if len(vg) != len(tg):
                verdict.append(f"{len(vg)} val file(s) vs {len(tg)} test")
            if verdict:
                print(f"  !! SUSPECT: {'; '.join(verdict)}")
                print("     Safe to SHIP at the current weight (the champion "
                      "gate covers that).")
                print("     Do NOT change this component's weight until the "
                      "pairing is understood.")
            else:
                print("  OK -- val and test look like the same object; a "
                      "weight change is measurable.")
        print()


if __name__ == "__main__":
    main()

"""
find_champion_factors.py

WHICH ARTEFACT + DECODE PRODUCED THE CHAMPION'S FACTOR COLUMN?

Factors are where the score actually is. S1 is 0.8047 and S2 is 0.6774, and
prior analysis already found that factors account for the whole gap to the
top scorer. But every factor decision made so far was judged against
`fac_oof` = 0.6399 --
which is BELOW the entire range of arms available.

**A baseline that scores worse than everything it is used to rank is not a
baseline, it is a different model.** That is precisely the shape of a risk
defect found previously: `champion_oof.csv` at 0.786 versus the deployed
0.826, where the missing piece turned out to be a whole symbolic layer.
A correction can look good offline purely because it duplicates work the
deployed pipeline already does, and that duplication is invisible unless the
shipped column is reproduced first.

Same remedy, same order:
  1. reproduce the SHIPPED factor column exactly from files on disk
  2. rebuild its validation twin from the same members
  3. re-judge every shelved factor lane on that twin, not on fac_oof

This script is step 1. It tries each candidate probability artefact under
several decodes and reports EXACT-set agreement per post against
`NIDHI_factfit.csv`:

  argmax-free thresholding is the only decode factors have -- a global
  threshold, a per-label threshold vector (artifacts/factor_thresholds_*.npy),
  and the shrunk-prevalence decode that [[factor-decode-calibration-bug]]
  measured as a gain. Whichever reproduces 378/378 IS the shipped decode, and
  its parameters are then known rather than guessed.

Reading the output: `exact` is posts whose FULL factor set matches. `jacc` is
mean Jaccard over posts, which degrades gracefully -- a file at exact 40% but
jacc 0.95 is the right model with the wrong threshold, while exact 40% /
jacc 0.55 is the wrong model.

Usage:
  python src/find_champion_factors.py
  python src/find_champion_factors.py --sub submissions/NIDHI_factfit.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402


def canon(name):
    return str(name).strip().lower().replace("_", " ")


CANON = {canon(f): f for f in FACTOR_TAXONOMY}


def as_set(cell):
    return frozenset(canon(x) for x in parse_factors(cell) if canon(x))


def load_factor_probs(path):
    """Return (row_ids, Pi[n,K], labels) for any file carrying one column per
    factor, or (row_ids, None, sets) for a file carrying a factors column."""
    try:
        d = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return None
    low = {canon(c): c for c in d.columns}
    rid = next((d.columns[i] for i, c in enumerate(d.columns)
                if canon(c) in ("row id", "row_id", "id")), None)
    if rid is None:
        return None
    d[rid] = d[rid].astype(str)
    d = d.drop_duplicates(subset=[rid])
    cols, labs = [], []
    for f in FACTOR_TAXONOMY:
        c = low.get(canon(f)) or low.get(canon("prob_" + f)) \
            or low.get(canon("p_" + f))
        if c is not None:
            cols.append(c)
            labs.append(f)
    if len(cols) >= max(3, len(FACTOR_TAXONOMY) // 2):
        X = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if np.nanmax(X) <= 1.0001:
            return d[rid].to_numpy(), np.nan_to_num(X), labs
    fc = next((low[c] for c in ("factors", "pred factors", "factor")
               if c in low), None)
    if fc is not None:
        return d[rid].to_numpy(), None, [as_set(v) for v in d[fc]]
    return None


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--sub", default="submissions/NIDHI_factfit.csv")
    a_.add_argument("--glob", default="artifacts/*factor*.csv,"
                                      "artifacts/*factors*.csv")
    a_.add_argument("--thr-glob", default="artifacts/factor_thresholds*.npy")
    args = a_.parse_args()

    sub = pd.read_csv(args.sub, keep_default_na=False)
    sub["row_id"] = sub["row_id"].astype(str)
    fcol = "factors" if "factors" in sub.columns else "pred_factors"
    tgt = {r: as_set(v) for r, v in zip(sub["row_id"], sub[fcol])}
    n_lab = sum(len(v) for v in tgt.values())
    print(f"target: {args.sub}   {len(tgt)} rows, {n_lab} factor labels "
          f"({n_lab/len(tgt):.2f} per post)")
    empt = sum(1 for v in tgt.values() if not v)
    print(f"  {empt} posts with NO factors\n")

    thrs = {"global 0.5": np.full(len(FACTOR_TAXONOMY), 0.5)}
    for p in glob.glob(args.thr_glob):
        try:
            v = np.load(p)
            if v.shape[-1] == len(FACTOR_TAXONOMY):
                thrs[os.path.basename(p)] = v.reshape(-1)
        except Exception:
            pass
    for t in (0.3, 0.35, 0.4, 0.45, 0.55, 0.6):
        thrs[f"global {t}"] = np.full(len(FACTOR_TAXONOMY), t)
    print(f"decodes: {len(thrs)}  ({', '.join(list(thrs)[:6])}...)\n")

    cands = sorted({g for pat in args.glob.split(",")
                    for g in glob.glob(pat.strip())})
    print(f"  {'file':<44}{'rows':>6}{'best decode':<22}{'exact':>7}"
          f"{'jacc':>7}")
    best = []
    for p in cands:
        got = load_factor_probs(p)
        if got is None:
            continue
        rids, X, labs = got
        keep = [i for i, r in enumerate(rids) if r in tgt]
        if len(keep) < 0.9 * len(tgt):
            continue
        rids = [rids[i] for i in keep]

        def score(sets):
            ex = sum(1 for r, s in zip(rids, sets) if s == tgt[r])
            jc = float(np.mean([
                (len(s & tgt[r]) / max(len(s | tgt[r]), 1)) if (s or tgt[r])
                else 1.0 for r, s in zip(rids, sets)]))
            return ex, jc

        if X is None:
            sets = [labs[i] for i in keep]
            ex, jc = score(sets)
            rec = ("as-written", ex, jc)
        else:
            Xk = X[keep]
            idx = [FACTOR_TAXONOMY.index(l) for l in labs]
            rec = ("", -1, -1.0)
            for nm, t in thrs.items():
                tt = t[idx]
                sets = [frozenset(canon(labs[j]) for j in range(len(labs))
                                  if Xk[i, j] >= tt[j])
                        for i in range(len(Xk))]
                ex, jc = score(sets)
                if ex > rec[1]:
                    rec = (nm, ex, jc)
        mark = "  <== EXACT" if rec[1] == len(rids) else ""
        print(f"  {os.path.basename(p):<44}{len(rids):>6}{rec[0]:<22}"
              f"{rec[1]:>7}{rec[2]:>7.3f}{mark}")
        best.append((rec[1], rec[2], p, rec[0], len(rids)))

    if not best:
        print("\nno candidate factor artefacts parsed")
        return
    best.sort(reverse=True)
    ex, jc, p, dec, n = best[0]
    print("\n" + "=" * 78)
    if ex == n:
        print(f"EXACT: {os.path.basename(p)} under `{dec}` reproduces all {n} "
              f"factor sets.")
        print("  -> build its validation twin next, then re-judge every")
        print("     shelved factor lane on THAT, not on fac_oof (0.6399).")
    else:
        print(f"NO EXACT MATCH. Best {os.path.basename(p)} under `{dec}`: "
              f"{ex}/{n} exact, Jaccard {jc:.3f}.")
        if jc > 0.9:
            print("  High Jaccard with low exact = RIGHT MODEL, WRONG "
                  "THRESHOLD. Sweep finer around the best decode; the shipped")
            print("  column probably used a per-label vector not on disk.")
        else:
            print("  Low Jaccard = wrong model. The shipped factors came from")
            print("  an artefact not in this glob, or through a decode with a")
            print("  step this script does not model (shrunk prevalence,")
            print("  a rank rule, or an LLM override).")
        print("  Do NOT judge factor changes until this reads EXACT -- that is")
        print("  the mistake that has cost risk submissions before.")


if __name__ == "__main__":
    main()

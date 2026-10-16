"""
conditioning_ablation.py

DOES CONDITIONING THE SPAN TAGGER ON THE RISK LEVEL ACTUALLY HELP?

The evidence subtask is claimed to be conditionally dependent on the risk
subtask: the clause that establishes `behavior` is not the clause that
establishes `ideation`, so a tagger told which level it is arguing for should
select better than one left to guess. That claim is load-bearing and is usually
asserted rather than measured.

This measures it directly. Several taggers were trained in matched pairs that
differ ONLY in whether the risk level was prepended to the post, and their
out-of-fold candidate dumps are on disk. Decoding both members of a pair the
same way and scoring both against the same gold spans isolates the
conditioning.

WHAT IS COMPARED. Runs are paired by name: a run whose tag contains `rc`
(risk-conditioned) is matched to the run with the same tag minus that marker.
Both are decoded identically -- each run's own confidence gate, then the
shipped cap of three spans per post -- so the only difference between the two
columns is the conditioning.

HOW IT IS JUDGED. Phrase-F1 under the official matching rules, computed per
post and averaged, then resampled over AUTHOR-DISJOINT partitions. The headline
is the WIN RATE across partitions and the per-split standard deviation, not a
confidence interval on the mean: that interval narrows with the number of
resamples, which is a quantity we choose, so it can be made to exclude zero at
a win rate barely above chance.

SECURITY: reads local files only and prints aggregate statistics. No post text
and no span text is emitted.

Usage:
  python src/conditioning_ablation.py --dir artifacts/evidence_oof
  python src/conditioning_ablation.py --dir artifacts/evidence_oof --splits 400
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1  # noqa: E402

RC_MARKERS = ["_rc", "rc_", "cond", "evcond"]


def decode(path, cap):
    """One candidate dump -> {row_id: [spans]}, using that run's own gate."""
    d = pd.read_csv(path, keep_default_na=False)
    if "row_id" not in d.columns or "span" not in d.columns:
        return None
    d["row_id"] = d["row_id"].astype(str)
    if "conf" in d.columns:
        if "min_conf" in d.columns:
            d = d[d["conf"].astype(float) >= d["min_conf"].astype(float)]
        d = d.sort_values("conf", ascending=False)
    out = {}
    for rid, grp in d.groupby("row_id", sort=False):
        out[rid] = [str(s) for s in grp["span"].head(cap) if str(s).strip()]
    return out


def score_per_post(pred, rows, gold):
    """Per-post phrase-F1 for every row, absent prediction meaning empty."""
    f = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        g = gold[i]
        p = pred.get(r, [])
        f[i] = float(phrase_f1([g], [p], average="macro").f1)
    return f


def base_tag(name):
    """Strip the conditioning marker so a pair can be recognised."""
    t = name.replace("_oof_candidates.csv", "").replace(".csv", "")
    for m in ("_rc", "rc_"):
        if m in t:
            return t.replace(m, "_").strip("_").replace("__", "_")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/evidence_oof")
    ap.add_argument("--labelled", default="artifacts/full_train.csv")
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--splits", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    lab = pd.read_csv(args.labelled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    rows = lab["row_id"].tolist()
    gold = [parse_evidence(str(c)) for c in lab["evidence"]]
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    authors = np.array([str(a) for a in lab[acol]])
    print(f"{len(rows)} labelled posts, {len(set(authors))} authors\n")

    files = sorted(f for f in os.listdir(args.dir)
                   if f.endswith("candidates.csv") and "oof" in f)
    runs = {}
    for f in files:
        dec = decode(os.path.join(args.dir, f), args.cap)
        if dec:
            runs[f] = dec

    # find matched pairs: one member carries a conditioning marker
    pairs = []
    for f in runs:
        if not any(m in f for m in RC_MARKERS):
            continue
        bt = base_tag(f)
        for g in runs:
            if g == f or any(m in g for m in RC_MARKERS):
                continue
            if base_tag(g) == bt:
                pairs.append((g, f))
                break

    if not pairs:
        print("No matched conditioned/unconditioned pair found in "
              f"{args.dir}.")
        print("Files present:")
        for f in sorted(runs):
            print(f"    {f}   {len(runs[f])} posts")
        print("\nPair them by hand with --dir pointing at a directory that "
              "holds both members, or rename so the conditioned run differs "
              "only by an `_rc` marker.")
        return

    rng = np.random.default_rng(args.seed)
    ua = np.array(sorted(set(authors)))

    print("=" * 76)
    print("RISK-CONDITIONED vs UNCONDITIONED, SAME TAGGER, SAME DECODE")
    print("=" * 76)
    for base, cond in pairs:
        fb = score_per_post(runs[base], rows, gold)
        fc = score_per_post(runs[cond], rows, gold)
        cov_b = sum(1 for r in rows if runs[base].get(r))
        cov_c = sum(1 for r in rows if runs[cond].get(r))
        diff = int((fb != fc).sum())

        print(f"\n  {base_tag(cond)}")
        print(f"    uncond {base[:34]:<36} macro {fb.mean():.4f}  "
              f"coverage {100*cov_b/len(rows):.0f}%")
        print(f"    cond   {cond[:34]:<36} macro {fc.mean():.4f}  "
              f"coverage {100*cov_c/len(rows):.0f}%")
        print(f"    full-sample delta {fc.mean()-fb.mean():+.4f}   "
              f"posts differing {diff}")

        d = []
        for _ in range(args.splits):
            h = set(rng.choice(ua, size=max(1, len(ua) // 3), replace=False))
            mk = np.array([a in h for a in authors])
            if mk.sum() >= 5:
                d.append(fc[mk].mean() - fb[mk].mean())
        if d:
            d = np.array(d)
            w = float((d > 0).mean())
            print(f"    held-out  mean {d.mean():+.4f}  sd {d.std():.4f}  "
                  f"WIN {100*w:.0f}%  over {len(d)} author splits")
            print(f"    composite estimate {0.30*(fc.mean()-fb.mean()):+.5f} "
                  f"at the 0.30 evidence weight")

    print("\n" + "=" * 76)
    print("HOW TO READ THIS")
    print("=" * 76)
    print("  Coverage is reported because a conditioned tagger that simply")
    print("  emits MORE spans is not demonstrating that conditioning helps;")
    print("  compare the two coverage figures before crediting the delta.")
    print("  A claim that conditioning matters needs a high WIN RATE across")
    print("  author-disjoint splits, not merely a positive full-sample mean.")


if __name__ == "__main__":
    main()

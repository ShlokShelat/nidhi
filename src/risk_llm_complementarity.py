"""
risk_llm_complementarity.py

Does the Qwen risk model fix posts the encoder chain gets wrong?

WHY THIS IS THE ONLY QUESTION LEFT IN THE RISK-LLM LANE. Fully replacing the
risk column with the Qwen risk bag was measured and found harmful. A forensic
on that result showed its class MARGINALS were already correct -- 154/118/94/12,
every one inside the
probe-measured 90% intervals (indicator 112-170, ideation 99-142, behavior
69-113, attempt 13-41). It disagreed with the champion on 75 of 378 posts and
was wrong on nearly all of them.

That matters because it rules out the rescue that worked on FACTORS. There, the
failure was thresholding -- 24 independent cutoffs on miscalibrated
probabilities -- and quantile decisions routed around it. Risk is argmax over 4
classes; the only calibration-sensitive quantity is the marginal allocation,
and the marginals were fine. So no decision rule reaches the residual failure,
which is ranking capability.

What was NEVER scored is Qwen as a low-weight MEMBER rather than a replacement.
A member only pays if it fixes posts the whole encoder chain misses. The
risk-architecture audit found the chain's members all tend to fail on the same
posts (38 of 39 confident errors have NEITHER head right), so this measures
whether Qwen breaks that pattern.

FALSIFIER, FIXED BEFORE LOOKING. The blended member must clear the chain by a
clear margin on a strong majority of author partitions. Merely positive is
not enough: the instrument's baseline is a SINGLE-SEED member while the
deployed chain averages three, so it systematically OVERSTATES the value of
adding anything. Below that bar, the risk-LLM lane closes permanently.

Usage:
  python src/risk_llm_complementarity.py
  python src/risk_llm_complementarity.py --weight 0.2 --splits 50
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402


def find_prob_cols(df):
    """Return the 4 class-probability columns, whatever they are named."""
    for pat in ("prob_{}", "{}", "p_{}", "prob{}"):
        cols = [pat.format(c) for c in RISK_LEVELS]
        if all(c in df.columns for c in cols):
            return cols
        cols = [pat.format(c.capitalize()) for c in RISK_LEVELS]
        if all(c in df.columns for c in cols):
            return cols
    return None


def load_probs(path_glob, k=5):
    """Concatenate per-fold OOF prediction files into one probability frame."""
    paths = sorted(glob.glob(path_glob))
    if not paths:
        return None, f"no files matching {path_glob}"
    fr = []
    for p in paths[:k]:
        d = pd.read_csv(p)
        if "row_id" not in d.columns:
            return None, f"{p} has no row_id (columns {list(d.columns)})"
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["row_id"] = d["row_id"].astype(str)
    return d, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="v2_rebuild/runs/risk_llm_*e6_f*/oof_predictions.csv")
    ap.add_argument("--chain", default="artifacts/champion_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--weight", type=float, default=0.2)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    llm, err = load_probs(args.llm)
    if err:
        raise SystemExit(f"LLM OOF: {err}")
    if not os.path.exists(args.chain):
        raise SystemExit(f"chain OOF missing: {args.chain}")
    ch = pd.read_csv(args.chain)
    ch["row_id"] = ch["row_id"].astype(str)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    lc, cc = find_prob_cols(llm), find_prob_cols(ch)
    if lc is None:
        raise SystemExit(f"no prob columns in the LLM OOF: {list(llm.columns)}")
    if cc is None:
        raise SystemExit(f"no prob columns in the chain OOF: {list(ch.columns)}")

    rows = sorted(set(llm.row_id) & set(ch.row_id) & set(lab.index))
    if not rows:
        raise SystemExit("no overlapping row_ids")
    L = llm.drop_duplicates("row_id").set_index("row_id").loc[rows, lc].to_numpy(float)
    C = ch.drop_duplicates("row_id").set_index("row_id").loc[rows, cc].to_numpy(float)
    y = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    auth = np.array([lab.loc[r, "anon_user_id"] for r in rows])

    pl = np.array([RISK_LEVELS[i] for i in L.argmax(1)])
    pc = np.array([RISK_LEVELS[i] for i in C.argmax(1)])
    okl, okc = pl == y, pc == y

    print(f"{len(rows)} OOF posts, {pd.unique(auth).size} authors")
    print(f"chain accuracy {okc.mean():.4f}   LLM accuracy {okl.mean():.4f}")
    print(f"chain wF1 {risk_weighted_f1(list(y), list(pc)):.4f}   "
          f"LLM wF1 {risk_weighted_f1(list(y), list(pl)):.4f}")

    fixes = int((okl & ~okc).sum())
    breaks = int((okc & ~okl).sum())
    both_wrong = int((~okl & ~okc).sum())
    print(f"\nCOMPLEMENTARITY")
    print(f"  LLM right, chain wrong   {fixes:>5}   <- the only way a member pays")
    print(f"  chain right, LLM wrong   {breaks:>5}")
    print(f"  both wrong               {both_wrong:>5}")
    print(f"  net                      {fixes - breaks:>+5}")
    print("\nThe audit found the chain's own members nearly all fail on the SAME")
    print("posts (38 of 39 confident errors had neither head right). A net far")
    print("above zero means Qwen breaks that pattern; near or below zero means")
    print("it fails where the chain fails and adds nothing.")

    # blended member, count-matched to the chain's own marginals so the change
    # is pure RANKING -- the device behind the only change that transferred
    B = (1 - args.weight) * C + args.weight * L
    tgt = pd.Series(pc).value_counts().reindex(RISK_LEVELS).fillna(0).astype(int)
    order = np.argsort(-(B.max(1) - np.sort(B, axis=1)[:, -2]))  # by margin
    left = tgt.to_dict()
    pb = np.array([None] * len(rows), dtype=object)
    for i in order:
        for j in np.argsort(-B[i]):
            c = RISK_LEVELS[j]
            if left.get(c, 0) > 0:
                pb[i] = c
                left[c] -= 1
                break
        if pb[i] is None:
            pb[i] = RISK_LEVELS[int(B[i].argmax())]
    pb = np.array([str(x) for x in pb])

    base = risk_weighted_f1(list(y), list(pc))
    blend = risk_weighted_f1(list(y), list(pb))
    print(f"\nCOUNT-MATCHED BLEND at w={args.weight}")
    print(f"  chain {base:.4f} -> blend {blend:.4f}   delta {blend - base:+.4f}")
    print(f"  posts changed {int((pb != pc).sum())}")

    uniq = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    wins, deltas = 0, []
    for _ in range(args.splits):
        pick = set(rng.choice(uniq, size=max(1, len(uniq) // 2), replace=False))
        m = np.array([a in pick for a in auth])
        if m.sum() < 20 or len(set(y[m])) < 2:
            continue
        d = (risk_weighted_f1(list(y[m]), list(pb[m]))
             - risk_weighted_f1(list(y[m]), list(pc[m])))
        deltas.append(d)
        wins += d > 0
    if deltas:
        d = np.array(deltas)
        print(f"\n{len(d)} author partitions: mean {d.mean():+.4f}  "
              f"median {np.median(d):+.4f}  won {100 * wins / len(d):.1f}%")

    ok = (blend - base) > 0.005 and deltas and (wins / len(deltas)) >= 0.70
    print("\n" + "=" * 68)
    if ok:
        print("PASSES. Qwen fixes posts the chain misses. A 3-family count-matched")
        print("MEMBER is worth ~10 GPU-hours and one submission slot.")
    else:
        print("FAILS the pre-stated bar (> +0.005 wF1 AND >= 70% of splits).")
        print("The risk-LLM lane CLOSES PERMANENTLY. Note the baseline here is a")
        print("single-seed member while the deployed chain averages three, so")
        print("this instrument OVERSTATES the value of adding anything -- a")
        print("failure at this bar is conclusive rather than marginal.")
    print("=" * 68)


if __name__ == "__main__":
    main()

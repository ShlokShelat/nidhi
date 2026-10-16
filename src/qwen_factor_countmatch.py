"""
qwen_factor_countmatch.py

Compare the Qwen factor RANKING against the incumbent bag's, with thresholding
removed from the comparison.

Ranking quality and cutoff calibration are separable failures, and any
comparison that applies a probability threshold confounds them. Count matching
removes the cutoff entirely: each rule copies the incumbent's exact per-label
prediction COUNT and changes only WHICH posts get the label. Density, per-label
precision/recall trade and zero-prediction safety are all held fixed, so ranking
is the only moving part.

INPUTS ON DISK. Two zero-/few-shot probes over 1635 OOF posts on 8 labels, with
row_ids:

    artifacts/qwen_rare_probe.npz    probs (1635, 8)
    artifacts/qwen_shot3_probe.npz   probs (1635, 8)

FOUR RULES COMPARED, all zero-parameter:
    incumbent   the bag's ranking (the baseline)
    qwen        Qwen's ranking, incumbent counts
    mean        mean of the two probability columns, incumbent counts
    rank-mean   mean of the two RANKS, incumbent counts -- invariant to either
                model's calibration, so a calibration difference between the two
                cannot drive the result

Reported per label and pooled over the 8, with a paired AUTHOR bootstrap.

Usage:
  python src/qwen_factor_countmatch.py
  python src/qwen_factor_countmatch.py --probe artifacts/qwen_shot3_probe.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_factors  # noqa: E402

D = "artifacts/factor_runs"


def load_arm_oof(arm: str, rows: list[str]) -> np.ndarray | None:
    """Out-of-fold probs for one arm, aligned to `rows`.

    The five fold dumps are DISJOINT -- each holds only that fold's held-out
    posts (330/330/323/323/329), so they CONCATENATE to the 1635-post OOF set.
    Averaging them is a shape error, and would be wrong even if the shapes
    matched: a post is scored out-of-fold by exactly one model.
    """
    fs = sorted(glob.glob(f"{D}/{arm}_cv_oof_f*.npz"))
    if not fs:
        return None
    chunks, ids = [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        if "row_id" not in z.files:
            print(f"  {os.path.basename(f)}: no row_id, cannot align")
            return None
        chunks.append(np.asarray(z["probs"], float))
        ids += [str(x) for x in z["row_id"]]
    P = np.concatenate(chunks, axis=0)
    idx = {}
    for i, r in enumerate(ids):
        idx.setdefault(r, i)          # first fold wins on any duplicate
    absent = [r for r in rows if r not in idx]
    if absent:
        print(f"  {arm}: {len(absent)} probe rows absent from its OOF dumps "
              f"(have {len(idx)}), skipped")
        return None
    return P[[idx[r] for r in rows]]


def f1_at_count(score: np.ndarray, y: np.ndarray, k: int) -> float:
    """Take the top-k posts by score; F1 against y. k fixed = counts matched."""
    if k <= 0:
        return 0.0 if y.sum() else 1.0
    k = min(k, len(score))
    order = np.argsort(-score, kind="stable")
    pred = np.zeros(len(score), bool)
    pred[order[:k]] = True
    tp = int((pred & (y == 1)).sum())
    if tp == 0:
        return 0.0
    pr, rc = tp / k, tp / max(int(y.sum()), 1)
    return 2 * pr * rc / (pr + rc)


def ap(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    if ys.sum() == 0:
        return float("nan")
    csum = np.cumsum(ys)
    prec = csum / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--probe", default="artifacts/qwen_rare_probe.npz")
    ap_.add_argument("--arms", default="mnli,dmnli,cmine,prot,d4")
    ap_.add_argument("--labeled", default="artifacts/full_train.csv")
    ap_.add_argument("--boot", type=int, default=3000)
    ap_.add_argument("--seed", type=int, default=0)
    args = ap_.parse_args()

    z = np.load(args.probe, allow_pickle=True)
    Q = np.asarray(z["probs"], float)
    labels = [str(x) for x in z["factors"]]
    rows = [str(x) for x in z["row_id"]]
    print(f"{args.probe}: {Q.shape[0]} posts x {len(labels)} labels")

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    missing = [r for r in rows if r not in lab.index]
    if missing:
        raise SystemExit(f"{len(missing)} probe rows not in {args.labeled}")
    gold = {f: np.array([f in set(parse_factors(lab.loc[r, "factors"]))
                         for r in rows], int) for f in labels}
    author = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])

    arms = [a.strip() for a in args.arms.split(",")]
    mats, used = [], []
    for a in arms:
        M = load_arm_oof(a, rows)
        if M is None:
            print(f"  arm {a}: no usable OOF dump, skipped")
            continue
        mats.append(M)
        used.append(a)
    if not mats:
        raise SystemExit(
            f"no arm OOF dumps found in {D}/. Expected {D}/<arm>_cv_oof_f*.npz")
    print(f"  incumbent = mean of {len(used)} arms: {used}")

    # taxonomy order of the incumbent matrix must be resolved to probe labels
    from scoring import FACTOR_TAXONOMY  # noqa: E402
    tax = list(FACTOR_TAXONOMY)
    cols = []
    for f in labels:
        if f not in tax:
            raise SystemExit(f"probe label {f!r} not in FACTOR_TAXONOMY")
        cols.append(tax.index(f))
    INC = np.mean(mats, axis=0)[:, cols]

    def ranks(x):
        o = np.argsort(np.argsort(x, kind="stable"), kind="stable")
        return o / max(len(x) - 1, 1)

    rules = {}
    rules["incumbent"] = INC
    rules["qwen"] = Q
    rules["mean"] = (INC + Q) / 2
    rules["rank-mean"] = np.stack(
        [(ranks(INC[:, j]) + ranks(Q[:, j])) / 2 for j in range(len(labels))],
        axis=1)

    # counts are the incumbent's own F1-optimal fire counts, held fixed
    counts = {}
    for j, f in enumerate(labels):
        y = gold[f]
        best_k, best_f1 = 0, -1.0
        for k in range(0, len(rows) + 1, max(1, len(rows) // 200)):
            v = f1_at_count(INC[:, j], y, k)
            if v > best_f1:
                best_f1, best_k = v, k
        counts[f] = best_k

    print(f"\n{'label':<42}{'n':>5}{'k':>5}" +
          "".join(f"{r:>11}" for r in rules) + f"{'AP inc':>9}{'AP qwen':>9}")
    print("-" * 118)
    per = {r: np.zeros(len(labels)) for r in rules}
    for j, f in enumerate(labels):
        y = gold[f]
        line = f"{f[:41]:<42}{int(y.sum()):>5}{counts[f]:>5}"
        for r, M in rules.items():
            v = f1_at_count(M[:, j], y, counts[f])
            per[r][j] = v
            line += f"{v:>11.3f}"
        line += f"{ap(INC[:, j], y):>9.3f}{ap(Q[:, j], y):>9.3f}"
        print(line)
    print("-" * 118)
    print(f"{'MEAN over these labels':<42}{'':>5}{'':>5}" +
          "".join(f"{per[r].mean():>11.3f}" for r in rules))

    # paired author bootstrap on the pooled macro over these labels
    rng = np.random.default_rng(args.seed)
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}
    base = "incumbent"
    print("\n" + "=" * 70)
    print(f"PAIRED AUTHOR BOOTSTRAP vs {base}  ({len(auths)} authors)")
    print("=" * 70)
    for r in rules:
        if r == base:
            continue
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            a_ = np.mean([f1_at_count(rules[r][sel, j], gold[f][sel],
                                      max(1, int(round(counts[f] * len(sel)
                                                       / len(rows)))))
                          for j, f in enumerate(labels)])
            b_ = np.mean([f1_at_count(INC[sel, j], gold[f][sel],
                                      max(1, int(round(counts[f] * len(sel)
                                                       / len(rows)))))
                          for j, f in enumerate(labels)])
            draws.append(a_ - b_)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        obs = per[r].mean() - per[base].mean()
        print(f"{r:<12}{obs:>+9.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"P(better) {np.mean(np.array(draws) > 0):.3f}")
    print("=" * 70)
    print(f"\nThese are {len(labels)} of 24 labels, restricted to the weak/rare")
    print("ones, so the comparison is on the subset most favourable to the LLM")
    print("and does not generalise to the full taxonomy on its own.")


if __name__ == "__main__":
    main()

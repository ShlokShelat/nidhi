"""
factor_ensemble.py

Ensemble the eight regime-diverse factor models. Factors are the last block.

WHY THIS, AND WHY IT MIGHT WORK WHERE OTHER FACTOR ATTEMPTS FAILED.

  risk     ~0.82  0.4 weight   previously-shipped attempts, both negative -- stop
  evidence ~0.75  0.3 weight   voting works, and has delivered gains so far
  factors   0.63  0.3 weight   gains available, has NEVER moved

  Failed so far: risk-consistency, threshold recalibration, a 14B LLM, and
  blending with the friend model. All four were post-processing or one
  alternative model.

  What worked on EVIDENCE was an ensemble of models differing in TRAINING
  REGIME, and the decisive detail was that seed variants clustered while regime
  changes did not. The previous factor "ensembles" on disk
  (ensembled_factors_seeds.csv, nli_ens3_factors.csv) are seed variants, which
  is why they went nowhere.

  The eight models here all change the regime -- learning rate, epochs,
  sequence length, negative sampling ratio, and crucially the LOSS (bce,
  bce_perlabel, asl). They came out spread 0.5693-0.6341 and predict between
  2.62 and 4.42 factors per post. That is real disagreement.

THE TRAP THIS AVOIDS, learned from src/factor_blend.py.

  That script reported the blend beating the baseline and it was an
  artefact: the baseline had been cross-fitted while the deployed system had
  its 24 thresholds fitted on everything. On equal footing the blend was
  negative. So here EVERY arm, baseline included, gets thresholds by the SAME
  procedure, reported both ways:

      xfit     thresholds fitted on the other author half   (honest, low)
      fit-all  thresholds fitted on everything              (optimistic, but
                                                             matches how the
                                                             deployed system
                                                             was built)

  Judge arms against the baseline WITHIN a column, never across.

ARMS
  best single      the strongest individual model
  incumbent        runs/factors_nli, what is actually shipped
  mean-prob        average probabilities over all models  (no free parameters)
  vote>=k          label-level agreement, k swept          (one small integer)
  mean top-N       average the N best models only

Usage:
  # first dump the incumbent so it is in the comparison
  python src/predict_factors_nli.py --test artifacts/val_split.csv \
      --nli-run runs/factors_nli \
      --dump-probs artifacts/factor_runs/factors_nli_val.npz \
      --out artifacts/factor_runs/factors_nli_val_labels.csv

  python src/factor_ensemble.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def macro_f1(P, Y):
    out = []
    for j in range(Y.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        fp = int((P[:, j] & (1 - Y[:, j])).sum())
        fn = int(((1 - P[:, j]) & Y[:, j]).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out)), out


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def fit_per_label(PR, Y, rows):
    th = np.full(PR.shape[1], 0.5)
    for j in range(PR.shape[1]):
        y, p = Y[rows, j], PR[rows, j]
        best, bt = -1.0, 0.5
        for t in GRID:
            pred = (p >= t).astype(int)
            tp = int((pred & y).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((pred & (1 - y)).sum())
                           + int(((1 - pred) & y).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt
    return th


def decide(PR, Y, half, mode):
    """mode 'fitall' -> thresholds from everything (optimistic, matches how the
    deployed system was built). mode 'xfit' -> fitted on the other author half
    (honest). Both reported; never compare one against the other."""
    if mode == "fitall":
        return (PR >= fit_per_label(PR, Y, np.ones(len(Y), bool))).astype(int)
    P = np.zeros_like(Y)
    for h in ("A", "B"):
        m = half == h
        P[m] = (PR[m] >= fit_per_label(PR, Y, ~m)).astype(int)
    return P


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    pk = next((k for k in ("probs", "p", "y", "scores") if k in keys), None)
    rk = next((k for k in ("row_id", "row_ids", "ids") if k in keys), None)
    if pk is None or rk is None:
        raise SystemExit(f"{path}: expected prob/row_id arrays, got {d.files}")
    return list(d[rk]), np.asarray(d[pk], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/val_split.csv")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, f"*_{args.split}.npz")))
    if not paths:
        raise SystemExit(f"no *_{args.split}.npz under {args.dir}")

    models, rid0 = {}, None
    for p in paths:
        name = os.path.basename(p)[: -len(f"_{args.split}.npz")]
        rids, PR = load_npz(p)
        if rid0 is None:
            rid0 = rids
        elif list(rids) != list(rid0):
            order = {r: i for i, r in enumerate(rids)}
            PR = PR[[order[r] for r in rid0]]
        models[name] = PR
        print(f"  {name:<16} {PR.shape}")

    lab = pd.read_csv(args.labeled).set_index("row_id")
    rows = [r for r in rid0 if r in lab.index]
    keep = [i for i, r in enumerate(rid0) if r in lab.index]
    for n in models:
        models[n] = models[n][keep]

    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rows), len(FACTOR_TAXONOMY)), dtype=int)
    for i, r in enumerate(rows):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    half = author_halves(list(lab.loc[rows, "anon_user_id"]))
    print(f"\n{len(rows)} posts, {len(models)} models, "
          f"halves {int((half=='A').sum())}/{int((half=='B').sum())}")

    # individual models, ranked
    singles = {}
    for n, PR in models.items():
        singles[n] = macro_f1(decide(PR, Y, half, "fitall"), Y)[0]
    order = sorted(singles, key=lambda n: -singles[n])
    print(f"\n{'model':<18}{'fit-all macro-F1':>18}")
    for n in order:
        print(f"{n:<18}{singles[n]:>18.4f}")

    arms = {}
    # THE BASELINE IS BOTH NLI SEEDS AVERAGED, not factors_nli alone. The
    # deployed factor column is artifacts/nli_ensemble_factors.csv, which
    # ensembles factors_nli and factors_nli_s2. Benchmarking against one seed
    # would understate the bar by exactly the value of that ensemble.
    inc = [n for n in ("factors_nli", "factors_nli_s2") if n in models]
    if inc:
        arms[f"INCUMBENT ({'+'.join(inc)})"] = np.mean(
            [models[n] for n in inc], axis=0)
    best_single = order[0]
    arms[f"best single ({best_single})"] = models[best_single]
    arms["mean-prob ALL"] = np.mean(list(models.values()), axis=0)
    for N in (3, 5):
        if N < len(order):
            arms[f"mean-prob top{N}"] = np.mean(
                [models[n] for n in order[:N]], axis=0)
    # the new models only, excluding both incumbents
    new = [n for n in order if n not in inc]
    if new:
        arms["mean-prob new-only"] = np.mean([models[n] for n in new], axis=0)

    # label-level vote, using each model's own fit-all thresholds
    binar = {n: decide(PR, Y, half, "fitall") for n, PR in models.items()}
    S = np.sum(list(binar.values()), axis=0)
    votes = {f"vote>={k}": (S >= k).astype(int)
             for k in range(2, len(models) + 1)}

    # EVERY arm scored the same way. The first version of this table took the
    # half-splits of probability arms from CROSS-FITTED decisions and those of
    # vote arms from FIT-ALL decisions, then compared both against a
    # cross-fitted baseline -- which made the vote arms look +0.10 better on
    # half A. Same mistake factor_blend.py made. Decisions are computed once,
    # under one mode, and every number below comes from them.
    res = {}
    for n, PR in arms.items():
        res[n] = decide(PR, Y, half, "fitall")
    res.update(votes)

    def trio(P):
        return (macro_f1(P, Y)[0],
                macro_f1(P[half == "A"], Y[half == "A"])[0],
                macro_f1(P[half == "B"], Y[half == "B"])[0])

    base = next(n for n in res if n.startswith("INCUMBENT"))
    bf, ba, bb = trio(res[base])

    print(f"\nALL ARMS, thresholds fitted on everything (the way the deployed")
    print(f"system was built). Baseline = {base}\n")
    print(f"{'arm':<30}{'macro-F1':>10}{'halfA':>9}{'halfB':>9}"
          f"{'delta':>9}{'composite':>11}")
    print("-" * 78)
    win = []
    for n, P in res.items():
        m, a, b = trio(P)
        d, da, db = m - bf, a - ba, b - bb
        ok = n != base and d > 0 and da > 0 and db > 0
        if ok:
            win.append((n, d))
        tag = "  <- BASELINE" if n == base else (
            "  <- both halves" if ok else "")
        print(f"{n:<30}{m:>10.4f}{a:>9.4f}{b:>9.4f}{d:>+9.4f}"
              f"{0.3 * d:>+11.4f}{tag}")

    print("\n" + "=" * 78)
    if win:
        n, d = max(win, key=lambda t: t[1])
        print(f"BEST: {n}   {d:+.4f} macro   ({0.3 * d:+.4f} composite)")
        print("\nDISCOUNT IT. Offline gains on this kind of val split tend")
        print("to transfer to a submission at only a fraction of their")
        print("measured value. Expect roughly "
              f"{0.3 * d * 0.17:+.4f} composite, not {0.3 * d:+.4f}.")
        print("\nNext: rebuild the test factor column with this arm and patch")
        print("factors ONLY on top of the current champion.")
    else:
        print("No arm beats the baseline on both halves. Factors close, and")
        print("the 0.020 in S2 is not reachable with these models.")
    print("=" * 78)


if __name__ == "__main__":
    main()

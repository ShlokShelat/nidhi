"""
factor_lora_ensemble.py

Does a fine-tuned Qwen-7B add anything to the DeBERTa cross-encoder?

WHY A FINE-TUNED MEMBER AND NOT A ZERO-SHOT ONE. Ensemble value depends on both
DECORRELATION and rough PARITY: a member far below the incumbent drags the
average down by more than its independence buys back, so a zero-shot LLM cannot
contribute however uncorrelated it is.

`runs/factors_lora7b` is Qwen2.5-7B FINE-TUNED on this task with LoRA (r=16,
q/v/k/o projections, SEQ_CLS head), so it should be near parity. And the two
readers could hardly be more architecturally different -- a 435M cross-encoder
scoring (post, hypothesis) pairs versus a 7B decoder with a classification head
-- so their errors have every reason to be independent. That is the
configuration in which an ensemble should pay. The probabilities are already on
disk, so the test costs nothing.

THE ALIGNMENT HAZARD, AND WHY THIS SCRIPT IS PARANOID ABOUT IT. val_probs.npy is
a bare array with NO row_id column, so the only way to line it up with gold is
to assume it is in val_split.csv order. If that assumption is wrong every number
below is noise dressed as a result. Two guards: the shape must match val_split
exactly, and the LoRA's STANDALONE macro-F1 must be plausible (a misaligned
matrix scores near chance, roughly 0.05-0.15, not 0.4+). If the standalone
number looks like chance the script refuses to report an ensemble.

CALIBRATION. The two models' probabilities are on different scales -- a
fine-tuned sigmoid and a LoRA classification head have no reason to agree on
what 0.5 means. So a RANK average is reported alongside the probability average;
it is invariant to any monotone rescaling and is the safer read.

EVERYTHING IS CROSS-FITTED over author halves. Both models fitted their
thresholds on this same val split, so applying those saved thresholds here would
be in-sample for both. Fresh cutoffs are fitted on one author half and applied
to the other.

CAVEAT ON POWER. This is the 372-post / 32-author instrument, whose null band is
about +/-0.0559 paired. It cannot prove a small effect. It CAN reveal a large
one, and the question here -- does a 7B fine-tune add anything at all -- is a
large-effect question.

Usage:
  python src/factor_lora_ensemble.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def gold_matrix(df):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(df), N), dtype=np.int8)
    for i, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def per_label_f1(P, Y):
    tp = (P & Y).sum(0).astype(float)
    fp = (P & (1 - Y)).sum(0).astype(float)
    fn = ((1 - P) & Y).sum(0).astype(float)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def macro(P, Y):
    return float(per_label_f1(P, Y).mean())


def argmax_threshold(pr, y):
    npos = int(y.sum())
    if npos == 0:
        return 1.01
    o = np.argsort(-pr, kind="stable")
    cum = np.cumsum(y[o])
    ks = np.searchsorted(-pr[o], -GRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    f1 = np.where(ks + npos > 0, 2.0 * tp / (ks + npos), 0.0)
    return float(GRID[int(np.argmax(f1))])


def author_halves(auth):
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        p = "A" if size["A"] <= size["B"] else "B"
        half[a] = p
        size[p] += c[a]
    return s.map(half).to_numpy()


def xfit(Pr, Y, A, B):
    X = np.zeros_like(Y)
    for src, dst in ((B, A), (A, B)):
        th = np.array([argmax_threshold(Pr[src, j], Y[src, j])
                       for j in range(N)])
        X[dst] = (Pr[dst] >= th).astype(np.int8)
    return X


def ranks(x):
    return pd.DataFrame(x).rank(axis=0, pct=True).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="runs/factors_lora7b/val_probs.npy")
    ap.add_argument("--enc-runs",
                    default="factors_nli_mnli,factors_nli_mnli_s2",
                    help="the deployed factor encoder pair")
    ap.add_argument("--enc-dir", default="artifacts/factor_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)

    # ---- the encoder pair, averaged the deployed way ------------------------
    PR, rid = [], None
    for r in [x.strip() for x in args.enc_runs.split(",")]:
        p = os.path.join(args.enc_dir, f"{r}_val.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        z = np.load(p, allow_pickle=True)
        PR.append(np.asarray(z["probs"], dtype=float))
        ids = [str(x) for x in z["row_id"]]
        if rid is None:
            rid = ids
        elif rid != ids:
            raise SystemExit(f"{r}: row_id order differs")
    E = np.mean(PR, axis=0)

    # ---- the LoRA, which carries NO row ids ---------------------------------
    if not os.path.exists(args.lora):
        raise SystemExit(f"missing {args.lora}")
    L = np.load(args.lora).astype(float)
    print(f"encoder {E.shape} (row_ids present)   lora {L.shape} (NO row_ids)")
    if L.shape != E.shape:
        raise SystemExit(
            f"shape mismatch {L.shape} vs {E.shape}. The LoRA dump cannot be "
            f"aligned to the encoder's posts; refusing to guess.")

    # gold in the ENCODER's row order, and in val_split's own order, so we can
    # tell which one the LoRA matrix is actually in
    g_enc = gold_matrix(va.set_index("row_id").loc[rid].reset_index())
    g_file = gold_matrix(va)
    same_order = rid == [str(x) for x in va["row_id"]]
    print(f"val_split order == encoder dump order: {same_order}")

    auth_enc = (va.set_index("row_id").loc[rid].reset_index()
                ["anon_user_id"].to_numpy())
    half = author_halves(auth_enc)
    A, B = half == "A", half == "B"

    # ---- ALIGNMENT GUARD -----------------------------------------------------
    # Score the LoRA against both candidate orderings. A misaligned matrix lands
    # near chance; the correct one should be far above it.
    print("\n" + "=" * 70)
    print("ALIGNMENT CHECK -- the LoRA has no row ids, so this is inferred")
    print("=" * 70)
    cand = {"val_split file order": g_file}
    if not same_order:
        cand["encoder dump order"] = g_enc
    scores = {}
    for name, G in cand.items():
        s = macro(xfit(L, G, A, B), G)
        scores[name] = s
        print(f"  {name:<26} standalone macro-F1 {s:.4f}")
    best_name = max(scores, key=scores.get)
    best = scores[best_name]
    if best < 0.25:
        print(f"\n  *** {best:.4f} is near chance under every ordering.")
        print("      Either the LoRA dump is misaligned in a way that cannot")
        print("      be recovered, or the model never learned the task.")
        print("      Refusing to report an ensemble built on it.")
        return
    G = cand[best_name]
    print(f"\n  using {best_name!r} (macro {best:.4f}) -- plausible, proceeding")

    # ---- the comparison ------------------------------------------------------
    e_pred = xfit(E, G, A, B)
    e = macro(e_pred, G)
    variants = {"encoder alone": e_pred,
                "lora alone": xfit(L, G, A, B)}
    variants["mean of probs"] = xfit(0.5 * E + 0.5 * L, G, A, B)
    variants["mean of ranks"] = xfit(0.5 * ranks(E) + 0.5 * ranks(L), G, A, B)

    print("\n" + "=" * 70)
    print("CROSS-FITTED macro-F1 on the 372-post val split")
    print("=" * 70)
    print(f"{'variant':<18}{'macro':>9}{'d vs encoder':>14}{'fac/post':>10}")
    print("-" * 55)
    for k, P in variants.items():
        print(f"{k:<18}{macro(P, G):>9.4f}{macro(P, G) - e:>+14.4f}"
              f"{P.sum(1).mean():>10.2f}")

    # ---- paired author bootstrap --------------------------------------------
    uniq = pd.unique(auth_enc)
    by = {a: np.flatnonzero(auth_enc == a) for a in uniq}
    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.boot) for k in variants}
    for i in range(args.boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        ix = np.concatenate([by[uniq[k]] for k in pick])
        Gb = G[ix]
        for k, P in variants.items():
            boot[k][i] = macro(P[ix], Gb)

    print("\n" + "=" * 70)
    print(f"PAIRED delta vs the encoder   ({args.boot} author bootstrap)")
    print("=" * 70)
    print(f"{'variant':<18}{'delta':>9}{'95% CI':>21}{'win%':>8}")
    print("-" * 60)
    for k in variants:
        if k == "encoder alone":
            continue
        d = boot[k] - boot["encoder alone"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{k:<18}{macro(variants[k], G) - e:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>21}"
              f"{100*float((d > 0).mean()):>7.1f}%")

    # ---- the number that actually decides it ---------------------------------
    print("\n" + "=" * 70)
    print("ERROR CORRELATION -- the thing that decides whether this can work")
    print("=" * 70)
    ee = (variants["encoder alone"] != G)
    le = (variants["lora alone"] != G)
    both = float((ee & le).sum())
    print(f"  encoder wrong on {int(ee.sum())} label-decisions")
    print(f"  lora    wrong on {int(le.sum())}")
    print(f"  BOTH wrong on    {int(both)}")
    exp = float(ee.sum()) * float(le.sum()) / ee.size
    print(f"  expected if independent: {exp:.0f}")
    print(f"  overlap ratio {both / max(exp, 1e-9):.2f}  "
          f"(1.0 = fully independent, >1 = correlated errors)")
    print("\n  A ratio near 1 means the errors really are independent and an")
    print("  ensemble should pay once the members are near parity. A ratio well")
    print("  above 1 means they fail on the same posts and no amount of")
    print("  averaging will help.")
    print("\nNOTE 372 posts / 32 authors: the paired null band on this")
    print("instrument is about +/-0.0559. Treat anything smaller as unproven.")


if __name__ == "__main__":
    main()

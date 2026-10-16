"""
factor_llm_ensemble.py

Does averaging the LLM with the encoder beat the encoder, even on the labels
where the LLM loses head-to-head?

WHY ASK WHEN THE ZERO-SHOT LLM SCORES BELOW THE ENCODER PER LABEL. Ensemble
gain depends on ERROR CORRELATION, not on either member's standalone score. A
weaker member with uncorrelated mistakes can still add; a stronger member that
fails on the same posts adds nothing. The two readers here could hardly be more
different -- one is a fine-tuned cross-encoder trained on 1015 posts, the other
is a frozen instruction model that has never seen this dataset -- so their
errors have every reason to be independent.

The test is cheap: it trains nothing and reuses probabilities that already
exist.

WHAT IS COMPARED, all with thresholds CROSS-FITTED over author halves so no
variant is scored on the data that chose its cutoffs:

  encoder     the OOF arm alone, refit -- the honest baseline here
  llm         LLM probabilities on its 8 labels, encoder elsewhere
  mean        equal average of the two probability sets on those 8
  wsweep      the best fixed encoder weight over a grid, chosen on the other
              author half so the choice is not free
  rank        average of within-label percentile RANKS rather than raw
              probabilities -- immune to the two models being on different
              scales, which they are (a fine-tuned sigmoid and a Yes/No logit
              ratio have no reason to be comparable)

NOTE ON THE BASELINE. `encoder` here is refit cross-fitted rather than read off
the deployed operating point, because every variant must be scored the same way
for the comparison to mean anything. Read the DELTAS, not the absolute levels.

THE BAR. Paired author bootstrap, 95% CI excluding zero AND at least 80% of
resamples won -- and it must also clear the null band measured between two
identically configured training runs (about +/-0.014 at k=5). An ensemble that
gains less than seed noise is not a finding.
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
WGRID = np.round(np.arange(0.0, 1.01, 0.1), 2)


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


def ranks(x):
    """Within-column percentile rank, ties averaged."""
    return pd.DataFrame(x).rank(axis=0, pct=True).to_numpy()


def author_halves(auth):
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        p = "A" if size["A"] <= size["B"] else "B"
        half[a] = p
        size[p] += c[a]
    return s.map(half).to_numpy()


def xfit_predict(Pr, Y, A, B):
    """Cross-fitted binarisation: cutoffs from one author half, applied to the
    other."""
    X = np.zeros_like(Y)
    for src, dst in ((B, A), (A, B)):
        th = np.array([argmax_threshold(Pr[src, j], Y[src, j])
                       for j in range(N)])
        X[dst] = (Pr[dst] >= th).astype(np.int8)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mnli")
    ap.add_argument("--dir", default="artifacts/factor_oof")
    ap.add_argument("--llm", default="artifacts/qwen_rare_probe.npz")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    enc, rid = [], []
    for f in range(args.k):
        p = os.path.join(args.dir, f"{args.arm}_f{f}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        z = np.load(p, allow_pickle=True)
        enc.append(np.asarray(z["probs"], dtype=float))
        rid += list(z["row_id"])
    E = np.concatenate(enc, axis=0)

    z = np.load(args.llm, allow_pickle=True)
    L_raw = np.asarray(z["probs"], dtype=float)
    L_fac = [str(x) for x in z["factors"]]
    L_rid = [str(x) for x in z["row_id"]]

    # align the LLM's posts to the encoder's fold order
    pos = {r: i for i, r in enumerate(L_rid)}
    missing = [r for r in rid if r not in pos]
    if missing:
        raise SystemExit(f"LLM dump is missing {len(missing)} posts, "
                         f"e.g. {missing[:3]}")
    order = [pos[r] for r in rid]
    L_raw = L_raw[order]

    cols = [FACTOR_TAXONOMY.index(f) for f in L_fac]
    print(f"{len(rid)} posts   encoder {E.shape}   LLM {L_raw.shape} on "
          f"{len(cols)} of {N} labels")
    print("LLM labels:", ", ".join(L_fac))

    lab = pd.read_csv(args.labeled, keep_default_na=False).set_index("row_id")
    ev = lab.loc[rid].reset_index()
    Y = gold_matrix(ev)
    auth = ev["anon_user_id"].to_numpy()
    half = author_halves(auth)
    A, B = half == "A", half == "B"

    # LLM probabilities placed into a full 24-wide matrix; untouched labels keep
    # the encoder so every variant differs ONLY on the LLM's 8 columns
    def blend(mode, w=None):
        P = E.copy()
        for c, j in enumerate(cols):
            e, l = E[:, j], L_raw[:, c]
            if mode == "llm":
                P[:, j] = l
            elif mode == "mean":
                P[:, j] = 0.5 * e + 0.5 * l
            elif mode == "rank":
                P[:, j] = 0.5 * ranks(e[:, None])[:, 0] + \
                          0.5 * ranks(l[:, None])[:, 0]
            elif mode == "w":
                P[:, j] = w * e + (1 - w) * l
        return P

    print(f"\n{'variant':<12}{'xfit macro':>12}{'d vs encoder':>14}"
          f"{'on LLM labels':>15}")
    print("-" * 55)
    base_P = xfit_predict(E, Y, A, B)
    base = macro(base_P, Y)
    bl = float(per_label_f1(base_P, Y)[cols].mean())
    print(f"{'encoder':<12}{base:>12.4f}{'':>14}{bl:>15.4f}")

    preds = {"encoder": base_P}
    for mode in ("llm", "mean", "rank"):
        P = xfit_predict(blend(mode), Y, A, B)
        preds[mode] = P
        print(f"{mode:<12}{macro(P, Y):>12.4f}{macro(P, Y) - base:>+14.4f}"
              f"{float(per_label_f1(P, Y)[cols].mean()):>15.4f}")

    # weight swept on one author half, applied to the other -- the choice of w
    # is itself cross-fitted, so it is not free
    def fit_and_score(Pw, rows):
        """Cutoffs fitted on `rows`, scored on `rows`. In-sample by design --
        used only to CHOOSE w from data the other half never sees."""
        th = np.array([argmax_threshold(Pw[rows, j], Y[rows, j])
                       for j in range(N)])
        return macro((Pw[rows] >= th).astype(np.int8), Y[rows]), th

    X = np.zeros_like(Y)
    picked = {}
    for src, dst, name in ((B, A, "A"), (A, B, "B")):
        best_w, best_s = 1.0, -1.0
        for w in WGRID:
            s, _ = fit_and_score(blend("w", w), src)
            if s > best_s:
                best_w, best_s = w, s
        Pw = blend("w", best_w)
        _, th = fit_and_score(Pw, src)
        X[dst] = (Pw[dst] >= th).astype(np.int8)
        picked[name] = best_w
    preds["wsweep"] = X
    print(f"{'wsweep':<12}{macro(X, Y):>12.4f}{macro(X, Y) - base:>+14.4f}"
          f"{float(per_label_f1(X, Y)[cols].mean()):>15.4f}"
          f"   encoder weight {picked}")

    # ---- paired author bootstrap --------------------------------------------
    uniq = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uniq}
    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.boot) for k in preds}
    for b in range(args.boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        ix = np.concatenate([by[uniq[k]] for k in pick])
        Yb = Y[ix]
        for k, P in preds.items():
            boot[k][b] = macro(P[ix], Yb)

    print("\n" + "=" * 72)
    print(f"PAIRED delta vs encoder   ({args.boot} author-bootstrap resamples)")
    print("=" * 72)
    print(f"{'variant':<12}{'delta':>9}{'95% CI':>21}{'win%':>8}{'verdict':>20}")
    print("-" * 72)
    for k in preds:
        if k == "encoder":
            continue
        d = boot[k] - boot["encoder"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        win = float((d > 0).mean()) * 100
        pt = macro(preds[k], Y) - base
        good = lo > 0 and win >= 80
        print(f"{k:<12}{pt:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>21}{win:>7.1f}%"
              f"{('clears CI' if good else 'no'):>20}")
    print("\nA 'clears CI' verdict is NOT sufficient. It must also exceed the")
    print("replica null band (~0.014 at k=5), or it is smaller than the")
    print("difference between two identically configured training runs.")

    print("\n" + "=" * 72)
    print("PER LABEL on the LLM's labels")
    print("=" * 72)
    fe = per_label_f1(preds["encoder"], Y)
    fl = per_label_f1(preds["llm"], Y)
    fm = per_label_f1(preds["mean"], Y)
    fr = per_label_f1(preds["rank"], Y)
    print(f"{'factor':<40}{'gold':>6}{'enc':>8}{'llm':>8}{'mean':>8}{'rank':>8}")
    for c, j in enumerate(cols):
        print(f"{FACTOR_TAXONOMY[j]:<40}{int(Y[:, j].sum()):>6}{fe[j]:>8.3f}"
              f"{fl[j]:>8.3f}{fm[j]:>8.3f}{fr[j]:>8.3f}")


if __name__ == "__main__":
    main()

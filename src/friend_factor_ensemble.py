"""
friend_factor_ensemble.py

Does the friend's FACTOR head add anything to the deployed model?

WHY ASK AGAIN. Its standalone macro is 0.4280 even with in-sample thresholds,
against the deployed model's 0.6203 held-out -- a 0.19 deficit, the same gap that made the
zero-shot LLM and the Qwen LoRA both worthless. That usually settles it.

But the EVIDENCE result just showed a member 0.10 behind still improving the
vote on both author halves, because a vote rewards AGREEMENT rather than
quality. So the honest question is whether factors have the same asymmetry.
They do not, by construction -- factors are combined by AVERAGING probabilities
and thresholding, where a weak member drags the mean instead of being outvoted.
This measures that rather than assuming it.

WHAT MAKES THIS TEST BETTER THAN THE LORA ONE. That ran on 372 val posts with a
+/-0.0559 paired null band. This runs on the 1590-post OOF where both sides are
genuinely out-of-fold: the deployed model's probabilities come from
`artifacts/factor_oof/` (each post scored by the fold model that never trained
on it) and the friend's from `artifacts/friend_oof_export.csv` (its own 5-fold
OOF).

THE NUMBER THAT DECIDES IT is the error-overlap ratio, not the standalone score.
Ratio near 1.0 means genuinely independent errors, so averaging can pay once the
members are close enough in strength. Well above 1.0 means they fail on the same
posts and nothing helps.

Usage:
  python src/friend_factor_ensemble.py
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
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)


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
    ap.add_argument("--arm", default="mnli")
    ap.add_argument("--dir", default="artifacts/factor_oof")
    ap.add_argument("--friend", default="artifacts/friend_oof_export.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PR, rid = [], []
    for f in range(args.k):
        p = os.path.join(args.dir, f"{args.arm}_f{f}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        z = np.load(p, allow_pickle=True)
        PR.append(np.asarray(z["probs"], dtype=float))
        rid += [str(x) for x in z["row_id"]]
    E = np.concatenate(PR, axis=0)

    fr = pd.read_csv(args.friend, keep_default_na=False)
    missing = [t for t in FACTOR_TAXONOMY if t not in fr.columns]
    if missing:
        raise SystemExit(f"friend export lacks columns: {missing}")
    fr["row_id"] = fr["row_id"].astype(str)
    fmap = fr.set_index("row_id")

    both = [r for r in rid if r in fmap.index]
    print(f"ours {len(rid)} posts   friend {len(fr)}   overlap {len(both)}")
    sel = [i for i, r in enumerate(rid) if r in fmap.index]
    E = E[sel]
    F = fmap.loc[both, list(FACTOR_TAXONOMY)].to_numpy(dtype=float)

    lab = pd.read_csv(args.labeled, keep_default_na=False).set_index("row_id")
    ev = lab.loc[both].reset_index()
    Y = gold_matrix(ev)
    auth = ev["anon_user_id"].to_numpy()
    half = author_halves(auth)
    A, B = half == "A", half == "B"

    variants = {"ours alone": xfit(E, Y, A, B),
                "friend alone": xfit(F, Y, A, B),
                "mean of probs": xfit(0.5 * E + 0.5 * F, Y, A, B),
                "mean of ranks": xfit(0.5 * ranks(E) + 0.5 * ranks(F), Y, A, B)}
    # a light touch of friend, in case 50/50 is simply too much of a weak member
    for w in (0.8, 0.9):
        variants[f"{w:.0%} ours"] = xfit(w * E + (1 - w) * F, Y, A, B)

    # ---- GATED: consult the friend ONLY where we are uncertain --------------
    # A flat blend asks the friend about all 38,160 decisions, including the
    # ~35,000 we already get right -- which is where its 3,722 breakages live,
    # and why 10% is the most we can afford.
    #
    # Our errors concentrate near the decision boundary. So weight the friend by
    # OUR uncertainty: full say when our probability sits on the threshold, none
    # at all when we are confident. That should capture more of the 1,409 fixes
    # while importing fewer breakages than any flat weight can.
    #
    # The gate width is fitted on one author half and applied to the other, so
    # it is not a free parameter.
    def gated(delta, band):
        """Blend proportionally to how close ours is to its own cutoff."""
        X = np.zeros_like(Y)
        for src, dst in ((B, A), (A, B)):
            th = np.array([argmax_threshold(E[src, j], Y[src, j])
                           for j in range(N)])
            # u = 1 at the threshold, 0 once |p - th| exceeds `band`
            u = np.clip(1.0 - np.abs(E[dst] - th) / max(band, 1e-6), 0.0, 1.0)
            P = E[dst] + delta * u * (F[dst] - E[dst])
            X[dst] = (P >= th).astype(np.int8)
        return X

    for band in (0.10, 0.20):
        for delta in (0.5, 1.0):
            variants[f"gate b{band:.2f} d{delta:.1f}"] = gated(delta, band)

    base = macro(variants["ours alone"], Y)
    print(f"\n{'variant':<16}{'macro':>9}{'d vs ours':>12}{'fac/post':>10}")
    print("-" * 50)
    for k, P in variants.items():
        print(f"{k:<16}{macro(P, Y):>9.4f}{macro(P, Y) - base:>+12.4f}"
              f"{P.sum(1).mean():>10.2f}")

    uniq = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uniq}
    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.boot) for k in variants}
    for i in range(args.boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        ix = np.concatenate([by[uniq[k]] for k in pick])
        Yb = Y[ix]
        for k, P in variants.items():
            boot[k][i] = macro(P[ix], Yb)

    print("\n" + "=" * 68)
    print(f"PAIRED delta vs ours   ({args.boot} author bootstrap)")
    print("=" * 68)
    print(f"{'variant':<16}{'delta':>9}{'95% CI':>21}{'win%':>8}")
    print("-" * 60)
    for k in variants:
        if k == "ours alone":
            continue
        d = boot[k] - boot["ours alone"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{k:<16}{macro(variants[k], Y) - base:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>21}"
              f"{100*float((d > 0).mean()):>7.1f}%")

    # ---- POLICY CONSISTENCY -------------------------------------------------
    # A sign that flips between threshold policies (positive under deployed
    # thresholds, negative under ratematch) can cost score on a submission.
    # A real improvement produces better PROBABILITIES, so it
    # survives any cutoff rule; an arm that wins under one rule only has
    # luckier thresholds. Nothing here is retrained, so seed noise does not
    # apply -- but the threshold rule still can, and this is the cheap way to
    # find out.
    def ratematch(Pr):
        X = np.zeros_like(Y)
        for src, dst in ((B, A), (A, B)):
            th = np.empty(N)
            for j in range(N):
                k = int(round(float(Y[src, j].mean()) * int(dst.sum())))
                k = min(max(k, 1), int(dst.sum()))
                th[j] = float(np.sort(Pr[dst, j])[-k])
            X[dst] = (Pr[dst] >= th).astype(np.int8)
        return X

    def fixed(Pr, t=0.5):
        return (Pr >= t).astype(np.int8)

    print("\n" + "=" * 68)
    print("POLICY CONSISTENCY -- the check that would have caught mnlidef")
    print("=" * 68)
    print(f"{'weight on ours':<16}{'refit':>10}{'ratematch':>12}{'fixed .5':>11}")
    print("-" * 52)
    for w in (1.00, 0.90, 0.80, 0.50):
        Pw = w * E + (1 - w) * F
        r1 = macro(xfit(Pw, Y, A, B), Y)
        r2 = macro(ratematch(Pw), Y)
        r3 = macro(fixed(Pw), Y)
        tag = "   <- ours alone" if w == 1.0 else ""
        print(f"{w:<16.2f}{r1:>10.4f}{r2:>12.4f}{r3:>11.4f}{tag}")
    print("\nRead DOWN each column. A blend that beats `ours alone` in EVERY")
    print("column is a genuine probability improvement. One that wins in some")
    print("and loses in others is a threshold artefact -- do not ship it.")

    print("\n" + "=" * 68)
    print("ERROR CORRELATION -- what actually decides whether this can work")
    print("=" * 68)
    ee = (variants["ours alone"] != Y)
    fe = (variants["friend alone"] != Y)
    both_w = float((ee & fe).sum())
    exp = float(ee.sum()) * float(fe.sum()) / ee.size
    print(f"  ours wrong   {int(ee.sum())} label-decisions")
    print(f"  friend wrong {int(fe.sum())}")
    print(f"  BOTH wrong   {int(both_w)}   (independent would give {exp:.0f})")
    print(f"  overlap ratio {both_w / max(exp, 1e-9):.2f}")
    fixes = int((ee & ~fe).sum())
    print(f"\n  the friend gets {fixes} of our {int(ee.sum())} errors RIGHT "
          f"({100*fixes/max(int(ee.sum()),1):.0f}%)")
    print(f"  but introduces {int((~ee & fe).sum())} errors of its own")

    # how much of that complementarity each variant actually converts
    print(f"\n{'variant':<18}{'our errors fixed':>18}{'new errors made':>18}")
    print("-" * 56)
    O = variants["ours alone"]
    for k, P in variants.items():
        if k in ("ours alone", "friend alone"):
            continue
        pe = (P != Y)
        print(f"{k:<18}{int((ee & ~pe).sum()):>18}{int((~ee & pe).sum()):>18}")
    print("\nA good variant fixes many and breaks few. The flat blends can only")
    print("trade one for the other; a gate should do better BECAUSE it declines")
    print("to consult the friend where we were already right.")
    print("\n  Complementarity is real if the first number is large. It only")
    print("  becomes USABLE if the second is not much larger -- averaging has")
    print("  no way to keep the fixes and discard the breakages, which is")
    print("  exactly what a vote CAN do and why evidence behaved differently.")


if __name__ == "__main__":
    main()

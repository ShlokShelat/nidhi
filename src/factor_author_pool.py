"""
factor_author_pool.py

Tests whether pooling factor probabilities across an author's posts improves
the factor decode.

Many factors are person-attributes rather than events: socio-economic status,
family situation, sexual orientation, physical health, trauma history do not
change between a person's posts. Risk level is expected to change, which is why
the same structure predicts opposite things for the two components. A
post-level factor model discards every other post by the same author, so a
trait mentioned once here but stated plainly in the author's next post is
information the model cannot use.

Both splits carry roughly ten posts per author, and author IDs are present on
the test set, so pooling is DEPLOYABLE rather than merely diagnostic.

WHAT IT MEASURES
  1. per-label AUTHOR CONSISTENCY: among authors who have the label at all,
     what fraction of their posts carry it? Near 1.0 means a trait; near the
     base rate means an event. Traits are what pooling can exploit; events are
     not, and pooling should be expected to HURT them.
  2. the blend p' = (1-w)*p_post + w*mean(p over the author's posts), decoded
     at the reference firing rates so the arms are comparable.
       w = 0.00  the unpooled model
       w = 0.25 / 0.50  fixed, ZERO fitted parameters
       w = argmax  one fitted parameter, so it is optimistic by construction
  3. the same blend applied ONLY to the labels that are author-consistent,
     which is the version with a mechanism behind it rather than a grid.
  4. a cross-fitted check: choose the label set on held-out AUTHORS, score on
     the rest, so the selection is not made and rewarded on the same noise.

THE POOL INCLUDES THE POST ITSELF, matching deployment: at inference the whole
test set and its author IDs are available at once. This is transductive rather
than leaky -- no gold label is consulted anywhere.

Usage (CPU):
  python src/factor_author_pool.py
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
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
PROTECTIVE = ["social support", "coping strategy", "psychological capital",
              "sense of responsibility", "meaning in life"]
PROT_J = [IDX[t] for t in PROTECTIVE]


def load_oof(d, arms, k):
    O, rid = [], None
    for a in arms:
        pr, ids = [], []
        for f in range(k):
            p = f"{d}/{a}_cv_oof_f{f}.npz"
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            z = np.load(p, allow_pickle=True)
            pr.append(z["probs"].astype(float))
            ids.extend(str(x) for x in z["row_id"])
        pr = np.concatenate(pr)
        if rid is None:
            rid = ids
        elif ids != rid:
            o = {x: i for i, x in enumerate(ids)}
            pr = pr[[o[x] for x in rid]]
        O.append(pr)
    return rid, np.stack(O)


def logit_mean(S):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, cell in enumerate(col):
        for x in set(parse_factors(str(cell))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def topk(P, rates):
    n = len(P)
    on = np.zeros((n, N), dtype=np.int8)
    for j in range(N):
        k = min(max(int(round(rates[j] * n)), 1), n - 1)
        cut = np.sort(P[:, j])[-k]
        on[:, j] = (P[:, j] >= cut).astype(np.int8)
    return on


def f1pl(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def author_mean(P, groups):
    """Mean probability over each author's posts, broadcast back to posts."""
    M = np.empty_like(P)
    for ix in groups.values():
        M[ix] = P[ix].mean(0)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--consist", type=float, default=0.55,
                    help="author-consistency threshold for the trait subset")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    uauth = pd.unique(auth)
    groups = {a: np.flatnonzero(auth == a) for a in uauth}

    ch = pd.read_csv(args.champion, keep_default_na=False)
    champ = cells(ch["factors"], len(ch))
    rates = champ.sum(0) / len(ch)

    sizes = np.array([len(v) for v in groups.values()])
    print(f"OOF {n} posts / {len(uauth)} authors   "
          f"posts per author: median {int(np.median(sizes))}, "
          f"min {sizes.min()}, max {sizes.max()}")

    # ---- 1. author consistency of the GOLD labels --------------------------
    print("\n" + "=" * 96)
    print("1. IS THE LABEL A TRAIT OR AN EVENT?")
    print("=" * 96)
    print(f"{'label':<44}{'gold':>6}{'base':>7}{'within':>8}{'lift':>7}  kind")
    consist = np.zeros(N)
    for j in range(N):
        base = Y[:, j].mean()
        num = den = 0
        for ix in groups.values():
            if Y[ix, j].sum() > 0:
                num += Y[ix, j].sum()
                den += len(ix)
        consist[j] = num / den if den else 0.0
        lift = consist[j] / base if base else 0.0
        kind = "TRAIT" if consist[j] >= args.consist else (
            "event" if lift < 2.0 else "mixed")
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{int(Y[:, j].sum()):>6}"
              f"{base:>7.3f}{consist[j]:>8.3f}{lift:>7.1f}x  {kind}")
    print("  within = P(label | same author has it somewhere).  * = protective")
    print("  A high LIFT means an author's other posts are informative -- that "
          "is the signal")
    print("  pooling can use. Lift near 1 means the label is independent "
          "across a person's posts.")

    # ---- 2. the blend, whole taxonomy --------------------------------------
    A = author_mean(P, groups)
    base_pred = topk(P, rates)
    base_f = f1pl(base_pred, Y)

    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([groups[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]

    print("\n" + "=" * 96)
    print("2. BLEND ALL 24 LABELS   p' = (1-w) p_post + w mean_author(p)")
    print("=" * 96)
    print(f"{'w':<8}{'macro':>9}{'protective':>12}{'d vs w=0':>11}"
          f"{'95% CI':>20}{'win':>7}")
    print(f"{'0.00':<8}{base_f.mean():>9.4f}{base_f[PROT_J].mean():>12.4f}"
          f"{'--':>11}{'--':>20}{'--':>7}")
    grid = [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]
    best_w, best_m = 0.0, base_f.mean()
    for w in grid:
        pred = topk((1 - w) * P + w * A, rates)
        f = f1pl(pred, Y)
        d = np.array([f1pl(pred[ix], Y[ix]).mean()
                      - f1pl(base_pred[ix], Y[ix]).mean() for ix in IXS])
        lo, hi = np.percentile(d, [2.5, 97.5])
        flag = "  <- fixed, 0 params" if w in (0.25, 0.50) else ""
        print(f"{w:<8.2f}{f.mean():>9.4f}{f[PROT_J].mean():>12.4f}"
              f"{f.mean() - base_f.mean():>+11.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{(d > 0).mean():>7.0%}{flag}")
        if f.mean() > best_m:
            best_w, best_m = w, f.mean()

    # ---- 3. trait labels only ----------------------------------------------
    trait = [j for j in range(N) if consist[j] >= args.consist]
    print("\n" + "=" * 96)
    print(f"3. BLEND ONLY THE TRAIT LABELS (consistency >= {args.consist}): "
          f"{len(trait)} of {N}")
    print("=" * 96)
    if not trait:
        print("  none qualify at this threshold")
    else:
        print("  " + ", ".join(FACTOR_TAXONOMY[j] for j in trait))
        for w in (0.25, 0.50, 1.00):
            Q = P.copy()
            for j in trait:
                Q[:, j] = (1 - w) * P[:, j] + w * A[:, j]
            pred = topk(Q, rates)
            f = f1pl(pred, Y)
            d = np.array([f1pl(pred[ix], Y[ix]).mean()
                          - f1pl(base_pred[ix], Y[ix]).mean() for ix in IXS])
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  w={w:.2f}  macro {f.mean():.4f}  protective "
                  f"{f[PROT_J].mean():.4f}  d {f.mean() - base_f.mean():+.4f}  "
                  f"[{lo:+.4f},{hi:+.4f}]  win {(d > 0).mean():.0%}")

    # ---- 4. per-label, at a fixed w ----------------------------------------
    w = 0.50
    predw = topk((1 - w) * P + w * A, rates)
    fw = f1pl(predw, Y)
    print(f"\n{'label':<44}{'w=0':>8}{'w=0.5':>8}{'delta':>8}{'consist':>9}")
    for j in np.argsort(-(fw - base_f)):
        if abs(fw[j] - base_f[j]) < 0.002:
            continue
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{base_f[j]:>8.3f}{fw[j]:>8.3f}"
              f"{fw[j] - base_f[j]:>+8.3f}{consist[j]:>9.3f}")

    # ---- 5. cross-fitted trait selection -----------------------------------
    print("\n" + "=" * 96)
    print("5. CROSS-FIT -- pick the trait set on held-out AUTHORS")
    print("=" * 96)
    rng2 = np.random.default_rng(args.seed + 1)
    dx = []
    for _ in range(args.splits):
        order = list(uauth)
        rng2.shuffle(order)
        cut = int(0.8 * len(order))
        fit = np.concatenate([groups[a] for a in order[:cut]])
        ev = np.concatenate([groups[a] for a in order[cut:]])
        sel = []
        for j in range(N):
            b = Y[fit, j].mean()
            num = den = 0
            for a in order[:cut]:
                ix = groups[a]
                if Y[ix, j].sum() > 0:
                    num += Y[ix, j].sum()
                    den += len(ix)
            c = num / den if den else 0.0
            if c >= args.consist and b > 0:
                sel.append(j)
        Q = P.copy()
        for j in sel:
            Q[:, j] = 0.5 * P[:, j] + 0.5 * A[:, j]
        a_ = f1pl(topk(Q[ev], rates), Y[ev]).mean()
        b_ = f1pl(topk(P[ev], rates), Y[ev]).mean()
        dx.append(a_ - b_)
    dx = np.array(dx)
    se = dx.std(ddof=1) / np.sqrt(len(dx))
    print(f"  w=0.5 on the cross-fitted trait set: {dx.mean():+.4f} "
          f"[{dx.mean() - 1.96 * se:+.4f},{dx.mean() + 1.96 * se:+.4f}]  "
          f"win {(dx > 0).mean():.0%}   composite {0.3 * dx.mean():+.4f}")

    print("\n" + "=" * 96)
    print("HOW TO DECIDE")
    print("=" * 96)
    print("Prefer a FIXED w (0.25 or 0.50) on the trait subset -- 0 fitted "
          "parameters, which is")
    print("the only category that has transferred above its estimate here. The "
          "argmax-w row is")
    print("1 parameter and historically delivers ~17%.")
    print("If section 1 shows high lift but sections 2-5 show nothing, the "
          "signal is real but")
    print("already captured by the post text, and the lane closes cleanly.")


if __name__ == "__main__":
    main()

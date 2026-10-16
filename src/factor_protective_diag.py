"""
factor_protective_diag.py

The protective labels are RANK-limited, not decode-limited: four of the five
worst APs in the taxonomy are protective (meaning in life 0.232, sense of
responsibility 0.556, social support 0.582, psychological capital 0.642), and
`meaning in life` caps at 0.292 even with a perfect decode. Re-decoding cannot
touch any of it.

THIS ASKS THE ONE QUESTION THAT IS FREE TO ANSWER: is the ranking bad in every
arm, or is it bad only in the POOL? We have an arm literally trained as the
protective specialist (`prot`). If it ranks these labels better than the 8-arm
logit-mean does, then the pool is DILUTING it -- the absorption law running
against us -- and the fix costs nothing to build and no GPU.

WHY THIS IS NOT THE CLOSED "per-label arm selection" LANE. That was judged on
F1 after decoding, which entangles ranking with fitted thresholds. AP is
threshold-free, so it isolates the ranking question the earlier pass could not
see. And the candidate here is FIXED, not searched: use `prot` for the five
protective labels, the pool everywhere else, decoded at the champion's existing
rates. Zero fitted parameters -- the only category that has ever transferred
above its estimate (route_attempt, evcond, bag8).

WHAT IT REPORTS
  1. per-arm AP for all 24 labels, so the dilution is visible or is not
  2. four candidates scored at CHAMPION rates with a paired author bootstrap:
       current   the 8-arm pool everywhere            (what we ship)
       prot5     `prot` on the 5 protective labels    (0 parameters)
       bestAP    best single arm per protective label, chosen IN-SAMPLE
                 (optimistic -- printed as a ceiling, never as a candidate)
       xfit      that same choice made on held-out authors, which is what
                 selecting would actually deliver
  3. the `meaning in life` error file: the posts we rank highest that are NOT
     gold, and the gold posts we rank lowest. AP 0.232 on 45 examples is either
     a definition that misses the concept or a label the annotators applied
     inconsistently, and reading twenty posts distinguishes those two cheaply.

Usage (cluster, CPU, ~2-3 min):
  python src/factor_protective_diag.py
  python src/factor_protective_diag.py --show 15
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
PROTECTIVE = [
    "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
]


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


def cells_from_factor_col(col, n):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    M = np.zeros((n, N), dtype=np.int8)
    for i, cell in enumerate(col):
        for x in set(parse_factors(cell)):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def topk_pred(P, rates, n):
    on = np.zeros((n, N), dtype=np.int8)
    for j in range(N):
        k = min(max(int(round(rates[j] * n)), 1), n - 1)
        cut = np.sort(P[:, j])[-k]
        on[:, j] = (P[:, j] >= cut).astype(np.int8)
    return on


def f1_per_label(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def submission_scale_split(auth, rng, eval_frac):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = list(counts.index)
    rng.shuffle(order)
    target = eval_frac * len(auth)
    ev, got = set(), 0
    for a in order:
        if got >= target:
            break
        ev.add(a)
        got += counts[a]
    return s.isin(ev).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--focus", default="meaning in life")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    n = len(rid)
    A = len(args.arms)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells_from_factor_col([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])

    ch = pd.read_csv(args.champion, keep_default_na=False)
    champ = cells_from_factor_col(ch["factors"], len(ch))
    rates = champ.sum(0) / len(ch)
    eval_frac = len(ch) / (len(ch) + n)

    prot_j = [FACTOR_TAXONOMY.index(t) for t in PROTECTIVE]
    print(f"OOF {n} posts / {pd.unique(auth).size} authors   arms {args.arms}")

    # ---------------- 1. per-arm AP ----------------------------------------
    AP = np.zeros((A + 1, N))
    for a in range(A):
        for j in range(N):
            AP[a, j] = average_precision_score(Y[:, j], S[a][:, j]) \
                if Y[:, j].sum() else 0.0
    for j in range(N):
        AP[A, j] = average_precision_score(Y[:, j], P[:, j]) \
            if Y[:, j].sum() else 0.0

    print("\n" + "=" * 110)
    print("AVERAGE PRECISION BY ARM -- is the pool diluting a specialist?")
    print("=" * 110)
    hdr = "".join(f"{a:>8}" for a in args.arms)
    print(f"{'label':<44}{'sup':>5}{hdr}{'POOL':>8}{'best':>8}")
    for j in list(range(N)):
        tag = FACTOR_TAXONOMY[j]
        star = "* " if tag in PROTECTIVE else "  "
        row = "".join(f"{AP[a, j]:>8.3f}" for a in range(A))
        ba = int(np.argmax(AP[:A, j]))
        flag = "  <-- arm beats pool" if AP[ba, j] > AP[A, j] + 0.01 else ""
        print(f"{star}{tag:<42}{int(Y[:, j].sum()):>5}{row}{AP[A, j]:>8.3f}"
              f"{args.arms[ba]:>8}{flag}")
    print("  * = protective")

    # ---------------- 2. candidates at champion rates -----------------------
    def build(source_for_prot):
        """source_for_prot: None -> pool; else index into args.arms."""
        Q = P.copy()
        if source_for_prot is not None:
            for j, a in source_for_prot.items():
                Q[:, j] = S[a][:, j]
        return Q

    best_in = {j: int(np.argmax(AP[:A, j])) for j in prot_j}
    prot_arm = args.arms.index("prot") if "prot" in args.arms else None
    cands = {"current": None}
    if prot_arm is not None:
        cands["prot5"] = {j: prot_arm for j in prot_j}
    cands["bestAP"] = best_in

    print("\n" + "=" * 110)
    print("CANDIDATES at the champion's existing rates -- paired author "
          f"bootstrap ({args.boot} resamples)")
    print("=" * 110)
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    rng = np.random.default_rng(args.seed)
    picks = [rng.choice(len(uauth), size=len(uauth), replace=True)
             for _ in range(args.boot)]
    IX = [np.concatenate([by[uauth[i]] for i in pk]) for pk in picks]

    base_pred = topk_pred(P, rates, n)
    base_f = f1_per_label(base_pred, Y)
    print(f"  {'candidate':<12}{'macro':>9}{'protective':>12}{'d vs cur':>10}"
          f"{'95% CI':>20}{'win':>7}")
    for name, src in cands.items():
        Q = build(src)
        pred = topk_pred(Q, rates, n)
        f = f1_per_label(pred, Y)
        if name == "current":
            print(f"  {name:<12}{f.mean():>9.4f}{f[prot_j].mean():>12.4f}"
                  f"{'--':>10}{'--':>20}{'--':>7}")
            continue
        d = np.array([f1_per_label(pred[ix], Y[ix]).mean()
                      - f1_per_label(base_pred[ix], Y[ix]).mean() for ix in IX])
        lo, hi = np.percentile(d, [2.5, 97.5])
        note = "  OPTIMISTIC (chosen in-sample)" if name == "bestAP" else ""
        print(f"  {name:<12}{f.mean():>9.4f}{f[prot_j].mean():>12.4f}"
              f"{f.mean() - base_f.mean():>+10.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{(d > 0).mean():>7.0%}{note}")

    # ---------------- 3. cross-fitted selection -----------------------------
    rng2 = np.random.default_rng(args.seed + 1)
    dx = []
    for _ in range(args.splits):
        E = submission_scale_split(auth, rng2, eval_frac)
        F = ~E
        nE = int(E.sum())
        sel = {}
        for j in prot_j:
            aps = [average_precision_score(Y[F, j], S[a][F, j])
                   if Y[F, j].sum() else 0.0 for a in range(A)]
            aps.append(average_precision_score(Y[F, j], P[F, j])
                       if Y[F, j].sum() else 0.0)
            b = int(np.argmax(aps))
            if b < A:
                sel[j] = b
        Q = build(sel) if sel else P
        a_ = f1_per_label(topk_pred(Q[E], rates, nE), Y[E]).mean()
        b_ = f1_per_label(topk_pred(P[E], rates, nE), Y[E]).mean()
        dx.append(a_ - b_)
    dx = np.array(dx)
    se = dx.std(ddof=1) / np.sqrt(len(dx))
    print(f"\n  {'xfit':<12}{'':>9}{'':>12}{dx.mean():>+10.4f}"
          f"{f'[{dx.mean() - 1.96 * se:+.4f},{dx.mean() + 1.96 * se:+.4f}]':>20}"
          f"{(dx > 0).mean():>7.0%}  selection made on held-out authors")
    print(f"\n  composite = 0.3 x macro delta")

    # ---------------- 4. the error file -------------------------------------
    j = FACTOR_TAXONOMY.index(args.focus)
    print("\n" + "=" * 110)
    print(f"ERROR FILE -- '{args.focus}'   AP {AP[A, j]:.3f}   "
          f"{int(Y[:, j].sum())} gold of {n}")
    print("=" * 110)
    posts = lab.loc[rid, "post"].astype(str).to_numpy()
    golds = lab.loc[rid, "factors"].astype(str).to_numpy()
    p = P[:, j]

    def snip(t, w=110):
        t = " ".join(str(t).split())
        return t[:w] + ("..." if len(t) > w else "")

    fp = [i for i in np.argsort(-p) if Y[i, j] == 0][:args.show]
    print("\nRANKED HIGHEST BUT NOT GOLD -- what the definition over-matches:")
    for i in fp:
        print(f"  p={p[i]:.3f}  {rid[i]}  {snip(posts[i])}")
        print(f"          gold: {snip(golds[i], 95)}")

    fn = [i for i in np.argsort(p) if Y[i, j] == 1][:args.show]
    print(f"\nGOLD BUT RANKED LOWEST -- what the definition is missing:")
    for i in fn:
        print(f"  p={p[i]:.3f}  {rid[i]}  {snip(posts[i])}")

    print("\n" + "=" * 110)
    print("HOW TO DECIDE")
    print("=" * 110)
    print("If `prot5` wins with its CI clear of zero, ship it -- it is a "
          "0-parameter source swap")
    print("on 5 of 24 independent labels and cannot touch the other 19.")
    print("If it does not, the ranking is bad in EVERY arm, which means the "
          "definitions or the")
    print("training signal are the problem and a fine-tuned factor model is "
          "the only lever left.")
    print("Read the error file either way: if the misses look nothing alike, "
          "the label is")
    print("annotator-noisy and 0.29 may simply be its ceiling.")


if __name__ == "__main__":
    main()

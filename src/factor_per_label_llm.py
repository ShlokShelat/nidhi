"""
factor_per_label_llm.py

PER-LABEL, not scalar. Where is the factor column weak, and does the 72B beat
it THERE?

WHY THIS IS WORTH RE-ASKING. The factor-LLM lane was closed on an AGGREGATE
number. That is exactly how a risk-definitions lane was wrongly closed: a
model that IMPROVED indicator and ideation was closed on an aggregate loss
concentrated entirely in one class. A scalar mean hides per-label structure.

Factors make that hiding both more likely and more exploitable, because the
scorer MACRO-averages 24 labels:

  * every label carries 1/24 of the factor macro regardless of prevalence
  * factor weight is 0.3, so one label's F1 is worth 0.3/24 = 0.0125 composite
  * per-label F1 is INDEPENDENT: swapping the source for label j cannot change
    any other label's score. No dilution, no absorption.

That last property is why this is not the risk override family. Those failed
because a weighted-F1 vote lets weak members outvote a strong one on shared
posts, and because risk and evidence move together through the indicator
suppression rule. A factor label swap touches one column of a 24-column matrix
and nothing else. It is the cleanest single-variable change available.

WHAT IT COMPARES, and both sides are honest:

  ours  the 8-arm bag EXACTLY as deployed -- the same arms bag8 ships
        (mnli dmnli cmine prot d4 qnli mnli7 lnli), logit-mean pooled, decoded
        at the champion's per-label RATES. Zero fitted parameters. Verified
        against the shipped test column before anything else is reported.
  llm   artifacts/factor_llm_oof.csv, columns llm_0..llm_23, which map to
        FACTOR_TAXONOMY[j] by construction (factor_llm_opinions.py:177). Hard
        binary opinions -- there is no threshold to tune, so a swap costs
        ZERO parameters too.

DECISION RULE, PRE-STATED. Swap label j only if the paired author bootstrap on
(F1_llm - F1_ours) excludes zero. 153 authors is the unit, not 1635 posts --
posts by one author are not independent, and a post-level CI would be roughly
sqrt(1635/153) = 3.3x too narrow. Labels whose CI spans zero stay with ours,
however tempting the point estimate.

The ORACLE row is the ceiling if every label picked its better source with
hindsight. It is NOT achievable -- it is the number the honest selection has to
be judged against, so a selection that captures most of it is doing well and
one that captures all of it is overfitting.

Usage (cluster, CPU, ~1 min):
  python src/factor_per_label_llm.py
  python src/factor_per_label_llm.py --boot 2000
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
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]

# The organizers' five protective factors. Measured at 0.504 macro against the
# risk factors' 0.651, the largest per-component gap anywhere in the pipeline.
PROTECTIVE = {
    "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
}


# ---------------------------------------------------------------- loading

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


def resolve_test(d, a):
    for c in (f"{d}/{a}_cv_test.npz", f"{d}/{a}_test.npz",
              f"{d}/nli_ensemble_{a}_test.npz"):
        if os.path.exists(c):
            return c
    return None


def logit_mean(S):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def topk_pred(PE, k_vec, nE):
    """Quantile decode -- per label fire on the top-k probabilities, ties
    together, exactly as factor_bag_ship.py decodes the test set."""
    on = np.zeros((nE, N), dtype=np.int8)
    for j in range(N):
        k = min(max(int(k_vec[j]), 1), nE - 1)
        cut = np.sort(PE[:, j])[-k]
        on[:, j] = (PE[:, j] >= cut).astype(np.int8)
    return on


def cells_from_factor_col(col, n):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    M = np.zeros((n, N), dtype=np.int8)
    for i, cell in enumerate(col):
        for x in set(parse_factors(cell)):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


# ---------------------------------------------------------------- scoring

def f1_per_label(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    fp = (P * (1 - Y)).sum(0)
    fn = ((1 - P) * Y).sum(0)
    den = 2 * tp + fp + fn
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def prf(P, Y, j):
    tp = int((P[:, j] & Y[:, j]).sum())
    fp = int((P[:, j] & (1 - Y[:, j])).sum())
    fn = int(((1 - P[:, j]) & Y[:, j]).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--llm", default="artifacts/factor_llm_oof.csv")
    ap.add_argument("--llm-test", default="artifacts/factor_llm_oof.csv.extra.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    # ---- our deployed bag, on OOF -----------------------------------------
    rid, S = load_oof(args.dir, args.arms, args.k)
    O = logit_mean(S)
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells_from_factor_col([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])

    ch = pd.read_csv(args.champion, keep_default_na=False)
    champ = cells_from_factor_col(ch["factors"], len(ch))
    champ_rate = champ.sum(0) / len(ch)
    ours = topk_pred(O, np.round(champ_rate * n), n)

    print(f"OOF {n} posts / {pd.unique(auth).size} authors   arms {args.arms}")
    print(f"champion column {args.champion}: "
          f"{champ.sum() / len(ch):.2f} factors/post")
    print(f"our OOF bag decoded at champion rates: "
          f"{ours.sum() / n:.2f} factors/post")

    # ---- PROVENANCE GUARD: does this bag reproduce the shipped column? -----
    # A rebuild that silently uses the wrong run list can change many posts
    # without raising any error. Never trust a reconstruction that was not
    # checked against the artefact it claims to be.
    T, tid, ok = [], None, True
    for a in args.arms:
        p = resolve_test(args.dir, a)
        if p is None:
            ok = False
            break
        z = np.load(p, allow_pickle=True)
        T.append(z["probs"].astype(float))
        tid = [str(x) for x in z["row_id"]]
    if ok:
        TT = logit_mean(np.stack(T))
        nT = len(tid)
        repro = topk_pred(TT, np.round(champ_rate * nT), nT)
        order = {r: i for i, r in enumerate(ch["row_id"].astype(str))}
        perm = [order[r] for r in tid]
        diff = int((repro != champ[perm]).sum())
        print(f"\nPROVENANCE: reconstruction vs shipped factor column -- "
              f"{diff} of {nT * N} cells differ")
        if diff > 0.02 * nT * N:
            print("  *** WARNING: this is NOT the deployed bag. Every number "
                  "below describes a different model than the one we ship. "
                  "Fix the arm list before acting on any of it. ***")
        else:
            print("  reconstruction is faithful -- numbers below describe the "
                  "deployed column")
    else:
        print("\nPROVENANCE: no test dumps found, cannot verify the "
              "reconstruction. Treat everything below as provisional.")

    # ---- the LLM's opinions ------------------------------------------------
    lm = pd.read_csv(args.llm, keep_default_na=False)
    lm["row_id"] = lm["row_id"].astype(str)
    lm = lm.set_index("row_id")
    miss = [r for r in rid if r not in lm.index]
    if miss:
        raise SystemExit(f"LLM file lacks {len(miss)} OOF rows (e.g. {miss[:3]})")
    L = lm.loc[rid, [f"llm_{j}" for j in range(N)]].to_numpy().astype(np.int8)
    print(f"LLM opinions: {L.sum() / n:.2f} factors/post   "
          f"gold: {Y.sum() / n:.2f} factors/post")

    # ---- per-label table with a paired AUTHOR bootstrap --------------------
    f_ours = f1_per_label(ours, Y)
    f_llm = f1_per_label(L, Y)

    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    rng = np.random.default_rng(args.seed)
    D = np.zeros((args.boot, N))
    for b in range(args.boot):
        pick = rng.choice(len(uauth), size=len(uauth), replace=True)
        ix = np.concatenate([by[uauth[i]] for i in pick])
        D[b] = f1_per_label(L[ix], Y[ix]) - f1_per_label(ours[ix], Y[ix])
    lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
    winr = (D > 0).mean(0)

    print("\n" + "=" * 100)
    print("PER-LABEL: our deployed bag vs the 72B, paired author bootstrap "
          f"({args.boot} resamples, {len(uauth)} authors)")
    print("=" * 100)
    print(f"{'label':<44}{'sup':>5}{'ourP':>6}{'ourR':>6}{'ourF':>7}"
          f"{'llmP':>6}{'llmR':>6}{'llmF':>7}{'dF1':>8}{'95% CI':>18}{'win':>6}")
    order = np.argsort(-(f_llm - f_ours))
    swap = []
    for j in order:
        op, orr, of = prf(ours, Y, j)
        lp, lr, lf = prf(L, Y, j)
        d = f_llm[j] - f_ours[j]
        star = ""
        if lo[j] > 0:
            star = " <-- SWAP"
            swap.append(j)
        elif hi[j] < 0:
            star = "  (ours wins)"
        tag = FACTOR_TAXONOMY[j]
        if tag in PROTECTIVE:
            tag = "* " + tag
        print(f"{tag:<44}{int(Y[:, j].sum()):>5}{op:>6.2f}{orr:>6.2f}{of:>7.3f}"
              f"{lp:>6.2f}{lr:>6.2f}{lf:>7.3f}{d:>+8.3f}"
              f"{f'[{lo[j]:+.3f},{hi[j]:+.3f}]':>18}{winr[j]:>6.0%}{star}")
    print("  * = protective factor")

    # ---- what the selection is worth --------------------------------------
    print("\n" + "=" * 100)
    print("WHAT IT BUYS")
    print("=" * 100)
    m_ours = float(f_ours.mean())
    m_llm = float(f_llm.mean())
    prot_ix = [j for j in range(N) if FACTOR_TAXONOMY[j] in PROTECTIVE]
    risk_ix = [j for j in range(N) if FACTOR_TAXONOMY[j] not in PROTECTIVE]
    print(f"  our macro-F1            {m_ours:.4f}   "
          f"(risk {f_ours[risk_ix].mean():.4f} / protective "
          f"{f_ours[prot_ix].mean():.4f})")
    print(f"  LLM macro-F1            {m_llm:.4f}   "
          f"(risk {f_llm[risk_ix].mean():.4f} / protective "
          f"{f_llm[prot_ix].mean():.4f})")

    sel = f_ours.copy()
    for j in swap:
        sel[j] = f_llm[j]
    orac = np.maximum(f_ours, f_llm)
    print(f"\n  SELECTED swap ({len(swap)} labels, CI excludes zero)")
    print(f"    macro-F1            {sel.mean():.4f}   "
          f"delta {sel.mean() - m_ours:+.4f}   "
          f"COMPOSITE {0.3 * (sel.mean() - m_ours):+.4f}")
    print(f"  ORACLE (hindsight, NOT achievable -- the ceiling to judge against)")
    print(f"    macro-F1            {orac.mean():.4f}   "
          f"delta {orac.mean() - m_ours:+.4f}   "
          f"COMPOSITE {0.3 * (orac.mean() - m_ours):+.4f}")

    if swap:
        print("\n  labels to swap:")
        for j in swap:
            print(f"    {FACTOR_TAXONOMY[j]:<44}{f_ours[j]:.3f} -> "
                  f"{f_llm[j]:.3f}   ({f_llm[j] - f_ours[j]:+.3f})")
        print("\n  NEXT: build the test column with exactly these labels taken "
              "from")
        print(f"  {args.llm_test} and every other label left untouched, then "
              "verify")
        print("  that the untouched labels are bit-identical to the champion's.")
    else:
        print("\n  NO label clears the bar. The lane is closed on a PER-LABEL "
              "measurement now, not a scalar mean -- which is the check that "
              "was missing the first time.")

    print("\nREAD THE PRECISION AND RECALL COLUMNS, not just dF1. A label where "
          "the LLM wins on")
    print("recall while losing precision behaves differently under macro-F1 "
          "than one where it")
    print("wins on both, and only the second kind is safe to swap.")


if __name__ == "__main__":
    main()

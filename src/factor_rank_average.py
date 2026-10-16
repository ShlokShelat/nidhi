"""
factor_rank_average.py

Does RANK-averaging fix the documented "more members makes it worse" pathology?

THE MEASURED FACT THIS ADDRESSES. From the factor arm results:

    "Individual model quality does NOT predict pair quality. dmnli's two seeds
     score 0.6349/0.6220 against mnli's 0.6218/0.6116 and the dmnli PAIR is
     still worse. Averaging more models narrows the probability distribution
     and the averaged-threshold step degrades with it -- four runs are worse
     than two. Do not add members to this procedure."

That is a real, repeatedly observed pathology, and ensembling more members is
supposed to help, not hurt. `predict_factors_nli.py:63` averages PROBABILITIES
and line 69 averages THRESHOLDS separately -- two operations that do not
compose.

WHY THIS IS NOT THE THRESHOLD-POLICY LANE AGAIN. `champion_factor_redecide.py`
already tried the obvious fix: refit the threshold ON the averaged
probabilities. Measured to be essentially nothing, with a CI spanning zero.
That result is informative. If the pathology were purely a MISPLACED
CUTOFF, refitting would have fixed it. It did not. So either there is no
pathology, or the damage is in the RANKING itself -- and no threshold can
repair a ranking.

Probability-averaging lets a badly calibrated but confident member dominate the
mean: a model that outputs 0.9 where others say 0.4 moves the average far more
than one that outputs 0.5 where others say 0.4, regardless of which is right.
Rank-averaging equalises every member's influence by construction and is
invariant to any monotone recalibration of any member. It is the standard fix
for exactly this failure, it has ZERO parameters, and nothing in this
repository implements it.

ZERO PARAMETERS MATTERS HERE. The measured transfer law in this project is that
0-parameter changes over-deliver, 1 fitted parameter transfers only partially,
and heavily parameterised changes transfer negatively. Rank-averaging fits
nothing.

THE FALSIFIER, FIXED BEFORE LOOKING. A small positive delta would be
unconvincing -- the +-0.0157 null band is wide enough that noise alone can
produce one. So the test is a PREDICTED SIGN REVERSAL, which noise cannot
easily fake:

    under mean-prob:  4 members must score WORSE than 2  (the documented fact)
    under rank-avg:   4 members must score BETTER than 2 (the prediction)

If rank-averaging does not reverse that ordering, the pathology is not
calibration and this lane closes regardless of any headline delta.

All policies are judged with thresholds CROSS-FITTED over author-disjoint
halves. Rank space has a different scale from probability space, so a shared
cutoff would be meaningless -- every rule refits, which keeps the comparison
fair rather than flattering the incumbent.

Usage (on the cluster, CPU only, seconds):
  python src/factor_rank_average.py --runs mnli mnli_s2 dmnli dmnli_s2
  python src/factor_rank_average.py            # auto-discovers *_val.npz
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def f1_at(pred, y):
    tp = int((pred & y).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + int((pred & (1 - y)).sum())
                     + int(((1 - pred) & y).sum()))


def macro(P, Y):
    return float(np.mean([f1_at(P[:, j], Y[:, j]) for j in range(N)]))


def best_th(pr, y):
    best, bt = -1.0, 0.5
    for t in GRID:
        v = f1_at((pr >= t).astype(int), y)
        if v > best:
            best, bt = v, t
    return bt


def author_halves(auth):
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        k = "A" if size["A"] <= size["B"] else "B"
        half[a] = k
        size[k] += c[a]
    return s.map(half).to_numpy()


# ----------------------------- combination rules -------------------------
def r_mean(S):
    return np.mean(S, axis=0)


def r_median(S):
    return np.median(S, axis=0)


def r_logit(S):
    X = np.clip(np.asarray(S), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def r_rank(S):
    """Per member, per LABEL, replace probabilities by their normalised rank
    across posts, then average. Invariant to any monotone recalibration of any
    member, so a member cannot dominate by being loud."""
    out = np.zeros_like(np.asarray(S, dtype=float))
    for m, P in enumerate(S):
        for j in range(P.shape[1]):
            o = np.argsort(np.argsort(P[:, j], kind="stable"), kind="stable")
            out[m, :, j] = o / max(len(o) - 1, 1)
    return out.mean(axis=0)


# ---------------------------------------------------------------- families
# Decided by ARCHITECTURE TAXONOMY, not fitted from data -- so this is a
# 0-parameter change, the class most likely to over-deliver relative to
# more heavily parameterised alternatives, which tend to transfer only
# partially, or negatively.
DECODER_LLM = {"qnli", "mnli7", "lnli", "mist", "qnli7", "llama", "mistral"}


def family_of(tag):
    return "llm" if tag in DECODER_LLM else "encoder"


def r_flogit(S, tags):
    """Logit-average WITHIN each architecture family, then average the family
    means with EQUAL weight.

    WHY. The measured driver of ensemble gain is cross-family diversity, not
    member count: adding Qwen-14B helped more than adding additional DeBERTa
    variants did. But flat averaging gives the 5-member DeBERTa bloc 5/6 of
    the vote, muting exactly the member carrying the marginal value. Equal
    weight per FAMILY raises the decorrelated side's influence.

    This is NOT a plain rank-average approach, which equalises MEMBERS on a
    set stuffed with DeBERTa duplicates without addressing family imbalance;
    this equalises FAMILIES, which is the variable the qnli/aspv2/d6
    experiment isolated."""
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    L = np.log(X / (1 - X))
    fams = {}
    for i, t in enumerate(tags):
        fams.setdefault(family_of(t), []).append(i)
    means = [L[ix].mean(axis=0) for ix in fams.values()]
    return 1.0 / (1.0 + np.exp(-np.mean(means, axis=0)))


RULES = {"mean prob (SHIPPED)": lambda S, t: r_mean(S),
         "rank average": lambda S, t: r_rank(S),
         "logit average": lambda S, t: r_logit(S),
         "median prob": lambda S, t: r_median(S),
         "FAMILY-balanced logit": r_flogit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--source", default="auto", choices=["auto", "oof", "val"],
                    help="OOF concatenates {run}_cv_oof_f0..4.npz -- 1635 "
                         "honestly out-of-fold posts, the stronger instrument "
                         "and what factors_encoder.sh actually writes. VAL "
                         "reads {run}_val.npz, 372 posts. Auto prefers OOF.")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = args.source
    if src == "auto":
        src = "oof" if glob.glob(f"{args.dir}/*_cv_oof_f0.npz") else "val"
        print(f"source: {src} (auto-detected)")

    runs = args.runs
    if not runs:
        pat = "*_cv_oof_f0.npz" if src == "oof" else "*_val.npz"
        cut = len("_cv_oof_f0.npz") if src == "oof" else len("_val.npz")
        runs = sorted(os.path.basename(p)[:-cut]
                      for p in glob.glob(f"{args.dir}/{pat}"))
        print(f"auto-discovered {len(runs)} runs: {runs}")
    if len(runs) < 2:
        raise SystemExit(f"need >=2 runs in {args.dir} (source={src})")

    def load_run(r):
        """Return (probs, row_ids, own_thresholds) for one member.

        The saved thresholds are what the DEPLOYED policy averages, and they
        are the suspected site of the pathology -- so they must be loaded, not
        recomputed. For OOF the cutoff differs per fold; the member's own
        vector is the mean across its folds."""
        if src == "val":
            v = np.load(f"{args.dir}/{r}_val.npz", allow_pickle=True)
            return (v["probs"].astype(float), [str(x) for x in v["row_id"]],
                    v["thresholds"].astype(float))
        pr, ids, th = [], [], []
        for f in range(args.k):
            p = f"{args.dir}/{r}_cv_oof_f{f}.npz"
            if not os.path.exists(p):
                raise SystemExit(f"{r}: missing fold file {p} -- an incomplete "
                                 f"bag would silently compare different post "
                                 f"sets across members")
            v = np.load(p, allow_pickle=True)
            pr.append(v["probs"].astype(float))
            ids.extend(str(x) for x in v["row_id"])
            th.append(v["thresholds"].astype(float))
        return np.concatenate(pr, axis=0), ids, np.mean(th, axis=0)

    S, TH, rid = [], [], None
    for r in runs:
        pr, ids, th = load_run(r)
        if rid is None:
            rid = ids
        elif ids != rid:
            if set(ids) != set(rid):
                raise SystemExit(f"{r}: row_id set differs from the first run")
            order = {k: i for i, k in enumerate(ids)}
            pr = pr[[order[k] for k in rid]]
        S.append(pr)
        TH.append(th)
    S = np.stack(S)
    TH = np.stack(TH)
    print(f"{S.shape[0]} members x {S.shape[1]} {src} posts x {S.shape[2]} labels")

    gold_path = args.labeled if src == "oof" else args.val
    val = pd.read_csv(gold_path, keep_default_na=False)
    val["row_id"] = val["row_id"].astype(str)
    val = val.set_index("row_id")
    miss = [r for r in rid if r not in val.index]
    if miss:
        raise SystemExit(f"{len(miss)} row_ids absent from {gold_path}")
    val = val.loc[rid]
    Y = np.zeros((len(rid), N), dtype=int)
    for i, r in enumerate(rid):
        for f in parse_factors(val.loc[r, "factors"]):
            if f in FACTOR_TAXONOMY:
                Y[i, FACTOR_TAXONOMY.index(f)] = 1
    auth = val["anon_user_id"].to_numpy()
    half = author_halves(auth)
    A, B = half == "A", half == "B"
    print(f"gold {Y.sum()} positives, {Y.sum(1).mean():.2f}/post   "
          f"halves {int(A.sum())}/{int(B.sum())}   "
          f"{pd.unique(auth).size} authors")

    def xfit(idx, rule):
        """Combine members `idx` with `rule`, threshold cross-fitted by half."""
        C = RULES[rule](S[list(idx)], [runs[i] for i in idx])
        X = np.zeros_like(Y)
        for src, dst in ((B, A), (A, B)):
            for j in range(N):
                t = best_th(C[src, j], Y[src, j])
                X[dst, j] = (C[dst, j] >= t).astype(int)
        return X

    def deployed(idx, rule):
        """THE SHIPPED POLICY, and the suspected bug. Combine the members'
        probabilities, then apply the MEAN OF THEIR OWN saved thresholds.
        Fits nothing on the eval data, so unlike `xfit` it is directly
        comparable across member counts -- and it is the only policy in which
        the documented 'more members is worse' effect can appear at all, since
        averaging shrinks the combined distribution while the averaged cutoff
        stays where the individual models put it."""
        C = RULES[rule](S[list(idx)], [runs[i] for i in idx])
        t = TH[list(idx)].mean(axis=0)
        return (C >= t[None, :]).astype(int)

    # Known per-label prevalence. At deploy time this is a KNOWN quantity --
    # and the allfac probe proved test factor density matches train (0.2011
    # measured vs 0.1981 predicted), so transferring it is legitimate.
    RATE = Y.mean(axis=0)

    def ratematch(idx, rule):
        """Cutoff placed so the PREDICTED POSITIVE RATE per label reproduces
        known prevalence. Uses the marginal rate only -- never a per-post
        label -- so it fits NOTHING on the evaluation set and is directly
        comparable across member counts. This is the honest stand-in for the
        deployed policy, whose saved OOF thresholds were fitted on these very
        posts. Note there are two ratematch variants in this project; this is
        the cutoff-transfer one (0.6434), not exact-top-k (0.6093)."""
        C = RULES[rule](S[list(idx)], [runs[i] for i in idx])
        X = np.zeros_like(Y)
        for j in range(N):
            k = int(round(RATE[j] * len(C)))
            k = min(max(k, 1), len(C) - 1)
            X[:, j] = (C[:, j] >= np.sort(C[:, j])[-k]).astype(int)
        return X

    POLICY = {"refit (xfit, honest)": xfit,
              "ratematch (fits nothing)": ratematch}

    # ------------------- the sign-reversal test --------------------------
    kall = list(range(len(runs)))
    # The champion's real pair if present -- NOT the first two alphabetically,
    # which is a meaningless baseline and was the flaw in the first run of this
    # script (it compared cmine+d4 against everything).
    pref = [c for c in (("mnli", "dmnli"), ("mnli", "mnli_s2"),
                        ("mnli", "cmine")) if all(x in runs for x in c)]
    k2 = ([runs.index(x) for x in pref[0]] if pref
          else list(range(min(2, len(runs)))))
    print("\n" + "=" * 78)
    print("THE SIGN-REVERSAL TEST -- does adding members still hurt?")
    print("=" * 78)
    print(f"pair = {[runs[i] for i in k2]}   full = {len(kall)} members")
    print("\nSHIPPED policy (combined probs + AVERAGED member thresholds):")
    print("  CAVEAT: for OOF the saved thresholds were fitted on the very posts")
    print("  being scored, so these numbers are OPTIMISTIC. The REFIT block")
    print("  below is cross-fitted and is the honest one.")
    print(f"  {'rule':<22}{'pair':>10}{'full':>10}{'delta':>10}{'verdict':>12}")
    print("  " + "-" * 62)
    for rule in RULES:
        if rule == "rank average":
            # Applying a PROBABILITY cutoff to normalised RANKS is meaningless
            # -- the two live on different scales. Skip rather than print a
            # number that looks like a catastrophic result but is an artefact.
            print(f"  {rule:<22}{'n/a -- rank scale != probability scale':>44}")
            continue
        sa, sb = macro(deployed(k2, rule), Y), macro(deployed(kall, rule), Y)
        print(f"  {rule:<22}{sa:>10.4f}{sb:>10.4f}{sb - sa:>+10.4f}"
              f"{('helps' if sb > sa else 'HURTS'):>12}")
    print("\nREFIT policy (threshold fitted on the COMBINED probs, xfit):")
    print(f"  {'rule':<22}{'pair':>10}{'full':>10}{'delta':>10}{'verdict':>12}")
    print("  " + "-" * 62)
    store = {}
    for rule in RULES:
        a, b = xfit(k2, rule), xfit(kall, rule)
        sa, sb = macro(a, Y), macro(b, Y)
        store[rule] = (a, b, sa, sb)
        print(f"  {rule:<22}{sa:>10.4f}{sb:>10.4f}{sb - sa:>+10.4f}"
              f"{('helps' if sb > sa else 'HURTS'):>12}")

    print("\nWHAT DECIDES THE MECHANISM. If mean-prob HURTS under SHIPPED and")
    print("HELPS under REFIT, the pathology is the averaged-threshold step, not")
    print("probability averaging -- and 'do not add members' was a workaround")
    print("for a fixable bug rather than a property of ensembling. If it helps")
    print("under BOTH, the pathology does not reproduce here at all and the")
    print("original observation was specific to the 372-post val recipe.")

    # ------------------- head-to-head on all members ---------------------
    base = macro(store["mean prob (SHIPPED)"][1], Y)
    print("\n" + "=" * 74)
    print(f"ALL {len(runs)} MEMBERS -- paired author bootstrap vs mean-prob")
    print("=" * 74)
    uniq = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uniq}
    rng = np.random.default_rng(args.seed)
    idxs = [np.concatenate([by[uniq[j]] for j in
                            rng.choice(len(uniq), len(uniq), replace=True)])
            for _ in range(args.boot)]
    print(f"{'rule':<24}{'macroF1':>10}{'delta':>9}{'95% CI':>22}"
          f"{'win%':>7}{'compos':>9}")
    print("-" * 81)
    for rule in RULES:
        P = store[rule][1]
        d = np.array([macro(P[ix], Y[ix])
                      - macro(store["mean prob (SHIPPED)"][1][ix], Y[ix])
                      for ix in idxs])
        lo, hi = np.percentile(d, [2.5, 97.5])
        s = macro(P, Y)
        print(f"{rule:<24}{s:>10.4f}{s - base:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}"
              f"{100 * float((d > 0).mean()):>6.1f}%{0.3 * (s - base):>+9.4f}")

    # ------------------- best subset under each rule ---------------------
    if len(runs) <= 6:
        print("\n" + "=" * 74)
        print("EVERY SUBSET -- does rank-avg reward size where mean-prob "
              "punishes it?")
        print("=" * 74)
        print(f"{'members':<10}{'subset':<34}", end="")
        for rule in RULES:
            print(f"{rule.split()[0]:>11}", end="")
        print()
        print("-" * 78)
        for k in range(2, len(runs) + 1):
            for c in itertools.combinations(range(len(runs)), k):
                nm = "+".join(runs[i] for i in c)
                print(f"{k:<10}{nm[:33]:<34}", end="")
                for rule in RULES:
                    print(f"{macro(xfit(c, rule), Y):>11.4f}", end="")
                print()

    # ------------------- SIZE TREND, the robust read ---------------------
    # The real question is not which subset wins -- the max of N noisy cells
    # is guaranteed to look good, which is the winner's curse that has cost
    # nine submissions. It is whether the MEAN at each size rises. That uses
    # every subset, so noise cancels instead of being selected for.
    if len(runs) <= 8:
        print("\n" + "=" * 78)
        print("SIZE TREND -- mean over ALL subsets of each size (not the best)")
        print("=" * 78)
        for pname, pfn in POLICY.items():
            print(f"\n{pname}")
            print(f"  {'k':>3}{'subsets':>9}", end="")
            for rule in RULES:
                print(f"{rule.split()[0]:>10}", end="")
            print()
            print("  " + "-" * 52)
            acc = {r: ([], []) for r in RULES}
            for k in range(2, len(runs) + 1):
                combos = list(itertools.combinations(range(len(runs)), k))
                print(f"  {k:>3}{len(combos):>9}", end="")
                for rule in RULES:
                    vals = [macro(pfn(c, rule), Y) for c in combos]
                    print(f"{np.mean(vals):>10.4f}", end="")
                    acc[rule][0].extend([k] * len(vals))
                    acc[rule][1].extend(vals)
                print()
            print(f"  {'slope/member':<12}", end="")
            for rule in RULES:
                s = np.polyfit(acc[rule][0], acc[rule][1], 1)[0]
                print(f"{s:>+10.5f}", end="")
            print()
            print(f"  {'2->6 compos':<12}", end="")
            for rule in RULES:
                s = np.polyfit(acc[rule][0], acc[rule][1], 1)[0]
                print(f"{0.3 * 4 * s:>+10.4f}", end="")
            print()
        print("\nSHIP TEST: the slope must be POSITIVE under BOTH policies and")
        print("for the mean/logit rules. This is the same policy-consistency")
        print("check that catches a sign flipping between fitting procedures")
        print("before it reaches a submission. A positive slope under one")
        print("policy only is not evidence -- it is the same failure mode again.")

    print("\nNOTE. Val is 372 posts and these thresholds are cross-fitted but")
    print("still fitted -- this instrument's val-based predictions do not")
    print("always match what a submission would show. Read the SIGN REVERSAL, not the magnitude. A")
    print("clean reversal plus a positive all-member delta justifies ONE")
    print("submission; a small positive delta on its own does not.")


if __name__ == "__main__":
    main()

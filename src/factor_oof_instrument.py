"""
factor_oof_instrument.py

Pool the five author-disjoint folds into one 1635-post factor measurement, and
compare arms by PAIRED author-bootstrap deltas.

WHY IT REPLACES THE 372-POST VAL INSTRUMENT. The standard error of an absolute
macro-F1 on 372 posts is about 0.0215, which is why two runs of an identical
configuration came out 0.0207 apart and why every arm ranking this campaign sat
inside its own measurement error. Pooling to 1635 posts takes that to roughly
0.003. Nothing about the model changes -- only how many posts it is judged on,
and whether those posts were ever fitted on.

THREE THINGS THIS DOES THAT THE OLD ONE DID NOT.

1. It scores posts that no part of the fitting ever saw. In the old instrument
   the thresholds AND the best epoch were both chosen on the very posts being
   scored, so the level was flattered. Here EVAL_k is disjoint from TRAIN_k and
   THRESH_k by author. Expect the absolute numbers to come out LOWER than the
   0.6218-style figures; that is the flattery being removed, not a regression.

2. It pools rather than averages folds. Per fold the rarest label has 1-2 gold
   positives, so a per-fold macro is meaningless. Pooling gives every label its
   full 1635-post support before any F1 is computed.

3. It compares arms PAIRED on identical posts. Post difficulty, author mix and
   rare-label scarcity are shared, so they cancel in the delta. This is the
   single biggest reason the old readings were unusable: absolute numbers were
   being differenced by eye, which adds the two noises instead of cancelling
   them.

WHAT IS STILL NOT MEASURED. The bootstrap resamples authors, so it captures
evaluation sampling -- the term that turns out to dominate. It does NOT capture
training nondeterminism or threshold-fitting instability. To bound those, run
one arm twice under a different seed pair and pass it as --null; its delta is a
draw from the true no-difference distribution and is the honest bar. A CI
excluding zero is the weaker test.

KNOWN BIAS. TRAIN is ~1015 posts against deployment's 1263, so transfer-learning
arms (anything MNLI-pretrained) read somewhat high. Upper bound, not point
estimate. See src/make_factor_oof_splits.py.

Usage:
  python src/factor_oof_instrument.py --arm mnli
  python src/factor_oof_instrument.py --arm mnli --arm mnlicm --arm cminep \
      --null mnlirep --vs mnli --per-label
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


def gold_matrix(df: pd.DataFrame) -> np.ndarray:
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(df), N), dtype=np.int8)
    for i, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def per_label_f1(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """zero_division=0, matching scoring.factors_macro_f1 exactly."""
    tp = (P & Y).sum(0).astype(float)
    fp = (P & (1 - Y)).sum(0).astype(float)
    fn = ((1 - P) & Y).sum(0).astype(float)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def macro_f1(P: np.ndarray, Y: np.ndarray) -> float:
    return float(per_label_f1(P, Y).mean())


GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def author_halves(auth):
    """Split authors into two balanced halves, largest-first. Identical to the
    implementation in factor_threshold_policies.py and evidence_vote_oof.py, so
    every instrument in the project cross-fits on the SAME partition."""
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        p = "A" if size["A"] <= size["B"] else "B"
        half[a] = p
        size[p] += c[a]
    return s.map(half).to_numpy()


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


def binarise(Pr, Y, A, B, policy, dumped):
    """Turn probabilities into predictions under a policy applied IDENTICALLY
    to every arm. Cross-fitted: cutoffs come from the other author half."""
    if policy == "deployed":
        return dumped
    X = np.zeros_like(Y)
    for src, dst in ((B, A), (A, B)):
        th = np.empty(N)
        for j in range(N):
            if policy == "refit":
                th[j] = argmax_threshold(Pr[src, j], Y[src, j])
            else:      # ratematch: reproduce the prevalence implied count.
                       # Fits NOTHING -- prevalence is known in advance -- which
                       # is why it can be deployed with no held-out set at all.
                k = int(round(float(Y[src, j].mean()) * int(dst.sum())))
                k = min(max(k, 1), int(dst.sum()))
                th[j] = float(np.sort(Pr[dst, j])[-k])
        X[dst] = (Pr[dst] >= th).astype(np.int8)
    return X


def load_arm(tag: str, d: str, k: int):
    """Pool the k fold dumps. Each npz already carries the DEPLOYED decision
    inputs: probs averaged over the two runs, and the two runs' own thresholds
    averaged (predict_factors_nli.py:63-69). Applying one to the other here is
    the same operation that produces the submission.

    Returns the raw probabilities as well, so a policy other than `deployed`
    can rebinarise them."""
    Pb, rid, ths, PR = [], [], [], []
    for f in range(k):
        p = os.path.join(d, f"{tag}_f{f}.npz")
        if not os.path.exists(p):
            have = sorted(int(x.split("_f")[-1][0]) for x in os.listdir(d)
                          if x.startswith(f"{tag}_f") and x.endswith(".npz"))
            msg = (f"missing {p}\n"
                   f"  {tag} has folds {have} of {list(range(k))}; "
                   f"use --k {len(have)} for an interim read\n"
                   f"  sbatch --export=ALL,FOLD={f},TAG={tag},MODEL=<same as "
                   f"the other folds> scripts/factors_oof_fold.sh")
            # A replica arm MUST keep its own seed pair or it silently becomes a
            # duplicate of the reference and the null band collapses to zero.
            if tag.endswith("rep"):
                msg += ("\n  NOTE this looks like a replica arm -- it also "
                        "needs SEED1/SEED2 set to the\n  SAME values its other "
                        "folds used, or it is not a replica at all.")
            raise SystemExit(msg)
        z = np.load(p, allow_pickle=True)
        pr = np.asarray(z["probs"], dtype=float)
        th = np.asarray(z["thresholds"], dtype=float)
        Pb.append((pr >= th).astype(np.int8))
        PR.append(pr)
        rid += list(z["row_id"])
        ths.append(th)
    return (np.concatenate(Pb, axis=0), rid, np.mean(ths, axis=0),
            np.concatenate(PR, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="tag under --dir; repeatable")
    ap.add_argument("--null", default=None,
                    help="tag of an identical-config replica of --vs; its "
                         "delta measures the real noise floor")
    ap.add_argument("--vs", default=None, help="reference arm (default: first)")
    ap.add_argument("--dir", default="artifacts/factor_oof")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-label", action="store_true")
    ap.add_argument("--policy", default="deployed",
                    choices=["deployed", "refit", "ratematch"],
                    help="how EVERY arm is binarised before comparing. "
                         "`deployed` uses each arm's own dumped thresholds -- "
                         "wrong for an ALLDATA arm, whose thresholds were fitted "
                         "on data it memorised. `refit` and `ratematch` are "
                         "cross-fitted over author halves and applied "
                         "identically to all arms, which is the only fair way "
                         "to compare arms with different threshold provenance.")
    args = ap.parse_args()

    tags = list(args.arm) + ([args.null] if args.null else [])
    lab = pd.read_csv(args.labeled, keep_default_na=False).set_index("row_id")

    arms, probs, rid_ref = {}, {}, None
    for t in tags:
        P, rid, th, PR = load_arm(t, args.dir, args.k)
        if rid_ref is None:
            rid_ref = rid
        elif rid != rid_ref:
            # folds are deterministic, so a mismatch means different splits
            order = {r: i for i, r in enumerate(rid)}
            if set(rid) != set(rid_ref):
                raise SystemExit(f"{t}: scores a different post set")
            idx2 = [order[r] for r in rid_ref]
            P, PR = P[idx2], PR[idx2]
            print(f"NOTE {t}: fold order differed, realigned by row_id")
        arms[t] = P
        probs[t] = PR

    ev = lab.loc[rid_ref].reset_index()
    Y = gold_matrix(ev)
    auth = ev["anon_user_id"].to_numpy()
    uniq = pd.unique(auth)
    by_author = {a: np.flatnonzero(auth == a) for a in uniq}

    ref = args.vs or tags[0]
    if ref not in arms:
        raise SystemExit(f"--vs {ref!r} is not among {list(arms)}")

    if args.policy != "deployed":
        half = author_halves(auth)
        A, B = half == "A", half == "B"
        arms = {t: binarise(probs[t], Y, A, B, args.policy, arms[t])
                for t in arms}

    print(f"{len(ev)} OOF posts over {len(uniq)} authors, {args.k} folds, "
          f"{len(arms)} arms   policy={args.policy}")
    if args.policy == "deployed":
        print("NOTE `deployed` uses each arm's OWN dumped thresholds. For an "
              "ALLDATA arm those\n     were fitted on data it memorised -- use "
              "--policy refit or ratematch to compare\n     arms of different "
              "threshold provenance on equal terms.")
    print(f"reference: {ref}")
    print("Every post here was held out of its own fold's training AND its")
    print("threshold fitting, so these levels are NOT comparable to the")
    print("0.6218-style val numbers -- those were fitted in-sample.")

    print("\n" + "=" * 78)
    print("POOLED OOF macro-F1")
    print("=" * 78)
    base = {}
    for t, P in arms.items():
        base[t] = macro_f1(P, Y)
        dead = [FACTOR_TAXONOMY[j] for j in range(N)
                if P[:, j].sum() == 0 and Y[:, j].sum() > 0]
        print(f"  {t:<14}{base[t]:.4f}   {P.sum(1).mean():.2f} factors/post"
              + (f"   ZERO PREDICTIONS: {dead}" if dead else ""))
    if any(P.sum(1).mean() == 0 for P in arms.values()):
        raise SystemExit("an arm predicts nothing at all -- check its dumps")

    # ---- paired author bootstrap --------------------------------------------
    rng = np.random.default_rng(args.seed)
    boot = {t: np.empty(args.boot) for t in arms}
    for b in range(args.boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        ix = np.concatenate([by_author[uniq[k]] for k in pick])
        Yb = Y[ix]
        for t, P in arms.items():
            boot[t][b] = macro_f1(P[ix], Yb)

    print("\n" + "=" * 78)
    print(f"PAIRED delta vs {ref}   ({args.boot} author-bootstrap resamples)")
    print("=" * 78)
    print(f"{'arm':<14}{'delta':>9}{'95% CI':>20}{'win%':>8}{'verdict':>25}")
    print("-" * 78)

    null_band = None
    order = [t for t in arms if t != ref]
    if args.null and args.null in order:
        order = [args.null] + [t for t in order if t != args.null]

    for t in order:
        d = boot[t] - boot[ref]
        lo, hi = np.percentile(d, [2.5, 97.5])
        win = float((d > 0).mean()) * 100
        pt = base[t] - base[ref]
        if t == args.null:
            null_band = max(abs(lo), abs(hi))
            verdict = "<- NULL, all noise"
        elif null_band is not None:
            if pt > null_band and win >= 80:
                verdict = "REAL, clears the null"
            elif pt < -null_band:
                verdict = "WORSE than the null"
            else:
                verdict = "inside the null band"
        else:
            verdict = ("real (weak bar)" if lo > 0 and win >= 80
                       else "worse" if hi < 0 else "unresolved")
        print(f"{t:<14}{pt:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>20}"
              f"{win:>7.1f}%{verdict:>25}")

    sd = float(np.std(boot[ref]))
    print(f"\nbootstrap SD of the reference's ABSOLUTE score: {sd:.4f}")
    print(f"  For comparison the 372-post val instrument's implied SE is 0.0215,")
    print(f"  which is what produced the 0.0207 identical-config gap.")
    if null_band is not None:
        print(f"Measured null band (identical config, different seeds): "
              f"+/-{null_band:.4f}")
        print("  An arm must clear THAT, not zero.")
    else:
        print("No --null supplied, so training nondeterminism is unbounded here.")
        print("  Run one arm twice under a different seed pair before trusting")
        print("  any 'real' verdict -- that is exactly what the 0.0207 replica")
        print("  revealed on the old instrument.")

    if args.per_label:
        print("\n" + "=" * 78)
        print(f"PER LABEL -- {ref} against the others")
        print("=" * 78)
        F = {t: per_label_f1(arms[t], Y) for t in arms}
        others = [t for t in arms if t != ref]

        # Precision/recall and the fire rate say WHY a label is weak, which F1
        # alone hides. A label with high recall and low precision is a
        # THRESHOLD problem -- fixable with no retraining. Low recall means the
        # model cannot find the concept, which needs a better model or a better
        # definition. `rate` is predicted positives over gold positives: above
        # 1.0 the threshold is too permissive, below 1.0 too strict.
        P0 = arms[ref]
        tp = (P0 & Y).sum(0).astype(float)
        pp = P0.sum(0).astype(float)
        gp = Y.sum(0).astype(float)
        prec = np.where(pp > 0, tp / np.maximum(pp, 1), 0.0)
        rec = np.where(gp > 0, tp / np.maximum(gp, 1), 0.0)
        rate = np.where(gp > 0, pp / np.maximum(gp, 1), np.nan)

        hdr = (f"{'factor':<40}{'gold':>5}{'F1':>7}{'prec':>7}{'rec':>7}"
               f"{'rate':>7}")
        hdr += "".join(f"{t[:8]:>9}" for t in others)
        print(hdr)
        print("-" * len(hdr))
        for j in np.argsort(F[ref]):
            t = FACTOR_TAXONOMY[j]
            row = (f"{t:<40}{int(gp[j]):>5}{F[ref][j]:>7.3f}{prec[j]:>7.3f}"
                   f"{rec[j]:>7.3f}{rate[j]:>7.2f}")
            row += "".join(f"{F[o][j]:>9.3f}" for o in others)
            why = ""
            if F[ref][j] < 0.60:
                why = ("   THRESHOLD: over-fires" if rate[j] > 1.35 else
                       "   THRESHOLD: too strict" if rate[j] < 0.70 else
                       "   MODEL: can't find it" if rec[j] < 0.60 else "")
            print(row + why)
        print("\nsorted worst-first. rate = predicted/gold positives.")
        print("THRESHOLD rows are fixable with no retraining; MODEL rows need a")
        print("better definition or a different reader.")
        rare = Y.sum(0) < 40
        print(f"\n{int(rare.sum())} labels under 40 gold positives carry "
              f"{100*int(rare.sum())/N:.0f}% of macro weight and nearly all of")
        print("its variance. A gain confined to those is probably not real.")
        print(f"macro on the {int((~rare).sum())} common labels only: "
              + "  ".join(f"{t}={F[t][~rare].mean():.4f}" for t in arms))


if __name__ == "__main__":
    main()

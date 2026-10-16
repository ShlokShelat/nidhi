"""
factor_arm_ap_protective.py

Separates two competing explanations for weak performance on the protective
factors: a MODEL failure that a better architecture could fix, or a CEILING
imposed by the label itself.

  MODEL   Each protective concept is a multi-facet umbrella. The
          suicide-resilience concept analysis the taxonomy is drawn from
          defines meaning in life by THREE components (cognitive: making sense
          of experience; motivational: pursuing worthwhile goals; affective:
          satisfaction), responsibility by TWO (to others, to self), and
          psychological capital by FOUR (efficacy, optimism, hope, resilience).
          A single NLI hypothesis per umbrella can only catch one facet. The
          response would be faceted aspect hypotheses, or a protective
          specialist -- the source paper parameterises risk and protective
          separately because they "operate independently rather than being the
          reverse of risk factors".

  CEILING The annotation does not follow the published definition
          consistently, so posts that satisfy the written rule are labelled
          negative and posts that read as a neighbouring concept are labelled
          positive. If the target is inconsistent, no architecture recovers it.

The two make opposite predictions across architecturally diverse arms. Under
MODEL at least one arm ranks the label materially better than the pool, so a
real signal exists and pooling is diluting it. Under CEILING every arm lands in
the same narrow band regardless of architecture, because all of them are
fitting the same inconsistent target.

AP is the right statistic here because it is threshold-free and therefore
measures RANKING only, which is the part of the loss no threshold choice can
reach.

REPORTED PER LABEL:
  * AP of every individual arm, plus the logit-mean pool
  * best arm minus pool -- positive means dilution, and how much
  * the SPREAD across arms. A wide spread means architecture matters, so the
    concept is learnable and the pool is the wrong aggregator. A narrow spread
    at low AP is the ceiling signature.
  * a random-ranking baseline (= prevalence) so "low AP" is calibrated against
    what chance looks like for a rare label

Usage:
  python src/factor_arm_ap_protective.py
  python src/factor_arm_ap_protective.py --all-labels
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_structure import BAG8, cells, load_oof, logit_mean  # noqa: E402
from scoring import FACTOR_TAXONOMY  # noqa: E402

PROTECTIVE = ["meaning in life", "sense of responsibility", "social support",
              "psychological capital", "coping strategy"]


def ap(p, y):
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum()
                 / ys.sum())


def boot_ap(p, y, auth, n=200, seed=13):
    """Author-resampled AP.

    Posts by one author are not independent draws, so resampling posts
    understates the uncertainty. Resampling AUTHORS gives a spread that
    reflects how the estimate would move on a different set of people, which is
    what the held-out set actually varies.
    """
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ix = np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
        v = ap(p[ix], y[ix])
        if not np.isnan(v):
            out.append(v)
    return float(np.mean(out)), float(np.std(out))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--arms", nargs="+", default=BAG8)
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--all-labels", action="store_true")
    a_.add_argument("--boot", type=int, default=200)
    args = a_.parse_args()

    # load each arm SEPARATELY as well as pooled
    per_arm = {}
    for arm in args.arms:
        try:
            rid, S = load_oof(args.dir, [arm], args.k)
        except Exception as e:                              # noqa: BLE001
            print(f"  skip {arm}: {e}")
            continue
        per_arm[arm] = logit_mean(S)
    if not per_arm:
        raise SystemExit("no arms loaded")
    rid, S = load_oof(args.dir, list(per_arm), args.k)
    pool = logit_mean(S)
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    print(f"{n} posts / {len(pd.unique(auth))} authors, "
          f"{len(per_arm)} arms: {', '.join(per_arm)}\n")

    labels = FACTOR_TAXONOMY if args.all_labels else PROTECTIVE
    verdicts = []
    for name in labels:
        j = FACTOR_TAXONOMY.index(name)
        y = Y[:, j]
        if y.sum() < 2:
            continue
        base = y.mean()
        vals = {a: ap(P[:, j], y) for a, P in per_arm.items()}
        pv, psd = boot_ap(pool[:, j], y, auth, args.boot)
        bs = max(vals, key=lambda k: vals[k])
        bv, bsd = boot_ap(per_arm[bs][:, j], y, auth, args.boot)
        spread = max(vals.values()) - min(vals.values())
        print("=" * 84)
        print(f"{name}   {int(y.sum())} positives ({base:.1%})   "
              f"chance AP {base:.3f}")
        print("=" * 84)
        for a in sorted(vals, key=lambda k: -vals[k]):
            bar = "#" * int(40 * vals[a])
            print(f"  {a:<14}{vals[a]:>7.3f}  {bar}")
        print(f"  {'POOL':<14}{ap(pool[:, j], y):>7.3f}   "
              f"(author-resampled {pv:.3f} +- {psd:.3f})")
        print(f"  best arm `{bs}` {bv:.3f} +- {bsd:.3f}   "
              f"best-pool {bv - pv:+.3f}   spread across arms {spread:.3f}")
        verdicts.append((name, int(y.sum()), base, pv, bv, spread, bs))

    print("\n" + "=" * 84)
    print("VERDICT PER LABEL")
    print("=" * 84)
    print(f"  {'label':<26}{'pos':>5}{'chance':>8}{'pool AP':>9}{'best AP':>9}"
          f"{'best-pool':>11}{'spread':>8}  reading")
    for name, npos, base, pv, bv, spread, bs in verdicts:
        lift = pv / max(base, 1e-9)
        if bv - pv > 0.05 and spread > 0.10:
            rd = f"DILUTION -- use `{bs}`"
        elif spread < 0.06 and lift < 4:
            rd = "CEILING -- all arms agree and all are weak"
        elif lift < 3:
            rd = "barely above chance"
        else:
            rd = "learnable, pool is near best"
        print(f"  {name:<26}{npos:>5}{base:>8.3f}{pv:>9.3f}{bv:>9.3f}"
              f"{bv-pv:>+11.3f}{spread:>8.3f}  {rd}")
    print("\n  DILUTION -> the concept is learnable and the pool discards the")
    print("     signal. Responses: per-label arm weights, or dropping the")
    print("     harmful arms for that label only.")
    print("  CEILING -> architecture does not change the ranking, so neither")
    print("     faceted hypotheses nor a protective specialist can help.")
    print("\n  Read `low AP` against the chance column, not against 1.0: on a")
    print("  rare label, an AP several times prevalence is a model that has")
    print("  learned the concept and cannot separate the tail, not a model")
    print("  that has learned nothing.")


if __name__ == "__main__":
    main()

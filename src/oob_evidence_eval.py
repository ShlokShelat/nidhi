"""
oob_evidence_eval.py

SCORE EVIDENCE ON AN INSTRUMENT THAT MATCHES DEPLOYMENT.

`scripts/oob_evidence.sh` writes candidates with a `bag` column. Each post was
scored by the ~5 bagged models that EXCLUDED its author, so voting across
`bag` is structurally the same operation deployment performs when it votes
across the 5 fold models. Voting across five DIFFERENT tagger configs, each a
single model, is a different operation entirely, so a measurement built that
way does not describe what deployment actually does.

WHAT IT COMPARES
  vote>=k over bags, k = 1..5, plus the FORCE rule, suppression on PREDICTED
  risk, MACRO averaging. All the choices that were tuned on the single-model
  instrument, re-asked on the right one.

WHY min-models MATTERS. Coverage is median 5 but ranges 2..10. A post with 2
clean models is a 2-model ensemble, not the 5 deployment uses, and its vote
threshold means something different. `--min-models 4` restricts to posts whose
ensemble size is close to deployment's; the unrestricted number is printed too
so the difference is visible rather than hidden.

Usage:
  python src/oob_evidence_eval.py --tag oobrc
  python src/oob_evidence_eval.py --tag oobrc --min-models 4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span, parse_evidence,  # noqa
                     phrase_f1, token_len)


def dedupe(sp):
    out = []
    for s in sorted(sp, key=lambda x: (-token_len(x),
                                       -len(normalize_span(x)))):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--tag", default="oobrc")
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--risks", default="artifacts/nligaur/oof_base.csv",
                    help="PREDICTED risk for suppression. Gold would leak the "
                         "answer, producing phantom evidence that inflates "
                         "the score without being deployable.")
    a_.add_argument("--min-models", type=int, default=0)
    a_.add_argument("--boot", type=int, default=400)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    p = f"{args.dir}/{args.tag}_oof_candidates.csv"
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run scripts/oob_evidence.sh first")
    d = pd.read_csv(p, keep_default_na=False)
    if "bag" not in d.columns:
        raise SystemExit(f"{p} has no `bag` column; it is not an OOB dump")
    d["row_id"] = d["row_id"].astype(str)
    mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
    d = d[d["conf"] >= mc]

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rs = pd.read_csv(args.risks, keep_default_na=False)
    rs["row_id"] = rs["row_id"].astype(str)
    rcol = next(c for c in ("pred_risk_level", "risk_level")
                if c in rs.columns)
    rmap = dict(zip(rs["row_id"], rs[rcol]))

    nbag = d.groupby("row_id")["bag"].nunique().to_dict()
    rows = [r for r in lab.index if r in nbag]
    if args.min_models:
        rows = [r for r in rows if nbag[r] >= args.min_models]
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    ship = np.array([normalize_label(rmap[r]) for r in rows])
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])

    cand = {}
    for r, s, b in zip(d["row_id"], d["span"], d["bag"]):
        cand.setdefault(r, []).append((int(b), str(s)))
    sizes = np.array([nbag[r] for r in rows])
    print(f"{len(rows)} posts  (min-models {args.min_models or 'none'})")
    print(f"ensemble size per post: median {int(np.median(sizes))}  "
          f"mean {sizes.mean():.1f}  min {sizes.min()}  max {sizes.max()}")
    print(f"deployment ensembles 5. Suppression on PREDICTED risk, MACRO.\n")

    def build(k):
        out = []
        for i, r in enumerate(rows):
            if ship[i] == "indicator":
                out.append([])
                continue
            c = {}
            for b, s in cand.get(r, []):
                c.setdefault(normalize_span(s), [set(), s])[0].add(b)
            keep = dedupe([v[1] for v in c.values() if len(v[0]) >= k])
            if not keep:
                pool = [s for _, s in cand.get(r, [])]
                keep = sorted(pool, key=token_len)[:1] if pool else []
            out.append(keep)
        return out

    def f(P, ix):
        return phrase_f1([G[i] for i in ix], [P[i] for i in ix],
                         average="macro").f1

    allix = list(range(len(rows)))
    arms = {f"vote>={k}": build(k) for k in (1, 2, 3, 4, 5)}
    base = arms["vote>=2"]

    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in
                             rng.choice(ua, size=len(ua), replace=True)])
             for _ in range(args.boot)]
    bb = np.array([f(base, ix) for ix in boots])

    print("=" * 84)
    print("VOTE THRESHOLD ON THE OOB (DEPLOYMENT-SHAPED) INSTRUMENT")
    print("=" * 84)
    print(f"{'arm':<12}{'macro':>9}{'spans/post':>12}{'delta':>10}"
          f"{'win':>7}{'one-draw 95%':>24}")
    for nm, P in arms.items():
        v = f(P, allix)
        sp = float(np.mean([len(x) for x in P]))
        if nm == "vote>=2":
            print(f"{nm:<12}{v:>9.4f}{sp:>12.2f}{'-- (ref)':>10}"
                  f"{'--':>7}{'--':>24}")
            continue
        dd = np.array([f(P, ix) for ix in boots]) - bb
        sd = dd.std(ddof=1)
        print(f"{nm:<12}{v:>9.4f}{sp:>12.2f}{dd.mean():>+10.4f}"
              f"{(dd > 0).mean():>7.0%}"
              f"{f'[{dd.mean()-1.96*sd:+.4f},{dd.mean()+1.96*sd:+.4f}]':>24}")

    print("\n  Compare the LEVEL against the single-model instrument, which")
    print("  read 0.7636 for the conditioned arm. If the OOB level is close to")
    print("  the deployed system's implied ~0.784, this instrument finally")
    print("  tracks deployment and its rankings can be trusted.")
    print("  Compare the vote-k RANKING against what the single-model")
    print("  instrument chose. If they differ, the shipped k was fitted on")
    print("  the wrong ensemble shape.")


if __name__ == "__main__":
    main()

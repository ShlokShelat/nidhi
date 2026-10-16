"""
evidence_rep_sweep.py

DOES `--rep modal` STILL WIN AT THE NEW OPERATING POINT?

`rep` decides which string represents a cluster of agreeing spans. modal was
chosen when the pool was ungated (or gated per-run): modal beat every
alternative representative (median, capsafe, short) under that measurement.

**That choice predates the two changes in the current configuration.** `--cap 3`
and a uniform `--min-conf-override 0.65` took the pool from 2687 candidates to
2376. Different pool, different clusters, possibly a different best
representative. The 48-cell gate x k x cap grid held `rep` FIXED, so this cell
has never been measured at the shipped configuration.

The cap_sweep harness cannot answer it -- that harness reimplements the vote
with `dedupe_nested` and has no rep logic at all. So this drives the SHIPPED
`evidence_vote_test.py` once per rep on the val split and scores the outputs
against val gold with the official scorer, macro-averaged, using PREDICTED risk
for suppression (never gold -- that inflates every reading by ~0.05).

Judged the only way that has held up here: paired author-disjoint splits, WIN
RATE and per-split sd, never the CI of the mean.

Usage (CPU, a couple of minutes -- it shells out to the vote 4 times):
  python src/evidence_rep_sweep.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1  # noqa: E402

RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def f1_of(gold, pred):
    r = phrase_f1([gold], [pred], average="macro")
    return float(getattr(r, "f1", r if isinstance(r, float) else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--reps", nargs="+",
                    default=["modal", "median", "short", "capsafe"])
    ap.add_argument("--suppress-prob", default=None,
                    help="threshold on prob_indicator for the suppression "
                         "rule; passed through to evidence_vote_test.py")
    ap.add_argument("--merge-overlap", default=None,
                    help="shared-token fraction for the overlap merge; "
                         "passed through to evidence_vote_test.py")
    ap.add_argument("--dedupe-keep", default="long",
                    choices=["long", "short"],
                    help="passed through to evidence_vote_test.py; 'long' is "
                         "shipped and has never been compared to 'short'.")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--min-conf", type=float, default=0.65)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = [str(x) for x in va["row_id"]]
    gold = [parse_evidence(e) for e in va["evidence"]]

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    amap = dict(zip(lab["row_id"], lab[acol].astype(str)))
    authors = np.array([amap.get(r, r) for r in rows])

    print(f"{len(rows)} val posts, {len(set(authors))} authors")
    print(f"config: k={args.k} cap={args.cap} min_conf={args.min_conf} "
          f"(the SHIPPED operating point)\n")

    F = {}
    for rep in args.reps:
        out = f"artifacts/_rep_{rep}_val.csv"
        cmd = [sys.executable, "src/evidence_vote_test.py",
               "--runs", ",".join(RC7), "--dir", args.dir,
               "--split", "val", "--test", args.val,
               "--risk-from", args.risk,
               "--k", str(args.k), "--cluster", "--rep", rep,
               "--cap", str(args.cap),
               "--min-conf-override", str(args.min_conf),
               "--dedupe-keep", args.dedupe_keep,
               "--out", out]
        if args.merge_overlap is not None:
            cmd += ["--merge-overlap", str(args.merge_overlap)]
        if args.suppress_prob is not None:
            cmd += ["--suppress-prob", str(args.suppress_prob)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {rep}: vote FAILED")
            print(r.stdout[-600:], r.stderr[-600:])
            continue
        d = pd.read_csv(out, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        m = dict(zip(d["row_id"], d[d.columns[-1]]))
        pred = [parse_evidence(m.get(x, "")) for x in rows]
        F[rep] = np.array([f1_of(gold[i], pred[i]) for i in range(len(rows))])
        sp = np.mean([len(p) for p in pred])
        print(f"  {rep:<9} macro {F[rep].mean():.4f}   {sp:.2f} spans/post")

    if "modal" not in F:
        raise SystemExit("modal did not run; nothing to compare against")

    rng = np.random.default_rng(args.seed)
    ua = np.array(sorted(set(authors)))
    holds = []
    for _ in range(args.splits):
        h = set(rng.choice(ua, size=max(1, len(ua) // 3), replace=False))
        mk = np.array([a in h for a in authors])
        if mk.sum() >= 5:
            holds.append(mk)

    print(f"\n  paired vs modal, {len(holds)} author-disjoint splits")
    print(f"  {'rep':<9}{'d macro':>10}{'win':>7}{'sd':>9}{'posts diff':>12}")
    base = F["modal"]
    for rep, f in F.items():
        if rep == "modal":
            continue
        dl = np.array([f[mk].mean() - base[mk].mean() for mk in holds])
        w = float((dl > 0).mean())
        print(f"  {rep:<9}{f.mean()-base.mean():>+10.4f}{100*w:>6.0f}%"
              f"{dl.std():>9.4f}{int((f != base).sum()):>12}"
              + ("   <-- CLEARS 90%" if w >= 0.90 else ""))
    print("\n  modal is shipped. A challenger needs >=90% win AND enough")
    print("  margin to matter, converted to composite at the 0.30 evidence")
    print("  weight.")


if __name__ == "__main__":
    main()

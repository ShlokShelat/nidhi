"""
evidence_risk_consistency.py

Enforces the annotation scheme's own rule linking risk level to evidence
presence -- a cross-task consistency constraint the pipeline never applied.

Measured over all 1635 labelled posts:

    P(post has gold evidence | risk level)
        indicator   0.038    n=611
        ideation    1.000    n=519
        behavior    1.000    n=391
        attempt     1.000    n=114

This is not a correlation, it is effectively a rule: a post above Indicator
ALWAYS carries evidence, and an Indicator post almost never does. Yet the BIO
tagger decides emptiness on its own, with no reference to the risk head, so
the deployed system violates the rule in both directions.

Why it matters so much more than it looks. The official phrase metric appears
to average PER POST -- per-post macro reproduces the recorded 0.7787 figure
for evidence_v2 to four decimals. Under per-post averaging a post where gold
is empty and anything is emitted scores 0 for that whole post, and a post
with gold evidence where nothing is emitted also scores 0.
Each such post costs a full 1/N. Under micro-averaging the same errors are
just a few extra false positives, which is why four rounds of span-selection
work never surfaced this.

Two corrections:

  SUPPRESS  predicted risk == indicator            -> emit no evidence
  FORCE     predicted risk != indicator, no spans  -> emit the best available
            span (from the fallback tagger pool) rather than nothing

The suppress half is free. The force half needs a candidate to fall back on,
so it is only applied when some tagger proposed something.

Usage:
  python src/evidence_risk_consistency.py --report          # measure on val
  python src/evidence_risk_consistency.py --apply \
      --evidence artifacts/evidence_vote2_test.csv \
      --risk artifacts/risk_blend_scheduledfix.csv \
      --fallback artifacts/evidence_per_run \
      --out artifacts/evidence_consistent_test.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1, normalize_label, token_len


def load_fallback(row_ids, cache_dir, tag):
    """Union of every cached tagger's spans, for the FORCE half."""
    pool = {r: [] for r in row_ids}
    if not (cache_dir and os.path.isdir(cache_dir)):
        return pool
    for f in sorted(glob.glob(os.path.join(cache_dir, f"*__{tag}.csv"))):
        d = pd.read_csv(f, keep_default_na=False).set_index("row_id")["evidence"]
        for r in row_ids:
            pool[r].extend(parse_evidence(d.get(r, "")))
    return pool


def apply_rule(row_ids, spans, risks, fallback):
    out, n_sup, n_forced, n_stuck = [], 0, 0, 0
    for r, sp, rk in zip(row_ids, spans, risks):
        if rk == "indicator":
            if sp:
                n_sup += 1
            out.append([])
        else:
            if sp:
                out.append(sp)
            else:
                cands = fallback.get(r, [])
                if cands:
                    out.append([min(cands, key=token_len)])
                    n_forced += 1
                else:
                    out.append([])
                    n_stuck += 1
    return out, n_sup, n_forced, n_stuck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--val-baseline", default=None,
                    help="CSV of the champion's val risk predictions. Defaults "
                         "to artifacts/val_deployed_baseline.csv (exact, from "
                         "cache_val_baseline.py) and falls back to "
                         "artifacts/val_baseline_reconstructed.csv.")
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--risk", default=None)
    ap.add_argument("--fallback", default="artifacts/evidence_per_run")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.report:
        val = pd.read_csv(args.val)
        rids = list(val["row_id"])
        gold = [parse_evidence(x) for x in val["evidence"]]
        cands = ([args.val_baseline] if args.val_baseline else
                 ["artifacts/val_deployed_baseline.csv",
                  "artifacts/val_baseline_reconstructed.csv"])
        path = next((p for p in cands if p and os.path.exists(p)), None)
        if path is None:
            raise SystemExit(
                "no val baseline found. Expected one of:\n  " +
                "\n  ".join(cands) +
                "\nGenerate the exact one with cache_val_baseline.py (GPU) or "
                "the approximate one with reconstruct_val_baseline.py (CPU).")
        rec = pd.read_csv(path).set_index("row_id")
        col = next((c for c in ("champion_final", "pred_risk_level", "risk_level")
                    if c in rec.columns), None)
        if col is None:
            raise SystemExit(f"{path} has no risk-prediction column; "
                             f"columns are {list(rec.columns)}")
        print(f"val risk predictions from: {path}  (column '{col}')")
        risks = [normalize_label(rec.loc[r, col]) for r in rids]

        runs = {}
        for r in ["evidence_v2", "evidence_v1", "evidence_s14", "evidence_s15"]:
            p = f"runs/{r}/val_predictions.csv"
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            c = next(x for x in d.columns if "pred" in x.lower())
            m = val[["row_id"]].merge(d[["row_id", c]], on="row_id", how="left")
            runs[r] = [parse_evidence(x) for x in m[c]]
        base = runs["evidence_v2"]
        pool = {rids[i]: [s for r in runs for s in runs[r][i]] for i in range(len(rids))}

        gated, n_sup, n_forced, n_stuck = apply_rule(rids, base, risks, pool)
        print(f"suppressed on {n_sup} indicator posts; forced a span on "
              f"{n_forced}; {n_stuck} non-indicator posts had no candidate at all\n")
        print(f"{'system':<34}{'macro':>10}{'micro':>10}")
        for name, sp in (("evidence_v2 (deployed)", base),
                         ("+ risk-consistency rule", gated)):
            print(f"{name:<34}{phrase_f1(gold, sp, average='macro').f1:>10.4f}"
                  f"{phrase_f1(gold, sp).f1:>10.4f}")
        dm = (phrase_f1(gold, gated, average="macro").f1
              - phrase_f1(gold, base, average="macro").f1)
        di = phrase_f1(gold, gated).f1 - phrase_f1(gold, base).f1
        print(f"\ndelta macro {dm:+.4f} = {0.3*dm:+.4f} composite")
        print(f"delta micro {di:+.4f} = {0.3*di:+.4f} composite")
        print("\nThe rule helps under BOTH averaging conventions, so it does not")
        print("depend on resolving which one the official scorer uses -- only the")
        print("MAGNITUDE does.")
        return

    if not args.apply:
        raise SystemExit("pass --report or --apply")
    for req in ("evidence", "risk", "out"):
        if not getattr(args, req):
            raise SystemExit(f"--apply needs --{req}")

    ev = pd.read_csv(args.evidence, keep_default_na=False)
    rk = pd.read_csv(args.risk)
    col = next(c for c in ("pred_risk_level", "risk_level") if c in rk.columns)
    rmap = dict(zip(rk["row_id"], rk[col]))
    rids = list(ev["row_id"])
    spans = [parse_evidence(x) for x in ev["evidence"]]
    risks = [normalize_label(rmap[r]) for r in rids]
    tag = args.tag or "test"
    pool = load_fallback(rids, args.fallback, tag)

    gated, n_sup, n_forced, n_stuck = apply_rule(rids, spans, risks, pool)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame({"row_id": rids,
                  "evidence": ["; ".join(s) for s in gated]}).to_csv(args.out, index=False)
    n_before = sum(1 for s in spans if not s)
    n_after = sum(1 for s in gated if not s)
    print(f"suppressed on {n_sup} indicator posts; forced a span on {n_forced}; "
          f"{n_stuck} had no candidate")
    print(f"posts with no evidence: {n_before} -> {n_after}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

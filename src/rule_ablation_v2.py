"""
rule_ablation_v2.py

Stage 2 of the symbolic-discovery pipeline, rebuilt around three changes to
the original rule_ablation.py:

1. NO TORCH, NO MODEL RELOAD. The original re-scored the 2x400M-parameter
   risk-NLI ensemble on val for EVERY invocation just to rebuild the same
   baseline predictions. Those are now computed once by cache_val_baseline.py
   and read from a CSV, so iterating on candidate rules costs milliseconds
   and runs anywhere (no GPU, no HF cache).

2. ONE SOURCE OF RULE LOGIC. The original re-declared URGENCY_PHRASES /
   METHOD_PHRASES / etc. locally -- a second copy of the deployed regexes
   that could silently drift from what actually ships. This imports
   apply_rules()/ALL_DEFS from apply_rules_to_test.py instead.

3. AUTHOR-LEVEL A/B STABILITY GATING. Decisions still come from the FULL-val
   net and delta-F1 (the only statistic with enough matches to mean anything
   at these rule sizes). On top of that, val is split into two halves and the
   rule must be non-negative on BOTH -- a sign test that catches the
   "memorised specific val posts" failure mode without the power collapse of
   deciding on half the data.

   The split is by anon_user_id, NOT by post. val is 372 posts from only 32
   authors, and the families these rules target are heavily author-clustered
   (the entire preparatory-farewell family is essentially two authors, U0044
   and U0092, writing in a consistent personal idiom). A post-level split
   puts near-identical posts by the same author on both sides, so a rule that
   has merely memorised one author's voice still looks "stable". Splitting by
   author makes the gate test what it is supposed to test: does this pattern
   survive on writers it was not derived from.

Usage:
  python src/rule_ablation_v2.py --baseline artifacts/val_deployed_baseline.csv
  python src/rule_ablation_v2.py --baseline ... --rules farewell_note,third_person_death
  python src/rule_ablation_v2.py --baseline ... --combined urgency_guarded,farewell_note
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, risk_weighted_f1
from apply_rules_to_test import ALL_DEFS, RULE_DEFS, apply_rules

BASELINE_COL_CANDIDATES = ["champion_final", "baseline_pred", "pred_risk_level"]


def load_baseline(path, val_path):
    """Returns a frame with row_id, anon_user_id, post, gold, baseline."""
    b = pd.read_csv(path)
    val = pd.read_csv(val_path)

    col = next((c for c in BASELINE_COL_CANDIDATES if c in b.columns), None)
    if col is None:
        raise SystemExit(
            f"{path} has none of {BASELINE_COL_CANDIDATES}; columns are {list(b.columns)}")

    keep = ["row_id", col] + [c for c in ("approximate", "pre_rule_pred")
                              if c in b.columns and c != col]
    df = val[["row_id", "anon_user_id", "post", "risk_level"]].merge(
        b[keep], on="row_id", how="left")
    if df[col].isna().any():
        missing = df.loc[df[col].isna(), "row_id"].tolist()[:5]
        raise SystemExit(f"baseline missing predictions for {df[col].isna().sum()} "
                         f"val rows, e.g. {missing}")

    df["gold"] = [normalize_label(x) for x in df["risk_level"]]
    df["baseline"] = [normalize_label(x) for x in df[col]]
    return df, col


def author_split(df):
    """Deterministic author-disjoint 2-way split, balanced on post count.

    Authors are sorted by post count (desc, then id for ties) and greedily
    assigned to whichever half currently holds fewer posts. No author ever
    appears in both halves.
    """
    counts = df.groupby("anon_user_id").size().sort_values(
        ascending=False, kind="mergesort")
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return df["anon_user_id"].map(half), size


def score_rule(df, mask_half, rule_key):
    """Apply one candidate on top of the baseline; return stats overall + per half."""
    new = []
    for post, base in zip(df["post"], df["baseline"]):
        lbl, _ = apply_rules(post, base, (rule_key,))
        new.append(lbl)
    df = df.assign(new=new)

    def bucket(sub):
        fired = sub[sub["new"] != sub["baseline"]]
        corrected = int(((fired["baseline"] != fired["gold"]) & (fired["new"] == fired["gold"])).sum())
        broken = int(((fired["baseline"] == fired["gold"]) & (fired["new"] != fired["gold"])).sum())
        still_wrong = len(fired) - corrected - broken
        return {"changed": len(fired), "corrected": corrected, "broken": broken,
                "still_wrong": still_wrong, "net": corrected - broken}

    overall = bucket(df)
    # `changed` counts label changes; `matches` counts trigger fires regardless
    # of whether the label was eligible -- both are worth seeing.
    overall["matches"] = int(sum(
        1 for p in df["post"]
        if apply_rules(p, "__none__", (rule_key,))[1][0]["fired"]))
    overall["f1_base"] = risk_weighted_f1(list(df["gold"]), list(df["baseline"]))
    overall["f1_new"] = risk_weighted_f1(list(df["gold"]), list(df["new"]))
    overall["f1_delta"] = overall["f1_new"] - overall["f1_base"]

    halves = {h: bucket(df[mask_half == h]) for h in ("A", "B")}
    changed_rows = df[df["new"] != df["baseline"]][
        ["row_id", "anon_user_id", "gold", "baseline", "new"]]
    return overall, halves, changed_rows


def verdict(overall, halves, min_net=3):
    reasons = []
    if overall["net"] < min_net:
        reasons.append(f"net {overall['net']:+d} below bar (+{min_net})")
    if overall["f1_delta"] <= 0:
        reasons.append(f"delta-F1 {overall['f1_delta']:+.4f} not positive")
    neg = [h for h in ("A", "B") if halves[h]["net"] < 0]
    if neg:
        reasons.append("negative on half " + "+".join(neg) + " (stability gate)")
    return ("PASS", reasons) if not reasons else ("REJECT", reasons)


def report(name, overall, halves, changed_rows, show_rows, min_net):
    v, reasons = verdict(overall, halves, min_net)
    print("\n" + "=" * 92)
    print(f"CANDIDATE: {name}    [{v}]")
    print("=" * 92)
    print(f"  trigger fired on {overall['matches']} posts; changed the label on "
          f"{overall['changed']}")
    print(f"  CORRECTED {overall['corrected']}   BROKEN {overall['broken']}   "
          f"changed-but-still-wrong {overall['still_wrong']}   NET {overall['net']:+d}")
    print(f"  weighted-F1 {overall['f1_base']:.4f} -> {overall['f1_new']:.4f}  "
          f"(delta {overall['f1_delta']:+.4f})")
    print(f"  stability (author-disjoint halves):")
    for h in ("A", "B"):
        s = halves[h]
        print(f"      half {h}: changed {s['changed']:>2}  corrected {s['corrected']:>2}  "
              f"broken {s['broken']:>2}  net {s['net']:+d}")
    if reasons:
        print("  REJECT because: " + "; ".join(reasons))
    if show_rows and len(changed_rows):
        print("  rows changed:")
        for _, r in changed_rows.iterrows():
            flag = ""
            if r["baseline"] != r["gold"] and r["new"] == r["gold"]:
                flag = "  <- CORRECTED"
            elif r["baseline"] == r["gold"] and r["new"] != r["gold"]:
                flag = "  <- BROKEN"
            print(f"      {r['row_id']}  {str(r['anon_user_id']):<7} gold={r['gold']:<9} "
                  f"{r['baseline']} -> {r['new']}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="CSV of the deployed champion's val predictions "
                         "(cache_val_baseline.py output)")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--rules", default=None,
                    help="comma-separated candidate keys (default: every "
                         "candidate not already deployed)")
    ap.add_argument("--combined", default=None,
                    help="comma-separated ORDERED keys applied cumulatively "
                         "ON TOP of the champion's final predictions")
    ap.add_argument("--chain", default=None,
                    help="comma-separated ORDERED keys forming a COMPLETE rule "
                         "chain rebuilt from the pre-rule predictions. Use this "
                         "(not --combined) to evaluate a rule that REPLACES a "
                         "deployed one, e.g. a guarded urgency variant: stacking "
                         "such a variant on top of the champion is meaningless, "
                         "because the rule it modifies has already fired.")
    ap.add_argument("--chain-ref", default="urgency,completed_act",
                    help="reference chain that --chain is compared against")
    ap.add_argument("--min-net", type=int, default=3,
                    help="acceptance bar on full-val net corrections")
    ap.add_argument("--show-rows", action="store_true", default=True)
    args = ap.parse_args()

    df, col = load_baseline(args.baseline, args.val)
    mask_half, sizes = author_split(df)

    approx = "approximate" in df.columns and bool(df["approximate"].any())
    print(f"baseline column   : {col}   ({len(df)} val rows)")
    if approx:
        print("  *** WARNING: baseline is RECONSTRUCTED, not exact. Numbers below are")
        print("  *** a pre-screen only -- re-run against cache_val_baseline.py output")
        print("  *** before trusting any PASS verdict.")
    print(f"baseline val wF1  : "
          f"{risk_weighted_f1(list(df['gold']), list(df['baseline'])):.4f}")
    print(f"author split      : half A = {sizes['A']} posts, half B = {sizes['B']} posts, "
          f"{df['anon_user_id'].nunique()} authors, disjoint")
    print(f"acceptance bar    : net >= +{args.min_net}, delta-F1 > 0, "
          f"non-negative on BOTH halves")

    if args.chain:
        if "pre_rule_pred" not in df.columns:
            raise SystemExit("--chain needs a 'pre_rule_pred' column in the baseline CSV")
        pre = [normalize_label(x) for x in df["pre_rule_pred"]]
        ref_keys = tuple(k.strip() for k in args.chain_ref.split(","))
        new_keys = tuple(k.strip() for k in args.chain.split(","))
        for k in ref_keys + new_keys:
            if k not in ALL_DEFS:
                raise SystemExit(f"unknown rule key: {k}")
        ref = [apply_rules(p, b, ref_keys)[0] for p, b in zip(df["post"], pre)]
        new = [apply_rules(p, b, new_keys)[0] for p, b in zip(df["post"], pre)]
        gold = list(df["gold"])
        f1r, f1n = risk_weighted_f1(gold, ref), risk_weighted_f1(gold, new)
        diff = [(r, g, a, b) for r, g, a, b in zip(df["row_id"], gold, ref, new) if a != b]
        print("\n" + "=" * 92)
        print("FULL-CHAIN COMPARISON (both chains rebuilt from pre-rule predictions)")
        print("=" * 92)
        print(f"  reference chain : {','.join(ref_keys):<40} wF1 = {f1r:.4f}")
        print(f"  candidate chain : {','.join(new_keys):<40} wF1 = {f1n:.4f}"
              f"   (delta {f1n - f1r:+.4f})")
        corrected = sum(1 for _, g, a, b in diff if a != g and b == g)
        broken = sum(1 for _, g, a, b in diff if a == g and b != g)
        print(f"  posts where the two chains disagree: {len(diff)}   "
              f"corrected {corrected}  broken {broken}  net {corrected - broken:+d}")
        for r, g, a, b in diff:
            tag = ("  <- CORRECTED" if a != g and b == g
                   else "  <- BROKEN" if a == g and b != g else "")
            print(f"      {r}  gold={g:<9} {a} -> {b}{tag}")
        if not diff:
            print("      (none -- the two chains are identical on this val split)")
        return

    if args.combined:
        keys = [k.strip() for k in args.combined.split(",")]
        for k in keys:
            if k not in ALL_DEFS:
                raise SystemExit(f"unknown rule key: {k}")
        print("\n" + "=" * 92)
        print("SEQUENTIAL / COMBINED ABLATION (applied on top of the champion)")
        print("=" * 92)
        cur = list(df["baseline"])
        print(f"{'champion baseline':<52}wF1 = "
              f"{risk_weighted_f1(list(df['gold']), cur):.4f}")
        for i, k in enumerate(keys, 1):
            prev = cur
            cur = [apply_rules(p, b, (k,))[0] for p, b in zip(df["post"], prev)]
            corrected = sum(1 for a, b, g in zip(prev, cur, df["gold"]) if a != g and b == g)
            broken = sum(1 for a, b, g in zip(prev, cur, df["gold"]) if a == g and b != g)
            changed = sum(1 for a, b in zip(prev, cur) if a != b)
            print(f"{'+ ' + ','.join(keys[:i]):<52}wF1 = "
                  f"{risk_weighted_f1(list(df['gold']), cur):.4f}   "
                  f"(this stage: {changed} changed, {corrected} corrected, {broken} broken)")
        base_f1 = risk_weighted_f1(list(df["gold"]), list(df["baseline"]))
        print(f"\nchampion {base_f1:.4f} -> combined "
              f"{risk_weighted_f1(list(df['gold']), cur):.4f} "
              f"(delta {risk_weighted_f1(list(df['gold']), cur) - base_f1:+.4f})")
        return

    keys = ([k.strip() for k in args.rules.split(",")] if args.rules
            else [k for k in ALL_DEFS if k not in RULE_DEFS])
    results = []
    for k in keys:
        if k not in ALL_DEFS:
            raise SystemExit(f"unknown rule key: {k}. known: {sorted(ALL_DEFS)}")
        overall, halves, rows = score_rule(df, mask_half, k)
        report(k, overall, halves, rows, args.show_rows, args.min_net)
        results.append((k, overall, halves))

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(f"{'candidate':<26}{'chg':>5}{'corr':>6}{'brk':>5}{'net':>6}"
          f"{'dF1':>10}{'netA':>7}{'netB':>7}  verdict")
    for k, o, h in sorted(results, key=lambda r: -r[1]["net"]):
        v, _ = verdict(o, h, args.min_net)
        print(f"{k:<26}{o['changed']:>5}{o['corrected']:>6}{o['broken']:>5}"
              f"{o['net']:>+6d}{o['f1_delta']:>+10.4f}"
              f"{h['A']['net']:>+7d}{h['B']['net']:>+7d}  {v}")
    if approx:
        print("\n*** reconstructed baseline -- pre-screen only, see warning above ***")


if __name__ == "__main__":
    main()

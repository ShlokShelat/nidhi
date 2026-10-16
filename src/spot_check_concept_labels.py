"""
spot_check_concept_labels.py

Checks a RANDOM sample of concept labels (not just hard cases —
random checks tell you about overall reliability, which is what
matters for trusting these as training signal).

Usage:
  Step 1 — generate the review sheet:
    python src/spot_check_concept_labels.py --mode sheet \
        --labels artifacts/train_concept_labels.csv \
        --posts artifacts/train_split.csv \
        --n 75 --out artifacts/concept_spotcheck_sheet.csv

  Step 2 — after filling in the 'correct_1_or_0' column by hand:
    python src/spot_check_concept_labels.py --mode score \
        --filled artifacts/concept_spotcheck_sheet.csv
"""

from __future__ import annotations

import argparse
import random

import pandas as pd


def stratified_sample(df, n, seed=42):
    """Sample ~n rows, proportionally by risk_level, matching the same
    stratification used at label-generation time.

    NOTE: normalizes risk_level first (train_split.csv has inconsistent
    casing/whitespace — same fix as generate_concept_labels.py)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from scoring import normalize_label
    df = df.copy()
    if "risk_level" not in df.columns or n >= len(df):
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    frac = n / len(df)
    df["_strat_key"] = df["risk_level"].map(normalize_label)
    sampled = df.groupby("_strat_key", group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed) if len(g) > 0 else g)
    return sampled.drop(columns=["_strat_key"]).reset_index(drop=True)

CONCEPTS = [
    "method_mechanism", "plan_present", "completed_act", "urgency_imminence",
    "speaker_is_author", "farewell_framing", "protective_factor", "negation",
    "past_attempt_history",
]


def make_sheet(args):
    labels_df = pd.read_csv(args.labels)
    posts_df = pd.read_csv(args.posts)
    merge_cols = ["row_id", "post"] + (["risk_level"] if "risk_level" in posts_df.columns else [])
    merged = posts_df[merge_cols].merge(labels_df, on="row_id", how="inner")

    sample = stratified_sample(merged, args.n)
    print(f"Sampled {len(sample)} posts, stratified by risk_level: "
         f"{dict(sample['risk_level'].value_counts()) if 'risk_level' in sample.columns else 'N/A'}")

    rows = []
    for _, r in sample.iterrows():
        for c in CONCEPTS:
            rows.append({
                "row_id": r["row_id"],
                "post_preview": str(r["post"])[:args.preview_len],
                "concept": c,
                "predicted": bool(r[c]),
                "model_confidence": r.get("confidence", ""),
                "correct_1_or_0": "",
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(sample)} posts x {len(CONCEPTS)} concepts "
         f"= {len(out_df)} rows to review")
    print("Open in Excel, read each post_preview, fill 'correct_1_or_0' with 1 or 0 "
         "for whether the 'predicted' value is actually right, save, then run --mode score")


def score_sheet(args):
    df = pd.read_csv(args.filled)
    df = df[df["correct_1_or_0"].notna() & (df["correct_1_or_0"] != "")]
    if len(df) == 0:
        print("No filled-in rows found yet — fill in 'correct_1_or_0' first")
        return

    df["correct_1_or_0"] = df["correct_1_or_0"].astype(int)

    print(f"\n{len(df)} rows scored so far\n")
    print(f"{'concept':<25}{'n':>6}{'accuracy':>10}{'pos_rate':>10}")
    overall_correct, overall_n = 0, 0
    weak_concepts = []
    degenerate_concepts = []
    for c in CONCEPTS:
        sub = df[df["concept"] == c]
        if len(sub) == 0:
            continue
        acc = sub["correct_1_or_0"].mean()
        pos_rate = sub["predicted"].astype(str).str.lower().eq("true").mean()
        print(f"{c:<25}{len(sub):>6}{acc:>10.1%}{pos_rate:>10.1%}")
        overall_correct += sub["correct_1_or_0"].sum()
        overall_n += len(sub)
        if acc < 0.75:
            weak_concepts.append((c, acc))
        # flag near-constant predictions — high "accuracy" can just mean
        # "always predicts No" with near-zero actual training signal
        if pos_rate < 0.03 or pos_rate > 0.97:
            degenerate_concepts.append((c, pos_rate))

    print(f"\n{'OVERALL':<25}{overall_n:>6}{overall_correct/max(overall_n,1):>10.1%}")

    if weak_concepts:
        print(f"\n{'!' * 60}")
        print("WEAK CONCEPTS (accuracy < 75%) — consider dropping or fixing the prompt:")
        for c, acc in weak_concepts:
            print(f"  {c}: {acc:.1%}")
        print(f"{'!' * 60}")

    if degenerate_concepts:
        print(f"\n{'!' * 60}")
        print("DEGENERATE POSITIVE RATE (<3% or >97%) — high accuracy here may just")
        print("mean the concept predicts the same value almost everywhere, providing")
        print("little real training signal even if 'correct' most of the time:")
        for c, rate in degenerate_concepts:
            print(f"  {c}: positive_rate={rate:.1%}")
        print(f"{'!' * 60}")

    if not weak_concepts and not degenerate_concepts:
        print("\nAll concepts clear both bars (accuracy >=75%, positive rate 3-97%) "
             "— safe to proceed to full labeling / training.")

    if "model_confidence" in df.columns:
        conf = pd.to_numeric(df["model_confidence"], errors="coerce")
        t = args.confidence_threshold
        low_conf = df[conf < t]
        if len(low_conf) > 0:
            low_conf_acc = low_conf["correct_1_or_0"].mean()
            high_conf_acc = df[conf >= t]["correct_1_or_0"].mean()
            print(f"\nConfidence check: rows with model_confidence<{t:.0f} "
                 f"(n={len(low_conf)}) had accuracy {low_conf_acc:.1%}, "
                 f"vs {high_conf_acc:.1%} for confidence>={t:.0f}.")
            print("If low-confidence rows are notably less accurate, that "
                 "confirms the confidence field is meaningful and worth using "
                 "later to filter/weight training labels.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sheet", "score"], required=True)
    ap.add_argument("--labels", help="required for --mode sheet")
    ap.add_argument("--posts", help="required for --mode sheet")
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--out", default="artifacts/concept_spotcheck_sheet.csv")
    ap.add_argument("--filled", help="required for --mode score")
    ap.add_argument("--preview-len", type=int, default=500,
                    help="characters of post text shown in the review sheet — "
                         "important signal (method, farewell, completed act) "
                         "often appears late in a post")
    ap.add_argument("--confidence-threshold", type=float, default=70,
                    help="split point for the low-vs-high confidence accuracy check")
    args = ap.parse_args()

    if args.mode == "sheet":
        make_sheet(args)
    else:
        score_sheet(args)


if __name__ == "__main__":
    main()

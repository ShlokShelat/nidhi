"""
Data audit + leakage-safe splitting.

Usage:
  python src/data_audit.py --train path/to/labeled.csv [--unlabeled path/to/unlabeled.csv]

Outputs (to --outdir, default artifacts/):
  audit_report.txt      human-readable audit
  train_split.csv       user-level stratified train split
  val_split.csv         user-level stratified validation split
  label_issues.csv      rows with problems (bad labels, ungrounded gold evidence, etc.)

Why user-level splitting: posts from the same anon_user_id are correlated
(sometimes near-duplicates). A row-level random split leaks between train and
val and inflates local scores that will NOT hold on the hidden test set.
Stratification target: the user's MAXIMUM risk level (severity is ordinal),
so both splits contain users across the severity range.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scoring import (
    RISK_LEVELS, FACTOR_TAXONOMY,
    normalize_label, normalize_span, parse_evidence, parse_factors,
)

SEVERITY = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}


# Real column names in the official release -> internal names.
# The released CSVs are left byte-identical on disk; all renaming happens here.
COLUMN_ALIASES = {
    "suicide risk": "risk_level",
    "risk level": "risk_level",
    "evidence for suicide risk level": "evidence",
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    # Canonicalize once here; all downstream files (splits) inherit it, so
    # training and inference share one text space. On the official data this
    # also makes ~145 of the 171 gold spans that are otherwise not substrings
    # of their post locatable, since the mismatch is typographic punctuation.
    from text_norm import clean_text
    if "post" in df:
        df["post"] = df["post"].astype(str).map(clean_text)
    if "evidence" in df:
        df["evidence"] = df["evidence"].astype(str).map(clean_text)
    return df


def audit(df: pd.DataFrame, out: list[str]) -> pd.DataFrame:
    issues = []

    out.append(f"rows: {len(df)}")
    if "anon_user_id" in df:
        out.append(f"unique users: {df['anon_user_id'].nunique()}")
        ppu = df.groupby("anon_user_id").size()
        out.append(f"posts/user: min={ppu.min()} median={ppu.median():.0f} max={ppu.max()}")

    # --- risk label hygiene ---
    if "risk_level" in df:
        raw_counts = Counter(df["risk_level"].astype(str))
        out.append(f"\nraw risk_level values (pre-normalization): {dict(raw_counts)}")
        df["risk_level_norm"] = df["risk_level"].map(normalize_label)
        bad = df[~df["risk_level_norm"].isin(RISK_LEVELS)]
        for i, row in bad.iterrows():
            issues.append({"row": i, "issue": "unknown_risk_label", "value": row["risk_level"]})
        out.append(f"normalized distribution: {dict(Counter(df['risk_level_norm']))}")
        out.append(f"rows with unrecognized labels: {len(bad)}")

    # --- post length by class: a length/class association is a confound a
    #     model can exploit instead of the content, so it is worth seeing ---
    if "post" in df and "risk_level_norm" in df:
        df["n_words"] = df["post"].astype(str).str.split().str.len()
        by = df.groupby("risk_level_norm")["n_words"].agg(["count", "mean", "median"])
        out.append(f"\npost length by class:\n{by.to_string()}")

    # --- gold evidence groundedness: the scorer matches a predicted span by
    #     containment in the post, so if a gold span is not a substring of its
    #     post under this normalization that assumption does not hold ---
    if "evidence" in df and "post" in df:
        ungrounded = 0
        span_lens = []
        for i, row in df.iterrows():
            post_n = normalize_span(row["post"])
            for span in parse_evidence(row.get("evidence")):
                span_lens.append(len(str(span).split()))
                if normalize_span(span) not in post_n:
                    ungrounded += 1
                    issues.append({"row": i, "issue": "gold_span_not_in_post", "value": span[:80]})
        out.append(f"\ngold evidence spans: {len(span_lens)} total, "
                   f"{ungrounded} NOT found in post text (investigate if > 0)")
        if span_lens:
            arr = np.array(span_lens)
            out.append(f"gold span token length: p25={np.percentile(arr,25):.0f} "
                       f"median={np.median(arr):.0f} p75={np.percentile(arr,75):.0f} max={arr.max()}")

    # --- factor distribution: the taxonomy is macro-averaged, so every label
    #     counts equally regardless of how rare it is in the data ---
    if "factors" in df:
        c = Counter()
        unknown = Counter()
        for cell in df["factors"]:
            for f in parse_factors(cell):
                (c if f in FACTOR_TAXONOMY else unknown)[f] += 1
        out.append("\nfactor counts (sorted):")
        for f in FACTOR_TAXONOMY:
            out.append(f"  {c.get(f,0):5d}  {f}")
        if unknown:
            out.append(f"labels OUTSIDE taxonomy (fix mapping!): {dict(unknown)}")
            for f, n in unknown.items():
                issues.append({"row": -1, "issue": "factor_not_in_taxonomy", "value": f"{f} (x{n})"})

    # --- near-duplicate posts within users ---
    if "anon_user_id" in df and "post" in df:
        dup = df.duplicated(subset=["anon_user_id", "post"]).sum()
        out.append(f"\nexact duplicate (user, post) rows: {dup}")

    return pd.DataFrame(issues)


def user_level_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 13):
    """Stratified by each user's max severity."""
    df = df.copy()
    df["risk_level_norm"] = df["risk_level"].map(normalize_label)
    df["sev"] = df["risk_level_norm"].map(SEVERITY).fillna(-1)
    user_max = df.groupby("anon_user_id")["sev"].max()

    rng = np.random.default_rng(seed)
    val_users = []
    for sev_val, users in user_max.groupby(user_max):
        u = users.index.to_numpy()
        rng.shuffle(u)
        k = max(1, round(len(u) * val_frac)) if len(u) > 1 else 0
        val_users.extend(u[:k])
    val_users = set(val_users)

    train = df[~df["anon_user_id"].isin(val_users)].drop(columns=["sev"])
    val = df[df["anon_user_id"].isin(val_users)].drop(columns=["sev"])
    assert set(train["anon_user_id"]) & set(val["anon_user_id"]) == set(), "user leakage!"
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--unlabeled", default=None)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out: list[str] = ["=== LABELED DATA AUDIT ==="]
    df = load(args.train)
    issues = audit(df, out)

    if args.unlabeled:
        out.append("\n=== UNLABELED / TEST DATA ===")
        udf = load(args.unlabeled)
        out.append(f"rows: {len(udf)}, users: {udf['anon_user_id'].nunique() if 'anon_user_id' in udf else 'n/a'}")
        if "anon_user_id" in udf and "anon_user_id" in df:
            overlap = set(udf["anon_user_id"]) & set(df["anon_user_id"])
            out.append(f"users appearing in BOTH labeled and unlabeled: {len(overlap)} "
                       f"(if > 0, their labeled history is usable context at test time)")

    train, val = user_level_split(df, args.val_frac, args.seed)
    out.append(f"\n=== SPLIT ===\ntrain: {len(train)} posts / {train['anon_user_id'].nunique()} users")
    out.append(f"val:   {len(val)} posts / {val['anon_user_id'].nunique()} users")
    out.append(f"train risk dist: {dict(Counter(train['risk_level_norm']))}")
    out.append(f"val risk dist:   {dict(Counter(val['risk_level_norm']))}")

    train.to_csv(f"{args.outdir}/train_split.csv", index=False)
    val.to_csv(f"{args.outdir}/val_split.csv", index=False)
    issues.to_csv(f"{args.outdir}/label_issues.csv", index=False)
    report = "\n".join(out)
    with open(f"{args.outdir}/audit_report.txt", "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()

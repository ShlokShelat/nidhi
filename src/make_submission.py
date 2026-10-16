"""
Build and validate the leaderboard submission CSV.

Two functions, both mandatory before every upload:
  build_submission()    assembles predictions into the required format
  validate_submission() strict checks; refuses to write a file that would
                        score zero or fail to parse on their side

Format (per task description): file named YourTeamName.csv with columns
  row_id, risk_level, evidence, factors
evidence = semicolon-separated verbatim spans (empty string if none)
factors  = Python-list string, e.g. "['hopelessness', 'coping strategy']"
           ("[]" if none) — matches the organizers' labeled-file format;
           required because their parser reads a semicolon-separated list
           as a single unrecognized label.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, FACTOR_TAXONOMY, normalize_span, parse_evidence, parse_factors
from train_evidence import find_span_in_post

REQUIRED_COLS = ["row_id", "risk_level", "evidence", "factors"]


def build_submission(test_df: pd.DataFrame,
                     risk_preds: list[str],
                     evidence_preds: list[list[str]],
                     factor_preds: list[list[str]],
                     team_name: str,
                     outdir: str = "submissions") -> str:
    assert len(test_df) == len(risk_preds) == len(evidence_preds) == len(factor_preds)
    sub = pd.DataFrame({
        "row_id": test_df["row_id"].values,
        "risk_level": risk_preds,
        "evidence": ["; ".join(e) for e in evidence_preds],
        # Python-list-string format, matching the organizers' own labeled
        # file, and required by their parser: a semicolon-separated list is
        # read as one unrecognized label, which silently collapses the
        # factors score.
        "factors": [str(list(f)) for f in factor_preds],
    })
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{team_name}.csv")
    sub.to_csv(path, index=False)
    return path


def validate_submission(sub_path: str, test_df: pd.DataFrame) -> list[str]:
    """Return list of problems. Empty list = safe to upload."""
    problems = []
    sub = pd.read_csv(sub_path, keep_default_na=False)

    for c in REQUIRED_COLS:
        if c not in sub.columns:
            problems.append(f"missing column: {c}")
    if problems:
        return problems

    if len(sub) != len(test_df):
        problems.append(f"row count {len(sub)} != test rows {len(test_df)}")
    if set(sub["row_id"]) != set(test_df["row_id"]):
        problems.append("row_id set does not match the test file")
    if sub["row_id"].duplicated().any():
        problems.append("duplicate row_id values")

    bad_labels = sub[~sub["risk_level"].astype(str).str.strip().str.lower()
                     .isin(RISK_LEVELS)]
    if len(bad_labels):
        problems.append(f"{len(bad_labels)} rows with invalid risk_level, "
                        f"e.g. {bad_labels['risk_level'].iloc[0]!r}")

    post_by_id = dict(zip(test_df["row_id"], test_df["post"].astype(str)))
    ungrounded = long_spans = 0
    for _, row in sub.iterrows():
        post = post_by_id.get(row["row_id"], "")
        for sp in parse_evidence(row["evidence"]):
            if find_span_in_post(sp, post) is None:
                ungrounded += 1
            if len(str(sp).split()) > 15:
                long_spans += 1
    if ungrounded:
        problems.append(f"{ungrounded} evidence spans are NOT substrings of "
                        f"their post — they will score zero")
    if long_spans:
        problems.append(f"{long_spans} evidence spans exceed 15 tokens — "
                        f"likely wasted precision under the 3x cap")

    unknown_factors = set()
    for cell in sub["factors"]:
        for f in parse_factors(cell):
            if f not in FACTOR_TAXONOMY:
                unknown_factors.add(f)
    if unknown_factors:
        problems.append(f"factors outside taxonomy: {sorted(unknown_factors)[:5]}")

    return problems


def main():
    ap = argparse.ArgumentParser(description="Validate an existing submission CSV")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--test", required=True, help="the unlabeled test CSV")
    args = ap.parse_args()
    test_df = pd.read_csv(args.test)
    test_df.columns = [c.strip().lower() for c in test_df.columns]
    problems = validate_submission(args.submission, test_df)
    if problems:
        print("BLOCKED — fix before uploading:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("submission passes all checks")


if __name__ == "__main__":
    main()

"""
compare_real_vs_synthetic.py

The cheap sanity check before building a full error-neighbor pipeline:
put real, genuinely-missed model errors side by side with the
already-generated synthetic examples for the SAME factor, and just
look at them. Are they the same kind of writing, or does the
synthetic data look cleaner/more explicit/more "textbook" than the
real failures?

No training, no API calls — pure inspection. Run this before deciding
whether the full hybrid (real-error + spec) generation pipeline is
worth building.

Usage:
  python compare_real_vs_synthetic.py \
      --errors artifacts/train_error_seeds.csv \
      --synthetic outputs/verified/mental_health_issues/positive.jsonl \
      --factor "mental health issues" \
      --n 8
"""

from __future__ import annotations

import argparse
import json
import textwrap

import pandas as pd


def wrap(text, width=90):
    return "\n    ".join(textwrap.wrap(str(text), width=width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", required=True,
                    help="train_error_seeds.csv from mine_train_errors.py")
    ap.add_argument("--synthetic", required=True,
                    help="a verified .jsonl file, e.g. "
                         "outputs/verified/<factor>/positive.jsonl")
    ap.add_argument("--factor", required=True,
                    help="factor name as it appears in --errors (e.g. "
                         "'mental health issues')")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    errors = pd.read_csv(args.errors)
    factor_errors = errors[errors["factor"] == args.factor]
    fn = factor_errors[factor_errors["error_type"] == "false_negative"]

    synth_rows = []
    with open(args.synthetic) as f:
        for line in f:
            if line.strip():
                synth_rows.append(json.loads(line))

    print("=" * 100)
    print(f"REAL MODEL ERRORS (false negatives) — {args.factor}")
    print(f"({len(fn)} available, showing up to {args.n})")
    print("=" * 100)
    for _, r in fn.head(args.n).iterrows():
        print(f"\n  [prob={r['model_prob']:.2f} vs threshold={r['threshold']:.2f}]")
        print(f"    {wrap(r['post'])}")

    print("\n\n" + "=" * 100)
    print(f"SYNTHETIC POSITIVE EXAMPLES — {args.factor}")
    print(f"({len(synth_rows)} available, showing up to {args.n})")
    print("=" * 100)
    for r in synth_rows[:args.n]:
        print(f"\n  [model={r.get('model','?')}]")
        print(f"    {wrap(r['post'])}")

    print("\n\n" + "=" * 100)
    print("EYEBALL TEST")
    print("=" * 100)
    print("Read both groups above. Ask:")
    print("  - Do the synthetic examples read as more EXPLICIT/obvious than the real errors?")
    print("  - Do the real errors rely on INDIRECT phrasing the synthetic ones don't use?")
    print("  - Would YOU expect a model trained only on the synthetic group to catch the real ones?")
    print("If synthetic looks noticeably 'cleaner' -> build the hybrid real-error+spec generator.")
    print("If they already look similar -> the problem is likely in training/integration, not the data.")


if __name__ == "__main__":
    main()

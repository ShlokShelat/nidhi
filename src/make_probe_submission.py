"""
make_probe_submission.py

Spend ONE submission to measure something no amount of offline work can reveal:
how S1 actually splits between risk and evidence.

THE PROBLEM. S1 = (4/7)*risk_wF1 + (3/7)*evidence_F1 = 0.7944 for the champion.
That is one equation with two unknowns, and every estimate of the split so far
has come from the OOF instrument, which is not reliable enough on its own to
determine which component is weak.

THE MEASUREMENT. Submit the champion with the evidence column emptied. Every
predicted span disappears, so evidence phrase-F1 is 0 by construction (micro
averaging: no predictions -> precision 0 -> F1 0), and

    S1_empty = (4/7) * risk_wF1

which gives

    risk_wF1     = S1_empty * 7/4
    evidence_F1  = (S1_champion * 7 - 4 * risk_wF1) / 3

Both components, exactly, from a single number. No model, no instrument, no
noise -- the official scorer reports S1 to four decimals and this arithmetic is
exact.

WHY IT IS WORTH A SLOT. Submission slots are limited, so knowing which of a
0.40-weighted and a 0.30-weighted component is the weak one redirects effort
far more efficiently than guessing.

THIS SUBMISSION WILL SCORE BADLY -- around 0.75*0.7944 = 0.60 S1 -- and that is
expected. It is a probe, not an attempt. The champion is unaffected; nothing is
overwritten.

THE SAME TRICK FOR THE OTHER SIDE. --mode risk-only empties evidence AND
flattens risk to a single class, which is a useful cross-check: the two probes
must agree on risk_wF1 to within rounding, and if they do not, one of the
assumptions about the scorer is wrong.

Usage:
  python src/make_probe_submission.py --mode no-evidence \
      --out submissions/NIDHI_probe_noev.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submissions/NIDHI_canon_on_nlimnli.csv")
    ap.add_argument("--mode", default="no-evidence",
                    choices=["no-evidence", "no-factors"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--s1", type=float, default=0.7944,
                    help="the champion's S1, for the arithmetic printout")
    ap.add_argument("--s2", type=float, default=0.6323)
    args = ap.parse_args()

    b = pd.read_csv(args.base, keep_default_na=False)
    n = b.copy()

    if args.mode == "no-evidence":
        n["evidence"] = ""
        touched, kept = "evidence", "risk_level and factors"
    else:
        n["factors"] = "[]"
        touched, kept = "factors", "risk_level and evidence"

    # the probe is only interpretable if EXACTLY one column moved
    for c in b.columns:
        if c == touched:
            continue
        if not b[c].equals(n[c]):
            raise SystemExit(f"{c} changed -- probe would be uninterpretable")
    assert (n[touched] != b[touched]).sum() > 0, "probe changed nothing"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print(f"  emptied {touched}; {kept} byte-identical to {args.base}")
    print(f"  {len(n)} rows")

    print("\n" + "=" * 70)
    if args.mode == "no-evidence":
        print("WHAT TO DO WITH THE SCORE")
        print("=" * 70)
        print("Read S1 from the result, call it E. Then:")
        print("    risk_wF1    = E * 7/4")
        print(f"    evidence_F1 = ({args.s1} * 7 - 4 * risk_wF1) / 3")
        print("\nWorked examples, so the answer is readable at a glance:")
        print(f"{'S1 returned':>12}{'-> risk_wF1':>14}{'-> evidence_F1':>16}")
        for e in (0.40, 0.42, 0.44, 0.4457, 0.46, 0.48, 0.50):
            r = e * 7 / 4
            ev = (args.s1 * 7 - 4 * r) / 3
            print(f"{e:>12.4f}{r:>14.4f}{ev:>16.4f}")
        print("\nS2 must come back EXACTLY "
              f"{args.s2} -- factors were not touched. If it moves, the")
        print("scorer is doing something not accounted for and the whole")
        print("arithmetic is void.")
        print("\nTHEN: whichever component is further from its ceiling is where")
        print("the remaining 18 days go. Risk carries 0.40 of the composite,")
        print("evidence 0.30 -- so a 0.01 gain in risk is worth 1.33x the same")
        print("gain in evidence.")
    else:
        print("S1 must come back EXACTLY "
              f"{args.s1}; S2 should be 0.0000 (no factors predicted).")
        print("If S2 is not 0, the scorer credits empty predictions somehow,")
        print("which would change how every factor arm should be judged.")
    print("=" * 70)


if __name__ == "__main__":
    main()

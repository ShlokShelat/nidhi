"""
conf_column_check.py

Checks whether the confidence gate is inert in the candidate dumps.

Sweeping the per-run `evidence_min_conf` offset is only meaningful if the gate
can actually change the candidate set. Two things can make such a sweep a null
measurement rather than a negative result:

  (a) the dumps are written ALREADY FILTERED at each run's gate, so lowering the
      offset cannot recover candidates that were never written, and raising it
      only bites if some kept candidate sits below the raised gate;
  (b) the `conf` column is degenerate -- evidence_enum_candidates.py writes conf
      as the constant 1.0, and a gate on a constant column can never fire.

This prints the actual distribution so the question is settled by data before
any conclusion is drawn from a sweep.
"""
import argparse
import os

import numpy as np
import pandas as pd

RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--runs", nargs="+", default=RC7)
    args = ap.parse_args()

    print(f"{args.dir}  split={args.split}\n")
    print(f"{'run':<20}{'rows':>7}{'min_conf':>10}{'conf min':>10}"
          f"{'conf p50':>10}{'conf max':>10}{'below gate':>12}{'const?':>8}")
    any_headroom = False
    for run in args.runs:
        p = os.path.join(args.dir, f"{run}_{args.split}_candidates.csv")
        if not os.path.exists(p):
            print(f"{run:<20}  MISSING {p}")
            continue
        d = pd.read_csv(p, keep_default_na=False)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else float("nan")
        if "conf" not in d.columns:
            print(f"{run:<20}{len(d):>7}{mc:>10.2f}    NO conf COLUMN")
            continue
        c = pd.to_numeric(d["conf"], errors="coerce").to_numpy(float)
        below = int(np.sum(c < mc)) if mc == mc else -1
        const = "YES" if np.nanstd(c) < 1e-9 else "no"
        if below > 0:
            any_headroom = True
        print(f"{run:<20}{len(d):>7}{mc:>10.2f}{np.nanmin(c):>10.3f}"
              f"{np.nanmedian(c):>10.3f}{np.nanmax(c):>10.3f}"
              f"{below:>12}{const:>8}")

    print("\nHOW TO READ THIS")
    print("  const=YES anywhere -> conf carries no information for that run;")
    print("      any gate on it is inert and the offset sweep measured nothing.")
    print("  'below gate' = 0 everywhere -> the dumps are PRE-FILTERED. Lowering")
    print("      the offset is impossible without re-dumping (GPU); raising it")
    print("      is still meaningful only if conf varies above the gate.")
    print("  conf min == conf max == 1.0 -> conf was written as a constant.")
    if not any_headroom:
        print("\n  VERDICT: no candidate sits below its run's gate, so the")
        print("  filter is already a no-op on these files. A LOOSER")
        print("  gate cannot be tested from disk -- it needs a re-dump with")
        print("  min_conf=0 (GPU). A STRICTER gate is testable iff conf varies.")


if __name__ == "__main__":
    main()

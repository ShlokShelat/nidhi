"""
retrain_evidence_cond.py

Retrain the champion's shipped evidence taggers with --risk-cond, replaying each
one's own hyperparameters.

WHY REPLAY RATHER THAN RETYPE. The eight shipped taggers differ in seed, lr,
epochs and val mode -- evidence_lr2 and evidence_ep9 are named for exactly those
differences. Retyping the flags is how a "single-variable change" quietly
becomes three. Each run's `run_config.json` is `vars(args)` from the training
that produced it, so the honest way to add ONE variable is to read that file back
and change only `--risk-cond` and `--outdir`.

WHAT IT SKIPS AND WHY. Flags that describe WHERE the run went (outdir) or what it
measured (best_val_phrase_f1, evidence_min_conf) are not training inputs and are
dropped. Everything else is replayed verbatim, including --seed, so the arm
differs from its parent in exactly one respect.

CONDITIONING AT TRAIN TIME USES GOLD LEVELS -- that is teacher forcing and is
correct; the train CSV carries risk_level. Only INFERENCE needs predicted levels,
and that is evidence_dump_all.py --risk-cond.

Usage:
  python src/retrain_evidence_cond.py --dry-run
  python src/retrain_evidence_cond.py
  python src/retrain_evidence_cond.py --runs evidence_v2,evidence_lr2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SHIPPED = ["evidence_v2", "evidence_v1", "evidence_s14", "evidence_s15",
           "evidence_s16", "evidence_s17", "evidence_lr2", "evidence_ep9"]

# not training inputs -- these describe the output or the measurement
DROP = {"outdir", "best_val_phrase_f1", "evidence_min_conf", "trained_on",
        "risk_cond", "val_risk_from", "batch_note"}

# store_true flags: emit the flag alone when true, nothing when false
FLAGS = {"no_val"}


def cmd_for(run: str, cfg: dict, outdir: str) -> list[str]:
    c = [sys.executable, "src/train_evidence.py"]
    for k, v in cfg.items():
        if k in DROP or v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if k in FLAGS:
            if v:
                c.append(flag)
            continue
        if isinstance(v, bool):
            if v:
                c.append(flag)
            continue
        c += [flag, str(v)]
    c += ["--outdir", outdir, "--risk-cond"]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(SHIPPED))
    ap.add_argument("--suffix", default="_rc")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="train identical configs anyway (see the warning)")
    args = ap.parse_args()

    runs = [r.strip() for r in args.runs.split(",")]
    missing = [r for r in runs
               if not os.path.exists(os.path.join("runs", r, "run_config.json"))]
    if missing:
        raise SystemExit(f"no run_config.json for {missing}")

    cfgs = {r: json.load(open(os.path.join("runs", r, "run_config.json")))
            for r in runs}

    # DUPLICATE CONFIGS ARE A CORRECTNESS BUG, NOT JUST WASTED GPU.
    #
    # evidence_v1 and evidence_v2 replay to byte-identical commands (same seed,
    # lr, epochs, data). Two identical models agree on EVERYTHING, and the
    # shipped vote is k=2 -- so every span either one emits would clear the
    # threshold on its own. The ensemble would stop filtering.
    #
    # The originals may still be genuinely different models, trained at
    # different times under different code; run_config.json cannot show that.
    # Their recorded val scores can, so print those and let the operator judge.
    seen, bumped = {}, {}
    for r in runs:
        key = tuple(sorted((k, str(v)) for k, v in cfgs[r].items()
                           if k not in DROP))
        if key in seen:
            twin = seen[key]
            f1_a = cfgs[twin].get("best_val_phrase_f1")
            f1_b = cfgs[r].get("best_val_phrase_f1")
            print(f"!! {r} has the SAME training config as {twin}")
            print(f"   recorded val phrase F1: {twin}={f1_a}  {r}={f1_b}")
            if not args.allow_duplicates:
                new_seed = int(cfgs[r].get("seed", 13)) - 1
                while any(int(c.get("seed", 13)) == new_seed
                          for c in cfgs.values()):
                    new_seed -= 1
                bumped[r] = (int(cfgs[r].get("seed", 13)), new_seed)
                cfgs[r]["seed"] = new_seed
                print(f"   -> seeding {r} at {new_seed} instead, to keep eight "
                      f"DISTINCT voters")
        else:
            seen[key] = r
    if bumped:
        print(f"\nseed changes: {bumped}")
        print("This is a deliberate deviation from a pure replay. The champion "
              "has\neight separately-trained taggers; a pure replay would give "
              "seven plus a\nclone, and a clone breaks k=2 voting. Re-run with "
              "--allow-duplicates to\noverride.\n")

    for run in runs:
        cfg = cfgs[run]
        out = os.path.join("runs", run + args.suffix)
        if os.path.exists(os.path.join(out, "model_final.pt")):
            print(f"--- {out}: exists, skipping ---")
            continue
        c = cmd_for(run, cfg, out)
        print(f"\n{'=' * 70}\n{run} -> {out}\n{'=' * 70}")
        print(" ".join(c))
        if args.dry_run:
            continue
        r = subprocess.run(c)
        if r.returncode != 0:
            raise SystemExit(f"FAILED {run} (exit {r.returncode})")

    print("\nall arms done" if not args.dry_run else "\ndry run only")


if __name__ == "__main__":
    main()

# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm and
# the second risk member can be rebuilt from source without the original
# project tree.
#
# LAYOUT NOTE. In the original project the stage modules live in a `src/`
# subdirectory beside this file. Here they are siblings of it, so the path
# insert below points at this file's own directory instead. Nothing else is
# changed.
"""Run the whole pipeline with one command:

    python src/friend_pipeline/run_pipeline.py --team_name MyTeam

Stages:
  1. clean data, build user-grouped folds, build evidence lexicon
  2. fine-tune the tri-head model, one per fold (the long part)
  3. tune thresholds on out-of-fold predictions, print honest CV score
  4. ensemble-predict the evaluation set, write <TeamName>.csv

Flags let you re-run any single stage without redoing the rest.

Stage 3 writes outputs/risk_probs_val.csv and stage 4 writes
outputs/risk_probs.csv. Those two files are the second risk member consumed by
src/ensemble_risk.py as --friend-probs; stage 3 also produces outputs/oof.npy,
from which friend_export_oof.py builds the flat OOF table.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team_name", default="MyTeam")
    ap.add_argument("--stages", default="1,2,3,4",
                    help="comma list, e.g. '3,4' to skip retraining")
    args = ap.parse_args()
    stages = {s.strip() for s in args.stages.split(",")}

    for f in (C.TRAIN_XLSX, C.LEADERBOARD_XLSX):
        if not os.path.exists(f):
            sys.exit(f"ERROR: missing {f}. Put train.xlsx and leaderboard.xlsx "
                     f"into the data/ folder first.")

    t0 = time.time()
    if "1" in stages:
        import stage1_data
        stage1_data.run()
    if "2" in stages:
        import stage2_train
        stage2_train.run()
    if "3" in stages:
        import stage3_tune
        stage3_tune.run()
    if "4" in stages:
        import stage4_predict
        stage4_predict.run(args.team_name)
    print(f"\nAll requested stages done in {(time.time() - t0) / 60:.1f} min.")
    print(f"Submission file (if stage 4 ran): outputs/{args.team_name}.csv")


if __name__ == "__main__":
    main()

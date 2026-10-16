"""
Dump probabilities from trained multi-label factor arms
(src/train_factors_multilabel.py) in the exact .npz format the pool tools
consume: keys `probs` (n, 24 float), `row_id` (object), `thresholds` --
what `factor_structure.load_oof` / `resolve_test` read.

Mirrors predict_factors_nli.py's interface: --run accepts comma-separated
run dirs and averages probs + thresholds across them, so the same script
produces both the per-fold OOF dump (one dir, or several seeds of one fold)
and the 5-fold-bagged test dump (all fold dirs). Using one code path for
both is what makes the OOF dump a valid instrument for the test dump: a
threshold or a weight tuned on probabilities produced by a different
averaging rule is tuned for a different distribution than the one it is
applied to.

Usage:
  python src/predict_factors_multilabel.py \
      --test artifacts/folds/val_f0.csv \
      --run runs/factors_mlc_s21_cv_f0 \
      --max-len 320 \
      --dump-probs artifacts/factor_runs/mlc_cv_oof_f0.npz \
      --out artifacts/factor_runs/mlc_cv_oof_f0_labels.csv
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load                                    # noqa: E402
from scoring import FACTOR_TAXONOMY                            # noqa: E402
from train_factors_nli import N                                # noqa: E402
from train_factors_multilabel import MultiLabelFactorArm, score_posts  # noqa: E402

# buffers that live in the LOSS, not the model -- a checkpoint may carry them
# while a bare inference instance does not (or vice versa). Anything else
# missing/unexpected means the run dir is not a multilabel arm: refuse.
LOSS_BUFFERS = {"pos_weight_vec", "nu_vec", "cb_vec"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--run", required=True,
                    help="one run dir, OR comma-separated dirs to bag "
                         "(avg probs + avg thresholds) -- pass all 5 fold "
                         "dirs for the test dump")
    ap.add_argument("--out", default=None,
                    help="optional (row_id, factors) CSV of the decoded labels")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--thresholds", default=None,
                    help="optional .npy threshold array to use INSTEAD of "
                         "the runs' own averaged thresholds.npy")
    ap.add_argument("--dump-probs", default=None,
                    help=".npz with probs/row_id/thresholds -- the artifact "
                         "the pool tools read")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    if not args.dump_probs and not args.out:
        raise SystemExit("nothing to write: pass --dump-probs and/or --out")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    run_dirs = [r.strip() for r in args.run.split(",") if r.strip()]
    df = load(args.test)

    all_probs, ths_list = [], []
    for rd in run_dirs:
        cfg_path = os.path.join(rd, "run_config.json")
        if not os.path.exists(cfg_path):
            raise SystemExit(f"{rd}: no run_config.json -- not a multilabel "
                             f"arm dir (NLI arms would load with a silently "
                             f"wrong head). Refusing.")
        cfg = json.load(open(cfg_path))
        if cfg.get("multilabel_arm") is not True:
            raise SystemExit(f"{rd}: run_config.json lacks the "
                             f"'multilabel_arm' marker -- another experiment "
                             f"owns this dir. Refusing.")
        from transformers import AutoTokenizer
        model_name = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(model_name)
        state = torch.load(os.path.join(rd, "model.pt"), map_location=device)
        # A --split-heads arm saves head_risk.*/head_prot.* instead of head.*.
        # The architecture is read from the checkpoint's own parameter names
        # rather than from run_config.json, because the parameter names are
        # what load_state_dict must match and a config is only a description
        # of them. Mis-detecting here is not silent: the head keys would land
        # in `missing`/`unexpected` and the strict check below refuses rather
        # than scoring with a randomly initialised head.
        split = any(k.startswith("head_risk.") for k in state)
        if split:
            print(f"  {rd}: split-head arm (19 risk + 5 protective)")
        model = MultiLabelFactorArm(model_name, split_heads=split).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad = ([k for k in missing if k.split(".")[-1] not in LOSS_BUFFERS]
               + [k for k in unexpected if k.split(".")[-1] not in LOSS_BUFFERS])
        if bad:
            raise SystemExit(f"{rd}: checkpoint/model mismatch beyond loss "
                             f"buffers: {bad}")
        model.eval()
        all_probs.append(score_posts(model, df, tok, args.max_len, device,
                                     use_bf16, batch=args.batch))
        ths_list.append(np.load(os.path.join(rd, "thresholds.npy")))
        print(f"  scored {rd}  ({model_name})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    probs = np.mean(all_probs, axis=0)
    if args.thresholds:
        ths = np.load(args.thresholds)
        print(f"using threshold override: {args.thresholds}")
    else:
        ths = np.mean(ths_list, axis=0)
    if len(run_dirs) > 1:
        print(f"bagged {len(run_dirs)} models (avg probs + avg thresholds)")

    row_id = np.array([str(x) for x in df["row_id"].tolist()], dtype=object)
    if args.dump_probs:
        os.makedirs(os.path.dirname(args.dump_probs) or ".", exist_ok=True)
        np.savez(args.dump_probs, probs=probs, thresholds=ths, row_id=row_id)
        print(f"dumped probabilities -> {args.dump_probs}  {probs.shape}")

    if args.out:
        on = (probs >= ths).astype(int)
        fac_lists = [[FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]]
                     for i in range(len(df))]
        out = pd.DataFrame({"row_id": df["row_id"].tolist(),
                            "factors": [str(list(f)) for f in fac_lists]})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}: mean {on.sum(1).mean():.2f} factors/post, "
              f"0-factor posts {(on.sum(1) == 0).sum()}/{len(df)}")


if __name__ == "__main__":
    main()

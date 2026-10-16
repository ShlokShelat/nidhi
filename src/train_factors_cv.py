"""
User-level k-fold CV for the factor classifier.

Why this exists: a single 20% val split leaves rare labels with 0-2
positives (sexual orientation=0, poor school perf=1, exposure=2 in our
split), so their local F1 is pure noise and we cannot tune their
thresholds. 5-fold CV gives EVERY label its full positive count across the
concatenated out-of-fold (OOF) predictions, so per-label F1 and per-label
thresholds become trustworthy.

Pipeline:
  1. Split TRAIN users into k folds (user-level, no leakage).
  2. For each fold: optionally augment the 4 training folds (never the held
     fold), train a factor model, predict the held fold -> OOF probs.
  3. Concatenate OOF probs over all folds = one honest prediction per train
     post. Tune per-label thresholds on the FULL OOF set (every label now
     has real support). Report per-label F1 that we can actually trust.
  4. Optionally train a FINAL model on ALL train data (+aug) with those
     thresholds, saved for submission.

Augmentation is done INSIDE each fold on that fold's training rows only, so
no synthetic row derived from a held-out post ever leaks into training.

Usage (GPU node, riskcup):
  python src/train_factors_cv.py \
      --train artifacts/train_split.csv \
      --val artifacts/val_split.csv \
      --model microsoft/deberta-v3-large \
      --k 5 --epochs 6 \
      --aug artifacts/train_aug.csv \
      --final-outdir runs/factors_cv_final

--aug is an OPTIONAL pre-generated augmented file; if given, its synthetic
rows are matched to folds by their synth_source's user, so augmentation
respects the fold boundary. Omit --aug to run CV on real data only.
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from scoring import FACTOR_TAXONOMY, parse_factors, factors_macro_f1, normalize_label
from train_factors import (
    FactorDataset, MultiLabelModel, tune_thresholds, to_factor_lists,
    predict_probs, N, F2I,
)


def user_folds(df: pd.DataFrame, k: int, seed: int = 13) -> list[np.ndarray]:
    """Assign each user to one of k folds; return per-fold row-index arrays.
    Stratify roughly by user's post count so folds are balanced in size."""
    users = list(df["anon_user_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    user_fold = {u: i % k for i, u in enumerate(users)}
    folds = []
    for f in range(k):
        mask = df["anon_user_id"].map(user_fold) == f
        folds.append(np.where(mask.values)[0])
    return folds


def train_one(tr_df, model_name, max_len, epochs, lr, batch, grad_accum,
              device, use_bf16, seed, condition_col=None):
    tr_ds = FactorDataset(tr_df, AutoTokenizer.from_pretrained(model_name), max_len, condition_col)
    # pos_weight from this fold's training data, clamped [1, 20]
    pos = tr_ds.labels.sum(0)
    pos_weight = np.clip((len(tr_ds) - pos) / np.maximum(pos, 1), 1.0, 20.0)
    loader = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0)
    model = MultiLabelModel(model_name, pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = (len(loader) // grad_accum + 1) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    torch.manual_seed(seed)
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        for b, batch_d in enumerate(loader):
            batch_d.pop("idx")
            batch_d = {k: v.to(device) for k, v in batch_d.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch_d)["loss"]
            (loss / grad_accum).backward()
            if (b + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
    return model


def predict_df(model, df, model_name, max_len, device, use_bf16, condition_col=None):
    ds = FactorDataset(df, AutoTokenizer.from_pretrained(model_name), max_len, condition_col)
    loader = DataLoader(ds, batch_size=batch_eval(), shuffle=False, num_workers=0)
    return predict_probs(model, loader, device, use_bf16), ds.labels


def batch_eval():
    return 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None, help="held-out val for a final sanity number")
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--aug", default=None,
                    help="optional augmented CSV (has is_synthetic, synth_source)")
    ap.add_argument("--external", default=None,
                    help="external labeled CSV (new users, no gold-fold membership); added to EVERY training fold — leak-safe because these rows never appear in any held-out fold")
    ap.add_argument("--final-outdir", default="runs/factors_cv_final")
    ap.add_argument("--save-oof", action="store_true",
                    help="save OOF probs+gold as oof_probs.npz for stacking")
    ap.add_argument("--condition-col", default=None,
                    help="column with predicted risk to prepend as [RISK=..] (must exist in --train/--val/--aug files)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    df = pd.read_csv(args.train).reset_index(drop=True)
    folds = user_folds(df, args.k, args.seed)

    # optional augmentation, mapped to the fold of its source user
    aug_df = None
    if args.aug:
        aug_df = pd.read_csv(args.aug)
        aug_df = aug_df[aug_df.get("is_synthetic", 0) == 1].reset_index(drop=True)
        src_user = dict(zip(df["row_id"], df["anon_user_id"]))
        aug_df["src_user"] = aug_df["synth_source"].map(src_user)
        users = list(df["anon_user_id"].unique())
        rng = np.random.default_rng(args.seed); rng.shuffle(users)
        user_fold = {u: i % args.k for i, u in enumerate(users)}
        aug_df["fold"] = aug_df["src_user"].map(user_fold)
        print(f"augment: {len(aug_df)} synthetic rows mapped to folds")

    ext_df = None
    if args.external:
        ext_df = pd.read_csv(args.external, keep_default_na=False)
        # ensure required columns exist; external posts have no anon_user_id/post_id
        for col, default in [("anon_user_id", "EXT"), ("post_id", ""),
                             ("risk_level", "Indicator"), ("evidence", "")]:
            if col not in ext_df.columns:
                ext_df[col] = default
        print(f"external: {len(ext_df)} rows added to EVERY training fold (leak-safe)")

    oof_probs = np.zeros((len(df), N), dtype=np.float32)
    gold = np.zeros((len(df), N), dtype=np.int64)

    for f in range(args.k):
        held = folds[f]
        train_mask = np.ones(len(df), dtype=bool); train_mask[held] = False
        tr_df = df[train_mask].copy()
        if aug_df is not None:
            add = aug_df[aug_df["fold"] != f][
                ["row_id", "anon_user_id", "post_id", "post",
                 "risk_level", "evidence", "factors"]]
            tr_df = pd.concat([tr_df, add], ignore_index=True)
        if ext_df is not None:
            ext_add = ext_df[["row_id", "anon_user_id", "post_id", "post",
                              "risk_level", "evidence", "factors"]]
            tr_df = pd.concat([tr_df, ext_add], ignore_index=True)
        print(f"\n=== fold {f+1}/{args.k}: train {len(tr_df)} "
              f"(held {len(held)}) ===")
        model = train_one(tr_df, args.model, args.max_len, args.epochs,
                          args.lr, args.batch, args.grad_accum,
                          device, use_bf16, args.seed + f, args.condition_col)
        held_df = df.iloc[held].reset_index(drop=True)
        probs, labels = predict_df(model, held_df, args.model, args.max_len,
                                   device, use_bf16, args.condition_col)
        oof_probs[held] = probs
        gold[held] = labels
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # honest per-label thresholds on the FULL OOF set (every label has support)
    ths = tune_thresholds(oof_probs, gold)
    pred = (oof_probs >= ths).astype(int)
    macro, per_label = factors_macro_f1(
        to_factor_lists(gold), to_factor_lists(pred), return_per_label=True)
    macro05 = factors_macro_f1(to_factor_lists(gold),
                               to_factor_lists((oof_probs >= 0.5).astype(int)))

    print(f"\n===== {args.k}-fold OOF results (TRUSTWORTHY) =====")
    print(f"OOF macro F1 tuned = {macro:.4f}  @0.5 = {macro05:.4f}")
    print("per-label F1 (worst first) — now with real support:")
    for lab, v in sorted(per_label.items(), key=lambda kv: kv[1]):
        n_pos = int(gold[:, F2I[lab]].sum())
        print(f"  {v:.3f}  th={ths[F2I[lab]]:.2f}  n={n_pos:4d}  {lab}")

    os.makedirs(args.final_outdir, exist_ok=True)
    np.save(os.path.join(args.final_outdir, "thresholds.npy"), ths)
    if args.save_oof:
        np.savez(os.path.join(args.final_outdir, "oof_probs.npz"),
                 probs=oof_probs, gold=gold)
        print(f"saved OOF probs+gold to {args.final_outdir}/oof_probs.npz")

    # final model on ALL train (+aug), using OOF-tuned thresholds
    print("\n=== training FINAL model on all train (+aug) ===")
    full_df = df.copy()
    if aug_df is not None:
        add = aug_df[["row_id", "anon_user_id", "post_id", "post",
                      "risk_level", "evidence", "factors"]]
        full_df = pd.concat([full_df, add], ignore_index=True)
    final = train_one(full_df, args.model, args.max_len, args.epochs,
                      args.lr, args.batch, args.grad_accum,
                      device, use_bf16, args.seed, args.condition_col)
    torch.save(final.state_dict(), os.path.join(args.final_outdir, "model_final.pt"))
    with open(os.path.join(args.final_outdir, "run_config.json"), "w") as fh:
        json.dump({"model": args.model, "max_len": args.max_len,
                   "k": args.k, "epochs": args.epochs,
                   "aug": args.aug, "oof_macro_f1": float(macro),
                   "trained_on": "all_train_plus_aug"}, fh, indent=2)

    # optional: score the final model on the untouched val split for a
    # comparison point against earlier single-split numbers
    if args.val:
        vp, vg = predict_df(final, pd.read_csv(args.val), args.model,
                            args.max_len, device, use_bf16, args.condition_col)
        vmacro = factors_macro_f1(to_factor_lists(vg.astype(int)),
                                  to_factor_lists((vp >= ths).astype(int)))
        print(f"\nfinal model on held-out val split: macro F1 = {vmacro:.4f} "
              f"(comparable to earlier 0.4165/0.4342 numbers)")

    print(f"\nsaved final model + OOF thresholds to {args.final_outdir}")


if __name__ == "__main__":
    main()

"""
Factor classifier: 24 independent sigmoid outputs + per-label threshold tuning.

Why thresholds matter here: the metric is MACRO F1, so the rarest label
counts as much as the most common. A blanket 0.5 cutoff systematically
under-predicts rare labels; tuning each label's threshold on validation
directly optimizes the deployed metric. This is usually worth several
macro-F1 points on long-tailed label sets, for free.

Usage:
  python src/train_factors.py --train artifacts/train_split.csv \
      --val artifacts/val_split.csv --model microsoft/deberta-v3-large \
      --outdir runs/factors_v1
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no hub calls; cache only (override: HF_HUB_OFFLINE=0)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from scoring import FACTOR_TAXONOMY, parse_factors, factors_macro_f1

N = len(FACTOR_TAXONOMY)
F2I = {f: i for i, f in enumerate(FACTOR_TAXONOMY)}


class FactorDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, condition_col=None):
        """condition_col: optional column name holding a per-row risk label
        to prepend as '[RISK=<level>] '. Uses PREDICTED risk (not gold) so
        train and test see the same signal quality. None = no conditioning
        (backward compatible)."""
        posts = df["post"].astype(str).tolist()
        if condition_col is not None and condition_col in df.columns:
            cond = df[condition_col].astype(str).tolist()
            self.texts = [f"[RISK={c.strip().lower()}] {p}"
                          for c, p in zip(cond, posts)]
        else:
            self.texts = posts
        self.labels = np.zeros((len(df), N), dtype=np.float32)
        for r, cell in enumerate(df["factors"]):
            for f in parse_factors(cell):
                if f in F2I:
                    self.labels[r, F2I[f]] = 1.0
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i]),
            "idx": torch.tensor(i),
        }


class MultiLabelModel(nn.Module):
    def __init__(self, model_name, pos_weight=None, focal_gamma=0.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(h, N)
        self.focal_gamma = focal_gamma  # 0.0 = plain BCE; >0 enables focal
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.float) if pos_weight is not None else None,
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        pooled = pooled.to(self.head.weight.dtype)  # mixed-precision safety
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            y = labels.to(logits.dtype)
            if self.focal_gamma > 0:
                # per-element BCE (no reduction), then focal modulation
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=self.pos_weight, reduction="none")
                # p_t = model prob of the TRUE class per element
                p = torch.sigmoid(logits)
                p_t = p * y + (1 - p) * (1 - y)
                focal = (1 - p_t).clamp(min=1e-6) ** self.focal_gamma
                loss = (focal * bce).mean()
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=self.pos_weight)
        return {"loss": loss, "logits": logits}


def tune_thresholds(probs: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Per-label threshold sweep maximizing that label's binary F1 on val."""
    ths = np.full(N, 0.5)
    grid = np.arange(0.05, 0.95, 0.025)
    MIN_POS = 5
    for j in range(N):
        if gold[:, j].sum() < MIN_POS:
            continue  # too few positives: tuning would fit noise; keep 0.5
                      # (v1 tuned 'exposure to others suicide' to 0.13 on ~3
                      # positives and it fired on a large share of test posts)
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (probs[:, j] >= t).astype(int)
            tp = int((pred & gold[:, j].astype(int)).sum())
            fp = int((pred & (1 - gold[:, j].astype(int))).sum())
            fn = int(((1 - pred) & gold[:, j].astype(int)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        ths[j] = best_t
    return ths


def to_factor_lists(binary: np.ndarray) -> list[list[str]]:
    return [[FACTOR_TAXONOMY[j] for j in range(N) if row[j]] for row in binary]


@torch.no_grad()
def predict_probs(model, loader, device, use_bf16):
    model.eval()
    out = []
    for batch in loader:
        batch.pop("idx"); batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=use_bf16):
            logits = model(**batch)["logits"]
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--pos-weighting", action="store_true", default=True)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--condition-col", default=None,
                    help="column with predicted risk to prepend as [RISK=..]")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (args.precision == "bf16" and device == "cuda"
                and torch.cuda.is_bf16_supported())
    print(f"[precision] {'bf16 autocast' if use_bf16 else 'fp32'} on {device}")
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tr_df, va_df = pd.read_csv(args.train), pd.read_csv(args.val)
    tr_ds = FactorDataset(tr_df, tok, args.max_len, args.condition_col)
    va_ds = FactorDataset(va_df, tok, args.max_len, args.condition_col)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.num_workers)
    va_loader = DataLoader(va_ds, batch_size=args.batch * 2, shuffle=False,
                           num_workers=args.num_workers)

    pos_weight = None
    if args.pos_weighting:
        pos = tr_ds.labels.sum(0)
        # clamp to [1, 20]: upweight rare labels, NEVER downweight common ones
        # (n_neg/n_pos < 1 for majority labels like hopelessness suppressed
        # them badly: 0.70 F1 on a 98%-base-rate label in v1)
        pos_weight = np.clip((len(tr_ds) - pos) / np.maximum(pos, 1), 1.0, 20.0)

    model = MultiLabelModel(args.model, pos_weight, args.focal_gamma).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(tr_loader) // args.grad_accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    best = {"macro": -1.0, "state": None, "ths": None}
    step = 0
    gold_bin = va_ds.labels.astype(int)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        opt.zero_grad()
        for b, batch in enumerate(tr_loader):
            idx = batch.pop("idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                rows = tr_df.iloc[idx.tolist()]
                print(f"\nNON-FINITE LOSS at epoch {epoch} batch {b}")
                print("row_ids:", rows.get("row_id", rows.index).tolist())
                rows.to_csv(os.path.join(args.outdir, "poison_batch.csv"), index=False)
                sys.exit("aborted; poison_batch.csv written")
            (loss / args.grad_accum).backward()
            running.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gn):
                    sys.exit(f"NON-FINITE GRAD NORM at epoch {epoch} batch {b}")
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} "
                          f"loss {np.mean(running[-40:]):.4f} grad_norm {gn:.2f}")

        probs = predict_probs(model, va_loader, device, use_bf16)
        ths = tune_thresholds(probs, va_ds.labels)
        macro = factors_macro_f1(to_factor_lists(gold_bin),
                                 to_factor_lists((probs >= ths).astype(int)))
        macro05 = factors_macro_f1(to_factor_lists(gold_bin),
                                   to_factor_lists((probs >= 0.5).astype(int)))
        print(f"== epoch {epoch}: macro F1 tuned={macro:.4f} @0.5={macro05:.4f}")
        if macro > best["macro"]:
            best = {"macro": macro,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
                    "ths": ths}

    model.load_state_dict(best["state"]); model.to(device)
    probs = predict_probs(model, va_loader, device, use_bf16)
    ths = best["ths"]
    macro, per_label = factors_macro_f1(
        to_factor_lists(gold_bin), to_factor_lists((probs >= ths).astype(int)),
        return_per_label=True)
    print(f"\nbest val macro F1 (tuned) = {macro:.4f}")
    print("per-label F1 (worst first) — the augmentation to-do list:")
    for f, v in sorted(per_label.items(), key=lambda kv: kv[1]):
        print(f"  {v:.3f}  th={ths[F2I[f]]:.2f}  {f}")

    va_out = va_df.copy()
    va_out["pred_factors"] = ["; ".join(fl) for fl in
                              to_factor_lists((probs >= ths).astype(int))]
    va_out.to_csv(os.path.join(args.outdir, "val_predictions.csv"), index=False)
    np.save(os.path.join(args.outdir, "thresholds.npy"), ths)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model_final.pt"))
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "best_val_macro_f1_tuned": float(macro)}, f, indent=2)


if __name__ == "__main__":
    main()

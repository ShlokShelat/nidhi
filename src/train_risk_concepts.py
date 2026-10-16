"""
Risk-level classifier with AUXILIARY CONCEPT SUPERVISION (Option 1:
shared encoder, separate heads, no bottleneck).

                 DeBERTa Encoder
                        |
      +-----------------+---------------------------+
      |                 |                           |
      v                 v                           v
  Risk Head       Concept Heads (x8)           (CORN Head - unchanged)
      |                 |                           |
  Risk Loss       Concept Losses (BCE)         CORN Loss

Risk head does NOT consume concept predictions — this tests only
whether shared-representation multi-task learning helps, isolated
from any bottleneck-style dependency. Same discipline as every other
experiment this campaign: one clean variable at a time.

Requires a --concept-labels CSV from generate_concept_labels.py
(row_id + 8 boolean columns, same row order as --train).

Usage:
  python src/train_risk_concepts.py --train artifacts/train_split.csv \
      --val artifacts/val_split.csv --model microsoft/deberta-v3-large \
      --outdir runs/risk_concepts \
      --concept-labels artifacts/train_concept_labels.csv \
      --focal-gamma 2.0 --concept-loss-weight 0.3 --seed 13
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

LABEL2ID = {l: i for i, l in enumerate(RISK_LEVELS)}
CONCEPTS = [
    "method_mechanism", "plan_present", "completed_act", "urgency_imminence",
    "speaker_is_author", "farewell_framing", "protective_factor", "negation",
    "past_attempt_history",
]


def build_text(row, use_user_context):
    if not use_user_context:
        return str(row["post"])
    k = row.get("post_index_in_user", "?")
    n = row.get("user_post_count", "?")
    return f"[USER_POST {k}/{n}] {row['post']}"


class PostDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, use_user_context, concept_df=None):
        df = df.copy()
        if "anon_user_id" in df:
            df["user_post_count"] = df.groupby("anon_user_id")["post"].transform("size")
            df["post_index_in_user"] = df.groupby("anon_user_id").cumcount() + 1
        self.texts = [build_text(r, use_user_context) for _, r in df.iterrows()]
        self.labels = [LABEL2ID[normalize_label(x)] for x in df["risk_level"]]
        self.tok = tokenizer
        self.max_len = max_len
        self.has_concepts = concept_df is not None
        if self.has_concepts:
            merged = df[["row_id"]].merge(concept_df, on="row_id", how="left")
            for c in CONCEPTS:
                merged[c] = merged[c].fillna(False).astype(float)
            self.concepts = merged[CONCEPTS].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i]),
            "idx": torch.tensor(i),
        }
        if self.has_concepts:
            item["concept_labels"] = torch.tensor(self.concepts[i], dtype=torch.float)
        return item


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


class ConceptRiskModel(nn.Module):
    """Shared encoder -> risk head (softmax + CORN, focal loss) +
    8 independent concept heads (BCE). Concept predictions are NEVER
    fed into the risk head — Option 1, not a bottleneck."""
    def __init__(self, model_name, n_classes=4, alpha_ordinal=0.5,
                 class_weights=None, focal_gamma=2.0, concept_loss_weight=0.3,
                 n_concepts=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.cat_head = nn.Linear(h, n_classes)
        self.ord_head = nn.Linear(h, n_classes - 1)
        self.concept_head = nn.Linear(h, n_concepts)  # one linear layer, 8 outputs
        self.alpha = alpha_ordinal
        self.n_classes = n_classes
        self.concept_loss_weight = concept_loss_weight
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None,
        )
        self.focal = FocalLoss(gamma=focal_gamma, weight=self.class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None, concept_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        pooled = pooled.to(self.cat_head.weight.dtype)
        cat_logits = self.cat_head(pooled)
        ord_logits = self.ord_head(pooled)
        concept_logits = self.concept_head(pooled)  # (B, 8) — NOT used by risk head

        loss = None
        if labels is not None:
            fl = self.focal(cat_logits, labels)
            corn = self._corn_loss(ord_logits, labels)
            risk_loss = (1 - self.alpha) * fl + self.alpha * corn

            concept_loss = torch.tensor(0.0, device=cat_logits.device)
            if concept_labels is not None:
                concept_loss = nn.functional.binary_cross_entropy_with_logits(
                    concept_logits.to(torch.float32), concept_labels)

            loss = risk_loss + self.concept_loss_weight * concept_loss

        return {"loss": loss, "logits": self._combine(cat_logits, ord_logits),
                "concept_logits": concept_logits}

    def _corn_loss(self, ord_logits, labels):
        total, n_terms = 0.0, 0
        for k in range(self.n_classes - 1):
            mask = labels >= k
            if mask.sum() == 0:
                continue
            target = (labels[mask] > k).float()
            total = total + nn.functional.binary_cross_entropy_with_logits(
                ord_logits[mask, k], target, reduction="sum")
            n_terms += mask.sum().item()
        return total / max(n_terms, 1)

    def _combine(self, cat_logits, ord_logits):
        cat_p = torch.softmax(cat_logits, dim=-1)
        cond = torch.sigmoid(ord_logits)
        cum = torch.cumprod(cond, dim=-1)
        ones = torch.ones_like(cum[:, :1])
        upper = torch.cat([ones, cum], dim=-1)
        lower = torch.cat([cum, torch.zeros_like(cum[:, :1])], dim=-1)
        ord_p = (upper - lower).clamp_min(1e-8)
        return torch.log(0.5 * cat_p + 0.5 * ord_p)


@torch.no_grad()
def evaluate(model, loader, device, use_bf16):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch.pop("idx")
        batch.pop("concept_labels", None)
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(**batch)["logits"]
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    gold = [RISK_LEVELS[i] for i in labels.tolist()]
    pred = [RISK_LEVELS[i] for i in logits.argmax(-1).tolist()]
    return risk_weighted_f1(gold, pred), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--alpha-ordinal", type=float, default=0.5)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--concept-loss-weight", type=float, default=0.3,
                    help="how much the concept heads' loss contributes overall")
    ap.add_argument("--concept-labels", required=True,
                    help="CSV from generate_concept_labels.py")
    ap.add_argument("--class-weighting", action="store_true")
    ap.add_argument("--user-context", action="store_true", default=True)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (args.precision == "bf16" and device == "cuda"
                and torch.cuda.is_bf16_supported())
    print(f"[precision] {'bf16 autocast' if use_bf16 else 'fp32'} on {device}")
    print(f"[focal] gamma={args.focal_gamma}  [concept_loss_weight]={args.concept_loss_weight}")
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tr_df, va_df = pd.read_csv(args.train), pd.read_csv(args.val)
    concept_df = pd.read_csv(args.concept_labels)
    for c in CONCEPTS:
        assert c in concept_df.columns, f"missing concept column: {c}"

    weights = None
    if args.class_weighting:
        counts = tr_df["risk_level"].map(normalize_label).value_counts()
        weights = [len(tr_df) / (len(RISK_LEVELS) * counts.get(l, 1))
                   for l in RISK_LEVELS]

    g = torch.Generator()
    g.manual_seed(args.seed)

    tr_loader = DataLoader(
        PostDataset(tr_df, tok, args.max_len, args.user_context, concept_df=concept_df),
        batch_size=args.batch, shuffle=True, generator=g, num_workers=args.num_workers)
    va_loader = DataLoader(
        PostDataset(va_df, tok, args.max_len, args.user_context, concept_df=None),
        batch_size=args.batch * 2, shuffle=False, num_workers=args.num_workers)

    model = ConceptRiskModel(args.model, alpha_ordinal=args.alpha_ordinal,
                             class_weights=weights, focal_gamma=args.focal_gamma,
                             concept_loss_weight=args.concept_loss_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(tr_loader) // args.grad_accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    best_f1, best_state, step = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        opt.zero_grad()
        for b, batch in enumerate(tr_loader):
            idx = batch.pop("idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch)["loss"]

            if not torch.isfinite(loss):
                rows = tr_df.iloc[idx.tolist()]
                print(f"\nNON-FINITE LOSS at epoch {epoch} batch {b}")
                rows.to_csv(os.path.join(args.outdir, "poison_batch.csv"), index=False)
                sys.exit("aborted; poison_batch.csv written")

            (loss / args.grad_accum).backward()
            running.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gn):
                    sys.exit(f"aborted, non-finite grad_norm epoch {epoch} batch {b}")
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} loss {np.mean(running[-40:]):.4f} "
                          f"grad_norm {gn:.2f} lr {sched.get_last_lr()[0]:.2e}")

        f1, _ = evaluate(model, va_loader, device, use_bf16)
        print(f"== epoch {epoch}: val weighted_f1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    f1, logits = evaluate(model, va_loader, device, use_bf16)
    print(f"\nbest val weighted_f1 = {f1:.4f}")

    va_out = va_df.copy()
    va_out["pred_risk_level"] = [RISK_LEVELS[i] for i in logits.argmax(-1).tolist()]
    probs = torch.softmax(logits, -1)
    for i, l in enumerate(RISK_LEVELS):
        va_out[f"prob_{l}"] = probs[:, i].numpy()
    va_out.to_csv(os.path.join(args.outdir, "val_predictions.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model_final.pt"))
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "best_val_weighted_f1": float(best_f1)}, f, indent=2)


if __name__ == "__main__":
    main()

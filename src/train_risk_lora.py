"""
LoRA fine-tuning of a Qwen decoder LLM for RISK classification.

This is an architecture change rather than a recombination of existing
components: prior risk models are all 400M-900M ENCODERs (deberta-v3-large,
deberta-v2-xlarge, MentalBERT). A 7B-14B decoder LLM has far stronger
pretrained representations than any of those encoders, and fine-tuning it
directly (rather than zero-shot prompting, which does not perform well for
this task) is untested.

Risk carries 0.4 of the composite score, the largest single weight, which
makes it the highest-leverage target for the largest model that can be
trained here.

Design:
  * Sequence-classification head on the decoder (AutoModelForSequenceClassification
    with num_labels=4), which is the standard, stable way to fine-tune a causal
    LM for classification — better calibrated here than generative decoding,
    and it produces softmax probabilities that blend the same way risk-NLI
    does.
  * The prompt carries the OFFICIAL risk definitions, so the model gets the
    same definition-grounding that made the NLI reformulation transfer.
  * LoRA (r=16) on attention projections, bf16, so a 7B fits comfortably and a
    14B fits on an 80GB card.
  * Saves softmax probabilities on val and test in the SAME format the existing
    blend machinery expects, so a successful model plugs straight into the
    proven augment-not-replace blend (artifacts/risk_blend.csv style).

READ THE VAL NUMBER HONESTLY: a small validation gain frequently does not
transfer to test. A genuinely stronger model should show a LARGE jump, the
scale a genuine architecture change produces, not a marginal one. A result
within noise of the existing risk-NLI member is not worth deploying.

Usage:
  python src/train_risk_lora.py \
      --train artifacts/train_split.csv --val artifacts/val_split.csv \
      --model Qwen/Qwen2.5-7B-Instruct \
      --outdir runs/risk_lora7b --epochs 3
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

LABEL2ID = {l: i for i, l in enumerate(RISK_LEVELS)}

# Official task-page definitions — the same grounding that made risk-NLI work.
DEFS = (
    "Risk levels:\n"
    "- Indicator: the post contains no explicit mentions of suicide.\n"
    "- Ideation: explicit suicidal expressions but no suicidal plan.\n"
    "- Behavior: explicit suicidal expressions and self-harm or a suicidal plan.\n"
    "- Attempt: explicit mentions of recent or past suicidal attempts.\n"
)


def build_prompt(post: str) -> str:
    return (f"{DEFS}\nClassify the suicide risk level of this post.\n\n"
            f"Post: {post}\n\nRisk level:")


class RiskDS(Dataset):
    def __init__(self, df, tok, max_len, has_labels=True):
        self.texts = [build_prompt(str(p)) for p in df["post"]]
        self.tok, self.max_len = tok, max_len
        self.labels = ([LABEL2ID[normalize_label(x)] for x in df["risk_level"]]
                       if has_labels else None)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i])
        return item


@torch.no_grad()
def predict_probs(model, df, tok, max_len, device, batch=4):
    model.eval()
    loader = DataLoader(RiskDS(df, tok, max_len, has_labels=False),
                        batch_size=batch, shuffle=False)
    out = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", required=True, help="path to Qwen model dir")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)   # LoRA likes higher LR
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--load-4bit", action="store_true",
                    help="QLoRA: load the base in NF4 so a 70B fits one 80GB "
                         "card (~40GB) instead of needing 140GB in bf16. A "
                         "LoRA'd 8B already matches the whole DeBERTa ensemble "
                         "(0.8006 vs 0.7985), so scale is the untried lever -- "
                         "and a single strong model needs no combination rule.")
    ap.add_argument("--no-val-select", action="store_true",
                    help="save the FINAL epoch's probabilities instead of the "
                         "best-on-val ones. Best-on-val makes val_probs.npy "
                         "optimistic on the very split it is later evaluated "
                         "against. Use this for any 5-fold OOF build, where the "
                         "pooled probs ARE the measurement.")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, TaskType

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("This trains a multi-billion-parameter model — GPU required.")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw = {"num_labels": len(RISK_LEVELS), "torch_dtype": torch.bfloat16}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        # a quantised model must be placed at load time; calling .to(device)
        # on it afterwards raises
        load_kw["device_map"] = {"": 0}
        print("[quant] NF4 4-bit base (QLoRA)")
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, **load_kw)
    base.config.pad_token_id = tok.pad_token_id
    if args.load_4bit:
        # casts layernorms to fp32 and enables input grads -- without this the
        # adapter trains on a frozen quantised graph and the loss barely moves
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True)
        base.config.use_cache = False
    lcfg = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(base, lcfg)
    if not args.load_4bit:
        model = model.to(device)      # 4-bit is already placed by device_map
    model.print_trainable_parameters()

    tr_df = pd.read_csv(args.train)
    va_df = pd.read_csv(args.val)
    loader = DataLoader(RiskDS(tr_df, tok, args.max_len),
                        batch_size=args.batch, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    total = (len(loader) // args.grad_accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    gold = [normalize_label(x) for x in va_df["risk_level"]]
    best_f1, best_probs = -1.0, None
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad()
        run = []
        for b, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            run.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (b + 1) % 100 == 0:
                print(f"epoch {epoch+1} step {b+1} loss {np.mean(run[-100:]):.4f}")

        probs = predict_probs(model, va_df, tok, args.max_len, device)
        pred = [RISK_LEVELS[i] for i in probs.argmax(1)]
        f1 = risk_weighted_f1(gold, pred)
        print(f"== epoch {epoch+1}: val weighted-F1 = {f1:.4f}")
        last_probs, last_f1 = probs, f1
        if args.no_val_select:
            # keep the final epoch, and keep its adapter with it
            best_f1, best_probs = f1, probs
            model.save_pretrained(os.path.join(args.outdir, "lora"))
        elif f1 > best_f1:
            best_f1, best_probs = f1, probs
            model.save_pretrained(os.path.join(args.outdir, "lora"))

    if args.no_val_select:
        best_probs, best_f1 = last_probs, last_f1
        print(f"[no-val-select] saved the FINAL epoch ({best_f1:.4f}), not the "
              f"best -- these probs are usable as honest OOF")

    if best_probs is not None:
        np.save(os.path.join(args.outdir, "val_probs.npy"), best_probs)
        # row_id-keyed copy: val_probs.npy alone forces every consumer to
        # assume the val CSV's row order, which is a silent-corruption risk.
        out = pd.DataFrame({"row_id": va_df["row_id"].astype(str)})
        for j, c in enumerate(RISK_LEVELS):
            out[f"prob_{c}"] = best_probs[:, j]
        out["pred_risk_level"] = [RISK_LEVELS[i] for i in best_probs.argmax(1)]
        out["gold"] = gold
        out.to_csv(os.path.join(args.outdir, "val_predictions.csv"),
                   index=False)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "best_val_weighted_f1": float(best_f1)}, f, indent=2)
    print(f"\nbest val weighted-F1 = {best_f1:.4f}")
    print("compare: risk-NLI (deberta-v3) = 0.8001, risk ensemble = 0.7498")
    print("A genuinely stronger model should show a LARGE jump, not +0.005.")


if __name__ == "__main__":
    main()

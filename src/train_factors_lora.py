"""
LoRA / QLoRA fine-tuning of a decoder LLM for the 24 FACTOR labels.

WHY, AND WHAT IT IS AIMED AT. Three CPU diagnostics closed every
cheap factor lane and left one precise target:

  * the 72B ZERO-SHOT loses on all 24 labels (macro 0.404 vs the deployed 0.642) and the
    hindsight oracle over per-label swaps shows zero benefit. Prompting is dead.
  * re-decoding is nearly exhausted: 18 of 24 labels sit at their decode
    ceiling, and an honest cross-fitted rate release is worth only a small
    composite gain.
  * the protective labels are RANK-limited, not decode-limited. Four of the five
    worst APs in the taxonomy are protective (meaning in life 0.232, sense of
    responsibility 0.556, social support 0.582, psychological capital 0.642).

The error file said why. On the posts ranked highest for `meaning in life`,
8 of 10 carry a DIFFERENT protective label in gold -- "you are what makes life
worth it" is annotated `psychological capital`, "he was my point to keep going"
is `sense of responsibility`. The model finds protective content correctly and
cannot guess which of five overlapping boxes the annotator used. That is a
DISCRIMINATION failure, and discrimination among confusable categories is what
training on examples fixes and what prompting cannot.

Prize if protective reaches the risk factors' level: 0.553 -> 0.665, which
would be worth a meaningful composite gain.

DESIGN, and where it differs from train_risk_lora.py
  * num_labels=24 with problem_type="multi_label_classification", so the head
    is 24 independent sigmoids under BCE -- matching a scorer that macro-
    averages 24 independent per-label F1s.
  * NO definitions in the prompt, unlike the risk trainer. Each output unit
    learns its own label from 1635 examples; a 24-definition preamble would eat
    ~260 of 512 tokens and starve the post. --defs turns it on as a second arm.
  * validation is decoded AT THE CHAMPION'S PER-LABEL RATES, 0 fitted
    parameters, so the printed macro-F1 is directly comparable to the deployed
    bag's 0.6417 rather than to a number from a different operating point.

THE FAINT-GOLD OPTION, measured but OFF by default. Gold repeats a factor once
per annotation, and recall tracks that count hard: 74% on single-mention golds,
88% on two, 94% on three or more -- while 63% of all golds are single-mention.
The scorer counts a faint factor exactly as much as a loud one, so --faint-
weight up-weights single-mention positives in the loss. It is OFF (1.0) for the
pilot because one variable at a time is the rule that survived this campaign;
run it as arm two.

Usage:
  # pilot, cheap, validates the pipeline before anything expensive
  python src/train_factors_lora.py \
      --train artifacts/folds/train_f3.csv --val artifacts/folds/val_f3.csv \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --outdir runs/fac_L8_f3 --epochs 3 --no-val-select

  # 70B
  ... --load-4bit --batch 1 --grad-accum 16
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
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, os.path.dirname(__file__))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
PROTECTIVE = [
    "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
]
PROT_J = [IDX[t] for t in PROTECTIVE]

# one-line definitions, only used with --defs
DEFS = {
    "mental health issues": "a diagnosed or described mental disorder",
    "physical health/characteristic": "illness, pain, weight, appearance",
    "substance use": "drugs, alcohol or tobacco use",
    "hopelessness": "feeling hopeless, trapped or stuck",
    "emotion dysregulation": "uncontrolled anger, anxiety or panic",
    "low self-esteem": "worthlessness, self-hate, feeling a burden",
    "poor school performance": "failing, bad grades, dropping out",
    "low socio-economic status": "unemployment, poverty, homelessness",
    "interpersonal violence": "violence or assault outside the home",
    "prior self-harm or suicidal thought/attempt": "a PAST self-harm act",
    "poor social support": "loneliness, isolation, rejection, no friends",
    "interpersonal difficulty": "trouble making friends or socialising",
    "dysfunctional family": "family conflict, neglect or abuse at home",
    "exposure to others' suicide": "someone ELSE's suicide or attempt",
    "stressful life event": "a specific difficult event: a move, a breakup",
    "traumatic experience": "an overwhelming event they cannot cope with",
    "cognitive deficits": "trouble thinking, concentrating or remembering",
    "suicide means (with access)": "a named or available method of dying",
    "sexual orientation related issues": "sexuality or gender identity",
    "social support": "PRESENT help from family, friends or professionals",
    "coping strategy": "something the author does to cope or get through",
    "psychological capital": "hope, confidence, resilience or optimism",
    "sense of responsibility": "duty to their own survival or to others",
    "meaning in life": "purpose, gratitude, pride, something worth living for",
}


def counts_and_targets(df):
    """Multi-hot targets plus the annotation COUNT per (post, label)."""
    n = len(df)
    C = np.zeros((n, N), dtype=np.float32)
    for i, cell in enumerate(df["factors"].astype(str)):
        for x in parse_factors(cell):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                C[i, j] += 1
    return (C > 0).astype(np.float32), C


def build_prompt(post: str, defs: bool) -> str:
    if defs:
        body = "\n".join(f"- {k}: {v}" for k, v in DEFS.items())
        return (f"Factors:\n{body}\n\nWhich factors are present in this post?"
                f"\n\nPost: {post}\n\nFactors:")
    return f"Which factors are present in this post?\n\nPost: {post}\n\nFactors:"


class FacDS(Dataset):
    def __init__(self, df, tok, max_len, defs, Yw=None):
        self.texts = [build_prompt(str(p), defs) for p in df["post"]]
        self.tok, self.max_len = tok, max_len
        self.Yw = Yw          # (targets, per-cell weights) or None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}
        if self.Yw is not None:
            Y, W = self.Yw
            item["labels"] = torch.from_numpy(Y[i])
            item["cell_w"] = torch.from_numpy(W[i])
        return item


@torch.no_grad()
def predict_probs(model, df, tok, max_len, defs, device, batch=4):
    model.eval()
    loader = DataLoader(FacDS(df, tok, max_len, defs), batch_size=batch,
                        shuffle=False)
    out = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out.append(torch.sigmoid(model(**b).logits.float()).cpu().numpy())
    return np.concatenate(out, axis=0)


def topk_pred(P, rates):
    n = len(P)
    on = np.zeros((n, N), dtype=np.int8)
    for j in range(N):
        k = min(max(int(round(rates[j] * n)), 1), n - 1)
        cut = np.sort(P[:, j])[-k]
        on[:, j] = (P[:, j] >= cut).astype(np.int8)
    return on


def macro_per_label(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv",
                    help="supplies the per-label decode RATES, 0 parameters")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--defs", action="store_true",
                    help="ARM TWO: put the 24 one-line definitions in the "
                         "prompt. Costs ~260 of 512 tokens, so raise --max-len "
                         "with it or the post gets truncated.")
    ap.add_argument("--faint-weight", type=float, default=1.0,
                    help="ARM THREE: loss weight on SINGLE-mention positives. "
                         "Recall is 74%% on those vs 94%% on 3+, and they are "
                         "63%% of all golds, but the scorer counts them the "
                         "same. 1.0 = off, which is the pilot.")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-val-select", action="store_true",
                    help="save the FINAL epoch. Mandatory for any 5-fold OOF "
                         "build, where the pooled probs ARE the measurement.")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, TaskType

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("multi-billion-parameter model -- GPU required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    tr_df = pd.read_csv(args.train, keep_default_na=False)
    va_df = pd.read_csv(args.val, keep_default_na=False)
    Ytr, Ctr = counts_and_targets(tr_df)
    Yva, _ = counts_and_targets(va_df)

    # per-cell loss weights: faint positives up-weighted, everything else 1.0
    Wtr = np.ones_like(Ytr, dtype=np.float32)
    if args.faint_weight != 1.0:
        Wtr[(Ctr == 1)] = args.faint_weight
    faint = float((Ctr == 1).sum()) / max(1.0, float((Ctr > 0).sum()))
    print(f"train {len(tr_df)} posts   {Ytr.sum() / len(tr_df):.2f} factors/post"
          f"   {faint:.0%} of golds are single-mention   "
          f"faint-weight {args.faint_weight}")

    # champion rates -- the decode is fixed, so val macro-F1 is comparable to
    # the deployed bag's 0.6417 rather than to a freely-chosen threshold
    ch = pd.read_csv(args.champion, keep_default_na=False)
    Cch = np.zeros(N)
    for cell in ch["factors"].astype(str):
        for x in set(parse_factors(cell)):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                Cch[j] += 1
    rates = Cch / len(ch)
    print(f"decode rates from {args.champion} ({Cch.sum() / len(ch):.2f} "
          f"factors/post)")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw = {"num_labels": N, "torch_dtype": torch.bfloat16,
               "problem_type": "multi_label_classification"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        load_kw["device_map"] = {"": 0}
        print("[quant] NF4 4-bit base (QLoRA)")
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, **load_kw)
    base.config.pad_token_id = tok.pad_token_id
    if args.load_4bit:
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
        model = model.to(device)
    model.print_trainable_parameters()

    loader = DataLoader(FacDS(tr_df, tok, args.max_len, args.defs, (Ytr, Wtr)),
                        batch_size=args.batch, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    total = (len(loader) // args.grad_accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    best_f1, best_probs, last = -1.0, None, None
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        run = []
        for b, batch in enumerate(loader):
            w = batch.pop("cell_w").to(device)
            y = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = (bce(logits.float(), y.float()) * w).mean()
            (loss / args.grad_accum).backward()
            run.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if (b + 1) % 100 == 0:
                print(f"epoch {epoch+1} step {b+1} "
                      f"loss {np.mean(run[-100:]):.4f}")

        probs = predict_probs(model, va_df, tok, args.max_len, args.defs,
                              device)
        per = macro_per_label(topk_pred(probs, rates), Yva)
        f1 = float(per.mean())
        print(f"== epoch {epoch+1}: val macro-F1 = {f1:.4f}   "
              f"protective {per[PROT_J].mean():.4f}   "
              f"(deployed bag: 0.6417 / 0.5530 on full OOF)")
        last = (probs, f1, per)
        if args.no_val_select or f1 > best_f1:
            best_f1, best_probs, best_per = f1, probs, per
            model.save_pretrained(os.path.join(args.outdir, "lora"))

    if args.no_val_select:
        best_probs, best_f1, best_per = last
        print(f"[no-val-select] saved the FINAL epoch ({best_f1:.4f}) -- these "
              f"probs are usable as honest OOF")

    out = pd.DataFrame({"row_id": va_df["row_id"].astype(str)})
    for j, c in enumerate(FACTOR_TAXONOMY):
        out[f"p_{j}"] = best_probs[:, j]
    out.to_csv(os.path.join(args.outdir, "val_predictions.csv"), index=False)
    np.save(os.path.join(args.outdir, "val_probs.npy"), best_probs)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "val_macro_f1": float(best_f1),
                   "val_protective_f1": float(best_per[PROT_J].mean())},
                  f, indent=2)

    print(f"\n{'label':<44}{'F1':>8}")
    for j in np.argsort(best_per):
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{best_per[j]:>8.3f}")
    print(f"\nval macro-F1 {best_f1:.4f}   protective "
          f"{best_per[PROT_J].mean():.4f}")
    print("\nBAR. This is ONE fold, so it is not comparable to the deployed "
          "bag's full-OOF")
    print("0.6417 directly -- run the same fold's bag number before judging. "
          "What would")
    print("justify the 5-fold spend is a clear PROTECTIVE gain, because that "
          "is the only")
    print("thing this model exists to fix. A model that matches overall but "
          "not on the five")
    print("protective labels has reproduced the bag, not improved it.")


if __name__ == "__main__":
    main()

"""
train_factors_llm_nli.py

Qwen as the factor cross-encoder -- with the definitions, which the earlier
LLM attempt threw away.

THE FLAW IN THE EARLIER EXPERIMENT. v2_rebuild/src/train_factors_llm.py trained
Qwen-14B with 24 plain sigmoid outputs and scored 0.3795 / 0.3809 against the
NLI model's 0.6172 / 0.5431 at the same threshold. The conclusion drawn was
"LLMs are worse at factors". That conclusion does not follow.

The whole reason the incumbent works is that it reads (post, DEFINITION) as a
pair. That is what lets it handle `sexual orientation related issues` from
eight examples -- it is reading the definition, not memorising instances. The
Qwen version had 24 anonymous outputs and no definitions at all, so it had to
learn 24 concepts from scratch from the same starved data with none of the
grounding. The comparison was definition-conditioned encoder vs unconditioned
LLM, not encoder vs LLM.

WHAT THIS DOES. Identical data pipeline to the incumbent -- it imports
make_pairs and hypothesis_for from train_factors_nli, so the positives, the
hard negatives drawn from the CONFUSABLE map, and the hypothesis strings are
byte-identical. The ONLY change is the encoder: DeBERTa-v3-large (435M) becomes
a LoRA-tuned Qwen.

WHY IT IS WORTH GPU WHEN NINE FACTOR IDEAS FAILED. src/factor_diagnosis.py
localised the deficit precisely:

    thresholds   small oracle ceiling; per-label and global tied    -> dead
    more data    AUC flat across prevalence bands                   -> dead
    ranking      average precision is what F1 tracks                -> here

Ranking is a property of the reader. Every other factor experiment changed the
data, the decision rule, or the wording. This changes the reader, with the
definitional grounding intact.

HONEST CAVEAT. The diagnosis also showed the worst labels are bounded by
ANNOTATION NOISE, not model capacity -- `meaning in life` has an oracle F1 of
0.235 no matter what reads it. So do not expect that label to move. Expect
gains, if any, on the mid-tier labels where the text genuinely supports the
annotation and the current model is merely imprecise.

PILOT FIRST. Two folds (0 and 3), not five. Fold 3 is the hardest for the
incumbent and fold 0 the easiest, so the pair brackets the range. If AP does
not improve on those, five folds will not rescue it.

Usage:
  python src/train_factors_llm_nli.py --fold 0 \
      --model Qwen/Qwen2.5-14B-Instruct \
      --outdir runs/factors_qwennli_f0
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scoring import FACTOR_TAXONOMY, parse_factors
from train_factors_nli import make_pairs, hypothesis_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=288)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--neg-ratio", type=int, default=3)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")

    tr = pd.read_csv(args.train)
    va = pd.read_csv(args.val)
    print(f"train {len(tr)}  val {len(va)}   model {args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    class PairDS(Dataset):
        """rows are (post, factor_index, label) from train_factors_nli.make_pairs"""

        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            post, j, y = self.rows[i]
            # truncation="only_first" trims the POST, never the hypothesis --
            # a truncated definition would change the question being asked
            enc = tok(str(post), hypothesis_for(FACTOR_TAXONOMY[j]),
                      truncation="only_first", max_length=args.max_len,
                      padding="max_length", return_tensors="pt")
            return ({k: v[0] for k, v in enc.items()},
                    torch.tensor(float(y)))

    kw = dict(num_labels=1, torch_dtype=torch.bfloat16)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    else:
        kw["device_map"] = None

    model = AutoModelForSequenceClassification.from_pretrained(args.model, **kw)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    if not args.load_4bit:
        model = model.to(device)

    lossf = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device=device))

    # every post x every factor, so val probabilities form a full matrix
    val_rows = [(str(p), j, 0) for p in va["post"] for j in range(24)]
    dl_va = DataLoader(PairDS(val_rows), batch_size=args.batch * 2)

    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Yva = np.zeros((len(va), 24), dtype=int)
    for i, c in enumerate(va["factors"]):
        for f in parse_factors(c):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Yva[i, j] = 1

    os.makedirs(args.outdir, exist_ok=True)
    steps_guess = 0
    best_ap, best_P = -1.0, None
    from sklearn.metrics import average_precision_score

    for ep in range(args.epochs):
        # fresh negative sample each epoch, exactly as the incumbent does
        rows = make_pairs(tr, args.neg_ratio, rng, args.hard_frac)
        dl_tr = DataLoader(PairDS(rows), batch_size=args.batch, shuffle=True)
        if not steps_guess:
            steps_guess = (len(dl_tr) // args.grad_accum + 1) * args.epochs
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr)
            sch = get_linear_schedule_with_warmup(
                opt, int(0.1 * steps_guess), steps_guess)
        n_pos = sum(1 for _, _, y in rows if y == 1)
        print(f"\nepoch {ep}: {len(rows)} pairs ({n_pos} pos)")

        model.train()
        opt.zero_grad()
        for step, (enc, y) in enumerate(dl_tr):
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).logits.squeeze(-1).float()
            loss = lossf(out, y.to(device))
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                opt.step()
                sch.step()
                opt.zero_grad()
            if step % 200 == 0:
                print(f"  step {step}/{len(dl_tr)} loss {loss.item():.4f}")

        model.eval()
        chunks = []
        with torch.no_grad():
            for enc, _ in dl_va:
                enc = {k: v.to(device) for k, v in enc.items()}
                chunks.append(torch.sigmoid(
                    model(**enc).logits.squeeze(-1).float()).cpu().numpy())
        P = np.concatenate(chunks).reshape(len(va), 24)
        ap = float(np.mean([average_precision_score(Yva[:, j], P[:, j])
                            if Yva[:, j].sum() else 0.0 for j in range(24)]))
        print(f"== epoch {ep}: val mean AP = {ap:.4f}")
        if ap > best_ap:
            best_ap, best_P = ap, P
            np.savez(os.path.join(args.outdir, "val_probs.npz"),
                     probs=P, thresholds=np.full(24, 0.5),
                     row_id=np.array(va["row_id"].tolist(), dtype=object))
            model.save_pretrained(os.path.join(args.outdir, "lora"))

    json.dump({**vars(args), "best_val_mean_ap": best_ap},
              open(os.path.join(args.outdir, "run_config.json"), "w"), indent=2)
    print(f"\nbest val mean AP = {best_ap:.4f}")
    print("The incumbent's mean AP on the full 1635 OOF is 0.653.")
    print("AP is the right comparison -- factor_diagnosis showed F1 tracks it,")
    print("and AUC is misleading on labels this imbalanced.")


if __name__ == "__main__":
    main()

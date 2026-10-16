"""
predict_risk_lora_test.py

Score the TEST posts with a 5-fold LoRA risk member and average the folds.

WHY THIS EXISTS. `risk_bag_eval.py` identified a majority vote over
encoder + cot72b + llama as robust against both baselines:

    encoder + cot72b + llama, MAJORITY vote
      vs oof_base      0.7985 -> 0.8259
      vs champion_oof  0.7858 -> 0.8206

`cot72b` already has test predictions; the LoRA member does not, because
scripts/risk_lora_5fold.sh only scored held-out folds. This closes that gap.

FOLD AVERAGING, not fold selection. Each of the five adapters saw a different
4/5 of the data; averaging their probabilities is the standard deployment and
matches how the factor arms build their test dumps. Picking one fold would
throw away 80% of the training signal and add a selection that cannot be
validated.

THE FAILURE MODE THIS CHECKS FIRST. `task_type=SEQ_CLS` should make PEFT save
the classifier head alongside the adapter, because `score.weight` is newly
initialised and TRAINED -- an adapter without it produces a random 4-way head
and confident garbage that looks like a working model. The script verifies the
head is present before scoring anything.

Prompt construction and truncation are imported from train_risk_lora, never
re-implemented: a second copy of the formatting is the drift bug this codebase
has already been bitten by.

Usage:
  python src/predict_risk_lora_test.py --tag loraL8
  python src/predict_risk_lora_test.py --tag loraL8 --test data/test.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="loraL8")
    ap.add_argument("--runs", default="runs/risk_{tag}_f{f}")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--model", default=None,
                    help="base model dir; read from run_config.json if unset")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--load-4bit", action="store_true",
                    help="load the base in NF4, as QLoRA training did. REQUIRED "
                         "for any 70B/72B adapter: bf16 needs ~140GB against a "
                         "93GB card and OOMs at .to(device).")
    ap.add_argument("--per-fold", action="store_true",
                    help="ALSO write each fold's test predictions separately.\n"
                         "WHY THIS MATTERS: the member's OOF is ONE fold model "
                         "per post, but the averaged test file is FIVE. "
                         "Averaging checkpoints narrows the distribution and "
                         "can move the argmax, so an agreement pattern measured "
                         "on OOF is not the same event as what is deployed on "
                         "test. A single-fold test file makes the two identical.")
    args = ap.parse_args()

    out = args.out or f"artifacts/lora_{args.tag}_test.csv"
    dirs = [args.runs.format(tag=args.tag, f=f) for f in range(args.folds)]
    missing = [d for d in dirs if not os.path.isdir(os.path.join(d, "lora"))]
    if missing:
        raise SystemExit(f"no lora/ adapter in: {missing}")

    # the trained classifier head must be in the adapter, or inference produces
    # a randomly-initialised 4-way head and confident nonsense
    for d in dirs:
        files = os.listdir(os.path.join(d, "lora"))
        print(f"{d}/lora: {files}")
    print("\nchecking the classifier head survived the save...")

    cfg = json.load(open(os.path.join(dirs[0], "run_config.json")))
    base = args.model or cfg["model"]
    max_len = args.max_len or int(cfg.get("max_len", 512))
    print(f"base model: {base}   max_len {max_len}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from train_risk_lora import predict_probs

    from data_audit import load
    te = load(args.test)
    te["row_id"] = te["row_id"].astype(str)
    # predict_probs reads a `risk_level` column for nothing but shape in some
    # versions; give it a dummy so a missing gold column cannot crash inference
    if "risk_level" not in te.columns:
        te = te.copy()
        te["risk_level"] = RISK_LEVELS[0]
    print(f"{len(te)} test posts")

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("GPU required")

    # A 70B in bf16 needs ~140GB and the card has 93GB, so scoring a QLoRA-
    # trained 70B OOMs at the .to(device) below unless the base is loaded in
    # NF4 exactly as it was for training. Mirrors train_risk_lora.py --load-4bit.
    load_kw = {"num_labels": len(RISK_LEVELS), "torch_dtype": torch.bfloat16}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        load_kw["device_map"] = {"": 0}
        print("[quant] NF4 4-bit base (matches QLoRA training)")

    acc = None
    for i, d in enumerate(dirs):
        print(f"\n--- fold {i}: {d}")
        m = AutoModelForSequenceClassification.from_pretrained(base, **load_kw)
        m.config.pad_token_id = tok.pad_token_id
        before = m.score.weight.detach().float().clone()
        m = PeftModel.from_pretrained(m, os.path.join(d, "lora"))
        # a quantised model is already placed by device_map; calling .to() on it
        # raises, and for a 70B in bf16 it is what OOMs
        m = (m if args.load_4bit else m.to(device)).eval()
        # PEFT's ModulesToSaveWrapper exposes TWO copies of the head:
        #   ...score.original_module.weight      the fresh, untouched one
        #   ...score.modules_to_save.default...  the TRAINED one, used in
        #                                        forward while the adapter is
        #                                        active
        # Checking the first match reports a false failure every time, because
        # the original is identical to a fresh init by construction.
        cands = [(n, p) for n, p in m.named_parameters()
                 if "score" in n and p.numel() == before.numel()]
        if not cands:
            raise SystemExit(
                "no classifier head found after loading the adapter -- "
                "inference would use a random head")
        trained = [(n, p) for n, p in cands if "modules_to_save" in n]
        check = trained or cands
        moved = [(n, p) for n, p in check
                 if not torch.allclose(p.detach().float().cpu(),
                                       before.cpu(), atol=1e-6)]
        if not moved:
            raise SystemExit(
                f"{d}: every classifier head is IDENTICAL to a fresh "
                f"initialisation, so none was saved with the adapter.\n"
                f"Candidates inspected: {[n for n, _ in cands]}\n"
                f"Those probabilities would be nonsense.")
        print(f"  trained head OK: {moved[0][0]}")

        P = predict_probs(m, te, tok, max_len, device)
        acc = P if acc is None else acc + P
        pred = [RISK_LEVELS[j] for j in P.argmax(1)]
        from collections import Counter
        print(f"  fold mix: {dict(Counter(pred))}")
        if args.per_fold:
            fo = pd.DataFrame({"row_id": list(te["row_id"])})
            for j, c in enumerate(RISK_LEVELS):
                fo[f"prob_{c}"] = P[:, j]
            fo["pred_risk_level"] = pred
            fp = out.replace(".csv", f"_f{i}.csv")
            fo.to_csv(fp, index=False)
            print(f"  wrote {fp}  (regime-matched to this member's OOF)")
        del m
        torch.cuda.empty_cache()

    P = acc / len(dirs)
    o = pd.DataFrame({"row_id": list(te["row_id"])})
    for j, c in enumerate(RISK_LEVELS):
        o[f"prob_{c}"] = P[:, j]
    o["pred_risk_level"] = [RISK_LEVELS[j] for j in P.argmax(1)]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o.to_csv(out, index=False)

    from collections import Counter
    mix = Counter(o["pred_risk_level"])
    print(f"\nwrote {out}")
    print(f"  averaged mix: {dict(mix)}")
    print("  probe 90% intervals on 378 test posts: indicator 112-170, "
          "ideation 99-142, behavior 69-113, attempt 13-41")
    PROBE = {"indicator": (112, 170), "ideation": (99, 142),
             "behavior": (69, 113), "attempt": (13, 41)}
    for c in RISK_LEVELS:
        lo, hi = PROBE[c]
        flag = "" if lo <= mix.get(c, 0) <= hi else "   <- OUTSIDE"
        print(f"    {c:<12}{mix.get(c, 0):>5}{flag}")
    print("\nA marginal outside its probe interval means this member is "
          "mis-calibrated\non test and should not enter the vote unexamined.")


if __name__ == "__main__":
    main()

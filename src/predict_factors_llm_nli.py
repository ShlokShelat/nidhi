"""
predict_factors_llm_nli.py

Test-set probabilities for the LoRA-tuned Qwen factor arm (`qnli`).

WHY THIS EXISTS. `predict_factors_nli.py` instantiates `NLIScorer` and loads a
`model.pt` state dict. The Qwen arm is an `AutoModelForSequenceClassification`
with a PEFT adapter saved by `model.save_pretrained(outdir/lora)`, so that
script cannot load it and `scripts/factor_arm_test_dumps.sh` fails on `qnli`.
Without a test dump the arm cannot ship, however well it scores out-of-fold --
and it PASSES its falsifier (ratematch 0.6740 vs the 5-arm bag's 0.6640).

EVERY DETAIL HERE MIRRORS THE TRAINER, because a scoring path that differs even
slightly from the one that produced the OOF probabilities makes the arm
incomparable with the bag members it will be averaged against:

  - `AutoModelForSequenceClassification(num_labels=1, bfloat16)`
  - `tok.pad_token = tok.eos_token` and `config.pad_token_id` set BEFORE the
    adapter is attached -- Qwen has no pad token and a wrong id silently
    pools over padding
  - `truncation="only_first"` so the POST is trimmed and never the hypothesis;
    a truncated definition changes the question being asked
  - `max_length=288`, `padding="max_length"`
  - hypotheses from `hypothesis_for`, post-major ordering (post, then factor)
  - `sigmoid(logits.squeeze(-1).float())`, reshaped to (n_posts, 24)

The PEFT SEQ_CLS adapter carries the `score` head in `modules_to_save`, so the
classifier is restored with the adapter rather than left randomly initialised.
This script ASSERTS that -- a random head would produce plausible-looking
garbage.

ENVIRONMENT. `FACTOR_DEFS_JSON`, `PROTECTIVE_TEMPLATE`, `FACTOR_ASPECTS_JSON`
and `FACTOR_SUBSET` are cleared, because the arm was trained with the published
definitions and `hypothesis_for` reads those variables at call time.

--verify IS THE POINT. It re-scores a fold's OWN val split and compares against
the `val_probs.npz` that fold saved during training. If the two agree to a few
decimals the inference path is provably identical to the training path. Run it
before trusting the test dump; it costs ~330 posts of compute.

Usage:
  python src/predict_factors_llm_nli.py --verify --folds 0
  python src/predict_factors_llm_nli.py --test data/test.csv --tag qnli
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("FACTOR_DEFS_JSON", "PROTECTIVE_TEMPLATE", "FACTOR_ASPECTS_JSON",
           "FACTOR_SUBSET"):
    _os.environ.pop(_v, None)

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY  # noqa: E402
from train_factors_nli import hypothesis_for  # noqa: E402

N = len(FACTOR_TAXONOMY)


def score_posts(model, tok, posts, max_len, batch, device):
    """(n_posts, 24) sigmoid probabilities, post-major exactly as the trainer."""
    import torch
    pairs = [(str(p), j) for p in posts for j in range(N)]
    hyp = [hypothesis_for(FACTOR_TAXONOMY[j]) for j in range(N)]
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            enc = tok([p for p, _ in chunk], [hyp[j] for _, j in chunk],
                      truncation="only_first", max_length=max_len,
                      padding="max_length", return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            lg = model(**enc).logits.squeeze(-1).float()
            out.append(torch.sigmoid(lg).cpu().numpy())
            if i % (batch * 100) == 0:
                print(f"    {i}/{len(pairs)} pairs", flush=True)
    return np.concatenate(out).reshape(len(posts), N)


def load_fold(base, lora_dir, device, four_bit=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"num_labels": 1, "torch_dtype": torch.bfloat16, "device_map": None}
    if four_bit:
        # A 72B in bf16 is ~140GB against a 93GB card. The trainer ran QLoRA
        # NF4, so 4-bit inference is also the path that REPRODUCES training --
        # --verify is what proves that, and it is not optional here.
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    m = AutoModelForSequenceClassification.from_pretrained(base, **kw)
    m.config.pad_token_id = tok.pad_token_id
    m = PeftModel.from_pretrained(m, lora_dir)

    # The SEQ_CLS classifier must come from the adapter, not be freshly
    # initialised. If it did not, every probability below is noise.
    saved = [n for n, _ in m.named_parameters()
             if "modules_to_save" in n and "score" in n]
    if not saved:
        raise SystemExit(
            f"{lora_dir}: no saved `score` head found in the adapter. The "
            f"classifier would be randomly initialised and every probability "
            f"meaningless. Refusing to score.")
    print(f"    restored classifier from adapter ({len(saved)} tensors)")
    # a 4-bit model is already placed by device_map; .to() would error
    return (m if four_bit else m.to(device)), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--run", default="runs/factors_{tag}_f{f}")
    ap.add_argument("--tag", default="qnli")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--outdir", default="artifacts/factor_runs")
    ap.add_argument("--max-len", type=int, default=288)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--load-4bit", action="store_true",
                    help="QLoRA NF4. REQUIRED for the 72B -- bf16 is ~140GB "
                         "against a 93GB card, and the trainer was 4-bit so "
                         "this is also the path that reproduces it.")
    ap.add_argument("--verify", action="store_true",
                    help="re-score each fold's OWN val split and compare with "
                         "the val_probs.npz it saved during training")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")
    folds = [int(x) for x in args.folds.split(",")]

    if args.verify:
        worst = 0.0
        for f in folds:
            rd = args.run.format(tag=args.tag, f=f)
            va = pd.read_csv(f"artifacts/folds/val_f{f}.csv")
            ref = np.load(os.path.join(rd, "val_probs.npz"), allow_pickle=True)
            print(f"\nfold {f}: re-scoring {len(va)} own-val posts")
            m, tok = load_fold(args.model, os.path.join(rd, "lora"), device,
                               args.load_4bit)
            P = score_posts(m, tok, va["post"].tolist(), args.max_len,
                            args.batch, device)
            d = float(np.abs(P - ref["probs"]).max())
            worst = max(worst, d)
            print(f"  max |new - saved| = {d:.5f}  "
                  f"corr {np.corrcoef(P.ravel(), ref['probs'].ravel())[0,1]:.6f}")
            del m
            torch.cuda.empty_cache()
        print(f"\nworst deviation across folds: {worst:.5f}")
        print("Under ~0.01 means the inference path reproduces training and the")
        print("test dump can be trusted. Large deviations mean a tokenizer, "
              "pad-token or hypothesis mismatch -- do NOT ship the dump.")
        return

    df = pd.read_csv(args.test)
    if "post" not in df.columns or "row_id" not in df.columns:
        raise SystemExit(f"{args.test} needs row_id and post columns")
    acc = []
    for f in folds:
        rd = args.run.format(tag=args.tag, f=f)
        lora = os.path.join(rd, "lora")
        if not os.path.isdir(lora):
            raise SystemExit(f"{lora} missing -- a partial bag is a DIFFERENT "
                             f"model from the one measured")
        print(f"\nfold {f}: scoring {len(df)} test posts")
        m, tok = load_fold(args.model, lora, device, args.load_4bit)
        acc.append(score_posts(m, tok, df["post"].tolist(), args.max_len,
                               args.batch, device))
        del m
        torch.cuda.empty_cache()

    P = np.mean(acc, axis=0)
    os.makedirs(args.outdir, exist_ok=True)
    dst = os.path.join(args.outdir, f"{args.tag}_cv_test.npz")
    np.savez(dst, probs=P, thresholds=np.full(N, 0.5),
             row_id=np.array(df["row_id"].astype(str).tolist(), dtype=object))
    print(f"\nwrote {dst}  {P.shape}  sd={P.std():.4f}")

    o = f"{args.outdir}/{args.tag}_cv_oof_f0.npz"
    if os.path.exists(o):
        so = float(np.load(o, allow_pickle=True)["probs"].std())
        print(f"oof sd {so:.4f}   ratio {P.std()/so:.3f}  "
              f"(a quantile cutoff ignores this)")
    print("\nNext:")
    print("  python src/factor_bag_ship.py --arms mnli dmnli cmine prot d4 "
          "qnli --rule logit \\")
    print("      --match-counts submissions/NIDHI_canon_on_nlimnli.csv \\")
    print("      --out artifacts/bag6q_matched_factors.csv")


if __name__ == "__main__":
    main()

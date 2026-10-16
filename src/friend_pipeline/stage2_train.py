# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm can
# be rebuilt from source without the original project tree.
"""Stage 2 - fine-tune the tri-head model, one model per user-grouped fold.

Catches handled here:
  * class weights on the risk loss (Attempt is rare)
  * fp16 mixed precision + gradient accumulation sized to your VRAM
  * separate learning rates: small for the pretrained encoder, larger for the
    freshly initialized heads
  * out-of-fold (OOF) predictions saved for honest threshold tuning later
  * model download falls back to roberta-base if mental-roberta is unavailable
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as C
from model import PostDataset, TriHeadModel, multitask_loss


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_encoder():
    from transformers import AutoModel, AutoTokenizer
    for name in (C.MODEL_NAME, C.FALLBACK_MODEL):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            enc = AutoModel.from_pretrained(name)
            print(f"[stage2] encoder = {name}")
            return tok, enc
        except Exception as e:
            print(f"[stage2] could not load {name}: {e}")
    raise RuntimeError("No encoder could be downloaded. Check internet/HF access.")


def train_fold(fold, df, hw):
    from transformers import get_linear_schedule_with_warmup
    set_seed(C.SEED + fold)
    tok, enc = load_encoder()
    hidden = enc.config.hidden_size
    model = TriHeadModel(enc, hidden).to(hw["device"])
    if hw["grad_ckpt"]:
        model.encoder.gradient_checkpointing_enable()

    tr_df = df[df.fold != fold]
    va_df = df[df.fold == fold]
    dl_tr = DataLoader(PostDataset(tr_df, tok), batch_size=hw["batch"],
                       shuffle=True, drop_last=True)

    # class weights: inverse frequency, normalized
    counts = tr_df["risk_level"].value_counts()
    w = torch.tensor([len(tr_df) / (len(C.RISK_LEVELS) * counts[l])
                      for l in C.RISK_LEVELS], dtype=torch.float).to(hw["device"])
    print(f"[fold {fold}] class weights: "
          + ", ".join(f"{l}={x:.2f}" for l, x in zip(C.RISK_LEVELS, w)))

    params = [
        {"params": model.encoder.parameters(), "lr": C.LR_ENCODER},
        {"params": list(model.risk_head.parameters())
                   + list(model.evid_head.parameters())
                   + list(model.fact_head.parameters()), "lr": C.LR_HEADS},
    ]
    opt = torch.optim.AdamW(params, weight_decay=C.WEIGHT_DECAY)
    steps = (len(dl_tr) // hw["accum"]) * C.EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(C.WARMUP_FRAC * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=hw["fp16"])
    import json as _json
    Y = np.zeros((len(tr_df), len(C.FACTORS)))
    for r, s in enumerate(tr_df["factors_clean"]):
        for f in _json.loads(s):
            Y[r, C.FACTORS.index(f)] = 1
    pos = Y.sum(0)
    fact_pw = torch.tensor(((len(tr_df) - pos) / np.maximum(pos, 1)).clip(1, 20),
                           dtype=torch.float).to(hw["device"])
    model.train()
    for ep in range(C.EPOCHS):
        tot = 0.0
        for step, batch in enumerate(dl_tr):
            batch = {k: v.to(hw["device"]) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=hw["fp16"]):
                out = model(batch["input_ids"], batch["attention_mask"])
                loss, parts = multitask_loss(out, batch, w, fact_pw)
                loss = loss / hw["accum"]
            scaler.scale(loss).backward()
            if (step + 1) % hw["accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(); sch.step()
            tot += loss.item() * hw["accum"]
        print(f"[fold {fold}] epoch {ep + 1}/{C.EPOCHS} mean loss {tot / len(dl_tr):.4f}")

    # ---- out-of-fold predictions (for threshold tuning + honest CV) ----
    model.eval()
    oof = predict(model, tok, va_df, hw)
    ck = os.path.join(C.OUT_DIR, f"model_fold{fold}.pt")
    torch.save({"state": model.state_dict(),
                "tok": tok.name_or_path, "hidden": hidden}, ck)
    print(f"[fold {fold}] saved {ck}")
    return oof


@torch.no_grad()
def predict(model, tok, df, hw, bs=None):
    """Returns raw arrays: risk probs, factor probs, token probs + offsets."""
    model.eval()
    ds = PostDataset(df, tok, is_train=False)
    bs = bs or max(8, hw["batch"])
    risk_p, fact_p, tok_p, offs = [], [], [], []
    for i in range(0, len(ds), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        ids = torch.stack([x["input_ids"] for x in items]).to(hw["device"])
        am = torch.stack([x["attention_mask"] for x in items]).to(hw["device"])
        with torch.cuda.amp.autocast(enabled=hw["fp16"]):
            out = model(ids, am)
        risk_p.append(torch.softmax(out["risk_logits"].float(), -1).cpu().numpy())
        fact_p.append(torch.sigmoid(out["fact_logits"].float()).cpu().numpy())
        tp = torch.sigmoid(out["evid_logits"].float()).cpu().numpy()
        for r, x in enumerate(items):
            tok_p.append(tp[r])
            offs.append(x["offsets"])
    return {"row_id": df["row_id"].tolist(),
            "risk": np.concatenate(risk_p),
            "fact": np.concatenate(fact_p),
            "tokp": tok_p, "offsets": offs,
            "post": df["post"].tolist()}


def run():
    hw = C.hardware_profile()
    print(f"[stage2] hardware: {hw['note']} | batch={hw['batch']} "
          f"accum={hw['accum']} fp16={hw['fp16']} grad_ckpt={hw['grad_ckpt']}")
    df = pd.read_csv(os.path.join(C.OUT_DIR, "train_clean.csv"))
    all_oof = {}
    for fold in range(C.N_FOLDS):
        oof = train_fold(fold, df, hw)
        all_oof[fold] = oof
        torch.cuda.empty_cache()
    np.save(os.path.join(C.OUT_DIR, "oof.npy"), all_oof, allow_pickle=True)
    print("[stage2] done; OOF predictions saved")


if __name__ == "__main__":
    run()

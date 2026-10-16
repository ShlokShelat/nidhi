"""
Multi-label factor classifier -- the FORMULATION arm.

WHAT IT IS. One forward pass per post emitting all 24 factor logits from a
shared [CLS] representation (a multi-label head). The other factor arms use a
different formulation: 24 separate (post, hypothesis) entailment questions per
post, scored by an NLI cross-encoder.

WHY IT EXISTS. The factor column is an ensemble over arms, and an ensemble
gains from a member whose errors are decorrelated from the rest. A multi-label
head makes all 24 decisions jointly from one encoding of the post, so its
errors are structurally different from pairwise entailment scoring even on the
identical backbone -- which is why this trainer defaults to the pool's own
backbone (microsoft/deberta-v3-large): it isolates the formulation as the only
changed axis.

DECORRELATION, NOT LABEL DEPENDENCIES, IS THE JUSTIFICATION. Learned label
co-occurrence cannot be the mechanism: an oracle given the other 23 gold labels
scores far below what the pool already reaches, and most of what that oracle
does capture is post cardinality rather than which factors co-occur. A side
benefit: one forward per post instead of 24 pairs is ~24x cheaper per epoch, so
more epochs/seeds fit in the same wall clock.

DROP-IN COMPATIBILITY. Thresholds are tuned by the NLI trainer's OWN
`tune_thresholds` (imported, not copied), gold comes from its `gold_matrix`,
and dumps are written by src/predict_factors_multilabel.py in the exact
probs/row_id/thresholds .npz format that `factor_structure.load_oof` reads.

LOSS. Plain BCE with per-label pos_weight clip((n-pos)/pos, 1, 20) is the
DEFAULT: it is the standard objective for a multi-label head, so the
formulation stays the only variable under test. --loss cbntr ports the NLI
trainer's CB-NTR to the multi-label head: identical math, just broadcast over
the (B, 24) logit matrix instead of indexed by a per-pair factor id.

NOTE ON ENV VARS. FACTOR_DEFS_JSON / FACTOR_ASPECTS_JSON /
PROTECTIVE_TEMPLATE / CONFUSABLE_JSON have NO effect here -- there are no
hypotheses. They are still recorded in run_config.json for provenance, and
FACTOR_SUBSET is refused outright rather than silently ignored.

Usage (one fold):
  python src/train_factors_multilabel.py \
      --train artifacts/folds/train_f0.csv --val artifacts/folds/val_f0.csv \
      --model microsoft/deberta-v3-large --outdir runs/factors_mlc_s21_cv_f0 \
      --seed 21
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from scoring import FACTOR_TAXONOMY, RISK_LEVELS, factors_macro_f1    # noqa: E402

# Defined locally rather than imported from train_factors_nli (44KB, pulls in
# the whole hypothesis machinery). Must stay identical to that module's set.
PROTECTIVE_FACTORS = {
    "social support",
    "coping strategy",
    "psychological capital",
    "sense of responsibility",
    "meaning in life",
}
from train_factors_nli import N, gold_matrix, to_lists, tune_thresholds  # noqa: E402

RISK_IDX = {str(r).strip().lower(): i for i, r in enumerate(RISK_LEVELS)}


class PostFactorDataset(Dataset):
    """One item per POST (not per pair): text -> 24-dim binary label vector."""

    def __init__(self, df, tok, max_len):
        self.texts = df["post"].astype(str).tolist()
        self.Y = gold_matrix(df).astype(np.float32)
        self.tok, self.max_len = tok, max_len
        # Gold risk for the multi-task auxiliary. -1 marks "no usable label",
        # which cross_entropy(ignore_index=-1) drops, so a df without the
        # column simply trains single-task instead of failing at step 1.
        if "risk_level" in df.columns:
            self.R = np.array(
                [RISK_IDX.get(str(r).strip().lower(), -1)
                 for r in df["risk_level"]], dtype=np.int64)
        else:
            self.R = np.full(len(self.texts), -1, dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.from_numpy(self.Y[i]),
                "risk": torch.tensor(self.R[i])}


class MultiLabelFactorArm(nn.Module):
    """Encoder + Linear(hidden, 24) on [CLS], with the same dropout as
    NLIScorer so the only variable that differs from the pool's arms is the
    formulation.

    loss_type:
      "bce"    per-label pos_weight BCE (default)
      "cbntr"  the NLI trainer's CB-NTR, broadcast over the label axis.
               nu_vec/cb_vec are computed by main() from POST-level prevalence,
               exactly as train_factors_nli.main does -- see its _cbntr
               docstring for the sign trap and the lambda mechanics.
    """

    def __init__(self, model_name, pos_weight_vec=None, loss_type="bce",
                 nu_vec=None, cb_vec=None, ntr_lambda=2.0, db_gamma=2.0,
                 lambda_risk=0.0, split_heads=False):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        # SPLIT HEADS: shared trunk, one Linear for the 19 risk factors and a
        # separate Linear for the 5 protective ones, logits reassembled in
        # taxonomy order so nothing downstream changes.
        #
        # WHY. The 5 protective labels are predicted markedly worse than the 19
        # risk factors, and macro-F1 averages all 24 equally, so a point of F1
        # on a protective label is worth exactly as much as one on a risk
        # factor while being much easier to find. Training a specialist on the
        # 5 protective labels alone removes the supervision the other 19 supply
        # to the shared representation, so the trunk stays shared and only the
        # head is split -- which is also the architecture the task description
        # sets out.
        #
        # NOTE a pure LINEAR split would be a NO-OP: Linear(h,19) concatenated
        # with Linear(h,5) IS Linear(h,24), same parameters and same
        # gradients. Each head therefore gets its own NONLINEAR projection, so
        # the protective group can learn a different representation of the
        # trunk rather than a different slice of the same one. That is what
        # "separate MLPs" in the organizers' architecture actually requires.
        self.split_heads = split_heads
        if split_heads:
            self.prot_idx = [i for i, t in enumerate(FACTOR_TAXONOMY)
                             if t in PROTECTIVE_FACTORS]
            self.risk_idx = [i for i in range(N) if i not in
                             set(self.prot_idx)]
            if len(self.prot_idx) != 5:
                raise SystemExit(
                    f"expected 5 protective labels, matched "
                    f"{len(self.prot_idx)} -- PROTECTIVE_FACTORS has drifted "
                    f"from the taxonomy")
            mid = h // 2
            self.head_risk = nn.Sequential(
                nn.Linear(h, mid), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(mid, len(self.risk_idx)))
            self.head_prot = nn.Sequential(
                nn.Linear(h, mid), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(mid, len(self.prot_idx)))
            self.head = None
        else:
            self.head = nn.Linear(h, N)
        # MULTI-TASK AUXILIARY. Risk level and factors are two views of the
        # same clinical judgement -- the factors are the reasons an annotator
        # assigned the level -- so predicting the level from the shared trunk
        # supervises the encoder a second time on a signal the factor labels
        # already depend on, which regularises the representation.
        # TRAINING-ONLY: risk_head.* is stripped before model.pt is written so
        # the saved artefact stays shape-identical to a single-task checkpoint
        # and predict_factors_multilabel.py needs no change.
        self.lambda_risk = lambda_risk
        self.risk_head = nn.Linear(h, len(RISK_LEVELS)) if lambda_risk > 0 \
            else None
        self.loss_type = loss_type
        self.ntr_lambda, self.db_gamma = ntr_lambda, db_gamma
        self.register_buffer("pos_weight_vec",
            torch.tensor(pos_weight_vec, dtype=torch.float)
            if pos_weight_vec is not None else None)
        self.register_buffer("nu_vec",
            torch.tensor(nu_vec, dtype=torch.float)
            if nu_vec is not None else None)
        self.register_buffer("cb_vec",
            torch.tensor(cb_vec, dtype=torch.float)
            if cb_vec is not None else None)

    def _cbntr(self, logits, target):
        """Identical math to NLIScorer._cbntr, with the (B,) per-pair factor
        index replaced by broadcasting nu/cb over the (B, 24) logit matrix.
        The bias lives in the LOSS only; inference emits the unshifted logit."""
        nu = self.nu_vec.to(logits.device).to(logits.dtype)      # (24,)
        w = self.cb_vec.to(logits.device).to(logits.dtype)       # (24,)
        lam = self.ntr_lambda
        z = logits - nu
        p = torch.sigmoid(z)
        pos = nn.functional.softplus(-z)
        neg = nn.functional.softplus(lam * z) / lam
        if self.db_gamma > 0:
            # the negative branch's focal q is sigmoid(lam*z), NOT sigmoid(z)
            # -- see the long comment in NLIScorer._cbntr for why using the
            # unscaled p would partially undo the negative tolerance.
            pos = pos * (1 - p).clamp(min=1e-6) ** self.db_gamma
            neg = neg * torch.sigmoid(lam * z).clamp(min=1e-6) ** self.db_gamma
        return (w * (target * pos + (1 - target) * neg)).mean()

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                risk=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop(out.last_hidden_state[:, 0])
        if self.split_heads:
            x = x.to(self.head_risk[0].weight.dtype)
            lr_ = self.head_risk(x)
            lp_ = self.head_prot(x)
            # reassemble in taxonomy order so every downstream consumer --
            # the loss, the threshold tuner, predict_factors_multilabel.py --
            # sees the same (batch, 24) tensor as a single-head model
            logits = torch.empty(x.shape[0], N, dtype=lr_.dtype,
                                 device=lr_.device)
            logits[:, self.risk_idx] = lr_
            logits[:, self.prot_idx] = lp_
        else:
            x = x.to(self.head.weight.dtype)
            logits = self.head(x)
        loss = None
        if labels is not None:
            y = labels.to(logits.dtype)
            if self.loss_type == "cbntr":
                if self.nu_vec is None or self.cb_vec is None:
                    raise ValueError("cbntr needs nu_vec and cb_vec")
                loss = self._cbntr(logits, y)
            elif self.loss_type == "bce":
                pw = (self.pos_weight_vec.to(logits.dtype)
                      if self.pos_weight_vec is not None else None)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pw)
            else:
                raise ValueError(f"unknown loss_type {self.loss_type}")

            # ignore_index=-1 drops rows with no usable gold risk, so the
            # auxiliary never invents a target. The factor logits returned
            # below are untouched -- inference is identical to single-task.
            if self.risk_head is not None and risk is not None:
                rl = self.risk_head(x)
                rloss = nn.functional.cross_entropy(
                    rl, risk.to(rl.device), ignore_index=-1)
                if torch.isfinite(rloss):
                    loss = loss + self.lambda_risk * rloss
        return {"loss": loss, "logits": logits}


@torch.no_grad()
def score_posts(model, df, tok, max_len, device, use_bf16, batch=64):
    """(n_posts, 24) probability matrix. One forward per post -- the whole
    point of the formulation. Shared by training eval and the predictor."""
    model.eval()
    texts = df["post"].astype(str).tolist()
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        # The training Dataset yields ONLY input_ids/attention_mask, so the
        # encoder never sees token_type_ids. DeBERTa's tokenizer emits them
        # anyway; forwarding them here would make eval compute something
        # training never did (and forward() rejects the kwarg outright).
        enc = {k: v for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=use_bf16):
            logits = model(**enc)["logits"]
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    # 320 matches the POOL'S DUMP LENGTH, and training at the serving length
    # means this arm has no train/dump length skew at all (the NLI arms train
    # at 384 and dump at 320; here train == dump == 320).
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=8,
                    help="one forward per post makes epochs ~24x cheaper than "
                         "the NLI trainer's, so more of them are affordable; "
                         "the best epoch is picked on tuned val macro-F1")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--split-heads", action="store_true",
                    help="shared trunk, separate Linear heads for the 19 risk "
                         "factors and the 5 protective ones. Logits are "
                         "reassembled in taxonomy order so the checkpoint "
                         "stays shape-compatible downstream. Default off "
                         "leaves the single-head model unchanged.")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--loss", choices=["bce", "cbntr"], default="bce",
                    help="'bce' (per-label pos_weight, clip 1..20) is the "
                         "default, so the FORMULATION stays the only variable "
                         "under test. 'cbntr' ports the NLI trainer's CB-NTR "
                         "to the multi-label head.")
    ap.add_argument("--max-pos-weight", type=float, default=20.0)
    ap.add_argument("--ntr-kappa", type=float, default=0.05)
    ap.add_argument("--ntr-lambda", type=float, default=2.0)
    ap.add_argument("--cb-beta", type=float, default=0.99)
    ap.add_argument("--db-gamma", type=float, default=2.0)
    ap.add_argument("--lambda-risk", type=float, default=0.0,
                    help="weight on a MULTI-TASK risk-level auxiliary head. "
                         "0 (default) = single-task, unchanged behaviour. Risk "
                         "level and factors are two views of one judgement, so "
                         "predicting the level from the shared trunk gives the "
                         "encoder a second supervision signal and regularises "
                         "it. The head is stripped before saving, so model.pt "
                         "stays single-task shaped.")
    args = ap.parse_args()

    if os.environ.get("FACTOR_SUBSET", "").strip():
        raise SystemExit("FACTOR_SUBSET is set but this trainer always emits "
                         "all 24 labels -- unset it (a silent no-op here would "
                         "poison any comparison against a subset NLI arm).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")
    if device == "cpu":
        print("WARNING: on CPU -- this trains a full model, use a GPU node.")

    os.makedirs(args.outdir, exist_ok=True)
    # run_config is written UP FRONT (and finalised after training) so the
    # sbatch script's TAG-collision guard and the predictor's marker check
    # never meet a dir that has model.pt but no marker -- a killed job must
    # stay resumable, not look like another experiment's directory.
    cfg = {"multilabel_arm": True, **vars(args),
           "best_val_macro_f1_tuned": None,
           "FACTOR_DEFS_JSON": os.environ.get("FACTOR_DEFS_JSON", ""),
           "FACTOR_ASPECTS_JSON": os.environ.get("FACTOR_ASPECTS_JSON", ""),
           "PROTECTIVE_TEMPLATE": os.environ.get("PROTECTIVE_TEMPLATE", ""),
           "CONFUSABLE_JSON": os.environ.get("CONFUSABLE_JSON", "")}
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr_df = pd.read_csv(args.train)
    va_df = pd.read_csv(args.val)
    tok = AutoTokenizer.from_pretrained(args.model)
    tr_ds = PostFactorDataset(tr_df, tok, args.max_len)
    va_gold = gold_matrix(va_df)

    Ymat = tr_ds.Y
    n_i = Ymat.sum(0).astype(float)
    Npost = len(tr_df)

    pw_vec = nu_vec = cb_vec = None
    if args.loss == "bce":
        # neg:pos per label from the POST-level matrix (there is no sampling
        # ratio here -- every post carries all 24 labels every epoch).
        pw_vec = np.clip((Npost - n_i) / np.clip(n_i, 1, None),
                         1.0, args.max_pos_weight)
        print(f"per-label pos_weight: min {pw_vec.min():.1f} "
              f"max {pw_vec.max():.1f} median {np.median(pw_vec):.1f}")
    else:
        # same construction as train_factors_nli.main: POST-level prevalence,
        # nu sign such that the RAREST label gets the LARGEST POSITIVE nu.
        p_i = np.clip(n_i / max(Npost, 1), 1e-6, 1 - 1e-6)
        nu_vec = -args.ntr_kappa * np.log(p_i / (1 - p_i))
        cb = (1 - args.cb_beta) / np.clip(
            1 - np.power(args.cb_beta, np.clip(n_i, 1, None)), 1e-8, None)
        cb_vec = cb / cb.mean()
        print(f"cbntr: kappa={args.ntr_kappa} lambda={args.ntr_lambda} "
              f"beta={args.cb_beta} gamma={args.db_gamma}")
        print(f"  nu   min {nu_vec.min():+.3f} max {nu_vec.max():+.3f}   "
              f"(RAREST label must have the LARGEST POSITIVE nu)")
        spread = float(cb_vec.max() / max(cb_vec.min(), 1e-9))
        print(f"  cb   min {cb_vec.min():.2f} max {cb_vec.max():.2f}   "
              f"spread {spread:.1f}x")
        if spread < 2.0:
            raise SystemExit(
                f"--cb-beta {args.cb_beta} gives a cb spread of only "
                f"{spread:.2f}x -- the class-balanced half of CB-NTR is a "
                f"no-op at this beta. Raise --cb-beta (0.99 caps the "
                f"effective number at 100, inside our support range).")

    model = MultiLabelFactorArm(
        args.model, pos_weight_vec=pw_vec, loss_type=args.loss,
        nu_vec=nu_vec, cb_vec=cb_vec, ntr_lambda=args.ntr_lambda,
        db_gamma=args.db_gamma, lambda_risk=args.lambda_risk,
        split_heads=args.split_heads).to(device)
    if args.split_heads:
        print(f"  SPLIT HEADS: {len(model.risk_idx)} risk-factor labels + "
              f"{len(model.prot_idx)} protective, shared trunk")
    print(f"objective: {args.loss}   {len(tr_df)} train posts, "
          f"{len(va_df)} val posts, max_len {args.max_len}")
    if args.lambda_risk > 0:
        nr = int((tr_ds.R >= 0).sum())
        if nr == 0:
            raise SystemExit(
                "--lambda-risk > 0 but no training post has a usable "
                "risk_level; the auxiliary would be a silent no-op.")
        print(f"multi-task: risk auxiliary at lambda {args.lambda_risk}, "
              f"{nr}/{len(tr_ds.R)} posts labelled "
              f"(head is stripped before save)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers)
    steps_per_epoch = len(loader) // args.grad_accum + 1
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    best_macro, best_state, best_ths = -1.0, None, None
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        run = []
        for b, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise SystemExit(f"NON-FINITE LOSS at epoch {epoch+1} batch {b}")
            (loss / args.grad_accum).backward()
            run.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (b + 1) % 40 == 0:
                print(f"epoch {epoch+1} step {b+1} loss {np.mean(run[-40:]):.4f}")
        vp = score_posts(model, va_df, tok, args.max_len, device, use_bf16)
        ths = tune_thresholds(vp, va_gold)          # the NLI trainer's tuner
        on = (vp >= ths).astype(int)
        macro = factors_macro_f1(to_lists(va_gold), to_lists(on))
        print(f"== epoch {epoch+1}: val macro-F1 (tuned) = {macro:.4f}")
        if macro > best_macro:
            best_macro = macro
            best_ths = ths
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is None:
        raise SystemExit("no epoch improved on -1.0 -- something is broken")
    # STRIP the multi-task auxiliary. risk_head is a training-time regulariser,
    # not part of the factor decision, and predict_factors_multilabel.py builds
    # a single-task model -- leaving the keys in would break it on load. The
    # in-memory best_state keeps them so the reload check below is exact.
    saved = {k: v for k, v in best_state.items()
             if not k.startswith("risk_head")}
    if len(saved) != len(best_state):
        print(f"stripped {len(best_state) - len(saved)} risk_head.* keys; "
              f"model.pt is shape-identical to a single-task checkpoint")
    torch.save(saved, os.path.join(args.outdir, "model.pt"))
    np.save(os.path.join(args.outdir, "thresholds.npy"), best_ths)
    with open(os.path.join(args.outdir, "tok"), "w") as f:
        f.write(args.model)

    # per-label F1 of the saved model, worst first (loaded back so the print
    # matches what ships, not the last epoch)
    model.load_state_dict(best_state)
    model.to(device)
    vp = score_posts(model, va_df, tok, args.max_len, device, use_bf16)
    on = (vp >= best_ths).astype(int)
    macro, per_label = factors_macro_f1(to_lists(va_gold), to_lists(on),
                                        return_per_label=True)
    print(f"\nbest val macro-F1 (tuned) = {best_macro:.4f} "
          f"(reload check {macro:.4f})")
    for fct, v in sorted(per_label.items(), key=lambda kv: kv[1])[:8]:
        print(f"  {v:.3f}  {fct}")

    cfg["best_val_macro_f1_tuned"] = float(best_macro)
    cfg["n_train_posts"] = int(Npost)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {args.outdir}: model.pt thresholds.npy tok run_config.json")


if __name__ == "__main__":
    main()

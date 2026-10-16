# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm can
# be rebuilt from source without the original project tree.
"""Multi-task model: one encoder, three heads.

  head 1: risk level        (4-way softmax on [CLS])
  head 2: evidence tagging  (per-token sigmoid: is this token part of evidence?)
  head 3: factors           (24 sigmoids on [CLS])

Head 2 is the 'give weight to the answer words' idea: forcing the encoder to
locate evidence tokens acts as a spotlight that also improves head 1, and its
output IS the Subtask-1b deliverable.
"""
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import config as C


class PostDataset(Dataset):
    """Tokenizes posts; builds char->token aligned evidence labels."""

    def __init__(self, df, tokenizer, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _char_mask(self, post, spans):
        mask = [0] * len(post)
        low = post.lower()
        for sp in spans:
            sp = sp.lower().strip()
            if not sp:
                continue
            start = 0
            while True:  # mark ALL occurrences of the span text
                i = low.find(sp, start)
                if i < 0:
                    break
                for j in range(i, min(i + len(sp), len(post))):
                    mask[j] = 1
                start = i + 1
        return mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        post = str(row["post"])
        enc = self.tok(post, truncation=True, max_length=C.MAX_LEN,
                       return_offsets_mapping=True, padding="max_length")
        item = {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
        }
        if self.is_train:
            # risk
            item["risk"] = torch.tensor(C.RISK_LEVELS.index(row["risk_level"]))
            # evidence token labels
            spans = json.loads(row["evidence_clean"])
            cmask = self._char_mask(post, spans)
            tok_lab = []
            for (a, b), am in zip(enc["offset_mapping"], enc["attention_mask"]):
                if am == 0 or a == b:
                    tok_lab.append(-100)          # padding / special tokens
                else:
                    tok_lab.append(1 if any(cmask[a:b]) else 0)
            item["evid"] = torch.tensor(tok_lab, dtype=torch.float)
            # factors
            fvec = torch.zeros(len(C.FACTORS))
            for f in json.loads(row["factors_clean"]):
                fvec[C.FACTORS.index(f)] = 1.0
            item["fact"] = fvec
        else:
            item["offsets"] = enc["offset_mapping"]
        return item


class TriHeadModel(nn.Module):
    def __init__(self, encoder, hidden):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.1)
        self.risk_head = nn.Linear(hidden, len(C.RISK_LEVELS))
        self.evid_head = nn.Linear(hidden, 1)
        self.fact_head = nn.Linear(hidden, len(C.FACTORS))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.drop(out.last_hidden_state)         # (B, T, H)
        cls = seq[:, 0]                                # (B, H)
        return {
            "risk_logits": self.risk_head(cls),
            "evid_logits": self.evid_head(seq).squeeze(-1),
            "fact_logits": self.fact_head(cls),
        }


def multitask_loss(out, batch, class_weights, fact_pos_weight):
    ce = nn.CrossEntropyLoss(weight=class_weights)
    l_risk = ce(out["risk_logits"], batch["risk"])

    # token-level BCE, ignoring -100 positions
    lab = batch["evid"]
    valid = lab != -100
    if valid.any():
        bce = nn.BCEWithLogitsLoss()
        l_evid = bce(out["evid_logits"][valid], lab[valid])
    else:
        l_evid = torch.tensor(0.0, device=lab.device)

    l_fact = nn.BCEWithLogitsLoss(pos_weight=fact_pos_weight)(out["fact_logits"], batch["fact"])
    total = C.W_RISK * l_risk + C.W_EVID * l_evid + C.W_FACT * l_fact
    return total, {"risk": l_risk.item(), "evid": float(l_evid), "fact": l_fact.item()}

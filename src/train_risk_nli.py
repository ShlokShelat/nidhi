"""
Definition-grounded entailment for RISK classification.

The same reformulation that took factor macro-F1 from 0.417 to 0.604 and
transferred to held-out evaluation with negligible shrinkage, now applied to
the highest-weighted component of the composite (risk = 0.4).

Why this should work here specifically: the risk confusion matrix shows
46% of all errors are ideation<->behavior — a DEFINITIONAL boundary, not a
lexical one. The same class of problem (stressful-event vs traumatic-
experience) is exactly what definition-grounding fixed for factors.

Formulation: for each post we build four premise-hypothesis pairs, one per
risk level, each hypothesis stating that level's clinical definition. A
shared encoder emits one scalar per pair; the four scalars are softmaxed
(the levels are mutually exclusive, unlike factors) and trained with
cross-entropy. At inference we take the argmax.

Optional ordinal smoothing (--label-smooth-adjacent) puts a small amount
of probability mass on adjacent severity levels, reflecting that
mislabeling Behavior as Attempt is less wrong than as Indicator.

NOTE ON DEFINITIONS: the four strings below follow the C-SSRS-derived
descriptions used by this task family. Verify them against the official
task page before the final run — the factor NLI gain came largely from
using the organizers' EXACT wording.

Usage:
  python src/train_risk_nli.py \
      --train artifacts/train_split.csv --val artifacts/val_split.csv \
      --model microsoft/deberta-v3-large --outdir runs/risk_nli \
      --epochs 4
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
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

# Severity-ordered definitions (Indicator < Ideation < Behavior < Attempt)
RISK_DEFINITIONS = {
    # EXACT definitions from the official task page (C-SSRS-grounded).
    # The Ideation/Behavior boundary hinges on presence of a PLAN — which
    # is exactly where 46% of risk errors occur.
    "indicator": "the post contains no explicit mentions of suicide",
    "ideation": "the post contains explicit suicidal expressions but no suicidal plan",
    "behavior": "the post contains explicit suicidal expressions and self-harm or a suicidal plan",
    "attempt": "the post contains explicit mentions of recent or past suicidal attempts",
}
K = len(RISK_LEVELS)
LABEL2ID = {l: i for i, l in enumerate(RISK_LEVELS)}


# ---------------------------------------------------------------------------
# GAUR-GROUNDED DEFINITIONS  (opt in with RISK_DEFS=gaur; default unchanged)
#
# WHY. The definitions above are the terse Table III one-liners, and the word
# "plan" is doing all the work at the boundary where 30% of the errors live
# (102 of 343 OOF errors are gold `behavior` predicted `ideation`).
#
# The problem is that "plan" means something much NARROWER in ordinary usage --
# and in C-SSRS, which is what a pretrained NLI encoder will have absorbed --
# than it means in this dataset. In C-SSRS, thinking of a method (level 3) and
# even a fully worked-out plan with intent (level 5) are still IDEATION; only
# acts from preparation onward are behavior, and self-harm without suicidal
# intent is a separate category entirely. A model reasoning from those priors
# systematically under-calls `behavior`, matching the observed error pattern.
#
# But this dataset does NOT follow C-SSRS directly. Both organizer papers cite
# **Gaur et al., WWW 2019** as the approved annotation criteria (Li et al. 2022,
# HKIE Transactions 29(4): "the adopted labels and their corresponding
# annotation criteria have been approved by domain experts working in psychology
# (Gaur et al., 2019)"). Gaur operationalises the class far more broadly:
#
#   "A user with suicidal behavior confesses active or historical self-harm, or
#    active planning to commit suicide, or a history of being institutionalized
#    for mental health. Actions include cutting or using blunt force violence
#    (self-punching and head strikes), heavy substance abuse, planning for
#    suicide attempt, or actions involving a means of death (holding guns or
#    knives, standing on ledges, musing over pills or poison, or driving
#    recklessly)."
#
# That paragraph accounts for every confident miss in the error dump: active
# or historical self-harm, active planning, musing over a means of death, and
# preparation actions such as writing notes or making a video -- categories
# the terse Table III wording does not name explicitly.
#
# The evidence that this is the real boundary and not a story: on the
# annotators' own quoted spans, plan/method/self-harm cues appear in 47.7% of
# spans on behavior posts predicted RIGHT, 31.8% on behavior posts MISSED, and
# only 5.3% on gold IDEATION. The cue separates the classes ~6:1, which means
# the model was being asked a narrower question than the annotators answered.
#
# This is the opposite move from simply swapping in the organizers' terser
# wording: it expands a terse definition with the operational content of the
# source it came from.
# ---------------------------------------------------------------------------
RISK_DEFINITIONS_GAUR = {
    "indicator":
        "the post shows distress or risk factors but never explicitly "
        "mentions suicide, self-harm, or wanting to die",
    "ideation":
        "the post explicitly says the author thinks about suicide or wants to "
        "die, but only as a thought or wish: no method is named, sought or at "
        "hand, no time or condition is stated, no self-harm is described, and "
        "nothing has been prepared",
    "behavior":
        "the post goes beyond suicidal thoughts to action: the author harms or "
        "has harmed themselves (cutting, burning, hitting themselves, scars, "
        "hurting myself), or names, asks for, or has access to a way of dying "
        "(how to die, ways to go, pills, gun, rope, jumping), or states a time "
        "or condition for killing themselves (tonight, soon, after an event, "
        "if something happens), or has prepared for death (a suicide note, a "
        "goodbye message or video, giving things away, arranging time off)",
    "attempt":
        "the post explicitly mentions a suicide attempt the author already "
        "made, recently or in the past",
}


# ---------------------------------------------------------------------------
# LADDER DEFINITIONS  (RISK_DEFS=ladder) -- the gaur arm's content, repackaged
#
# WHAT THE GAUR ARM ACTUALLY DID, per-class, pooled 5-fold OOF (1635 posts):
#
#     class        base F1   gaur F1
#     indicator      0.872     0.879
#     ideation       0.766     0.772
#     behavior       0.749     0.707
#     attempt        0.726     0.542   <- the whole loss
#
# Attempt RECALL fell 0.640 -> 0.421. Two classes improved; one collapsed and
# took the pooled mean down with it. The Gaur CONTENT was right. The
# PACKAGING had two defects:
#
#  1. NO UPPER BOUND. The gaur `behavior` text reads "harms OR HAS HARMED
#     themselves ... scars" -- past-tense self-injury, which is precisely what
#     gold ATTEMPT posts describe (risk_attempt_boundary.py: self-harm words in
#     20.3% of attempt spans vs 12.1% of behavior spans). Behavior was widened
#     into the class above it with nothing to stop it.
#
#  2. UNBALANCED SPECIFICITY. This model scores four hypotheses and softmaxes
#     four scalars. gaur `behavior` ran ~90 words with ~20 concrete surface
#     cues; gaur `attempt` was left at the terse 15-word original. In a
#     cue-matching competition the rich hypothesis wins on richness alone,
#     regardless of meaning.
#
# THE FIX, and it is the severity-ladder point the attempt-boundary analysis
# raised: the four classes are an ORDERED ladder and the annotators take the
# HIGHEST rung a post reaches. A post that both asks for a method and reports
# having already acted carries a behavior cue AND an attempt cue, and is gold
# ATTEMPT, because attempt is the higher rung. The earlier hypotheses described
# the four as independent options,
# so a post with two cues split its mass -- the 0.401-vs-0.426 near-tie on gold
# attempt posts.
#
# So: every definition EXCLUDES the rungs above it, and all four are written at
# comparable length and cue density. `attempt` gets the completed-act markers
# that measured 0.480 on attempt spans against 0.021 on behavior spans -- a
# 23x separation that was in the data and absent from the hypothesis.
#
# PRE-STATED FALSIFIER: attempt recall must return to within 0.02 of base
# (>= 0.620) AND pooled weighted-F1 must beat base (> 0.7985). Anything less
# and definition engineering closes for real.
# ---------------------------------------------------------------------------
RISK_DEFINITIONS_LADDER = {
    "indicator":
        "the post shows distress, hopelessness or life problems only: it never "
        "explicitly mentions suicide, self-harm or wanting to die, names no "
        "method, states no time, describes no self-injury, and reports no past "
        "attempt",
    "ideation":
        "the post explicitly says the author thinks about suicide or wants to "
        "die, and that is as far as it goes: a thought or a wish (want to die, "
        "wish I was dead, thinking about ending it) with no method named, "
        "sought or at hand, no time or condition stated, no self-injury "
        "described, nothing prepared, and no attempt already made",
    "behavior":
        "the post goes beyond the wish to action or preparation, but the "
        "author has NOT already made an attempt: they self-injure or urge to "
        "(cutting, burning, hitting themselves), or name, seek or possess a "
        "means of dying (how to die, ways to go, pills, gun, rope, jumping), "
        "or state a time or condition (tonight, soon, after an event), or have "
        "prepared for death (a note, a goodbye video, giving things away)",
    "attempt":
        "the post says the author ALREADY CARRIED OUT an act meant to end "
        "their life: they tried or attempted it, overdosed, were found, "
        "resuscitated or hospitalised, woke up in hospital, survived it, or "
        "refer to an earlier try (last time, tried once, it failed, again, "
        "this time it will work); the act is completed and in the past, not "
        "something planned for the future",
}


# ---------------------------------------------------------------------------
# FENCED DEFINITIONS  (RISK_DEFS=fenced) -- symmetric include / exclude
#
# WHAT THE LADDER SET STILL GETS WRONG. Its fences exist but they are ABSTRACT
# while its inclusions are CONCRETE:
#
#     includes: "cutting, burning, hitting themselves ... pills, gun, rope"
#     excludes: "but the author has NOT already made an attempt"
#
# This model matches surface cues. A list of ten concrete nouns fires hard; a
# bare negation of an abstract noun barely fires at all. So the include side
# dominates the exclude side by construction, which is precisely how gold
# ATTEMPT posts ("swallowed two packs of Zoloft", "attempted suicide and was in
# the hospital") kept landing on `behavior` -- 26 of 86 broken posts, the single
# largest cell in risk_arm_transitions.py.
#
# THE FIX: give both sides the same concreteness. Every class names what it
# covers AND names, with the same surface cues, what belongs to its neighbours.
#
# ON LENGTH. train_risk_nli.py encodes (post, hypothesis) with
# truncation="only_first" -- the hypothesis is preserved and the POST is cut.
# Measured against full_train.csv (median post 38 words, p90 150):
#
#     shipped   19-word hyp -> 262-word post budget ->  3.9% of posts cut
#     ladder    78           -> 203                  ->  6.5%
#     fenced   ~120          -> ~155                 -> ~9.5%
#
# Tolerable, and long posts were measured to score BETTER than short ones, so
# losing their tails is not the risk it appears to be. Definitions are capped
# near 115 words deliberately; do not let them sprawl further.
#
# PRE-STATED FALSIFIER: same bar as ladder -- attempt recall >= 0.620 AND
# pooled weighted-F1 > 0.7985 (base).
# ---------------------------------------------------------------------------
RISK_DEFINITIONS_FENCED = {
    "indicator":
        "IT COVERS distress with no suicidal content at all: hopelessness, "
        "depression, loneliness, failing at school or work, family or money "
        "trouble, feeling worthless or a burden, wanting the pain to stop. "
        "IT DOES NOT COVER any explicit mention of suicide or of wanting to be "
        "dead (want to die, wish I was dead, kill myself, end it, better off "
        "dead), nor any self-injury, nor any named way of dying, nor any "
        "stated time, nor any attempt already made — those belong to the "
        "higher levels",
    "ideation":
        "IT COVERS a suicidal thought or wish and nothing further: want to "
        "die, wish I was dead, thinking about killing myself, do not want to "
        "be alive, better off dead, hope I do not wake up. "
        "IT DOES NOT COVER self-injury (cutting, burning, hitting themselves, "
        "scars), nor a way of dying named, sought or at hand (how to die, ways "
        "to go, pills, gun, rope, jumping), nor a stated time or condition "
        "(tonight, soon, after an event), nor anything prepared (a note, a "
        "goodbye video, giving things away), nor an attempt already made — "
        "those belong to the higher levels",
    "behavior":
        "IT COVERS action or preparation short of an attempt: self-injury or "
        "the urge to it (cutting, burning, hitting themselves, scars, blades "
        "at hand), or a way of dying named, sought or possessed (how to die, "
        "ways to go, pills, gun, rope, jumping), or a stated time or condition "
        "(tonight, soon, after an event, if something happens), or preparation "
        "for death (a suicide note, a goodbye message or video, giving things "
        "away, arranging time off). "
        "IT DOES NOT COVER an attempt the author already carried out (tried "
        "it, attempted suicide, overdosed, swallowed pills, woke up in "
        "hospital, was found or resuscitated, survived it, last time, tried "
        "once, again) — that is the higher level",
    "attempt":
        "IT COVERS an act the author ALREADY CARRIED OUT to end their life: "
        "tried it, attempted suicide, overdosed, swallowed pills, hanged or "
        "shot themselves, woke up in hospital, was found or resuscitated, "
        "survived it, or refers to an earlier try (last time, tried once, it "
        "failed, again, this time it will work). "
        "IT DOES NOT COVER an act only wished for, planned or urged toward "
        "with nothing yet done, nor self-injury carried out without meaning to "
        "die — those belong to the lower levels",
}


_DEF_SETS = {
    "": RISK_DEFINITIONS,
    "gaur": RISK_DEFINITIONS_GAUR,
    "ladder": RISK_DEFINITIONS_LADDER,
    "fenced": RISK_DEFINITIONS_FENCED,
}


def _defs() -> dict:
    """RISK_DEFS selects the definition set: 'gaur', 'ladder', or unset for the
    shipped dict, so the champion path is untouched by default."""
    key = os.environ.get("RISK_DEFS", "").lower()
    if key not in _DEF_SETS:
        raise SystemExit(f"unknown RISK_DEFS={key!r}; "
                         f"expected one of {sorted(_DEF_SETS)}")
    return _DEF_SETS[key]


def hypothesis(level: str) -> str:
    return f"This post shows suicide risk level {level}: {_defs()[level]}"


class RiskPairDataset(Dataset):
    """One item = one post, expanded into K premise-hypothesis pairs."""

    def __init__(self, df, tok, max_len, has_labels=True):
        self.posts = df["post"].astype(str).tolist()
        self.tok = tok
        self.max_len = max_len
        self.labels = None
        if has_labels:
            self.labels = [LABEL2ID[normalize_label(x)] for x in df["risk_level"]]

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, i):
        post = self.posts[i]
        enc = self.tok([post] * K, [hypothesis(l) for l in RISK_LEVELS],
                       truncation="only_first", max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {"input_ids": enc["input_ids"],            # (K, L)
                "attention_mask": enc["attention_mask"]}  # (K, L)
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[i])
        return item


class RiskNLIScorer(nn.Module):
    def __init__(self, model_name, smooth_adjacent=0.0):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, 1)
        self.smooth = smooth_adjacent

    def forward(self, input_ids=None, attention_mask=None, label=None, **kw):
        B, k, L = input_ids.shape
        flat_ids = input_ids.view(B * k, L)
        flat_mask = attention_mask.view(B * k, L)
        out = self.enc(input_ids=flat_ids, attention_mask=flat_mask)
        x = self.drop(out.last_hidden_state[:, 0])
        x = x.to(self.head.weight.dtype)
        logits = self.head(x).view(B, k)  # one scalar per risk level

        loss = None
        if label is not None:
            if self.smooth > 0:
                # ordinal-aware soft targets: mass on the true level, a
                # little on its immediate neighbours
                tgt = torch.zeros_like(logits)
                for b in range(B):
                    y = int(label[b])
                    tgt[b, y] = 1.0 - self.smooth
                    nb = [j for j in (y - 1, y + 1) if 0 <= j < k]
                    for j in nb:
                        tgt[b, j] = self.smooth / len(nb)
                loss = -(tgt * torch.log_softmax(logits, -1)).sum(-1).mean()
            else:
                loss = nn.functional.cross_entropy(logits, label)
        return {"loss": loss, "logits": logits}


@torch.no_grad()
def score_posts(model, df, tok, max_len, device, use_bf16, batch=8):
    """Return (N, K) probability matrix over risk levels."""
    model.eval()
    ds = RiskPairDataset(df, tok, max_len, has_labels=False)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    probs = []
    for b in loader:
        ids = b["input_ids"].to(device)
        am = b["attention_mask"].to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=use_bf16):
            logits = model(input_ids=ids, attention_mask=am)["logits"]
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4, help="posts per batch (x4 pairs)")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--label-smooth-adjacent", type=float, default=0.0,
                    help="e.g. 0.1 puts 10%% of mass on adjacent severity levels")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")
    if device == "cpu":
        print("WARNING: on CPU — this trains a full model, use a GPU node.")

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr_df = pd.read_csv(args.train)
    va_df = pd.read_csv(args.val)
    tok = AutoTokenizer.from_pretrained(args.model)

    loader = DataLoader(RiskPairDataset(tr_df, tok, args.max_len),
                        batch_size=args.batch, shuffle=True, num_workers=2)
    model = RiskNLIScorer(args.model, args.label_smooth_adjacent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = (len(loader) // args.grad_accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    gold = [normalize_label(x) for x in va_df["risk_level"]]
    best_f1, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad()
        run = []
        for b, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss = model(**batch)["loss"]
            (loss / args.grad_accum).backward()
            run.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (b + 1) % 100 == 0:
                print(f"epoch {epoch+1} step {b+1} loss {np.mean(run[-100:]):.4f}")

        probs = score_posts(model, va_df, tok, args.max_len, device, use_bf16)
        pred = [RISK_LEVELS[i] for i in probs.argmax(1)]
        f1 = risk_weighted_f1(gold, pred)
        print(f"== epoch {epoch+1}: val weighted-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        torch.save(best_state, os.path.join(args.outdir, "model.pt"))
    with open(os.path.join(args.outdir, "tok"), "w") as f:
        f.write(args.model)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "best_val_weighted_f1": float(best_f1)}, f, indent=2)
    print(f"\nbest val weighted-F1 = {best_f1:.4f}")
    print("compare against current risk ensemble ~0.7646 (val, blended)")


if __name__ == "__main__":
    main()

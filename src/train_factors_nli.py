"""
Definition-aware factor classification (NLI / entailment style).

Per-label factor F1 is strongly correlated (r=0.806) with training-example
count, and the rare factors carry only n=8-35 examples. The organizers'
paper (arXiv 2507.10008) provides exact clinical DEFINITIONS for all 24
factors, so instead of 24 anonymous sigmoid outputs that must each be
learned from scratch, the task is posed as entailment:

    premise    = the post
    hypothesis = "This post shows <factor>: <definition>"
    label      = 1 if the factor is present, else 0

The encoder sees the factor's MEANING every time, so signal is shared
across labels: learning "traumatic experience" partly teaches the model
what "stressful life event" is NOT (their definitions contrast), and a
rare factor benefits from the model's general language understanding of
its definition rather than needing many examples. This is the standard
route to beating label scarcity (cf. entailment-based zero/few-shot
classification).

Cost: one forward pass per (post, factor) pair. Test set is 378 posts x
24 factors ~= 9072 pairs — trivial. Training builds pairs from the labeled
posts (balanced: all positives + sampled negatives).

Usage:
  python src/train_factors_nli.py \
      --train artifacts/train_split.csv --val artifacts/val_split.csv \
      --model microsoft/deberta-v3-large --outdir runs/factors_nli \
      --neg-ratio 3 --epochs 3
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from scoring import FACTOR_TAXONOMY, parse_factors, factors_macro_f1

# Five of the 24 labels are PROTECTIVE, not risk factors -- their published
# definitions make that explicit ("a sense of purpose, goals, or reasons for
# living"). They are also, as a group, the worst-performing labels on 1635
# out-of-fold posts:
#
#     meaning in life          0.324      social support     0.654
#     sense of responsibility  0.560      coping strategy    0.743
#     psychological capital    0.598      -> average 0.576
#     the 19 risk factors                 -> average ~0.70
#
# A plausible cause is the hypothesis template. "This post SHOWS meaning in
# life" is the right question for a risk factor -- the post shows hopelessness
# -- but the wrong one for a protective factor, where the question is whether
# the PERSON HAS it. On five of 24 labels, 21% of a macro-averaged score, the
# model is being asked a subtly malformed question.
#
# Set PROTECTIVE_TEMPLATE=1 in the environment to use the alternative phrasing.
# It is an env var rather than a CLI flag deliberately: the hypothesis is built
# in TWO places (training and inference), and predict_factors_nli.py imports
# this module. A flag threaded through only one path would produce train/serve
# skew -- the model trained on one question and asked another.
PROTECTIVE_FACTORS = {
    "social support",
    "coping strategy",
    "psychological capital",
    "sense of responsibility",
    "meaning in life",
}


def _definitions() -> dict:
    """FACTOR_DEFS_JSON, if set, replaces the published definitions with ones
    DERIVED FROM THE ANNOTATIONS -- see src/derive_definitions.py.

    The published definition is not always the rule the annotators applied. On
    labels such as `meaning in life`, a hypothesis that states the published
    definition faithfully still disagrees with gold, because the operational
    rule the annotators used exists only in their labels. Deriving the
    hypothesis text from the annotations targets that rule directly.

    The file MUST have been derived from the current fold's TRAINING portion
    only. A definition written from the posts it is then evaluated on is
    circular, and any apparent gain would be pure leakage.

    Loaded fresh each call rather than cached at import, so a fold loop can
    point at a different file per fold without restarting the process."""
    p = os.environ.get("FACTOR_DEFS_JSON")
    if not p:
        return FACTOR_DEFINITIONS
    import json as _json
    d = dict(FACTOR_DEFINITIONS)
    d.update({k: v for k, v in _json.load(open(p)).items() if v})
    return d


def _subset() -> list[int]:
    """FACTOR_SUBSET restricts training AND inference to some factors only.

    Accepts "protective", "risk", or a comma-separated list of factor names.
    Unset means all 24, i.e. the default full-taxonomy path.

    WHY A SPECIALIST IS WORTH TRAINING. Held-out F1 splits by polarity: the five
    protective factors average 0.504 against 0.651 for the nineteen risk ones,
    and the polar pairs are worse still (`social support` 0.533 vs `poor social
    support` 0.817). The organizers' own paper (arXiv 2507.10008) uses SEPARATE
    parameters for risk and protective factors, on the stated grounds that they
    "operate independently rather than being the reverse of risk factors".

    The cross-encoder shares every parameter across all 24, so the only thing
    telling it a question is protective is the hypothesis wording -- which has
    to fight a corpus where every post is distressed. A specialist never has to
    switch polarity: its entire training signal is positive content.

    Pair this with --init-from so the specialist starts from the fully-trained
    24-factor checkpoint rather than from scratch. Five factors is only about a
    quarter of the pairs, which is too thin to learn a representation from, but
    ample to specialise one.
    """
    s = os.environ.get("FACTOR_SUBSET", "").strip()
    if not s:
        return list(range(N))
    if s.lower() == "protective":
        names = list(PROTECTIVE_FACTORS)
    elif s.lower() == "risk":
        names = [t for t in FACTOR_TAXONOMY if t not in PROTECTIVE_FACTORS]
    else:
        names = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in names if x not in FACTOR_TAXONOMY]
    if bad:
        raise SystemExit(f"FACTOR_SUBSET names not in the taxonomy: {bad}")
    return sorted(FACTOR_TAXONOMY.index(x) for x in names)


def _aspects() -> dict:
    """FACTOR_ASPECTS_JSON, if set, replaces a factor's SINGLE hypothesis with
    SEVERAL concrete ones that are max-pooled at inference.

    WHY. Held-out per-label F1 splits sharply by polarity: the five protective
    factors average 0.504 against 0.651 for the nineteen risk factors, and the
    polar pairs are worse still -- `social support` 0.533 vs `poor social
    support` 0.817, `meaning in life` 0.231 vs `hopelessness` 0.816. Every post
    in this corpus is distressed, so the encoder learns a negative prior for the
    domain; a hypothesis asserting something positive fights it while a negative
    one rides it. `meaning in life` has precision 0.203 AND recall 0.267 while
    over-firing at 1.31x, so it is a discrimination failure, not a calibration
    one -- thresholds are already at their held-out optimum.

    Asking one abstract question ("does this post show psychological capital")
    is a large semantic leap. Asking several concrete ones ("does the person
    describe recovering from difficulty") is close to the text.

    Loaded fresh each call, not cached at import, so a sweep can point at a
    different file per run without restarting.
    """
    p = os.environ.get("FACTOR_ASPECTS_JSON")
    if not p:
        return {}
    import json as _json
    d = _json.load(open(p))
    return {k: list(v) for k, v in d.items()
            if k in FACTOR_TAXONOMY and isinstance(v, list) and v}


def hypotheses_for(factor: str) -> list[str]:
    """Every hypothesis for this factor. One element unless aspects are on."""
    return _aspects().get(factor) or [hypothesis_for(factor)]


def hypothesis_for(factor: str) -> str:
    """The NLI hypothesis for one factor. Single source of truth for both the
    training dataset and the scoring path."""
    d = _definitions()[factor]
    if (os.environ.get("PROTECTIVE_TEMPLATE") == "1"
            and factor in PROTECTIVE_FACTORS):
        return f"The person in this post has {factor}: {d}"
    return f"This post shows {factor}: {d}"


# exact definitions from arXiv 2507.10008 Table III (same as label_external.py)
FACTOR_DEFINITIONS = {
    "mental health issues": "existing mental disorders such as depression or personality disorders",
    "physical health/characteristic": "physical health issues such as illness, obesity, or being underweight",
    "substance use": "the uncontrollable use of drugs, alcohol, or tobacco",
    "hopelessness": "feelings of hopelessness, or feeling trapped or stuck",
    "emotion dysregulation": "emotion regulation difficulties such as uncontrolled anxiety or anger",
    "low self-esteem": "self-negative feelings such as worthlessness, being a burden, or self-hate",
    "poor school performance": "low school performance such as failing tests or bad grades",
    "low socio-economic status": "unemployment, poverty, or homelessness",
    "interpersonal violence": "violence or assault occurring outside home settings",
    "prior self-harm or suicidal thought/attempt": "past, resolved, previous self-harm or suicidal thought, plan, or attempt",
    "poor social support": "lacking friends, loneliness, being isolated, rejected, neglected, or abandoned",
    "interpersonal difficulty": "difficulty making new friends or socializing",
    "dysfunctional family": "family-related issues that negatively impact the individual",
    "exposure to others' suicide": "mentioning or describing others' suicidal thoughts, attempt, or death",
    "stressful life event": "events that pose challenges but do not lead to traumatic responses",
    "traumatic experience": "an event overwhelming an individual's ability to cope",
    "cognitive deficits": "difficulty in cognitive abilities",
    "suicide means (with access)": "description of potential suicide means or access to those means",
    "sexual orientation related issues": "gender or sexual identity issues, or same-sex relationship concerns",
    "social support": "support from family, partners, friends, or healthcare professionals",
    "coping strategy": "activities the individual engages in to deal with stressful situations",
    "psychological capital": "the individual's positive psychological state such as resilience or optimism",
    "sense of responsibility": "awareness of responsibility to one's own health or survival and to others",
    "meaning in life": "a sense of purpose, goals, or reasons for living",
}
N = len(FACTOR_TAXONOMY)


# Confusable-factor groups: definitions that are semantically close and easy
# to mistake. Sampling negatives from a post's confusable neighbors (given its
# positives) creates HARD negatives that teach the model the boundary, e.g.
# "stressful life event" vs "traumatic experience", the exact distinction the
# annotators' definitions hinge on.
CONFUSABLE = {
    "stressful life event": ["traumatic experience"],
    "traumatic experience": ["stressful life event", "interpersonal violence"],
    "interpersonal violence": ["traumatic experience", "dysfunctional family", "interpersonal difficulty"],
    "interpersonal difficulty": ["poor social support", "interpersonal violence"],
    "poor social support": ["interpersonal difficulty", "social support"],
    "social support": ["poor social support", "psychological capital"],
    "dysfunctional family": ["interpersonal violence", "stressful life event"],
    "hopelessness": ["low self-esteem", "emotion dysregulation", "meaning in life"],
    "low self-esteem": ["hopelessness", "emotion dysregulation"],
    "emotion dysregulation": ["hopelessness", "low self-esteem"],
    "meaning in life": ["hopelessness", "psychological capital", "sense of responsibility"],
    "psychological capital": ["coping strategy", "meaning in life", "social support"],
    "coping strategy": ["psychological capital", "sense of responsibility"],
    "sense of responsibility": ["meaning in life", "coping strategy"],
    "prior self-harm or suicidal thought/attempt": ["suicide means (with access)", "exposure to others' suicide"],
    "suicide means (with access)": ["prior self-harm or suicidal thought/attempt"],
    "exposure to others' suicide": ["prior self-harm or suicidal thought/attempt"],
    "substance use": ["mental health issues"],
    "mental health issues": ["substance use", "emotion dysregulation", "cognitive deficits"],
    "cognitive deficits": ["mental health issues"],
}
def _as_idx(m):
    return {FACTOR_TAXONOMY.index(k): [FACTOR_TAXONOMY.index(v) for v in vs
                                       if v in FACTOR_TAXONOMY]
            for k, vs in m.items() if k in FACTOR_TAXONOMY}


_CONF_IDX = _as_idx(CONFUSABLE)
_CONF_CACHE = {}


def _conf_idx():
    """CONFUSABLE_JSON, if set, REPLACES the hand-written map above with one
    mined from the model's own out-of-fold confusions -- see
    src/confusion_audit.py.

    The hand map was written by reading the published definitions and guessing
    which ones look alike; it decides half of every epoch's negatives and was
    never validated. The mined map measures, for each ordered pair (g, j), how
    much the presence of g inflates the model's score for an absent j.

    Read from the environment on each call, not cached at import, so a fold
    loop can point at a per-fold map without restarting the process. Parsed
    files are cached by path so the per-epoch rebuild stays cheap."""
    p = os.environ.get("CONFUSABLE_JSON")
    if not p:
        return _CONF_IDX
    if p not in _CONF_CACHE:
        import json as _json
        _CONF_CACHE[p] = _as_idx(_json.load(open(p)))
    return _CONF_CACHE[p]


def make_pairs(df, neg_ratio, rng, hard_frac=0.5, full_enum=False):
    """Build (text, factor_idx, label) rows. All positives + sampled negatives,
    where a fraction `hard_frac` of negatives are drawn from factors CONFUSABLE
    with the post's positives (hard negatives), the rest random (easy).

    Called once PER EPOCH with a fresh rng draw so negatives vary across epochs
    (regularization + better coverage of the easy-negative space).

    full_enum=True emits ALL 24 factors for every post and ignores neg_ratio
    and hard_frac entirely.

    Why that option exists. The sampled scheme sets n_neg = len(pos)*neg_ratio,
    so a post with one gold factor is trained against 3 of its 23 negatives
    while a post with six is trained against all 18 -- densely labelled posts
    get near-exhaustive coverage and sparse ones almost none. Inference, in
    score_all, evaluates all 24 for every post regardless. Full enumeration
    removes both the coverage imbalance and that train/serve mismatch, at
    roughly 31,300 pairs per epoch against 14,600 (about 2.1x the GPU time).

    It also makes hard-negative SAMPLING redundant: every hard negative is
    present by construction. The emphasis then belongs in the objective, so
    pair this with --loss asl (which down-weights easy negatives) or
    --loss bce_perlabel rather than plain bce, whose single pos_weight=3.0 was
    derived from the sampling ratio that no longer applies."""
    if full_enum:
        # FACTOR_SUBSET narrows the enumeration. With the 5 protective factors
        # that is 5 pairs per post instead of 24, so full enumeration costs
        # about a quarter of a normal epoch rather than 2.1x -- exhaustive
        # coverage becomes the cheap option, not the expensive one, and the
        # sampled scheme's coverage imbalance disappears.
        sub = _subset()
        rows = []
        for _, r in df.iterrows():
            post = str(r["post"])
            pos = {FACTOR_TAXONOMY.index(f) for f in parse_factors(r["factors"])
                   if f in FACTOR_TAXONOMY}
            rows.extend((post, j, 1 if j in pos else 0) for j in sub)
        return rows

    conf = _conf_idx()
    rows = []
    for _, r in df.iterrows():
        post = str(r["post"])
        pos = {FACTOR_TAXONOMY.index(f) for f in parse_factors(r["factors"])
               if f in FACTOR_TAXONOMY}
        for j in pos:
            rows.append((post, j, 1))
        n_neg = max(1, len(pos) * neg_ratio) if pos else neg_ratio
        # hard-negative candidates: confusable with any positive, not itself positive
        hard_pool = set()
        for j in pos:
            for c in conf.get(j, []):
                if c not in pos:
                    hard_pool.add(c)
        hard_pool = list(hard_pool)
        rng.shuffle(hard_pool)
        n_hard = min(len(hard_pool), int(round(n_neg * hard_frac)))
        chosen = hard_pool[:n_hard]
        # fill the rest with random easy negatives
        easy_pool = [j for j in range(N) if j not in pos and j not in set(chosen)]
        rng.shuffle(easy_pool)
        chosen += easy_pool[: n_neg - n_hard]
        for j in chosen:
            rows.append((post, j, 0))
    return rows


class PairDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        post, j, y = self.rows[i]
        # With aspects on, each training pair uses ONE of the factor's
        # hypotheses, chosen deterministically by (row, factor) so the run stays
        # reproducible while different posts see different aspects. Cycling
        # rather than sampling keeps the exposure even and needs no RNG.
        hs = hypotheses_for(FACTOR_TAXONOMY[j])
        hyp = hs[0] if len(hs) == 1 else hs[(i + j) % len(hs)]
        enc = self.tok(post, hyp, truncation="only_first", max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(float(y)),
                "factor": torch.tensor(j)}


class NLIScorer(nn.Module):
    """Definition-grounded NLI scorer with a pluggable objective.

    loss_type:
      "bce"          -- default. One global pos_weight shared by all 24
                        factors.
      "bce_perlabel" -- per-factor pos_weight taken from the ACTUAL neg:pos
                        ratio of the constructed training pairs.
      "asl"          -- Asymmetric Loss (Ridnik et al., arXiv 2009.14119).

    Why the alternatives exist. Subtask 2 is MACRO-F1 over 24 labels, so every
    factor counts equally regardless of prevalence, and the weakest labels are
    all rare ones. Plain BCE sets

        pos_weight = clip(neg_ratio, 1, 5) = 3.0

    from the NEGATIVE SAMPLING ratio, not from label prevalence, and applies
    that one constant to every factor. Prevalence spans 0.5% to 45.6%, so the
    per-factor neg:pos ratio of the constructed pairs spans roughly 0.8:1 to
    77:1. Handling a ~100x spread with a single constant under-weights exactly
    the rare positives macro-F1 cares most about.

    ASL attacks the same imbalance differently: with gamma_pos=0 the positive
    term keeps full gradient while easy negatives are down-weighted by
    gamma_neg, and the probability-shift clip discards very easy negatives
    outright. The 2024 IEEE BigData Cup winner used a related metric-aligned
    objective (macro double soft-F1) rather than plain BCE.

    Soft-macro-F1 itself is NOT offered here: this model is trained pair-wise,
    one (post, factor) logit at a time, so a batch of 32 pairs spread over 24
    factors gives about one example per label -- far too few to estimate the
    per-label F1 that a soft-F1 surrogate needs.
    """

    def __init__(self, model_name, pos_weight=None, loss_type="bce",
                 pos_weight_vec=None, gamma_neg=4.0, gamma_pos=0.0, clip=0.05,
                 nu_vec=None, cb_vec=None, ntr_lambda=2.0, db_gamma=2.0):
        nn.Module.__init__(self)
        self.enc = AutoModel.from_pretrained(model_name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, 1)
        self.loss_type = loss_type
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip
        self.ntr_lambda, self.db_gamma = ntr_lambda, db_gamma
        self.register_buffer("pos_weight",
            torch.tensor([pos_weight], dtype=torch.float) if pos_weight else None)
        self.register_buffer("pos_weight_vec",
            torch.tensor(pos_weight_vec, dtype=torch.float)
            if pos_weight_vec is not None else None)
        self.register_buffer("nu_vec",
            torch.tensor(nu_vec, dtype=torch.float)
            if nu_vec is not None else None)
        self.register_buffer("cb_vec",
            torch.tensor(cb_vec, dtype=torch.float)
            if cb_vec is not None else None)

    def _cbntr(self, logit, target, factor):
        """CB-NTR: class-balanced weight x negative-tolerant regularization.

        Huang et al., EMNLP 2021 (arXiv 2109.04712), the text-domain
        replication of Distribution-Balanced Loss (Wu et al., ECCV 2020,
        arXiv 2007.09654). Chosen because it is the only long-tail method
        validated at this project's scale in BOTH modalities:

            VOC-MLT   1142 training images, 20 classes, support 4-775
            here      1635 posts,           24 labels,  support 8-745
            Reuters   BERT + sigmoid, tail labels with <=8 positives
                      tail macro-F1  BCE 0.00 -> NTR-FL 22.3 -> CB-NTR 32.3

        THE MECHANISM MATCHES THE MEASURED FAILURE MODE. Under binary-per-label
        training a tail label's logit is buried by negative gradient, its
        logits collapse, and its RANKING never recovers -- `meaning in life`
        has AP 0.23 against `hopelessness` 0.92, and
        `factor_loss_metric_gap.py` attributes the deficit to ordering rather
        than to the cut point. A per-label threshold only moves the cut, so it
        cannot repair ordering; only the objective can.

        Two pieces, and the paper's ablation says the SECOND carries the gain
        (NTR alone 15.4 -> 22.3 tail; adding CB -> 32.3):

          nu_i   the paper's class-specific bias  v_i = -kappa * logit(p_i),
                 p_i the POST-level prevalence, applied as `logit - nu`.
                 SIGN MATTERS AND IS EASY TO GET BACKWARDS. For a rare label
                 logit(p_i) is very negative, so nu_i is POSITIVE and the
                 shifted logit starts LOW -- at the class prior instead of at
                 p=0.5. That is the point: a rare label's negatives are then
                 already well-classified at initialisation and contribute
                 almost no gradient, so they stop drowning its few positives.
                 Shifting the other way (raising rare logits) would maximise
                 negative gradient and do the exact opposite. kappa = 0.05.

                 The bias lives in the LOSS, not the model, so inference emits
                 the unshifted logit. A constant per-label offset changes
                 neither the within-label ranking (AP) nor anything `tfit`
                 cannot absorb, so no decode change is needed.
          1/lam  the negative half of the loss is divided by lambda and its
                 saturation sharpened by lambda, so an ocean of negatives can
                 no longer drown a rare positive. lambda = 2.
          cb_i   class-balanced weight (1-beta)/(1-beta^n_i), beta = 0.9,
                 normalised to mean 1 so the overall loss scale is unchanged.

        The focal factor (gamma = 2) wraps both halves, as in the paper's
        framework. gamma = 0 disables it.
        """
        nu = self.nu_vec.to(logit.device)[factor].to(logit.dtype)
        w = self.cb_vec.to(logit.device)[factor].to(logit.dtype)
        lam = self.ntr_lambda
        z = logit - nu
        p = torch.sigmoid(z)
        # softplus(-z) = log(1+e^-z) is the positive term; the negative term is
        # log(1+e^(lam*z))/lam -- note BOTH the scale and the saturation use
        # lambda, which is what makes it negative-TOLERANT rather than merely
        # down-weighted.
        pos = nn.functional.softplus(-z)
        neg = nn.functional.softplus(lam * z) / lam
        if self.db_gamma > 0:
            # THE FOCAL WEIGHT USES A DIFFERENT q PER BRANCH. The paper defines
            #     q = sigmoid(z - nu)          for positives
            #     q = sigmoid(lam * (z - nu))  for negatives
            # so the negative branch's focal factor is sigmoid(lam*z)^gamma,
            # NOT sigmoid(z)^gamma. Since sigmoid(lam*z) < sigmoid(z) for
            # z < 0, using the unscaled p would give well-classified negatives
            # a LARGER weight than the method specifies -- partially undoing
            # the negative tolerance that is the entire point of the loss, and
            # implementing a different method from the one cited.
            pos = pos * (1 - p).clamp(min=1e-6) ** self.db_gamma
            neg = neg * torch.sigmoid(lam * z).clamp(min=1e-6) ** self.db_gamma
        return (w * (target * pos + (1 - target) * neg)).mean()

    def _asl(self, logit, target):
        """Asymmetric Loss, per-example. gamma_pos=0 keeps rare positives sharp."""
        p = torch.sigmoid(logit)
        p_pos = p
        p_neg = (1 - p - self.clip).clamp(min=0) if self.clip > 0 else (1 - p)
        loss_pos = target * torch.log(p_pos.clamp(min=1e-8)) * \
            ((1 - p_pos) ** self.gamma_pos)
        loss_neg = (1 - target) * torch.log(p_neg.clamp(min=1e-8)) * \
            ((1 - p_neg) ** self.gamma_neg)
        return -(loss_pos + loss_neg).mean()

    def forward(self, input_ids=None, attention_mask=None, label=None, **kw):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop(out.last_hidden_state[:, 0])
        x = x.to(self.head.weight.dtype)
        logit = self.head(x).squeeze(-1)
        loss = None
        if label is not None and self.loss_type != "bce":
            tgt = label.to(logit.dtype)
            if self.loss_type == "asl":
                loss = self._asl(logit, tgt)
            elif self.loss_type == "cbntr":
                factor = kw.get("factor")
                if factor is None or self.nu_vec is None or self.cb_vec is None:
                    raise ValueError("cbntr needs `factor`, nu_vec and cb_vec")
                loss = self._cbntr(logit, tgt, factor.to(logit.device))
            elif self.loss_type == "bce_perlabel":
                factor = kw.get("factor")
                if factor is None or self.pos_weight_vec is None:
                    raise ValueError("bce_perlabel needs `factor` and pos_weight_vec")
                w = self.pos_weight_vec.to(logit.device)[factor.to(logit.device)]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logit, tgt, pos_weight=w.to(logit.dtype))
            else:
                raise ValueError(f"unknown loss_type {self.loss_type}")
            return {"loss": loss, "logit": logit}
        if label is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, label.to(logit.dtype), pos_weight=self.pos_weight)
        return {"loss": loss, "logit": logit}


@torch.no_grad()
def score_all(model, df, tok, max_len, device, use_bf16, batch=64):
    """Return (N_posts, 24) probability matrix by scoring every pair."""
    model.eval()
    posts = df["post"].astype(str).tolist()
    probs = np.zeros((len(posts), N), dtype=np.float32)
    # One entry per (post, factor, aspect). Without FACTOR_ASPECTS_JSON every
    # factor has exactly one aspect, so this reduces to one pair per
    # (post, factor) and the max below is taken over a single value.
    HYP = [hypotheses_for(FACTOR_TAXONOMY[j]) for j in range(N)]
    # A specialist scores only its own factors; the rest stay 0 so a splice can
    # tell trained columns from untouched ones. Unset FACTOR_SUBSET -> all 24.
    sub = _subset()
    pairs = [(pi, j, a) for pi in range(len(posts))
             for j in sub for a in range(len(HYP[j]))]
    for s in range(0, len(pairs), batch):
        chunk = pairs[s : s + batch]
        texts = [posts[pi] for pi, _, _ in chunk]
        hyps = [HYP[j][a] for _, j, a in chunk]
        enc = tok(texts, hyps, truncation="only_first", max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            logit = model(**enc)["logit"]
        p = torch.sigmoid(logit.float()).cpu().numpy()
        # MAX-pool across a factor's aspects: the factor is present if ANY of
        # its concrete statements is entailed. Mean would let three quiet
        # aspects drown the one the post actually expresses.
        for (pi, j, _), pv in zip(chunk, p):
            if pv > probs[pi, j]:
                probs[pi, j] = pv
    return probs


def tune_thresholds(probs, gold):
    ths = np.full(N, 0.5)
    for j in range(N):
        best_t, best_f = 0.5, -1
        for t in np.arange(0.1, 0.9, 0.05):
            pred = (probs[:, j] >= t).astype(int)
            tp = int(((pred == 1) & (gold[:, j] == 1)).sum())
            fp = int(((pred == 1) & (gold[:, j] == 0)).sum())
            fn = int(((pred == 0) & (gold[:, j] == 1)).sum())
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f:
                best_f, best_t = f1, t
        ths[j] = best_t
    return ths


def gold_matrix(df):
    Y = np.zeros((len(df), N), dtype=int)
    for r, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            if f in FACTOR_TAXONOMY:
                Y[r, FACTOR_TAXONOMY.index(f)] = 1
    return Y


def to_lists(binary):
    return [[FACTOR_TAXONOMY[j] for j in range(N) if row[j]] for row in binary]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--neg-ratio", type=int, default=3)
    ap.add_argument("--external", default=None,
                    help="external labeled CSV (new users) — added to training pairs, never to val")
    ap.add_argument("--hard-frac", type=float, default=0.5,
                    help="fraction of negatives drawn from confusable factors (hard negs)")
    ap.add_argument("--full-enum", action="store_true",
                    help="emit all 24 (post, factor) pairs instead of sampling "
                         "negatives. Removes the coverage imbalance and the "
                         "train/serve mismatch; ~2.1x the pairs per epoch. "
                         "Pair with --loss asl or bce_perlabel -- the default "
                         "pos_weight=3.0 is derived from the sampling ratio "
                         "this flag discards.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--init-from", default=None,
                    help="run dir whose model.pt initialises this run. Use with "
                         "FACTOR_SUBSET to specialise an existing 24-factor "
                         "checkpoint instead of training one from scratch.")
    ap.add_argument("--loss", choices=["bce", "bce_perlabel", "asl", "cbntr"],
                    default="bce",
                    help="objective. 'bce' is a single global pos_weight and "
                         "is the default. 'cbntr' "
                         "is class-balanced weighting x negative-tolerant "
                         "regularization (Huang et al. EMNLP 2021, the text "
                         "replication of Distribution-Balanced Loss) -- the "
                         "only long-tail method validated at this corpus size "
                         "(VOC-MLT 1142 instances) AND this per-label support "
                         "(Reuters tail <=8 positives), lifting tail macro-F1 "
                         "from 0.00 to 32.3 there.")
    ap.add_argument("--ntr-kappa", type=float, default=0.05,
                    help="cbntr: scale of the per-label logit shift "
                         "nu_i = -kappa * logit(prevalence_i)")
    ap.add_argument("--ntr-lambda", type=float, default=2.0,
                    help="cbntr: negative-tolerance. The negative half of the "
                         "loss is divided by lambda and saturates lambda times "
                         "sharper, so negatives cannot drown a rare positive.")
    ap.add_argument("--cb-beta", type=float, default=0.99,
                    help="cbntr: class-balanced weight (1-b)/(1-b^n_i). NOT "
                         "0.9: the effective number saturates at 1/(1-b), so "
                         "b=0.9 caps at 10 and gives 22 of the 24 labels an "
                         "IDENTICAL weight -- a silent no-op, and the paper's "
                         "ablation puts most of the gain in the CB half. "
                         "b=0.99 caps at 100, inside the 8-745 support range. "
                         "b=0.999 approaches 1/n, i.e. full inverse-frequency "
                         "re-weighting, which over-weights the rarest labels.")
    ap.add_argument("--db-gamma", type=float, default=2.0,
                    help="cbntr: focal exponent wrapping both halves; 0 = off")
    ap.add_argument("--max-pos-weight", type=float, default=20.0,
                    help="cap on per-factor pos_weight (bce_perlabel), so the "
                         "rarest labels do not explode the gradient")
    ap.add_argument("--asl-gamma-neg", type=float, default=4.0)
    ap.add_argument("--asl-gamma-pos", type=float, default=0.0)
    ap.add_argument("--asl-clip", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")
    if device == "cpu":
        print("WARNING: on CPU — this trains a full model, use a GPU node.")

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    tr_df = pd.read_csv(args.train)
    va_df = pd.read_csv(args.val)
    if args.external:
        ext = pd.read_csv(args.external, keep_default_na=False)
        keep = [c for c in ["row_id", "post", "factors"] if c in ext.columns]
        tr_df = pd.concat([tr_df, ext[keep]], ignore_index=True)
        print(f"external: +{len(ext)} posts merged into training (now {len(tr_df)})")
    tok = AutoTokenizer.from_pretrained(args.model)

    # pos_weight approximates neg:pos ratio so the head doesn't collapse to
    # predicting "absent" (clamped to avoid over-correction)
    pw = float(np.clip(args.neg_ratio, 1.0, 5.0))

    # per-factor pos_weight, measured from the ACTUAL constructed pairs rather
    # than assumed from the sampling ratio. Only used when --loss bce_perlabel.
    pw_vec = None
    if args.loss == "bce_perlabel":
        _probe = make_pairs(tr_df, args.neg_ratio, np.random.default_rng(args.seed),
                            args.hard_frac, args.full_enum)
        npos = np.zeros(N)
        nneg = np.zeros(N)
        for _, j, y in _probe:
            (npos if y else nneg)[j] += 1
        pw_vec = np.clip(nneg / np.clip(npos, 1, None), 1.0, args.max_pos_weight)
        print(f"per-factor pos_weight: min {pw_vec.min():.1f} max {pw_vec.max():.1f} "
              f"median {np.median(pw_vec):.1f}  (global constant was {pw})")
        worst = np.argsort(-pw_vec)[:5]
        for j in worst:
            print(f"    {FACTOR_TAXONOMY[j]:<44} pos {int(npos[j]):>5} "
                  f"neg {int(nneg[j]):>5}  w {pw_vec[j]:.1f}")

    # cbntr vectors. BOTH are computed from the POST-LEVEL label matrix, not
    # from pair counts: the prevalence p_i the theory needs is "what fraction
    # of documents carry label i", and the 3:1 negative sampling already
    # distorts pair counts away from that. Deriving them from pair counts would
    # make nu_i a function of the sampler rather than of the data.
    nu_vec = cb_vec = None
    if args.loss == "cbntr":
        Ymat = gold_matrix(tr_df)
        n_i = np.asarray(Ymat).sum(0).astype(float)
        Npost = len(tr_df)
        p_i = np.clip(n_i / max(Npost, 1), 1e-6, 1 - 1e-6)
        # v_i = -kappa * logit(p_i), applied as (logit - v_i). Rare label ->
        # logit(p_i) very negative -> v_i POSITIVE -> shifted logit starts at
        # the class prior, so its negatives are already easy and stop
        # dominating the gradient. See _cbntr for why the other sign is wrong.
        nu_vec = -args.ntr_kappa * np.log(p_i / (1 - p_i))
        cb = (1 - args.cb_beta) / np.clip(
            1 - np.power(args.cb_beta, np.clip(n_i, 1, None)), 1e-8, None)
        cb_vec = cb / cb.mean()          # mean 1 -> overall loss scale intact
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
                f"{spread:.2f}x across 24 labels -- the class-balanced half of "
                f"CB-NTR is a no-op at this beta and the run would silently "
                f"measure NTR alone. Raise --cb-beta (0.99 caps the effective "
                f"number at 100, inside our 8-745 support range).")
        for j in np.argsort(n_i)[:3]:
            print(f"    {FACTOR_TAXONOMY[j]:<44} n {int(n_i[j]):>4} "
                  f"nu {nu_vec[j]:+.3f} cb {cb_vec[j]:.2f}")
        for j in np.argsort(-n_i)[:2]:
            print(f"    {FACTOR_TAXONOMY[j]:<44} n {int(n_i[j]):>4} "
                  f"nu {nu_vec[j]:+.3f} cb {cb_vec[j]:.2f}")

    model = NLIScorer(args.model, pos_weight=pw, loss_type=args.loss,
                      pos_weight_vec=pw_vec, gamma_neg=args.asl_gamma_neg,
                      gamma_pos=args.asl_gamma_pos, clip=args.asl_clip,
                      nu_vec=nu_vec, cb_vec=cb_vec,
                      ntr_lambda=args.ntr_lambda,
                      db_gamma=args.db_gamma).to(device)
    print(f"objective: {args.loss}")

    # --init-from warm-starts from an already-trained checkpoint. Needed for a
    # FACTOR_SUBSET specialist: five factors is about a quarter of the pairs,
    # far too thin to learn a representation from but ample to SPECIALISE one
    # that already exists. Loss buffers (pos_weight, class_weights) are
    # recomputed from this run's own data, so a mismatch there is expected and
    # tolerated; anything else is not.
    if getattr(args, "init_from", None):
        ck = os.path.join(args.init_from, "model.pt")
        if not os.path.exists(ck):
            raise SystemExit(f"--init-from: no {ck}")
        sd = torch.load(ck, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loss_buf = {"pos_weight", "pos_weight_vec", "class_weights",
                    "nu_vec", "cb_vec"}
        bad = ([k for k in missing if k.split(".")[-1] not in loss_buf]
               + [k for k in unexpected if k.split(".")[-1] not in loss_buf])
        if bad:
            raise SystemExit(
                f"--init-from: checkpoint/model mismatch beyond loss buffers: "
                f"{bad}\nIs {args.init_from} the same --model architecture?")
        print(f"warm-started from {ck}"
              + (f"  (loss buffers reinitialised: "
                 f"{sorted(set(missing) | set(unexpected))})"
                 if (missing or unexpected) else ""))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # estimate steps from a first pair build (size is stable across epochs)
    probe = make_pairs(tr_df, args.neg_ratio, rng, args.hard_frac, args.full_enum)
    steps_per_epoch = len(probe) // args.batch // args.grad_accum + 1
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    _cj = os.environ.get("CONFUSABLE_JSON")
    print(f"~{len(probe)} pairs/epoch ({sum(r[2] for r in probe)} pos), "
          f"pos_weight={pw}, "
          + ("full-enum (hard_frac ignored)" if args.full_enum
             else f"hard_frac={args.hard_frac}")
          + f", confusable={_cj if _cj else 'hand-written'}")

    va_gold = gold_matrix(va_df)
    best_macro, best_state = -1.0, None
    for epoch in range(args.epochs):
        # FRESH negatives each epoch (regularization + coverage)
        train_rows = make_pairs(tr_df, args.neg_ratio, rng, args.hard_frac,
                                args.full_enum)
        loader = DataLoader(PairDataset(train_rows, tok, args.max_len),
                            batch_size=args.batch, shuffle=True, num_workers=2)
        model.train(); opt.zero_grad()
        run = []
        for b, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch)["loss"]
            (loss / args.grad_accum).backward()
            run.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (b + 1) % 200 == 0:
                print(f"epoch {epoch+1} step {b+1} loss {np.mean(run[-200:]):.4f}")
        # eval
        vp = score_all(model, va_df, tok, args.max_len, device, use_bf16)
        ths = tune_thresholds(vp, va_gold)
        on = (vp >= ths).astype(int)
        macro = factors_macro_f1(to_lists(va_gold), to_lists(on))
        print(f"== epoch {epoch+1}: val macro-F1 (tuned) = {macro:.4f}")
        if macro > best_macro:
            best_macro = macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            np.save(os.path.join(args.outdir, "thresholds.npy"), ths)

    if best_state:
        torch.save(best_state, os.path.join(args.outdir, "model.pt"))
    with open(os.path.join(args.outdir, "tok"), "w") as f:
        f.write(args.model)

    # RECORD THE EXACT QUESTIONS THIS MODEL WAS TRAINED ON.
    # The hypothesis is built from THREE env vars -- FACTOR_DEFS_JSON,
    # FACTOR_ASPECTS_JSON and PROTECTIVE_TEMPLATE -- and inference rebuilds it
    # from the same code. If the environment differs at inference the model is
    # asked a different question than it learned, and nothing in the code path
    # raises an error -- the run simply scores a mismatched hypothesis. This
    # file lets a downstream check compare the questions actually used at
    # inference against the ones trained on, and refuse on a mismatch.
    # Writing the strings themselves (not the file paths) makes that check
    # byte-exact and keeps it valid if a file is later edited or moved.
    import json as _json
    with open(os.path.join(args.outdir, "hypotheses.json"), "w") as f:
        _json.dump({t: hypotheses_for(t) for t in FACTOR_TAXONOMY}, f,
                   indent=1, sort_keys=True)
    print(f"\nbest val macro-F1 = {best_macro:.4f}")
    print("compare against plain-factor baseline ~0.4165 (unaug single split)")


if __name__ == "__main__":
    main()

# Pipeline

This document describes TRIDENT, the submitted system, component by component, with every weight and threshold and the reason for it. It is carried over from the README of the competition submission, where it was written alongside the code; only the framing is new.

Contents: [The pipeline](#the-pipeline) · [Guards](#guards) · [Notes on evaluation](#notes-on-evaluation) · [Getting the model weights](#getting-the-model-weights) · [Running on unseen data](#running-on-unseen-data)

## The pipeline
### Risk - weighted-F1, weight 0.4

```
3 x DeBERTa-v3-large   runs/risk_v1{,_s14,_s15}    RISK_CAT_ONLY=1
        |              dual-head model; inference uses the categorical head
        |
        +-- mean --> 0.5 x ours + 0.5 x MentalRoBERTa (5-fold bag)
                     src/ensemble_risk.py -> artifacts/ensembled_risk_catonly2.csv
                                    |
        M = 0.70 x ensembled + 0.30 x NLI (runs/risk_nli, runs/risk_nli_s2)
                                    |
                                 argmax
                                    |
        apply_rules(urgency, completed_act) on canonicalised text
                                    |
        route_attempt  (Qwen2.5-72B chain-of-thought says attempt,
                        and the label is not `indicator`)
```

The three DeBERTa seeds share an architecture and training set, so their errors
are correlated and averaging them mainly reduces seed variance. The
MentalRoBERTa member is a different architecture with mental-health domain
pretraining, and the NLI members pose the task as entailment against
per-class hypotheses - both contribute errors that are uncorrelated with the
DeBERTa block, which is what the blend exploits.

`route_attempt` is applied last and never touches `indicator`: evidence is
suppressed on `indicator` posts, so promoting one would leave an empty
evidence cell against non-empty gold.

`src/champion_risk_inject.py --verify` rebuilds this chain and confirms it
reproduces the shipped risk column before any change is attributed to a model.

### Evidence - phrase-F1, weight 0.3

Seven risk-conditioned BIO taggers (`evidence_{ep9,lr2,s14,s15,s16,s17,v1}_rc`).
Each is told the predicted risk level as part of its input, so candidates are
generated against the risk column that is actually submitted.

```
uniform confidence gate 0.65 across all seven taggers
vote k >= 2, clustered under the scorer's containment equivalence
representative = modal string within a cluster
at most 3 spans per post, ranked by agreement
emit nothing where the risk column says `indicator`
```

Clustering uses the scorer's own equivalence - a span matches if either
contains the other within a 3x token-length cap - so taggers that agree on
which clause matters but differ on its boundaries reinforce each other instead
of splitting the vote. The per-post cap matters because matching is one-to-one:
several predictions that all match the same gold span score one true positive
and the rest as false positives.

Suppression on `indicator` follows the annotation scheme, where an
`indicator` post carries no evidence span and an empty prediction scores 1.0.

`0.65 / k=2 / cap=3` is the maximum of a swept gate x k x cap grid
(`src/evidence_cap_sweep.py`).

### Factors - macro-F1, weight 0.3

Eleven arms, unweighted mean in log-odds space, then 24 independent per-label
thresholds chosen by maximising each label's F1 on the pooled out-of-fold
predictions (`tfit`).

```
mnli dmnli cmine prot d4 qnli mnli7 lnli   entailment scorers over per-factor
                                           hypotheses
ltail                                      class-balanced negative-tolerant loss
cbnsynapi                                  CB-NTR warm-started from synthetic data
friendfac                                  MentalRoBERTa tri-head factor head
```

`--match-oof-scale` rescales each arm before pooling. A mean of log-odds
weights each member by the dispersion of its scores, and the out-of-fold
predictions come from a single fold model while the test predictions average
five - an operation that compresses each arm's logits by an arm-specific
amount. Left uncorrected, an arm carries a different influence on the data
being scored than it did on the data the thresholds were fitted against. The
correction maps each arm's test logits onto its own out-of-fold mean and
standard deviation. It uses no labels and fits nothing.

A per-label floor guarantees that any label whose threshold fires on the
out-of-fold data also emits at least one prediction. Macro-F1 averages 24
per-label scores, so a label predicting nothing contributes a hard zero worth
1/24 of the metric.

## Guards
Each of these refuses rather than warning, because the failure it prevents is
silent - the pipeline would otherwise produce a valid-looking file built from
a different model than the one validated.

| Guard | Prevents |
|---|---|
| `champion_risk_inject.py` rebuild gate | attributing a rebuild difference to a model change |
| route-coverage refusal | applying `route_attempt` to only part of the test set |
| `ensemble_risk.py` friend-coverage refusal | blending at an effective weight of 1.0 when the second member is absent |
| `evidence_vote_test.py` missing-config refusal | defaulting every tagger's confidence gate to zero |
| `predict_factors_nli.py` hypothesis check | scoring a model with hypotheses it was not trained on |
| `factor_threshold_stabilise.py` row-set and coverage checks | pooling arms that cover different rows, or writing a partly-rebuilt column |
| `patch_submission_risk.py` single-component check | a change that moves two columns at once |
| `build_submission.py --verify` | shipping a pipeline that cannot rebuild its own output |
| `build_submission.py` risk-head check | shipping a risk column read from the ordinal-mixture head instead of the categorical one |
| `predict.py` state-dict check | loading a checkpoint whose head does not match the model |
| `predict.py` `user_context` check | scoring a model on a text format it was not trained on |

`scripts/cold_run_drill.sh` runs the whole chain on held-out rows the stored
artifacts do not cover, confirming it works on data it has not seen.

## Notes on evaluation
- **Split by `anon_user_id`, never by row.** Posts from one author are often
  near-duplicates of each other, so a row-level split places near-copies on
  both sides and inflates validation scores.
- **Evidence phrase-F1 is macro** - the mean of per-post F1, where a post with
  no gold and no prediction scores 1.0.
- **Span matching is containment in either direction** plus
  `token_len(pred) <= 3 x token_len(gold)`, resolved by maximum bipartite
  matching. Redundant near-duplicate predictions therefore cost precision and
  cannot both score.
- **Judge a configuration by win rate across author-disjoint splits and the
  per-split standard deviation**, not by the confidence interval of the mean:
  the splits are correlated, so that interval understates the spread of a
  single draw.

## Getting the model weights
**Start here if you intend to score new posts.** The code in this package does
not contain model weights. Scoring a new test set needs 62 trained
checkpoints, 86.6 GB in total, which is why they are distributed
separately. They are fp32 state dicts with no optimizer state, so there is
nothing in them to strip.

```bash
python src/fetch_weights.py --repo ShlokShlok123/nidhi-weights
python src/check_setup.py           # must print READY before you go further
```

That repository is **gated**. These weights are fine-tuned on a clinical
corpus and can retain fragments of their training text, so access is recorded
rather than anonymous. Approval is automatic and takes one click:

1. Sign in at <https://huggingface.co/ShlokShlok123/nidhi-weights>
2. Accept the terms shown on the model card
3. `huggingface-cli login` on the machine that will download, or pass
   `--token hf_...`

There is no waiting on us. If you get a 401 or 403, it means the terms have
not been accepted on that account yet, not that the repository name is wrong.

The download resumes, so re-run it after an interruption. It also rewrites the
three LoRA arms' `run_config.json` from the cluster-absolute base-model path
they were trained with to a Hub id, which is what makes them portable off the
machine that produced them.

The preflight lists exactly what is present and what is missing, and exits
non-zero until the set is complete:

```bash
python src/check_setup.py --test data/my_new_test.csv   # or against your file
```

The full manifest, all of it under `runs/`:

| Component | Directories | Count | Approx. size |
|---|---|---|---|
| Risk, seed ensemble | `risk_v1`, `risk_v1_s14`, `risk_v1_s15` | 3 | 5.0 GB |
| Risk, entailment arm | `risk_nli`, `risk_nli_s2` | 2 | 3.3 GB |
| Risk + factors, tri-head | `artifacts/friend_ckpt/model_fold0..4.pt` | 5 | 2.4 GB |
| Evidence taggers | `evidence_{ep9,lr2,s14,s15,s16,s17,v1}_rc` | 7 | 11.6 GB |
| Factor pool, encoder arms | `factors_{arm}_cv_f0` to `_cv_f4`, 7 arms | 35 | 61 GB |
| Factor pool, LoRA arms | `factors_{arm}_f0` to `_f4`, 3 arms | 15 | 2 GB |

The 11 factor arms are `mnli dmnli cmine prot d4 qnli mnli7 lnli ltail
cbnsynapi friendfac`, and they do **not** all follow one naming convention:

| Arms | Layout |
|---|---|
| `mnli dmnli cmine prot d4 ltail cbnsynapi` | `runs/factors_{arm}_cv_f0` to `_cv_f4` |
| `qnli mnli7 lnli` | `runs/factors_{arm}_f0` to `_f4` |
| `friendfac` | no `runs/` directory; the tri-head in `artifacts/friend_ckpt/` |

Every arm needs all 5 folds. The dump script refuses a partial bag, because
4 of 5 checkpoints is a different model from the one that was measured.
`src/check_setup.py` knows all 3 conventions and is the reliable way to tell
whether a member is complete; `src/run_new_data.py` handles all 3 and is the
entry point for a run on new posts.

`src/fetch_weights.py` places everything correctly. If you move files by hand
instead, each run directory needs its `model_final.pt` (or `model.pt`) and its
`run_config.json`, under `runs/` with exactly the names above.

Four pretrained models are pulled from the Hub at first use rather than
shipped. `Qwen2.5-72B-Instruct` (about **136 GB**) is used only by the routing
step, the slowest thing in the pipeline by a wide margin - see *The slow stage
- budget for it, do not skip it*. The other three are the base models the LoRA factor arms adapt, and two
of them are **gated**: accept their terms while signed in, or those arms
cannot load.

| Arm | Base model | Gated |
|---|---|---|
| `lnli` | `meta-llama/Llama-3.1-8B-Instruct` | **yes** |
| `mnli7` | `mistralai/Mistral-7B-Instruct-v0.3` | **yes** |
| `qnli` | `Qwen/Qwen2.5-14B-Instruct` | no |

`src/fetch_weights.py` prints the links for the gated two, and rewrites each
arm's recorded base path - an absolute path on the machine that trained it -
to the Hub id above.

The **gated** `mental/mental-roberta-base` is **not** needed: a copy of its
config and tokenizer ships in `artifacts/friend_ckpt/encoder/`.

### The whole download budget

Our Hugging Face repository holds **only the checkpoints we trained**. The
pretrained models underneath them belong to Meta, Mistral, Alibaba and
Microsoft and are not ours to redistribute, so they come from their own
repositories. Plan for roughly **280 GB** in total, or **145 GB** if you skip
the routing stage.

| What | From | Size | Needed for |
|---|---|---|---|
| Our 67 checkpoints | `ShlokShlok123/nidhi-weights` (gated, instant approval) | **86.6 GB** | everything |
| `Qwen/Qwen2.5-72B-Instruct` | Hub, ungated | **~136 GB** | routing only |
| `Qwen/Qwen2.5-14B-Instruct` | Hub, ungated | ~28 GB | `qnli` factor arm |
| `meta-llama/Llama-3.1-8B-Instruct` | Hub, **gated** | ~16 GB | `lnli` factor arm |
| `mistralai/Mistral-7B-Instruct-v0.3` | Hub, **gated** | ~15 GB | `mnli7` factor arm |
| `microsoft/deberta-v3-large` | Hub, ungated | ~0.9 GB | risk, evidence, 7 factor arms |

Sizes other than the first two are approximate. Only `src/fetch_weights.py`
needs running by hand; the rest download on first use, provided the machine has
internet. On an air-gapped compute node, fetch them on a login node first and
point `HF_HOME` at that cache, then pass `--offline`.

## Running on unseen data
One command scores a new test set end to end:

```bash
python src/fetch_weights.py --repo ShlokShlok123/nidhi-weights   # once, 86.6 GB
python src/check_setup.py --test your_test.csv              # must print READY
python src/run_new_data.py --test your_test.csv --out submissions/NIDHI_new.csv
```

Input is a CSV with `data/test.csv`'s header - `row_id,anon_user_id,post_id,post`.
Output is `row_id,risk_level,evidence,factors`.

This path has been run end to end on one H100, including the routing stage:
all 20 stages completed and produced a well-formed CSV. Everything below that
says "refuses" or "is checked" was observed doing so, not just written down.

### What a correct rerun looks like

We reran the whole chain on our own `data/test.csv` and compared the result
against the submitted predictions. **It does not come out identical, and it is
not supposed to.** Expect differences of about this size:

| Column | Rows differing, of 378 |
|---|---|
| `risk_level` | 15 |
| `evidence` | 9 |
| `factors` | 46 |

Factors came out at **3.19 per post against the submitted 3.16**.

Two causes account for all of it, both documented below. That run used
`--skip-route`, and the missing attempt promotions show up exactly as a
6-post shift: `behavior` 101 against 95, `attempt` 18 against 24, with
`indicator` and `ideation` identical to the row. The rest comes from the
`friendfac` tri-head, which is both a factor arm and the second risk member,
and does not rebuild bit-exactly.

So a rerun that lands within a few percent of these columns is working
correctly. A rerun that produces an empty column, or factor counts far from
3, is not - and every stage is built to refuse rather than reach that state.
If you want an exact reproduction of the submitted file instead, that is
*Reproduce the submission*, which is byte-identical and needs no GPU.

Before committing a day of GPU time, `--dry-run` prints all 20 stages in order
with the exact arguments each will receive, and runs nothing:

```bash
python src/run_new_data.py --test your_test.csv --dry-run
```

**Nothing is refitted on the new rows.** Every threshold, weight, member and
seed is the champion's; only the posts change. The decode thresholds come from
the shipped `*_cv_oof_f*.npz` and `full_train.csv`, which are keyed to the
*training* rows and so stay valid for any test set.

This needs a GPU. It is 62 model loads over your posts, and on CPU it will not
finish in reasonable time. Budget roughly a day, almost all of it in the
routing stage described below.

### What the script does, and why it is a script

`src/run_new_data.py` only calls scripts already in this package. It exists
because four details are easy to get wrong and fail *quietly* - producing a
plausible but wrong column instead of an error:

1. `FACTOR_DEFS_JSON`, `FACTOR_ASPECTS_JSON`, `PROTECTIVE_TEMPLATE`,
   `CONFUSABLE_JSON` and `FACTOR_SUBSET` are read at **inference** time.
   Inherited from a login shell they silently change what the factor arms
   predict. The script clears all five.
2. The `prot` arm was trained with `PROTECTIVE_TEMPLATE=1` and every other arm
   with it off. The wrong setting yields plausible garbage.
3. The 11 factor arms come in **three** families with three loaders and three
   on-disk layouts: 7 encoder arms (`runs/factors_{arm}_cv_f0..f4`), 3 LoRA
   arms (`runs/factors_{arm}_f0..f4`), and `friendfac` (vendored tri-head). One
   loader cannot read another's checkpoints.
4. The evidence taggers must be conditioned on the risk column *this run*
   produces. Conditioning them on a stored submission binds them to our 378
   `row_id`s, which do not exist in your data.

The stages, in order: the MentalRoBERTa tri-head (which yields both the second
risk member and the `friendfac` arm in one pass) → the 3-seed risk ensemble →
the risk entailment arm → chain-of-thought routing → the risk column → the 7
evidence taggers → the 11 factor arms → the cold-start factor decode →
assembly. Each stage refuses loudly if its inputs do not cover the test set.

`scripts/cold_run_drill.sh` runs the same chain on held-out rows and additionally
scores it against gold; it is how the four traps above were found.

### The slow stage - budget for it, do not skip it

**This stage is part of the champion. Skipping it does not reproduce our
system.** Please plan for the time rather than dropping it.

Chain-of-thought routing uses `Qwen2.5-72B-Instruct` as-is - prompted, not
fine-tuned - and takes roughly **20 hours on one GPU**, nearly the whole
runtime. It is slow because the model does not just emit a label: it writes out
its reasoning for every post, which is far more compute than a single forward
pass.

What it buys is one decision, on the label our other models are worst at. The
four risk levels are indicator → ideation → behavior → **attempt**, and attempt
is both the rarest and the most severe: **24 of the 378 posts, about 6%**. A
model trained on all four at once sees very little of it. So a much larger
model reads each post independently, and its opinion is used for exactly one
rule:

> if the routing model says `attempt`, and our own model did not say
> `indicator`, the label becomes `attempt`.

That is the whole rule. It can only promote a post to `attempt`; it can never
demote one, and it cannot touch anything already labelled `indicator`. Risk is
0.4 of the composite, so recovering the attempt class matters out of proportion
to its frequency.

**Point it at a local copy of the model.** `Qwen2.5-72B-Instruct` is about
**136 GB** on disk, and it is loaded in 4-bit, which alone takes roughly 1.5
hours before any post is scored. The default is the Hub id, which downloads
the whole thing; on a compute node without internet it fails outright. Once
you have a copy, name it:

```bash
python src/run_new_data.py --test your_test.csv \
    --qwen /path/to/Qwen2.5-72B-Instruct --offline
```

`--offline` resolves every pretrained model from the local Hugging Face cache
and never touches the network, which is what you want on an air-gapped compute
node. `QWEN=/path/to/model` works instead of the flag.

If you genuinely cannot spare the GPU time, the run does not fail:

```bash
python src/run_new_data.py --test your_test.csv --skip-route
```

The other stages are unaffected and a valid CSV is still produced, but posts
that should be `attempt` stay at `behavior` or `ideation` and the risk score
drops. Treat that as a degraded run, not as our system. The skip is announced
at the start and again at the end, and is never applied silently.

### Known limitations of a rebuild

Stated plainly, because they are measurable and you would otherwise find them
yourselves:

- **The `friendfac` factor arm does not rebuild bit-exactly.** Its five
  checkpoints are only four distinct models, and the shipped arm was a
  *weighted* bag; a rebuild averages them uniformly. Measured difference
  against the shipped dump is `max 0.0977, mean 0.0139` on our test set. It is
  1 of 11 arms and the effect on the factor column is small, but it is not zero.
- **Internet is needed on first use.** The risk, evidence and encoder factor
  arms build their tokenizer and config from `microsoft/deberta-v3-large`
  (public and ungated); only the fine-tuned weights come from `runs/`. On an
  air-gapped node, pre-populate the Hugging Face cache and set `HF_HOME`.
  The tri-head is the exception: it reads a vendored encoder copy from
  `artifacts/friend_ckpt/encoder/`, so the **gated** `mental/mental-roberta-base`
  is never contacted.
- **The three LoRA arms need their base model.** `src/fetch_weights.py`
  rewrites each `run_config.json` from the cluster-absolute path it was trained
  with to a Hub id, so the base downloads by name.

- **Score the whole evaluation set in one file.** Two mechanisms make a small
  batch behave differently from a large one, and they compound. The factor
  rescale is transductive, so each arm's scale is recomputed from the rows it
  is given. And the decode's zero-prediction floor lowers the threshold for
  any label that no row triggers - on a small batch, rare labels are absent by
  chance rather than because the model rejected them, so the floor fires and
  adds predictions. On a 20-row sample this produced 4.20 factors per post
  against the champion's 3.16; the effect shrinks as the batch grows.

One further property worth knowing:

- The risk models take a `[USER_POST k/n]` prefix, where `k` comes from the
  `post_id` column rather than file position, so predictions do not depend on
  how the input CSV is sorted.

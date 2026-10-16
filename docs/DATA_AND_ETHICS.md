# Data and ethics

The posts this system was built on are real crisis posts from a clinical
research dataset, annotated for suicide risk by clinicians. This page states what this
repository contains, what it deliberately does not, how the data was handled
during the work, and what the system must not be used for.

## What this repository contains

**No post text.** Not the test posts, not the training posts, not the evidence
spans the system quotes from posts, and not the model reasoning traces written
about posts. Those files exist and are needed for an exact rebuild, so they are
listed by name and SHA-256 hash in [`artifacts/MANIFEST.md`](../artifacts/MANIFEST.md)
and distributed only through an access-controlled channel to researchers who
have accepted the dataset's terms.

What is public is the code and purely numeric model output (probabilities,
thresholds and row identifiers), from which no post can be reconstructed.

The trained checkpoints are gated separately at
[`ShlokShlok123/nidhi-weights`](https://huggingface.co/ShlokShlok123/nidhi-weights).
They were fine-tuned on this corpus and can retain fragments of their training
text, so access is recorded rather than anonymous.

## How the data was handled

**The submitted system runs entirely locally.** Every model on the build path
and on the inference path for new data runs from local weights: DeBERTa-v3-large,
MentalRoBERTa, the entailment models, and Qwen2.5-72B-Instruct for routing. No
component of the submitted system sends text to an external service.

**Some exploratory experiments did.** During the audit, five experiments sent
post text to a commercial LLM API (DeepSeek). None of them is on any build path,
and none was used by the submitted system:

| Script | What it sent | Experiment |
|---|---|---|
| `src/llm_rare_factors.py` | posts | LLM labeling of rare factors |
| `src/derive_definitions.py` | labelled example posts | factor definitions derived from annotations |
| `src/generate_concept_labels.py` | posts | concept labels for the concept-bottleneck comparison |
| `src/evidence_llm_reranker.py` | posts with candidate spans | LLM as evidence reranker |
| `src/evidence_llm_score.py` | posts with candidate spans | graded variant of the reranker |

One further script, `src/gen_rare_factors_api.py`, called the same API to
generate synthetic posts, but its prompts contained only the published factor
definitions, never a dataset post. The `cbnsynapi` factor arm in the submitted
system was warm-started from that synthetic data.

The remaining files that mention the API (`deepseek_client.py`,
`style_match_synthetic.py`, `why_defs_failed.py`, `analyze_concept_bottleneck*.py`,
`train_concept_only_mlp.py`, `evidence_llm_propose.py`) make no API calls: they
are the client itself, or they read the outputs of the experiments above, or
they run a local model.

Anyone reusing these exploratory scripts on this or similar data should first
check that the dataset's terms permit sending its text to a third-party service.

## What this system is not

The system orders a queue; it does not make a clinical judgment, and deciding
who is looked at first is itself consequential triage. It was studied as an
offline research problem, and nothing here establishes that it is safe or
effective for operational use.

- Its outputs are not diagnoses and not clinical risk assessments.
- No output should trigger any action concerning a person without a human
  reading the post.
- A missed high-risk post and a false alarm carry different costs to different
  people, and a single aggregate score establishes an acceptable rate for
  neither.
- Evidence spans exist so that a reviewer can check the system against the post
  rather than defer to it. They do not by themselves explain risk.

## Foreseeable misuse

A system like this could be used to surveil rather than to help, applied to
people who have not asked for attention; that the posts are public does not
make that acceptable. A tool that appears to explain itself also invites its
reviewer to stop checking. Both argue for keeping such systems inside services
people have chosen to use, with an accountable person in the loop.

## Acknowledgement

The judgments this work depends on were made by clinicians reading distressing
material at length. The benefit of that work is entirely ours.

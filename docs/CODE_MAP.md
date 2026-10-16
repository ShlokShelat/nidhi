# Code map

Every source file, grouped by role, with the opening of its own docstring. This is a research codebase: 10 modules rebuild the submission, about 15 score new data, and the rest are the experiments behind the paper's 31 comparisons, kept so that each claim can be traced to the code that measured it (see `AUDIT_MAP.md`).

Five files marked **[API: posts]** sent post text to a commercial LLM API (DeepSeek) during exploration; one marked **[API: definitions only]** sent only the published factor definitions. None of them is on any build path, and none was used by the submitted system. See `DATA_AND_ETHICS.md`.

All paths are relative to the repository root. A handful of risk and factor experiments (`champion_oof.py`, `champion_plus_qwen.py`, `risk_ensemble_test.py`, `factor_risk_rules.py`, `make_hard_weights.py`, `risk_llm_complementarity.py`, `factor_free_diagnostics.py`, `factor_label_diagnosis.py`, `selective_override.py`, `compare_vs_champion_risk.py`, and `scripts/build_champion_qwen.sh`, `scripts/nli_risk_5fold.sh`) default to inputs under `v2_rebuild/`: the run directories and artifacts of an earlier-generation codebase. That tree is not published and is not needed for the build path or for scoring new data; pass these scripts' path arguments explicitly to point them elsewhere.

| Group | Files |
|---|---|
| [Build path (reproduces the submission)](#build-path-reproduces-the-submission) | 10 |
| [Scoring new data (inference)](#scoring-new-data-inference) | 14 |
| [Training](#training) | 23 |
| [Risk experiments](#risk-experiments) | 78 |
| [Evidence experiments](#evidence-experiments) | 90 |
| [Factor experiments](#factor-experiments) | 121 |
| [Shared diagnostics and utilities](#shared-diagnostics-and-utilities) | 40 |
| [Teammate tri-head pipeline (src/friend_pipeline)](#teammate-tri-head-pipeline-srcfriendpipeline) | 10 |

## Build path (reproduces the submission)

| File | Purpose |
|---|---|
| `src/apply_rules_to_test.py` | Builds the final risk prediction file: takes an EXISTING risk_blend CSV of test-set predictions from the equal-weight risk ensemble and applies the |
| `src/build_submission.py` | Rebuild all three submission columns from their component artefacts in one process, and check the result against the shipped submission. |
| `src/candidate_patterns.py` | CANDIDATE symbolic rules under evaluation -- NOT deployed. Imported by apply_rules_to_test.py into ALL_DEFS so that every consumer (ablation |
| `src/champion_risk_inject.py` | Rebuild the deployed risk column from its component artefacts, optionally inject a correction ahead of the argmax, and report exactly which labels move. |
| `src/data_audit.py` | Data audit + leakage-safe splitting. Usage: |
| `src/evidence_vote_test.py` | Build the TEST evidence column by voting across the evidence tagger runs: a span is emitted only when at least --k of the runs propose it. |
| `src/factor_threshold_stabilise.py` | Estimate the 24 per-label decision thresholds of the factor pool, compare four estimators of them under cross-fitting, and optionally build the factors column |
| `src/scoring.py` | Local reimplementation of the IEEE BigData Cup 2026 scoring. Composite = 0.4 * weighted_F1(risk_level) |
| `src/text_norm.py` | Shared text canonicalization. Applied ONCE to posts and evidence at load time; every downstream component (training, prediction, span extraction, |
| `src/train_evidence.py` | Evidence span extraction: BIO token tagger + hard grounding verifier. Design choice: extractive tagging, not generation. A tagger's output is a |

## Scoring new data (inference)

| File | Purpose |
|---|---|
| `src/_new_run_helpers.py` | Small stages of the new-data run that are not scripts of their own. Invoked by src/run_new_data.py; not intended to be run by hand. Each |
| `src/build_route_attempt.py` | Ship `route_attempt`: take the CoT-72B's ATTEMPT calls, keep the champion's risk label everywhere else. |
| `src/check_setup.py` | Preflight check. Run this BEFORE anything else. Reports exactly which model checkpoints and data files are present and which |
| `src/dump_nli_probs.py` | Saves the risk-NLI models' probabilities for ANY split (val or test), once. blend_risk_saved.py recomputes these from the checkpoints on every run and |
| `src/ensemble_risk.py` | Cross-model risk ensemble. Averages the DeBERTa 3-seed risk probabilities with an externally supplied |
| `src/fetch_weights.py` | Download the model checkpoints needed to score a NEW test set. The submission package carries code plus the ~4 MB of model outputs needed to |
| `src/friend_factor_predict.py` | Regenerate the `friendfac` factor arm INSIDE THIS REPO, on any test set. WHY THIS EXISTS. `friendfac` is the single best member of the factor pool |
| `src/predict.py` | Generate a competition submission CSV from the three trained checkpoints. Usage (on a GPU node): |
| `src/predict_factors_llm_nli.py` | Test-set probabilities for the LoRA-tuned Qwen factor arm (`qnli`). WHY THIS EXISTS. `predict_factors_nli.py` instantiates `NLIScorer` and loads a |
| `src/predict_factors_multilabel.py` | Dump probabilities from trained multi-label factor arms (src/train_factors_multilabel.py) in the exact .npz format the pool tools |
| `src/predict_factors_nli.py` | Predict factors on the test set with the trained definition-grounded NLI model. Produces a CSV (row_id, factors) consumable by |
| `src/predict_risk_nli.py` | Predict risk levels on the test set with the trained definition-grounded risk-NLI model. |
| `src/run_new_data.py` | Score a NEW test set end to end, in the champion configuration. python src/run_new_data.py --test your_test.csv --out submissions/NIDHI_new.csv |
| `src/upload_weights.py` |  |

## Training

| File | Purpose |
|---|---|
| `src/train_concept_head.py` | STAGE 1 of two-stage training. Trains encoder + concept_head ONLY, against the concept BCE loss (with per-concept pos_weight). There is |
| `src/train_concept_only_mlp.py` | DIAGNOSTIC EXPERIMENT -- NOT A DEPLOYABLE MODEL. Estimates the CEILING of what the 9 concept labels alone can explain |
| `src/train_evidence_nli.py` | Definition-grounded evidence extraction (NLI-style clause scoring). Every prior evidence attempt used BIO token-tagging, which labels tokens |
| `src/train_factors.py` | Factor classifier: 24 independent sigmoid outputs + per-label threshold tuning. Why thresholds matter here: the metric is MACRO F1, so the rarest label |
| `src/train_factors_cv.py` | User-level k-fold CV for the factor classifier. Why this exists: a single 20% val split leaves rare labels with 0-2 |
| `src/train_factors_llm_nli.py` | Qwen as the factor cross-encoder -- with the definitions, which the earlier LLM attempt threw away. |
| `src/train_factors_lora.py` | LoRA / QLoRA fine-tuning of a decoder LLM for the 24 FACTOR labels. WHY, AND WHAT IT IS AIMED AT. Three CPU diagnostics closed every |
| `src/train_factors_multilabel.py` | Multi-label factor classifier -- the FORMULATION arm. WHAT IT IS. One forward pass per post emitting all 24 factor logits from a |
| `src/train_factors_nli.py` | Definition-aware factor classification (NLI / entailment style). Per-label factor F1 is strongly correlated (r=0.806) with training-example |
| `src/train_factors_nli_debug.py` | Definition-aware factor classification (NLI / entailment style). Motivation: our per-label analysis shows factor F1 is r=0.806 correlated |
| `src/train_factors_nli_guarded.py` | Definition-aware factor classification (NLI / entailment style). Motivation: our per-label analysis shows factor F1 is r=0.806 correlated |
| `src/train_factors_nli_targeted.py` | Definition-aware factor classification (NLI / entailment style). Motivation: our per-label analysis shows factor F1 is r=0.806 correlated |
| `src/train_factors_stack.py` | Second-stage label-refinement model for factors. Motivation: the first-stage factor model treats 24 labels independently |
| `src/train_risk.py` | Risk-level classifier: encoder + dual head (categorical softmax + ordinal CORN). Rationale (from 2024/2025 editions of this cup): |
| `src/train_risk_boundary.py` | A dedicated binary model for the single costliest decision in the risk chain: does this post describe a PLAN or a SELF-HARM ACT? |
| `src/train_risk_boundary_sample.py` | Risk-level classifier: encoder + dual head (categorical softmax + ordinal CORN). IDENTICAL to train_risk.py in every way EXCEPT: |
| `src/train_risk_concepts.py` | Risk-level classifier with AUXILIARY CONCEPT SUPERVISION (Option 1: shared encoder, separate heads, no bottleneck). |
| `src/train_risk_concepts_bottleneck.py` | Risk-level classifier with CONCEPT-AUGMENTED FUSION, now with: (A) per-concept positive-class weighting on the concept BCE loss, |
| `src/train_risk_focal.py` | Risk-level classifier — IDENTICAL to train_risk.py except the categorical head's CrossEntropyLoss is replaced with FOCAL LOSS |
| `src/train_risk_lora.py` | LoRA fine-tuning of a Qwen decoder LLM for RISK classification. This is an architecture change rather than a recombination of existing |
| `src/train_risk_nli.py` | Definition-grounded entailment for RISK classification. The same reformulation that took factor macro-F1 from 0.417 to 0.604 and |
| `src/train_risk_rationale.py` | Risk classifier with RATIONALE SUPERVISION: teach the encoder WHERE to look. WHY. src/risk_head_diagnosis.py showed the risk errors are CONFIDENTLY wrong -- |
| `src/train_span_reranker.py` | A CROSS-ENCODER THAT READS THE SPAN, INSTEAD OF RE-WEIGHTING VOTE COUNTS. The selection oracle on the deployed pool is macro 0.8824 -> 0.9493: the |

## Risk experiments

| File | Purpose |
|---|---|
| `src/apply_span_rule.py` | Promotes a risk label from behavior to attempt when a completed-act pattern appears in the EVIDENCE CLAUSE. Zero parameters, and no probabilities needed. |
| `src/assemble_champion_qwen.py` | Build the test risk column: the base ensemble plus Qwen as an extra member. ensembled = artifacts/ensembled_risk.csv (already on disk: |
| `src/average_risk_nli.py` | Average two or more risk-NLI out-of-fold probability files into one, so a multi-architecture NLI member can be fed to src/champion_oof.py --nli. |
| `src/blend_risk_nli.py` | Blend risk-NLI softmax probabilities INTO the existing risk ensemble (augment, don't replace). |
| `src/blend_risk_saved.py` | Blend risk-NLI into the risk ensemble. The ensemble side is read from SAVED probability files rather than |
| `src/build_risk_majority.py` | Build the 3-way MAJORITY risk vote: champion + CoT-72B + LoRA-Llama-8B. THE RESULT THIS DEPLOYS. `risk_bag_eval.py`, measured on 1635 out-of-fold |
| `src/build_unanimous_route.py` | Ship the one rule that met all four criteria: overrule the champion to a SINGLE class, only when EVERY listed member agrees. |
| `src/champion_factor_redecide.py` | Keep the model that scores S2 = 0.6301. Change only how it decides. THE SUSPECTED BUG. predict_factors_nli.py averages the two runs' PROBABILITIES |
| `src/champion_oof.py` | Rebuild a champion-like risk ensemble over ~1590 OUT-OF-FOLD posts, so that every subsequent decision is made on roughly 4.5x the data of the 372-post |
| `src/champion_oof_twin.py` | BUILD THE VALIDATION-SIDE TWIN OF THE CHAMPION'S RISK VECTOR. `champion_risk_inject.py --verify` now proves the TEST vector: |
| `src/champion_plus_qwen.py` | Rebuild the CHAMPION risk ensemble on val, verify it, then ask whether Qwen adds anything as an extra member. |
| `src/combined_rule_ablation.py` | Companion to rule_ablation.py. Independent per-rule evaluation can hide overlap: two rules that each look good alone might correct the |
| `src/compare_risk_arms.py` | Checks whether a hard-mining or class-weighting arm reduces the misses it targets. |
| `src/compare_vs_champion_risk.py` | Compares a candidate risk arm against the risk system it would ACTUALLY be replacing. |
| `src/compute_boundary_weights.py` | Emits a per-example difficulty weight for risk training: difficulty-aware sampling only, with no loss or architecture changes. |
| `src/indicator_stack.py` | Fit P(gold evidence is empty) from risk logits PLUS evidence-tagger features, and use it to decide indicator suppression. |
| `src/indicator_stack_ship.py` | Apply the indicator stack fitted by `src/indicator_stack.py` to an existing submission and write the resulting file. |
| `src/joint_indicator_decode.py` | The risk prediction is scored twice by the composite objective, but is optimized as if it were scored once. |
| `src/make_hard_weights.py` | Build per-post training weights that emphasise the risk errors that are most plausibly learnable. |
| `src/mine_attempt_patterns.py` | Extend `completed_act` to cover the attempt phrasings it misses. THE MEASURED WEAKNESS. `attempt` is the worst cell in the risk system: |
| `src/patch_submission_risk.py` | Builds a new submission that differs from a base submission in the risk_level column ONLY, carrying evidence and factors across untouched. |
| `src/retest_routes_on_twin.py` | Re-judges LLM routing variants on a matched base that includes the full symbolic layer, rather than on a rules-free proxy. |
| `src/retest_rules_on_twin.py` | RE-JUDGE EVERY CANDIDATE SYMBOLIC RULE ON THE MATCHED BASE. The layer decomposition from `champion_oof_twin.py` + `retest_routes_on_twin.py` |
| `src/risk_argmax_audit.py` | Does the recorded risk label agree with its own probabilities? WHY THIS EXISTS. Reading the confident `behavior -> ideation` misses turned up |
| `src/risk_arm_transitions.py` | WHERE did an arm's predictions move, and was the move good or bad? WHY THIS EXISTS. The gaur arm lost weighted-F1 overall, a scalar that alone |
| `src/risk_attempt_boundary.py` | How did the annotators separate BEHAVIOR from ATTEMPT? WHY ASK. The per-class probability table on 1590 OOF posts shows the chain is |
| `src/risk_bag_eval.py` | Evaluate a MULTI-FAMILY generative risk bag against the encoder chain. WHY A MULTI-FAMILY BAG RATHER THAN AN INCREMENTAL CHANGE. Closing a |
| `src/risk_boundary_errors.py` | Why does behavior -> ideation resist a fix? Read the actual posts. The measurement says it is 102 of 343 out-of-fold errors (30%), behavior recall |
| `src/risk_boundary_examples.py` | WHAT ACTUALLY SEPARATES ideation FROM behavior? ASK THE ANNOTATOR. Ideation explicit suicidal expressions but NO suicidal plan |
| `src/risk_candidate_check.py` | Pre-submission sanity checks for a candidate risk prediction file when NO honest validation score exists for it. |
| `src/risk_cascade_decode.py` | THE FOUR RISK CLASSES ARE NOT FOUR CATEGORIES. THEY ARE THREE NESTED QUESTIONS. Read the organizers' definitions: |
| `src/risk_class_routing.py` | Route per class: trust the LLM where it is strong, the encoder where it is. WHY THE IDEA IS SOUND. On 1635 OOF posts the CoT 72B and the encoder chain are |
| `src/risk_conservative_override.py` | Overrule the champion only when the challengers are UNANIMOUS -- and validate against an OOF analogue of the DEPLOYED champion, not a weaker surrogate. |
| `src/risk_cot_probe.py` | Can a local instruct model REASON its way to the risk label without being told the answer? |
| `src/risk_crossmodal_cells.py` | DOES THE FACTOR MODEL SEE THE PLAN THAT THE RISK MODEL MISSED? `risk_error_anatomy.py` found the dominant confusion cell: |
| `src/risk_crossmodal_stack.py` | RISK CARRIES 0.4 AND IS PREDICTED FROM ONE MODEL'S OPINION OF THE TEXT. indicator_stack.py established that the EVIDENCE TAGGER knows things the risk |
| `src/risk_disagreement_analysis.py` | Instead of asking "what does the final prediction say", asks "how much did the 3 risk models agree before averaging". Per team's proposal: |
| `src/risk_ensemble_test.py` | Measures whether ADDING a Qwen probability source to a risk ensemble helps, as opposed to replacing an existing member with it. |
| `src/risk_ensemble_weight_sweep_v2.py` | Coarse, hand-picked weight sweep over the risk ensemble members: 8 combinations rather than a full grid. A small, fixed candidate set is far |
| `src/risk_error_anatomy.py` | Characterises WHERE a risk model's out-of-fold errors fall, so that effort goes to defects a fix can actually reach. |
| `src/risk_error_dump.py` | Dumps the risk model's errors as readable posts, grouped by confusion cell, so that the ceiling question is settled by inspection rather than inference. |
| `src/risk_error_stats.py` | What exactly is the risk chain getting wrong, per class? Scored on `artifacts/champion_oof.csv`, which carries the DEPLOYED chain's |
| `src/risk_evidence_coupling.py` | How much of the EVIDENCE score is decided by the RISK model? WHY THE TWO HEADS ARE NOT INDEPENDENT. The span taggers never see the risk |
| `src/risk_from_evidence.py` | THE RISK/EVIDENCE COUPLING, RUN BACKWARDS. Gold evidence is empty EXACTLY when gold risk is indicator (0.986 agreement, |
| `src/risk_head_diagnosis.py` | Dumps the CATEGORICAL and ORDINAL head probabilities of the risk model separately, and attributes the risk errors to one head or the other. |
| `src/risk_indicator_boundary.py` | Is the INDICATOR boundary lexical, and is that boundary being missed anyway? WHY THIS BOUNDARY MATTERS. risk_evidence_coupling.py showed the evidence |
| `src/risk_llm_complementarity.py` | Does the Qwen risk model fix posts the encoder chain gets wrong? WHY THIS IS THE ONLY QUESTION LEFT IN THE RISK-LLM LANE. Fully replacing the |
| `src/risk_llm_member_build.py` | Add the Qwen risk model as a LOW-WEIGHT, COUNT-MATCHED MEMBER of the risk chain -- the one risk-LLM configuration that was never scored. |
| `src/risk_lora_member.py` | Is the LoRA'd Qwen2.5-7B a useful RISK BLEND MEMBER? Free, CPU, on disk already. WHY THIS SUDDENLY MATTERS. `runs/risk_lora7b/run_config.json` records |
| `src/risk_member_forward.py` | SIXTY-FIVE TRAINED RISK MODELS. THE BLEND USES THREE OF THEM. The champion's encoder term is `mean(risk_v1, risk_v1_s14, risk_v1_s15)`. |
| `src/risk_member_loo.py` | Audits the risk ensemble's membership. Risk carries a weight of 0.4 in the composite score, more than any other component. |
| `src/risk_member_search.py` | Adds a genuinely different model to the risk blend, judged on a twin that reproduces the deployed model. |
| `src/risk_method_coverage.py` | IF METHOD WORDS ARE EXCLUSIVE TO behavior, WHY CAN WE ONLY FIX 14 OF 89? `risk_boundary_examples.py` counted method words inside GOLD EVIDENCE SPANS: |
| `src/risk_nli_pool_oof.py` | Pool the 5 fold models of a risk-NLI arm into an honest OOF prediction set. WHY THIS EXISTS. scripts/risk_nli_gaur.sh ended its comparison by reading |
| `src/risk_nli_weight_sweep.py` | Sweeps the NLI blend weight, a single parameter set once and otherwise never revisited: |
| `src/risk_nofriend_joint.py` | Dropping the friend member moves risk AND evidence; both must be measured together. |
| `src/risk_oof.py` | User-level k-fold OOF risk prediction. Needed for risk-conditioned factors: if the factor model were conditioned on |
| `src/risk_plan_rule.py` | THE behavior CRITERION IS METHOD-AND-TIMELINE, NOT SELF-HARM. `risk_boundary_examples.py` counted the words annotators put inside gold |
| `src/risk_prior_decode.py` | THE FACTOR LENS, POINTED AT RISK. Factors moved 0.6519 -> 0.6774 on nothing but the decode: each label was |
| `src/risk_probe_analyze.py` | Analyzes a signal in the evidence tagger's confidence that runs opposite to the naive expectation. |
| `src/risk_recal_on_twin.py` | Tests whether a linear recalibration of the risk model's own four logits generalizes, evaluated on a twin that reproduces the deployed model rather |
| `src/risk_route_composite.py` | Score risk-routing rules on the COMPOSITE, not on risk alone. THE OVERSIGHT THIS FIXES. `risk_class_routing.py` judged every rule by risk |
| `src/risk_rule_ablation.py` | Checks whether the symbolic risk rules help on held-out posts, or whether they were fitted to the public evaluation slice. |
| `src/risk_rule_ship.py` | Keeps the symbolic rule that helps and drops the one that hurts. risk_rule_ablation.py decomposed the symbolic layer on 1590 OOF posts. It |
| `src/risk_selfharm_rule.py` | ONGOING SELF-HARM IS THE BEHAVIOR CRITERION, AND NO EXISTING SIGNAL DETECTS IT. `risk_error_anatomy.py`: the dominant confusion cell is |
| `src/risk_span_rules.py` | SCOPE THE COMPLETED-ACT RULE TO THE EVIDENCE CLAUSE, NOT THE WHOLE POST. `symbolic_on_spans.py` measured, on 1590 OOF posts, the SAME `DONE` regex |
| `src/risk_stack_ship.py` | Recalibrates the risk head; only the risk column changes. Restacking the risk model's own logits (`src/risk_crossmodal_stack.py`, |
| `src/risk_symbolic_fences.py` | Apply symbolic fences AFTER the model instead of inside the hypothesis text. Editing hypothesis definitions is ZERO-SUM: the model scores K hypotheses and |
| `src/risk_tagger_probe.py` | ASK THE EVIDENCE TAGGER TO VOTE ON RISK, BY CONDITIONING IT ON ALL FOUR LEVELS. THE PIPELINE RUNS ONE WAY: |
| `src/risk_tagger_probe_test.py` | Build the test-side severity dump the risk-severity nudge consumes. THE KNOB. `champion_risk_inject.py --severity <csv> --w <x>` tilts the risk |
| `src/risk_vs_factor_evidence.py` | IS THE EVIDENCE THAT EXPLAINS RISK THE SAME EVIDENCE THAT IDENTIFIES FACTORS? The corpus supplies one set of gold evidence spans, and those spans were |
| `src/risk_wnli_joint.py` | Measure a change in the risk NLI blend weight on RISK AND EVIDENCE JOINTLY. The NLI weight `w_nli` is applied by the risk blend, and `rebuild_and_apply_ |
| `src/rule_ablation.py` | Per team's methodology: doesn't measure gold-label precision on raw text (that only tells you if a pattern correlates with a label, not |
| `src/rule_ablation_v2.py` | Stage 2 of the symbolic-discovery pipeline, rebuilt around three changes to the original rule_ablation.py: |
| `src/scope_transfer_study.py` | DOES SCOPING A SYMBOLIC RULE TO THE EVIDENCE CLAUSE GENERALISE ACROSS RULES? A symbolic rule normally reads the whole post. The evidence tagger exists to |
| `src/selective_override.py` | Can Qwen ADD to the champion rather than replace it? WHERE THIS COMES FROM. compare_vs_champion_risk.py settled the replacement |
| `src/symbolic_on_spans.py` | THE RULES SEARCH THE WHOLE POST INSTEAD OF THE EVIDENCE SPAN ALREADY LOCATED. Every symbolic rule in `risk_symbolic_fences.py` fires on the WHOLE POST: |
| `src/validate_rule_triggers.py` | Before writing ANY post-processing rule, check it against the FULL training set (not just the 44 hand-picked P1 cases where the ensemble |

## Evidence experiments

| File | Purpose |
|---|---|
| `src/apply_span_reranker.py` | Scores the deployed candidate pool with the span re-ranker and compares re-ranked selection against the `vote>=2` baseline. |
| `src/build_reranker_pairs_oof.py` | Training pairs for the span selector, built from REAL out-of-fold tagger candidates instead of synthetic negatives. |
| `src/canonicalize_evidence.py` | Fold submitted evidence spans into the same text space the annotators used. THE BUG. src/evidence_dump_all.py reads data/test.csv RAW, while the training |
| `src/cap_diff_check.py` | SEPARATE COSMETIC REORDERING FROM A REAL SPAN-SET CHANGE. `evidence_vote_test.py --cap N` sorts the kept spans by agreement before |
| `src/conditioning_ablation.py` | DOES CONDITIONING THE SPAN TAGGER ON THE RISK LEVEL ACTUALLY HELP? The evidence subtask is claimed to be conditionally dependent on the risk |
| `src/conf_column_check.py` | Checks whether the confidence gate is inert in the candidate dumps. Sweeping the per-run `evidence_min_conf` offset is only meaningful if the gate |
| `src/ensemble_evidence.py` | Cross-model evidence ensemble. Same lever that worked for risk and factors: combine two |
| `src/ensemble_evidence_seeds.py` | Evidence seed ensemble — confidence averaging across independently-seeded BIO taggers, with min_conf swept and validated on val_split gold. |
| `src/evidence_arm_swap_check.py` | IS THE CHAMPION SHIPPING THE WEAKER EVIDENCE ARM? The champion's evidence column is reproduced exactly by |
| `src/evidence_arms_honest.py` | RE-RANK EVERY EVIDENCE ARM UNDER THE METRIC AND THE REGIME ACTUALLY DEPLOYED. Every evidence arm in this project was chosen under TWO defects at once: |
| `src/evidence_backoff.py` | Evidence: fix the empty/non-empty decision (the real loss center). Diagnosis showed span MATCHING is fine (0.7485 on posts where both |
| `src/evidence_cap_sweep.py` | PER-POST SPAN CAP on the shipped RC7 decode. CPU, seconds. THE OBSERVATION. RC7 at k=2 emits **1.65 spans/post** while gold averages |
| `src/evidence_combo_gate.py` | Re-runs the evidence combination search AND the author-disjoint stability gate against the FRESH per-run predictions cached by evidence_vote_ensemble.py, |
| `src/evidence_cond4_detector.py` | Test whether the evidence taggers can flag posts where the predicted risk label is wrong. |
| `src/evidence_cond_pool.py` | Evaluate risk-conditioned span tagging over pooled out-of-fold predictions. Two measurement properties drive the design: |
| `src/evidence_cond_robust.py` | Score spans by CONDITIONING ROBUSTNESS: how many of the four risk-level conditionings propose the same span. |
| `src/evidence_conditioning_oracle.py` | Measure what conditioning the evidence tagger on the WRONG risk level costs. The pipeline is one-directional: risk -> evidence. The tagger is TOLD the |
| `src/evidence_conf_bands.py` | Test whether the deployed confidence cut discards candidate spans the metric would reward. |
| `src/evidence_conf_gate_sweep.py` | Sweep the candidate confidence gate jointly with the vote threshold k. WHY THE TWO KNOBS BELONG TOGETHER. Each deployed tagger contributes spans at |
| `src/evidence_conf_rule.py` | Compare span-selection rules built on confidence magnitude against the vote-count rule the decode currently uses. |
| `src/evidence_conf_vote.py` | The evidence vote counts taggers. It should weigh them. WHAT IS BEING THROWN AWAY. evidence_vote_oof.py filters each tagger's candidates |
| `src/evidence_conf_vote_val.py` | Test the confidence-weighted vote on the EIGHT TAGGERS THAT ACTUALLY SHIP. WHY THIS EXISTS RATHER THAN REUSING THE OOF RESULT. src/evidence_conf_vote.py |
| `src/evidence_decouple.py` | BREAK THE risk -> evidence DEPENDENCY WHERE RISK IS UNRELIABLE. THE ARCHITECTURAL DEFECT, measured and confirmed |
| `src/evidence_diagnose.py` | Evidence error diagnosis — where exactly is phrase-F1 lost? Evidence carries 0.3 of the composite (same as factors) and the val |
| `src/evidence_diagnosis.py` | Diagnose the evidence component. It has never been looked at. WHY IT IS NOW THE ONLY LIVE COMPONENT. |
| `src/evidence_dump_all.py` | One GPU pass that produces the two artefacts every downstream evidence experiment needs, so neither of them requires its own GPU job. |
| `src/evidence_ensemble_search.py` | Evidence carries 0.3 of the composite and the error decomposition (evidence_error_analysis.py) showed only ~4% of gold spans are boundary- |
| `src/evidence_enum_candidates.py` | CANDIDATE GENERATION FOR THE SPAN RERANKER -- from raw text, no model. WHY THIS FILE EXISTS. `evidence_tagger_rank.py` established that the seven |
| `src/evidence_enumerate_rank.py` | IF THE POOL IS SATURATED, HOW HARD IS THE RANKING? `evidence_generator_ceiling.py` established, on gold data alone, that a dumb |
| `src/evidence_error_analysis.py` | Applies the symbolic-discovery methodology to the EVIDENCE component, which carries the same 0.3 composite weight as factors and has never been analysed |
| `src/evidence_error_anatomy.py` | WHERE EVIDENCE POINTS ARE LOST, BROKEN DOWN BY CAUSE. The scorer matches a predicted span to a gold span iff ONE CONTAINS THE OTHER |
| `src/evidence_error_typology.py` | WHAT KIND of span error is responsible for the composite gap? `evidence_suppression_rule.py` proved all remaining evidence headroom sits in |
| `src/evidence_failure_anatomy.py` | WHY DOES A PREDICTED SPAN FAIL? THE SCORER'S RULE MAKES THIS GEOMETRIC. Evidence scores 0.777 overall, but that number is flattered: ~42% of posts are |
| `src/evidence_feature_ceiling.py` | IS THERE ANY SIGNAL LEFT IN THE FEATURES, OR IS THE VOTE RULE ALREADY OPTIMAL? Selection attempts have failed against a verified 0.8824 baseline: |
| `src/evidence_feature_selector.py` | A FEATURE-BASED candidate scorer for RC7. CPU, no GPU, no new model. THE MEASUREMENT THAT JUSTIFIES THIS (evidence_pool_oracle_compare.py): |
| `src/evidence_fm_decode.py` | PLUG-IN EXPECTED-F1 DECODING for evidence. Training-free, CPU, no new model. WHY THIS DIFFERS FROM A THRESHOLD-BASED EVIDENCE DECODE. |
| `src/evidence_gated_hedge.py` | HEDGE THE EVIDENCE CONDITIONING ONLY WHERE THE RISK LABEL IS LIKELY WRONG. `evidence_conditioning_oracle.py` confirmed the mechanism cleanly: |
| `src/evidence_generator_ceiling.py` | CAN A CANDIDATE GENERATOR WITH NO MODEL IN IT SATURATE EVIDENCE RECALL? THE BINDING CONSTRAINT, RESTATED. The candidate pool holds ~1.1 spans/post |
| `src/evidence_llm.py` | LLM evidence extraction — Qwen2.5-14B prompted as a psychiatrist to extract VERBATIM risk-evidence spans, grounded in the risk-level definitions. |
| `src/evidence_llm_confirm.py` | Use the LLM's spans to CONFIRM tagger candidates, not to propose new ones. WHY THE PROPOSER FRAMING FAILED. evidence_llm_propose.py measured, on 216 val |
| `src/evidence_llm_extract.py` | A GENERATIVE EVIDENCE EXTRACTOR. ALL EIGHT SHIPPED VOTERS ARE BIO TAGGERS. The evidence vote combines 8 checkpoints that are the SAME architecture -- a |
| `src/evidence_llm_propose.py` | Does an LLM, told the risk level, propose gold spans the taggers NEVER propose? WHY THIS IS NOT THE LANE THAT CLOSED. Attempts at evidence SELECTION have |
| `src/evidence_llm_reranker.py` **[API: posts]** | Uses an LLM as a SELECTOR over span candidates produced by the BIO taggers, not as an extractor. |
| `src/evidence_llm_score.py` **[API: posts]** | Graded variant of the LLM span selector. Asks the model to score each candidate 0-100 instead of making a binary keep/drop call, then chooses the |
| `src/evidence_loss_decompose.py` | Decompose the missing evidence macro-F1 into disjoint buckets. Macro phrase-F1 gives every post equal weight and scores an empty-vs-empty |
| `src/evidence_match_quality.py` | Report whether evidence points are earned by exact spans or by the scorer's containment slack. |
| `src/evidence_matchprob.py` | SELECT SPANS BY A CALIBRATED MATCH PROBABILITY, NOT BY A VOTE COUNT. Per-post phrase F1 is 2m / (n_pred + n_gold). Adding one more span with match |
| `src/evidence_mbr_decode.py` | SET-LEVEL MBR for evidence: enumerate subsets, score with the real metric. WHY A SET-LEVEL DECODER AND NOT ANOTHER PER-CANDIDATE RANKER. A large share of |
| `src/evidence_member_forward.py` | Test whether adding any unused trained tagger to the evidence vote helps. The deployed vote uses the risk-conditioned taggers, but |
| `src/evidence_member_loo.py` | Leave-one-out over the evidence tagger pool: does every member earn its place? The deployed decode votes k>=2 over eight `evidence_*_rc` runs. |
| `src/evidence_model_x_cond.py` | TWO INDEPENDENT AGREEMENT AXES: ACROSS MODELS, AND ACROSS QUESTIONS. The deployed decode keeps a span when >=2 of 7 taggers propose it, all |
| `src/evidence_pair_compare.py` | PAIRED AUTHOR-SPLIT TEST BETWEEN TWO VOTE OUTPUTS. Both configurations are supplied as already-built val prediction CSVs, so |
| `src/evidence_pool_analysis.py` | Diagnose what is missing from the evidence candidate POOL, and what would put it there. |
| `src/evidence_pool_ceiling.py` | Two falsifiers that decide whether the evidence POOL has anything left. A perfect selector can never exceed the gold-span recall of the candidate pool |
| `src/evidence_pool_oracle_compare.py` | WHERE IS THE EVIDENCE GAP -- selection, or candidates? CPU, seconds. THE UNASKED QUESTION. `evidence_true_oracle.py` measures the exact ceiling of |
| `src/evidence_rc_sweep.py` | SWEEP THE VOTE ON THE EIGHT TAGGERS ACTUALLY SHIPPED. The shipped evidence config is `k=2 --cluster --rep modal`. It was chosen on a |
| `src/evidence_redump.py` | THE GENERATOR THROWS AWAY SPANS THE SCORER WOULD HAVE ACCEPTED. `verify_spans(..., max_tokens=12)` in train_evidence.py:114-127 REJECTS any |
| `src/evidence_rep_sweep.py` | DOES `--rep modal` STILL WIN AT THE NEW OPERATING POINT? `rep` decides which string represents a cluster of agreeing spans. modal was |
| `src/evidence_reranker.py` | Candidate generation + reranking for evidence spans. Motivation, measured by headroom_analysis.py on val: |
| `src/evidence_risk_consistency.py` | Enforces the annotation scheme's own rule linking risk level to evidence presence -- a cross-task consistency constraint the pipeline never applied. |
| `src/evidence_selection_oracle.py` | THE CANDIDATES ARE RIGHT. THE SELECTOR PICKS THE WRONG ONES. `evidence_tagger_diagnosis.py` split every gold span three ways: |
| `src/evidence_signal_for_indicator.py` | Can the EVIDENCE tagger's silence improve the RISK indicator decision? THE OPENING. `evidence_suppression_rule.py` measured, on 1590 OOF posts: |
| `src/evidence_solo_pair.py` | TWO TAGGERS, THE SAME POSTS, ONE FLAG APART. `evidence_vote_oof.py` hardcodes `v2` as its solo baseline, so reordering |
| `src/evidence_span_budget.py` | KEEP THE RANKING. LEARN WHERE TO CUT. `evidence_selection_oracle.py`, against a baseline verified at 0.8824: |
| `src/evidence_structure_check.py` | WHAT IS DIFFERENT ABOUT THE TEST SPANS, WITHOUT NEEDING TEST GOLD. Val overstates every evidence defect 3-5x (span over-emission +27% vs +5%, |
| `src/evidence_suppress_pindicator.py` | Additionally suppress evidence on high-p_indicator posts. CPU, seconds. THE MECHANIC (measured, not assumed). Evidence phrase-F1 is MACRO over posts |
| `src/evidence_suppression_decouple.py` | UNWELD "IS THE EVIDENCE EMPTY?" FROM "IS THE RISK LABEL INDICATOR?" `evidence_match_quality.py` located the entire evidence deficit: |
| `src/evidence_suppression_rule.py` | Re-examine the evidence CONSISTENCY RULE under MACRO averaging. WHY NOW. `NIDHI_probe_noev.csv` proved the official evidence metric is |
| `src/evidence_sweep_rc.py` | SOFT-DECODE SWEEP FOR THE SEVEN TAGGERS THAT ACTUALLY SHIP. THE MEASURED CONSTRAINT. The candidate pool holds 1.02-1.30 spans/post against |
| `src/evidence_tagger_diagnosis.py` | WHAT, EXACTLY, WOULD MAKE THE EVIDENCE TAGGER BETTER? The tagger is responsible for more than the zero-scoring posts suggested. At |
| `src/evidence_tagger_rank.py` | THE ONE EVIDENCE LANE STILL SIZED TO THE GAP. WHERE THIS COMES FROM. Three measurements: |
| `src/evidence_true_oracle.py` | THE EXACT CEILING of any selector over the enumerated candidate pool. WHY. `span_rerank_upper_bound.py` reported a "pool oracle" of 0.8243, previously |
| `src/evidence_union.py` | Raise evidence recall by pooling spans across the evidence models. THE DIAGNOSIS THIS FOLLOWS FROM (src/evidence_diagnosis.py, 372 val posts): |
| `src/evidence_val_macro.py` | Score evidence CSVs on val, MACRO, with a paired author bootstrap. WHY THIS IS NEEDED BEFORE UPLOADING THE CONDITIONED ENSEMBLE. Everything |
| `src/evidence_vote_ensemble.py` | Builds a span-level VOTE ensemble over several evidence taggers and writes an override CSV consumable by predict.py --evidence-override or by |
| `src/evidence_vote_oof.py` | Scores the evidence span vote over the full out-of-fold set rather than a single held-out split. |
| `src/friend_evidence_arm.py` | THE TEAMMATE'S THIRD HEAD, NEVER USED. Their model is a TRI-head MentalRoBERTa: risk + factors + evidence tokens. The |
| `src/friend_evidence_eval.py` | Score the friend's EVIDENCE tagger, which has never been used. WHY IT MATTERS. Evidence is 0.30 of the composite and the least-modernised |
| `src/friend_evidence_export.py` | Export the friend's evidence spans in the candidate format the vote expects, so it can be tested as a 9th voter. |
| `src/llm_evidence_pilot_score.py` | Score the never-evaluated LLM evidence pilot against gold. WHY. Evidence is the only component with large headroom: the candidate pool's |
| `src/oob_evidence_eval.py` | SCORE EVIDENCE ON AN INSTRUMENT THAT MATCHES DEPLOYMENT. `scripts/oob_evidence.sh` writes candidates with a `bag` column. Each post was |
| `src/patch_submission_evidence.py` | Swap ONLY the evidence column of a submission, and prove nothing else moved. Built for the `evidence_v2_rc` drop: that tagger is the worst of the eight on |
| `src/predict_evidence_backoff.py` | Emit test-set evidence using the validated backoff decoding strategy. Validated on val_split against the OFFICIAL per-post macro metric: |
| `src/reground_evidence.py` | Make every submitted evidence span a VERBATIM substring of its post. THE PROBLEM. Auditing submissions/NIDHI_evidence_vote8.csv against data/test.csv: |
| `src/rerank_cache_from_enum.py` | Give RC7's candidates a reranker score WITHOUT a GPU, by joining to the enumerated pool that was already scored. |
| `src/retrain_evidence_cond.py` | Retrain the champion's shipped evidence taggers with --risk-cond, replaying each one's own hyperparameters. |
| `src/score_evidence_variants.py` | SCORE EVIDENCE VARIANT FILES AGAINST GOLD, IN THE SCORER'S OWN AGGREGATION. `evidence_vote_test.py --split val` writes an evidence column per cluster |
| `src/span_diagnosis.py` | Diagnose the ungrounded gold evidence spans found by data_audit.py. For every gold span that plain matching can't locate, try a ladder of |
| `src/span_rerank_cluster_select.py` | CLUSTER-THEN-RANK selection over the enumerated pool. CPU, no GPU. WHY THIS SHAPE. `evidence_true_oracle.py` proved the pool's exact ceiling is |
| `src/span_rerank_upper_bound.py` | FREE GO/NO-GO on the span-reranker lane. CPU, seconds, no GPU, no re-scoring. WHY THIS EXISTS. `train_span_reranker.py` reported OOF AUC 0.890 against |

## Factor experiments

| File | Purpose |
|---|---|
| `src/apply_ratematch_factors.py` | Turn factor probabilities into a factor column using PREVALENCE RATE-MATCHING instead of fitted thresholds. |
| `src/arm_complementarity.py` | DO THE FACTOR ARMS ACTUALLY MAKE DIFFERENT MISTAKES, AND WHAT DOES EACH ADD? Three diagnostics on the eleven arms of the shipped factor pool. None of them |
| `src/arm_robustness.py` | Re-measures each factor arm's cross-fitted macro-F1 over many random author partitions instead of the single partition compare_factor_arms.py reports. |
| `src/augment_factors.py` | Tail-label augmentation via local Qwen paraphrasing. The rarest factor labels (substance use, exposure to others' suicide, |
| `src/average_factor_seeds.py` | Average the factor probabilities of several seeds into one arm. WHY. The risk model averages several seeds; the factor model runs a single seed |
| `src/blend_factors_xl.py` | Blend the deberta-v2-xlarge factor-NLI model into the 2-seed deberta-v3 factor-NLI ensemble, validated on val_split gold. |
| `src/blend_nli_friend_factors.py` | Blend NLI factor probabilities with the teammate's MentalRoBERTa factor probabilities — validated honestly on val_split. |
| `src/build_factor_thresholds.py` | Writes a new factor threshold array, changing ONLY the labels whose CROSS-FITTED F1 gain is positive, and leaving every other label at its |
| `src/build_hybrid_defs.py` | Keep the published definition where it already works; use the derived one only where it does not. |
| `src/compare_factor_arms.py` | Re-scores every factor arm on EQUAL FOOTING. WHY THIS IS NEEDED. A threshold fitted on the same posts it is then scored on |
| `src/compare_factor_models.py` | Per-factor F1 comparison — baseline (no synthetic data) vs the synthetic-augmented model, on the SAME val_split gold. Macro-F1 is a |
| `src/compare_real_vs_synthetic.py` | The cheap sanity check before building a full error-neighbor pipeline: put real, genuinely-missed model errors side by side with the |
| `src/compression_theory_check.py` | DOES THE MEASURED THRESHOLD COMPRESSION MATCH WHAT AVERAGING PREDICTS? Thresholds are fitted on cross-validated predictions, where each post is scored |
| `src/derive_definitions.py` **[API: posts]** | Derives each factor's definition FROM THE ANNOTATIONS rather than taking the published wording as given. |
| `src/ensemble_factors.py` | Cross-model factor ensemble (cautious, ours-dominant). Blends our DeBERTa factor probabilities (factors_cv_final) with the |
| `src/ensemble_factors_seeds.py` | Factor seed ensemble: probability averaging across differently-seeded, identical-architecture models, applied to factors. |
| `src/factor_antonym.py` | Measures, and optionally corrects, the coupling between protective factors and the risk factors that are their surface-level polarity opposites. |
| `src/factor_arm_ap_protective.py` | Separates two competing explanations for weak performance on the protective factors: a MODEL failure that a better architecture could fix, or a CEILING |
| `src/factor_arm_weight.py` | Tests GLOBAL weighting of the factor ensemble arms against the equal-weight logit mean. |
| `src/factor_arm_weights.py` | WEIGHTED arm pooling for factors. The factor column pools its arms with `logit_mean`, an UNWEIGHTED mean of |
| `src/factor_author_pool.py` | Tests whether pooling factor probabilities across an author's posts improves the factor decode. |
| `src/factor_author_tboth.py` | AUTHOR POOLING WAS CLOSED UNDER A DECODE THAT STRUCTURALLY FORBIDS IT. factor_author_pool.py tested the blend under a QUANTILE decode, which fires |
| `src/factor_bag9.py` | Is bag8 + fenum better than bag8, on the scorer's own terms? WHY fenum AND NOTHING ELSE. Seven unused arms were re-tested per label on AP |
| `src/factor_bag9_tboth.py` | RE-TEST THE ABSORPTION LAW UNDER THE tboth DECODE, SINCE IT WAS ORIGINALLY MEASURED UNDER A DECODE THAT IS NO LONGER IN USE -- AND BECAUSE DIVERSITY, |
| `src/factor_bag_ship.py` | Build a multi-arm factor column using FIRE-RATE TRANSFER. WHAT THIS SHIPS. The member-count trend measured by `factor_rank_average.py` |
| `src/factor_blend.py` | Blend the two independent factor models. Factors are the last untouched block. WHERE THE GAP IS. Composite = 0.4 risk + 0.3 evidence + 0.3 factors. Backing |
| `src/factor_build_test.py` | Build the TEST factor column from the top-3 diverse factor models. VALIDATED ON VAL (372 posts, both author-disjoint halves, all arms thresholded |
| `src/factor_cooccurrence_stack.py` | The 24 factors are decided INDEPENDENTLY. They do not occur independently. Gold averages 2.92 factors per post and the labels travel in packs -- |
| `src/factor_count_target.py` | Move the champion's OPERATING POINT without touching its model. WHY. Three consecutive scores decline monotonically as factor counts drop: |
| `src/factor_diagnosis.py` | Attribute the factor deficit to DATA, MODEL, or THRESHOLD -- per label. WHY THIS INSTEAD OF ANOTHER EXPERIMENT. Nine factor ideas have been tried and |
| `src/factor_dupcluster.py` | Average factor scores WITHIN near-duplicate post clusters before decoding. THE MEASURED PREMISE. Annotators label near-identical text near-identically: |
| `src/factor_empty_test.py` | Does FORCING at least one factor onto every post help or hurt? THE STRUCTURAL FACT. In gold, 128 of 1635 posts (7.8%) carry NO factors at all. |
| `src/factor_ensemble.py` | Ensemble the eight regime-diverse factor models. Factors are the last block. WHY THIS, AND WHY IT MIGHT WORK WHERE OTHER FACTOR ATTEMPTS FAILED. |
| `src/factor_ensemble_audit.py` | Tests whether the deployed 2-seed factor ensemble is actually helping, or is being held back by an internal inconsistency in how its thresholds are formed. |
| `src/factor_error_dump.py` | Prints the actual false positives and false negatives for the weak factor labels, so a human can decide whether the errors share a linguistic or |
| `src/factor_error_read.py` | Read the actual mistakes on the worst factor labels. WHY THIS AND NOT ANOTHER METRIC. src/factor_diagnosis.py established what the |
| `src/factor_extra_arm_ap.py` | Do the SYNTHETIC / EXTERNAL arms help the STARVED labels, per label? WHY RE-ASK. The synthetic arms were previously rejected on one number: the |
| `src/factor_fold_baseline.py` | What the DEPLOYED 8-arm bag scores on ONE fold's validation posts. A factor pilot trained on fold 3 prints a val macro-F1 that cannot be compared |
| `src/factor_free_diagnostics.py` | Two cheap checks that can kill two factor ideas before any GPU is spent. A. TRUNCATION. `score_all` tokenizes with max_len=320 and |
| `src/factor_incumbent_id.py` | WHAT IS "THE INCUMBENT" IN THE QWEN-72B LOG? The Qwen-72B training log reports it beating an incumbent's mean AP on fold 3 |
| `src/factor_inventory.py` | Map what is on disk onto the six factor arms, before building any submission. WHY THIS EXISTS. The member-count trend measured by `factor_rank_average.py` |
| `src/factor_label_anatomy.py` | WHICH OF THE 24 LABELS IS THE MACRO-F1 ACTUALLY LOSING, AND IS IT THE THRESHOLD OR THE RANKING? |
| `src/factor_label_diagnosis.py` | Full 24-label breakdown of the factor model on OUT-OF-FOLD predictions over all 1635 labelled posts -- the diagnostic that should precede any per-label |
| `src/factor_label_structure.py` | Gold-only structural analysis of the 24 factor labels. Question: is there exploitable STRUCTURE among the 24 labels that a rare label |
| `src/factor_label_swap.py` | Apply a purpose-built factor arm ONLY to the labels it was built for. THE PROBLEM THIS SOLVES. Adding a 12th arm to an 11-arm logit mean gives it |
| `src/factor_lexical_knn.py` | A NON-NEURAL FACTOR ARM. Every existing arm is a cross-encoder: a transformer scoring (post, hypothesis). Those arms correlate 0.72-0.96 with the pool, so |
| `src/factor_lexical_rules.py` | Zero-parameter lexical gating for the factor labels that are lexically defined. THE ARGUMENT. |
| `src/factor_llm_ensemble.py` | Does averaging the LLM with the encoder beat the encoder, even on the labels where the LLM loses head-to-head? |
| `src/factor_llm_opinions.py` | Get a reasoning LLM's opinion on all 24 factors, then ROUTE per label. WHY ROUTING RATHER THAN A GLOBAL USE OF THE LLM. Forms that use the LLM's |
| `src/factor_lora_ensemble.py` | Does a fine-tuned Qwen-7B add anything to the DeBERTa cross-encoder? WHY A FINE-TUNED MEMBER AND NOT A ZERO-SHOT ONE. Ensemble value depends on both |
| `src/factor_loss_metric_gap.py` | HOW BADLY DOES THE TRAINING LOSS MISMATCH THE SCORING METRIC? The factor arms are trained with BCE on flattened (post, factor) pairs and |
| `src/factor_matched_dump_probe.py` | WHICH TEST DUMP IS MATCHED TO THE OOF? An offline diagnostic. THE SITUATION. Several arms (mnli, dmnli, prot) have MORE THAN ONE test dump on |
| `src/factor_member_forward.py` | FORWARD SELECTION OVER FACTOR ENSEMBLE MEMBERS. `artifacts/factor_runs/` holds many arms with 5-fold OOF, a subset of which |
| `src/factor_member_loo.py` | LEAVE-ONE-OUT MEMBERSHIP AUDIT FOR THE FACTOR ENSEMBLE. The logit-mean pool gives every member equal weight, so a member that is |
| `src/factor_oof_instrument.py` | Pool the five author-disjoint folds into one 1635-post factor measurement, and compare arms by PAIRED author-bootstrap deltas. |
| `src/factor_operating_point.py` | Where is the macro-F1 optimum as a function of how many factors we predict? WHY. Across three different models, predicted factor count and score move |
| `src/factor_paired_arms.py` | Judge factor arms by PAIRED author-bootstrap deltas instead of absolute macro-F1. WHY THIS EXISTS. Every factor arm this campaign has been ranked by its absolut... |
| `src/factor_per_label_arm.py` | Per-LABEL arm selection: use whichever factor arm is best at each label. WHY NOW. Per-CLASS routing on risk is a zero-parameter rule: hand one class |
| `src/factor_per_label_diagnostic.py` | Finds which of the 24 factor labels are dragging Subtask-2 down. Why this matters more than it looks: S2 is MACRO-F1 over 24 labels, so every |
| `src/factor_per_label_llm.py` | PER-LABEL, not scalar. Where is the factor column weak, and does the 72B beat it THERE? |
| `src/factor_perlabel_pool.py` | THE METRIC AVERAGES OVER LABELS. THE POOL DOES NOT VARY BY LABEL. Factors are scored by MACRO-F1 over 24 labels -- every label worth 1/24 |
| `src/factor_pool_diagnostic.py` | IS THE UNWEIGHTED LOGIT-MEAN ACTUALLY GIVING EACH ARM AN EQUAL VOTE? The champion pools 11 arms with `logit_mean` -- the arithmetic mean of |
| `src/factor_prevalence_decode.py` | THE DECODE WAS CALIBRATED AGAINST ITS OWN OUTPUT. `champ_rate = champion's predicted counts / 378`. That is circular: the fire |
| `src/factor_prior_match.py` | Fix the factor model's systematic over-prediction WITHOUT tuning anything. THE OBSERVATION. factor_label_diagnosis.py --by-risk shows recall exceeding |
| `src/factor_protective_diag.py` | The protective labels are RANK-limited, not decode-limited: four of the five worst APs in the taxonomy are protective (meaning in life 0.232, sense of |
| `src/factor_qwen_bag9.py` | CLOSE QWEN ON CROSS-FITTED F1, NOT ON AP -- USING ONLY THE FOLDS WE ALREADY HAVE. Qwen-72B loses to the pool on mean AP on one fold and roughly ties on another. |
| `src/factor_rank_average.py` | Does RANK-averaging fix the documented "more members makes it worse" pathology? THE MEASURED FACT THIS ADDRESSES. From the factor arm results: |
| `src/factor_rare_keywords.py` | THE RAREST LABELS ARE WORTH THE MOST AND ARE THE MOST LEXICALLY CLOSED. Factor macro-F1 weights all 24 labels equally, so ONE label is 1/24 = 0.0417 |
| `src/factor_rate_decompose.py` | For each of the 24 factor labels, split its weakness into the two causes that need OPPOSITE fixes: |
| `src/factor_regime_gap.py` | THE OOF AND TEST PROBABILITIES COME FROM DIFFERENT KINDS OF MODEL. OOF one fold model per post, each trained on 1308 posts |
| `src/factor_risk_consistency.py` | Applies the annotation scheme's risk-conditional factor structure as a post-hoc correction -- the same intervention class as |
| `src/factor_risk_rules.py` | Apply the evidence-consistency method to FACTORS, and measure it honestly. BACKGROUND. The evidence-consistency fix works because gold obeys a hard |
| `src/factor_sentence_screen.py` | Is the annotation unit the SENTENCE rather than the POST? THE EVIDENCE FOR ASKING. The organizers' 2022 ancestor paper (Li et al., HKIE |
| `src/factor_set_search.py` | EXHAUSTIVE set search over a shortlist, with an untouched outer holdout. WHY. Every membership test so far has been ONE SWAP from the shipped set -- |
| `src/factor_signal_for_risk.py` | Measures whether the plan/attempt distinctions the risk labels depend on are already carried by the hand-annotated factor column. |
| `src/factor_smallsample_thresholds.py` | Fits factor thresholds to expected F1 on a test-sized draw rather than to F1 on the pooled OOF sample. |
| `src/factor_span_count.py` | Measures whether the span multiplicity discarded by the binary factor target explains where the factor models lose macro-F1. |
| `src/factor_stacker.py` | Let each factor see the other 23. The NLI model cannot. THE STRUCTURAL GAP. train_factors_nli scores one (post, definition) pair at a |
| `src/factor_starved_labels.py` | Compares decode variants for the handful of factor labels whose score is decided by one or two predicted cells. |
| `src/factor_structure.py` | Evaluate factor-decode rules that exploit structure a per-label threshold cannot see, and optionally build a test factor column from one of them. |
| `src/factor_test_calibration.py` | Tests whether factor thresholds transfer from the OOF pool to the test pool, and optionally rescales the test logits so that they do. |
| `src/factor_threshold_decode.py` | Compares quantile factor decodes against probability-threshold decodes, and optionally builds the test factor column from the chosen rule. |
| `src/factor_threshold_diagnostic.py` | For each of the 24 factor labels, decides WHY it scores badly: UNMEASURABLE too few gold positives here to say anything. On the 372-post |
| `src/factor_threshold_policies.py` | How much macro-F1 is the thresholding rule throwing away? Free to answer -- it runs on the OOF dumps that already exist, and trains nothing. |
| `src/factor_threshold_refine.py` | PUSH THE LANE THAT IS ACTUALLY PAYING. Established: a THRESHOLD decode beats a QUANTILE decode out-of-sample. |
| `src/factor_threshold_search.py` | Factor threshold-strategy search — CPU-only, validated on gold OOF data. Macro-F1 weights all 24 labels equally. Deployed thresholds were tuned on the |
| `src/factor_threshold_shrink.py` | tfit FITS 24 INDEPENDENT THRESHOLDS. ON A LABEL WITH 8 POSITIVES THAT IS NOISE. `factor_label_anatomy.py` measured, cross-fitted by author, that the gap |
| `src/factor_threshold_stabilise2.py` | PUSH THE ONE AXIS THAT WORKED. `tboth` (bagged + smoothed thresholds) beat the shipped `tfit` with the CI |
| `src/factor_threshold_strategies.py` | Attacks the largest QUANTIFIED defect found in the whole system. factor_ensemble_audit.py measured per-label threshold tuning on val at |
| `src/factor_weak_label_errors.py` | WHAT is the model confusing the weak factor labels WITH? WHY THIS COMES BEFORE ANY NEW ARCHITECTURE. Our diagnostics say HOW OFTEN the |
| `src/factors_oof_cv.py` | Author-disjoint K-fold cross-validation for the NLI factor model. WHY NOT A PLAIN FULL-DATA RETRAIN. train_factors_nli.py requires --val and |
| `src/find_champion_factors.py` | WHICH ARTEFACT + DECODE PRODUCED THE CHAMPION'S FACTOR COLUMN? Factors are where the score actually is. S1 is 0.8047 and S2 is 0.6774, and |
| `src/fold_runs_to_bag.py` | Publish per-fold `val_probs.npz` files into the bag's naming, with checks. WHY THIS IS NOT JUST `cp`. `scripts/factors_qwen_nli.sh` writes |
| `src/friend_factor_arm.py` | THE FRIEND HAS A COMPLETE FACTOR ARM AND IT HAS NEVER BEEN IN THE POOL. `src/friend_pipeline` is the MentalRoBERTa tri-head system. Its |
| `src/friend_factor_ensemble.py` | Does the friend's FACTOR head add anything to the deployed model? WHY ASK AGAIN. Its standalone macro is 0.4280 even with in-sample thresholds, |
| `src/gen_rare_factors.py` | Generate synthetic training posts for the factor labels with the fewest real examples. |
| `src/gen_rare_factors_api.py` **[API: definitions only]** | Generate synthetic training posts for the rare factor labels via the DeepSeek API, with length and diversity designed into the generation. |
| `src/label_external.py` | External-data factor labeling — attack the r=0.806 scarcity wall. The winning 2025 approach (arXiv 2510.20085) added ~3700 externally-sourced, |
| `src/llm_rare_factors.py` **[API: posts]** | Ask a large LLM, zero-shot, which of the STARVED factors a post shows -- and measure it per label against the NLI model on the same 1635 OOF posts. |
| `src/make_factor_oof_splits.py` | Build author-disjoint TRAIN / THRESH / EVAL splits for the factor OOF instrument. WHAT THE DEPLOYED RECIPE DOES, and therefore what must be replicated: |
| `src/meaning_contamination_probe.py` | Is `meaning in life` failing because the ANNOTATION GUIDELINE sends its vocabulary to `hopelessness`? Zero GPU, minutes. |
| `src/pseudo_label_factors.py` | Pseudo-label a target set with a teacher that never saw it. `src/self_train_factors.py` does the same for `artifacts/external_raw.csv`, |
| `src/qwen_factor_countmatch.py` | Compare the Qwen factor RANKING against the incumbent bag's, with thresholding removed from the comparison. |
| `src/qwen_factor_verdict.py` | Test whether an average-precision gain from the Qwen factor arm survives contact with the metric Subtask 2 actually scores. |
| `src/qwen_rare_factor_probe.py` | Can a local LLM, given only the published definition, recognise the factors the supervised model cannot learn? |
| `src/rare_factor_rules.py` | Keyword ADD-rules for the rare factor labels, ported from the friend model's postprocess.py (RARE_FACTOR_RULES) and measured against gold before being |
| `src/rat_bag_to_oof.py` | Convert the rationale bag's per-fold val_probs.npz into the oof_predictions.csv format src/champion_oof.py expects, so the bag can be swapped in as the DeBERTa |
| `src/reconcile_derived_defs.py` | Reconcile the two contradictory readings of the derived-definitions arm, then build the hybrid on the CORRECT selection rule. |
| `src/rethreshold_test.py` | Re-threshold an existing test probability dump using thresholds fitted in the SAME regime the test probabilities were produced in. |
| `src/robust_threshold_search.py` | Joint threshold optimization for the champion 2-seed factor-NLI ensemble, targeting the ACTUAL competition metric (factor macro-F1, |
| `src/self_train_factors.py` | Self-training for factors — pseudo-label the already-crawled external posts using the existing champion factor-NLI ensemble as teacher, instead of Qwen. |
| `src/splice_factor_columns.py` | Combine a specialist's factor columns with the champion's for the rest. WHY. Held-out F1 splits by polarity -- the five protective factors average |
| `src/style_match_synthetic.py` | Strip the surface tells that let a model separate synthetic posts from real ones. THE PROBLEM, measured on artifacts/syn_rare_factors.csv against the real corpu... |
| `src/swap_factor_column.py` | Put a new factor column onto an existing submission, changing nothing else. WHY A DEDICATED SCRIPT. `splice_factor_columns.py` merges two npz probability |
| `src/threshold_audit.py` | Is the champion's factor column using the wrong thresholds? WHAT WAS FOUND. predict_factors_nli.py loads each run's own thresholds.npy |
| `src/threshold_regime_check.py` | Can a threshold decode measure a consistent, positive gain offline and still move the WRONG WAY on submission? (Case examined: the `paircm` tag.) |
| `src/threshold_rule_robustness.py` | Is the shrinkage threshold gain real, or one lucky author split? WHAT IS AVAILABLE. src/threshold_rules.py, on the incumbent's 1635 out-of-fold |
| `src/threshold_rules.py` | The threshold RULE is worth more than most of the arms tested so far. THE SIZE OF THE PRIZE. compare_factor_arms.py measures what each arm loses when |
| `src/throttle_decompose.py` | Is the --match-counts constraint on factor_bag_ship.py THROTTLING the 8-arm bag, or is the refit-xfit instrument simply over-optimistic at 8 arms? |
| `src/user_smoothing.py` | The factor model scores every post in isolation. It should not. THE SIGNAL. Measured on all 1635 labelled posts (153 users, 10.7 posts each), |
| `src/why_defs_failed.py` | Diagnose why factor definitions derived from the training annotations score worse than the organizers' published definitions. |

## Shared diagnostics and utilities

| File | Purpose |
|---|---|
| `src/add_pred_risk.py` | Annotate a CSV with a `pred_risk_level` column from the risk model(s). Used to condition the factor model on PREDICTED (not gold) risk, so the |
| `src/analyze_concept_bottleneck.py` | Runs the concept-bottleneck diagnostic suite on a trained ConceptRiskModel checkpoint. Steps 2-4 (ablation, fusion weights, fusion activation) |
| `src/analyze_concept_bottleneck_oracle_concepts.py` | Runs the concept-bottleneck diagnostic suite on a trained ConceptRiskModel checkpoint. Steps 2-4 (ablation, fusion weights, fusion activation) |
| `src/batch_repro.py` | Replay the poison batch through the EXACT training path and localize the first non-finite tensor. |
| `src/build_indicator_demote_ng.py` | `indicator_demote` moves a post from indicator to ideation when the text explicitly mentions suicide -- which by the task's own definition it cannot be |
| `src/cache_val_baseline.py` | Computes the deployed champion's val predictions ONCE and caches them, so rule_ablation_v2.py (and anything else that needs "what does the champion |
| `src/check_encoders.py` | Which encoder checkpoints can this environment actually load, offline? WHY THIS EXISTS. Three factor/evidence jobs were queued and all three died at |
| `src/clinical_reliability.py` | Measures the properties a deployed triage system needs that the competition composite does not score. |
| `src/compare_cv_vs_deployed.py` | Decides whether the 5-fold CV factor models are actually BETTER than the deployed 2-seed NLI ensemble, on a like-for-like basis. |
| `src/compare_head_combination.py` | Diagnostic, no retraining. The risk models train with alpha_ordinal=0.5 (categorical CE + CORN ordinal, averaged 50/50 at inference in |
| `src/confusion_audit.py` | Validates the hand-written CONFUSABLE map against the confusions the model actually makes. |
| `src/debug_nan.py` | Bisect the NaN. Runs in ~30s. Usage (any node, GPU preferred): python src/debug_nan.py |
| `src/deepseek_client.py` **[API client]** | Independent verifier client. Mirrors openai_client.py's public interface (generate, generate_json) so verify_generated.py can use |
| `src/dump_disagreement_cases_v5.py` | Writes two review files for manual error analysis on the val split. 1. disagreement_cases — posts bucketed by the SHAPE of the models' |
| `src/dump_nli_val_probs.py` | Saves the risk-NLI models' val probabilities to disk, once. WHY. Testing whether a candidate arm adds to the CHAMPION ensemble -- rather |
| `src/duplicate_consistency.py` | Measures whether the corpus contains near-duplicate posts and whether the model's predictions on them are self-consistent. |
| `src/error_analysis.py` | Error analysis: turns a validation prediction file into a diagnosis. Answers, in order of composite-score weight: |
| `src/error_predictability.py` | IS "THE MODEL WILL GET THIS WRONG" PREDICTABLE FROM THE TEXT? THE PRIZE IS MEASURED. `evidence_gated_hedge.py` established that hedging the |
| `src/family_prevalence.py` | Screens every candidate rule family from the symbolic notebook for TEST-SET PREVALENCE before any engineering effort is spent on it. |
| `src/find_champion_evidence_runs.py` | Recover WHICH evidence runs produced the champion's evidence column. WHY. The champion's evidence came from an 8-run vote>=2, but no log records |
| `src/find_champion_probs.py` | WHICH artifacts/ensembled_risk_*.csv PRODUCED THE CHAMPION'S RISK COLUMN? A prior correction was fitted on `champion_oof.csv` and applied to the |
| `src/friend_ablation.py` | Answers the only question that matters about the friend model: does it add INDEPENDENT information, or does it dilute stronger models? |
| `src/friend_coverage_check.py` | DOES THE FRIEND'S RISK FILE ACTUALLY COVER EVERY TEST ROW? `ensemble_risk.py` blends our seed-ensemble with the teammate's probabilities |
| `src/generate_concept_labels.py` **[API: posts]** | Generates weak binary labels for the auxiliary concepts that the 4-level risk taxonomy actually turns on — tense, concreteness, |
| `src/headroom_analysis.py` | Measures the CEILING of the existing models, so that an architecture proposal can be justified by an upper bound before it is built. |
| `src/isolate_poison.py` | Isolate which row(s) of the poison batch break the model, and how. Usage (GPU node): |
| `src/make_probe_submission.py` | Spend ONE submission to measure something no amount of offline work can reveal: how S1 actually splits between risk and evidence. |
| `src/make_submission.py` | Build and validate the leaderboard submission CSV. Two functions, both mandatory before every upload: |
| `src/merge_confirmed_wins.py` | Combine two previously-verified single-column changes into one submission. WHY THIS IS SAFE, and it is measured rather than assumed. Two single-variable |
| `src/mine_champion_failures.py` | Stage 1 of the Error-Driven Symbol Discovery framework: mine the CURRENT, real error set against the actual deployed champion (rules |
| `src/mine_train_errors.py` | Finds GENUINE model errors to seed "error-neighbor" synthetic generation — without ever touching val_split (that would leak val |
| `src/oob_bags.py` | MAKE THE INSTRUMENT MATCH DEPLOYMENT. THE DEFECT. Deployment scores each TEST post with an ENSEMBLE of 5 fold |
| `src/predict_risk_lora_test.py` | Score the TEST posts with a 5-fold LoRA risk member and average the folds. WHY THIS EXISTS. `risk_bag_eval.py` identified a majority vote over |
| `src/probe_suite.py` | Build degenerate submissions that turn the official scorer into an exact measuring instrument, and solve the returned score for the quantity each one |
| `src/reconstruct_val_baseline.py` | Produces an APPROXIMATE stand-in for cache_val_baseline.py's output on a machine with no GPU / no risk-NLI weights, so candidate rules can be |
| `src/spot_check_concept_labels.py` | Checks a RANDOM sample of concept labels (not just hard cases — random checks tell you about overall reliability, which is what |
| `src/truncation_probe.py` | FREE falsifier for the sequence-length lane, on existing OOF dumps. CPU only. THE FINDING IT TESTS. The factor encoders read max_len 320 (~205 words of |
| `src/validate_ensemble.py` | Honest local validation of the cross-model ensemble. Every blend weight submitted so far (risk 0.4/0.5, factors 0.7/0.5) was not |
| `src/verify_component_artifacts.py` | IS THIS COMPONENT'S VAL ARTEFACT THE SAME OBJECT AS ITS TEST ARTEFACT? Run this BEFORE any blend-weight change. Skipping it risks concrete failure |
| `src/verify_factor_test_dumps.py` | Pre-flight for any factor-column rebuild. Catches two silent hazards in `factor_threshold_stabilise.py --build-test` that nothing else checks. |

## Teammate tri-head pipeline (src/friend_pipeline)

| File | Purpose |
|---|---|
| `src/friend_pipeline/config.py` | Central configuration. Everything tunable lives here. |
| `src/friend_pipeline/friend_export_oof.py` | Export stage2's OOF predictions to a flat CSV. Reads outputs/oof.npy (written by stage 2) and flattens every fold's held-out |
| `src/friend_pipeline/model.py` | Multi-task model: one encoder, three heads. head 1: risk level (4-way softmax on [CLS]) |
| `src/friend_pipeline/postprocess.py` | The symbolic 'sandwich' guards + decoding of model outputs into final answers. Catches handled here: |
| `src/friend_pipeline/run_pipeline.py` | Run the whole pipeline with one command: python src/friend_pipeline/run_pipeline.py --team_name MyTeam |
| `src/friend_pipeline/scorer.py` | Local scorer for the IEEE Big Data Cup - Explainable Suicide Risk Detection. Implements the official rules from the competition page: |
| `src/friend_pipeline/stage1_data.py` | Stage 1 - Data cleaning, user-grouped folds, evidence lexicon. Catches handled here: |
| `src/friend_pipeline/stage2_train.py` | Stage 2 - fine-tune the tri-head model, one model per user-grouped fold. Catches handled here: |
| `src/friend_pipeline/stage3_tune.py` | Stage 3 - tune decision thresholds on OUT-OF-FOLD predictions and print the honest cross-validated composite score using the official metric rules. |
| `src/friend_pipeline/stage4_predict.py` | Stage 4 - ensemble the 5 fold-models on the leaderboard set, apply the symbolic guards + tuned thresholds, and write the official submission CSV. |

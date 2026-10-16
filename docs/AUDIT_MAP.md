# Audit map

The paper reports 31 pre-specified comparisons (Table 2 of the paper) and a set
of measurements that decide whether a comparison can be trusted at all. This
page links each one to the code that ran it, so that every verdict can be
traced and re-run.

Verdicts follow the paper's protocol (Section 4): a change holds when it wins on
at least 80 percent of author-disjoint resamples (90 percent if chosen from many
candidates), or clears a margin far outside any run-to-run variation, or, for
routing, gains on the blind leaderboard. Component codes: **F** factors,
**R** risk, **E** evidence.

Where a technique is an option of a shared training script, the option is
named. Files marked **[API: posts]** sent post text to a commercial LLM API during
exploration, and **[API: definitions only]** sent only factor definitions; see
[DATA_AND_ETHICS.md](DATA_AND_ETHICS.md).

## The 31 comparisons

| # | Technique | Verdict | Code |
|---|---|---|---|
| 1 | Task-adaptive pretraining | no effect (F) | `scripts/tapt_factors.sh` |
| 2 | Larger backbone, factors | helped once, then stopped | `scripts/factors_encoder.sh`, `scripts/factors_nli_xl.sh` |
| 3 | Larger backbone, risk | no effect | `scripts/risk_nli_xl.sh`, `scripts/risk_nli_xl6.sh` |
| 4 | Generated training data | no effect (F) | `src/gen_rare_factors.py` (local Qwen), `src/gen_rare_factors_api.py` **[API: definitions only]**, `src/style_match_synthetic.py`, `src/compare_real_vs_synthetic.py` |
| 5 | Self-training on own labels | no effect (F) | `src/self_train_factors.py`, `src/pseudo_label_factors.py`, `scripts/self_train_factors.sh` |
| 6 | Retrieval of similar posts | hurt (F) | `src/factor_lexical_knn.py` |
| 7 | Longer input window | no effect (E) | `src/truncation_probe.py` |
| 8 | Class-balanced loss | **held** (F) | `src/train_factors_nli.py --loss cbntr`, `scripts/factors_longtail.sh` (the shipped `ltail` arm) |
| 9 | Focal loss | did not replicate (R) | `src/train_risk_focal.py`, `scripts/train_risk_focal_gamma_sweep.sh` |
| 10 | Asymmetric loss | no effect (F) | `src/train_factors_nli.py --loss asl` |
| 11 | Difficulty-weighted sampling | no effect (R) | `src/compute_boundary_weights.py`, `src/make_hard_weights.py`, `scripts/risk_hardmine.sh` |
| 12 | Architecturally diverse ensemble | **held** (F) | `src/arm_complementarity.py`, `src/friend_factor_ensemble.py` |
| 13 | Seed ensembling | no effect (F) | `src/average_factor_seeds.py`, `src/ensemble_factors_seeds.py` |
| 14 | Greedy ensemble selection | lost to a fixed set (F) | `src/factor_member_forward.py` |
| 15 | Leave-one-out member pruning | no effect on R; E unresolved | `src/factor_member_loo.py`, `src/risk_member_loo.py`, `src/evidence_member_loo.py` |
| 16 | Hard majority voting | unresolved (E) | `src/evidence_vote_test.py` (`k`), `src/evidence_rc_sweep.py` |
| 17 | Stacking on model outputs | no effect (R) | `src/risk_crossmodal_stack.py`, `src/risk_stack_ship.py` |
| 18 | Shared encoder, separate heads | hurt (F) | `scripts/factors_2head.sh` |
| 19 | Concept bottleneck | no measurable contribution (R) | `src/train_risk_concepts_bottleneck.py`, `src/analyze_concept_bottleneck.py`; concept labels from `src/generate_concept_labels.py` **[API: posts]** |
| 20 | Ordinal regression head | hurt on one architecture (R) | `src/train_risk.py` (dual categorical + CORN head), `src/compare_head_combination.py` |
| 21 | Post-hoc symbolic rules | no effect (R) | `src/risk_rule_ablation.py`, `src/rule_ablation_v2.py` |
| 22 | Rules scoped to the evidence | **held** (R) | `src/risk_span_rules.py`, `src/scope_transfer_study.py` |
| 23 | Per-category thresholds | in-sample bias (F) | `src/threshold_audit.py`, `src/factor_threshold_policies.py` |
| 24 | Prevalence correction | no effect (F) | `src/factor_prior_match.py`, `src/factor_prevalence_decode.py` |
| 25 | Threshold estimator swap | hurt (composite, leaderboard) | `src/factor_threshold_shrink.py`, `src/factor_smallsample_thresholds.py` |
| 26 | Label definitions as input | **held**, largest in the audit (F) | `src/train_factors_nli.py` |
| 27 | Same, applied to evidence | hurt (E) | `src/train_evidence_nli.py` |
| 28 | Zero-shot prompting | hurt (F) | `src/qwen_rare_factor_probe.py`, `src/factor_llm_opinions.py` |
| 29 | Parameter-efficient tuning | no effect (R) | `src/train_risk_lora.py`, `scripts/risk_lora7b.sh` |
| 30 | Routing one weak class | **held** (composite, leaderboard) | `src/risk_cot_probe.py`, `src/build_route_attempt.py`, `scripts/risk_cot_member.sh` |
| 31 | LLM as reranker | unresolved (E) | `src/evidence_llm_reranker.py` **[API: posts]**, `src/evidence_llm_score.py` **[API: posts]** |

## The measurements behind the verdicts

| Measurement in the paper | Code |
|---|---|
| Author-disjoint splitting; author dependence of labels (odds 6.2 rarest, 1.1 commonest) | `src/data_audit.py`, `src/factor_label_anatomy.py` |
| Seed range and paired, author-level resampling (the null band) | `src/arm_robustness.py`, `src/factor_paired_arms.py`, `src/evidence_pair_compare.py` |
| Selection inflation of the best of several candidates | `src/factor_set_search.py` |
| Saturation check on 163 near-duplicate pairs | `src/duplicate_consistency.py` |
| Label co-occurrence oracle (0.2732) | `src/factor_label_structure.py`, `src/factor_cooccurrence_stack.py` |
| Strengthening-baseline check (5 failures added to the final pool) | `src/factor_member_forward.py`, `src/factor_member_loo.py` |
| Deployment-consistent calibration (spread ratio 0.804 to 0.970; +0.0137 F) | `src/factor_threshold_stabilise.py --match-oof-scale`, `src/compression_theory_check.py`, `src/factor_regime_gap.py` |
| Gold risk spans carry no factor signal (0.1897 vs 0.1900) | not included in this package |

## The submitted system

Each surviving technique is a component of TRIDENT; [PIPELINE.md](PIPELINE.md)
gives the exact configuration. The build path that rebuilds the submission is
ten modules: `build_submission.py`, `champion_risk_inject.py`,
`evidence_vote_test.py`, `factor_threshold_stabilise.py`,
`apply_rules_to_test.py`, `candidate_patterns.py`, `data_audit.py`,
`scoring.py`, `text_norm.py` and `train_evidence.py`.

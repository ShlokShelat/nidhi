# Artifact manifest

The reproduction reads two kinds of file, split by one rule: **anything that contains post text stays out of this public repository.**

## Public (in this repository)

68 files of numeric model output only: factor-arm score dumps (`probs`, `thresholds`, `row_id`) and risk probability tables. No post text and no evidence spans.

| File | SHA-256 (prefix) |
|---|---|
| `artifacts/factor_runs/cbnsynapi_cv_oof_f0.npz` | `01ea71f0f6c52728` |
| `artifacts/factor_runs/cbnsynapi_cv_oof_f1.npz` | `051bca0fa6f470d4` |
| `artifacts/factor_runs/cbnsynapi_cv_oof_f2.npz` | `957da378b8de4b68` |
| `artifacts/factor_runs/cbnsynapi_cv_oof_f3.npz` | `6a488aebc255569a` |
| `artifacts/factor_runs/cbnsynapi_cv_oof_f4.npz` | `5b57690d8025fcc4` |
| `artifacts/factor_runs/cbnsynapi_cv_test.npz` | `d50bebbcffcd46ca` |
| `artifacts/factor_runs/cmine_cv_oof_f0.npz` | `ec9b89822bca73bd` |
| `artifacts/factor_runs/cmine_cv_oof_f1.npz` | `5e494afe0d2b6e0a` |
| `artifacts/factor_runs/cmine_cv_oof_f2.npz` | `38aa9adca18206fb` |
| `artifacts/factor_runs/cmine_cv_oof_f3.npz` | `348f7830439db87b` |
| `artifacts/factor_runs/cmine_cv_oof_f4.npz` | `094d7a59dbaeeeb0` |
| `artifacts/factor_runs/cmine_cv_test.npz` | `ab424f58c9c21a4e` |
| `artifacts/factor_runs/d4_cv_oof_f0.npz` | `b552812ba2a15e5e` |
| `artifacts/factor_runs/d4_cv_oof_f1.npz` | `df7fa94fedb96477` |
| `artifacts/factor_runs/d4_cv_oof_f2.npz` | `f570c1bf21125b48` |
| `artifacts/factor_runs/d4_cv_oof_f3.npz` | `ae5496a542bb619d` |
| `artifacts/factor_runs/d4_cv_oof_f4.npz` | `eaff5befc6d631d9` |
| `artifacts/factor_runs/d4_cv_test.npz` | `f716cdf1c18f3691` |
| `artifacts/factor_runs/dmnli_cv_oof_f0.npz` | `a2435ecfa7b6b384` |
| `artifacts/factor_runs/dmnli_cv_oof_f1.npz` | `b157d3a89e90cced` |
| `artifacts/factor_runs/dmnli_cv_oof_f2.npz` | `a1ec25ef935eb06a` |
| `artifacts/factor_runs/dmnli_cv_oof_f3.npz` | `d9f42fbb6e37f118` |
| `artifacts/factor_runs/dmnli_cv_oof_f4.npz` | `d1de290450d4ccad` |
| `artifacts/factor_runs/friendfac_cv_oof_f0.npz` | `49449fe6dd1c5afd` |
| `artifacts/factor_runs/friendfac_cv_oof_f1.npz` | `7e83ad8ec59de442` |
| `artifacts/factor_runs/friendfac_cv_oof_f2.npz` | `eaa42197ff8e7ae9` |
| `artifacts/factor_runs/friendfac_cv_oof_f3.npz` | `c07427c8ede5e5b6` |
| `artifacts/factor_runs/friendfac_cv_oof_f4.npz` | `84593b32eae917f3` |
| `artifacts/factor_runs/friendfac_cv_test.npz` | `a1bd998ffb77a502` |
| `artifacts/factor_runs/lnli_cv_oof_f0.npz` | `e7f183fac21aa778` |
| `artifacts/factor_runs/lnli_cv_oof_f1.npz` | `7572573a86b5d50c` |
| `artifacts/factor_runs/lnli_cv_oof_f2.npz` | `d7250652143f421d` |
| `artifacts/factor_runs/lnli_cv_oof_f3.npz` | `8ad5f0edaf94781b` |
| `artifacts/factor_runs/lnli_cv_oof_f4.npz` | `01eef8dd220272f5` |
| `artifacts/factor_runs/lnli_cv_test.npz` | `34ca1ce76294a595` |
| `artifacts/factor_runs/ltail_cv_oof_f0.npz` | `4b8091ceb5112f51` |
| `artifacts/factor_runs/ltail_cv_oof_f1.npz` | `d1b57d3c8bc0077e` |
| `artifacts/factor_runs/ltail_cv_oof_f2.npz` | `78b0d3388d8bbd9c` |
| `artifacts/factor_runs/ltail_cv_oof_f3.npz` | `bfa0cd5a7a2d02ea` |
| `artifacts/factor_runs/ltail_cv_oof_f4.npz` | `763491cdd3d27f2c` |
| `artifacts/factor_runs/ltail_cv_test.npz` | `243795a136c33413` |
| `artifacts/factor_runs/mnli7_cv_oof_f0.npz` | `872ccacc3ab3e807` |
| `artifacts/factor_runs/mnli7_cv_oof_f1.npz` | `955201a58e2f0c3b` |
| `artifacts/factor_runs/mnli7_cv_oof_f2.npz` | `274cf0f4309f67bd` |
| `artifacts/factor_runs/mnli7_cv_oof_f3.npz` | `995374d373b62e7c` |
| `artifacts/factor_runs/mnli7_cv_oof_f4.npz` | `90124fc604367271` |
| `artifacts/factor_runs/mnli7_cv_test.npz` | `5068e47ecfaa74bf` |
| `artifacts/factor_runs/mnli_cv_oof_f0.npz` | `82d678b9b504ae79` |
| `artifacts/factor_runs/mnli_cv_oof_f1.npz` | `9cb04b22c90afd74` |
| `artifacts/factor_runs/mnli_cv_oof_f2.npz` | `d5bff9d3dcd22126` |
| `artifacts/factor_runs/mnli_cv_oof_f3.npz` | `39d6ce15be48b992` |
| `artifacts/factor_runs/mnli_cv_oof_f4.npz` | `fb21ab50fe3e38d9` |
| `artifacts/factor_runs/nli_ensemble_dmnli_test.npz` | `93f0f17cff9824cc` |
| `artifacts/factor_runs/nli_ensemble_mnli_test.npz` | `60dff9cff6233ddc` |
| `artifacts/factor_runs/prot_cv_oof_f0.npz` | `4e5f278aa72716e4` |
| `artifacts/factor_runs/prot_cv_oof_f1.npz` | `9d912444dca2eb53` |
| `artifacts/factor_runs/prot_cv_oof_f2.npz` | `4abd162a0a5e68d8` |
| `artifacts/factor_runs/prot_cv_oof_f3.npz` | `a8cd55716f31f984` |
| `artifacts/factor_runs/prot_cv_oof_f4.npz` | `30fd7c6f56e25465` |
| `artifacts/factor_runs/prot_cv_test.npz` | `5a2ff7cfe14b6c0b` |
| `artifacts/factor_runs/qnli_cv_oof_f0.npz` | `b41eb3d3b1f09b95` |
| `artifacts/factor_runs/qnli_cv_oof_f1.npz` | `a7bdf0b81b9a4397` |
| `artifacts/factor_runs/qnli_cv_oof_f2.npz` | `99cc1c97679fb0c1` |
| `artifacts/factor_runs/qnli_cv_oof_f3.npz` | `23bf1223228cb083` |
| `artifacts/factor_runs/qnli_cv_oof_f4.npz` | `4f0e8fe0f6e36c9b` |
| `artifacts/factor_runs/qnli_cv_test.npz` | `5d1a8c1b44e8b673` |
| `artifacts/ensembled_risk_catonly2.csv` | `593f16f3524a9793` |
| `artifacts/nli_risk_test_probs.csv` | `18be14ed9013a38a` |

## Gated (not in this repository)

34 files that carry verbatim post text: the organizers' posts, the annotated training set, evidence spans quoted from posts, or reasoning traces about posts. They are available only to researchers who have accepted the dataset's terms (see `docs/DATA_AND_ETHICS.md`). After placing them, run `python tools/verify_bundle.py`, which checks every file against the hashes below.

`artifacts/factors_scaled_test.csv` is not listed: step 1 of the reproduction writes it.

| File | Contains | SHA-256 |
|---|---|---|
| `data/test.csv` | organizers' test posts | `32eb5efefa6b8be0c68fb63e3671a1b59d62136fb2f74e3c5e8aa04938037af8` |
| `artifacts/full_train.csv` | annotated training set (posts and labels) | `399cd41c9520440bc0f0c202f72c83be1fbf81eac0056bd9a69b057d3a1aa12a` |
| `artifacts/cot_soft_oof.csv.extra.csv` | reasoning traces about posts | `9cc42f46da10dc304e44bd9994f2b72b17f3160c3a9fd90233dfe03c49c0d627` |
| `artifacts/factors_tfit_test.csv` | evidence spans | `7803eb99bc5ce77612a7f9fb0b80ddda62e223a9e94e631a92629d7b04eaf31b` |
| `submissions/NIDHI_scaled.csv` | the submitted predictions (contain evidence spans) | `5a088e536a6c0e6146aa7f4b29623fc0dfb9b28ed2c35177810f1e0b67a87a44` |
| `submissions/NIDHI.csv` | identical copy of the submitted predictions | `5a088e536a6c0e6146aa7f4b29623fc0dfb9b28ed2c35177810f1e0b67a87a44` |
| `artifacts/evidence_runs_ff11/evidence_ep9_rc_test_candidates.csv` | evidence-tagger candidate spans | `a969fdd01f218f0e5fe512fc22d4a85444537832b895688815eefe63b5217b59` |
| `artifacts/evidence_runs_ff11/evidence_ep9_rc_test_pred.csv` | evidence-tagger candidate spans | `65b11bd1e50bff644b19ae5576ed8fcfde47aa680b1c1c802cd75389c59296c6` |
| `artifacts/evidence_runs_ff11/evidence_ep9_rc_val_candidates.csv` | evidence-tagger candidate spans | `f9f44531b1920cc2cf6c12a735637391e5b309ee7d6bf0be94d99bcdb05fb058` |
| `artifacts/evidence_runs_ff11/evidence_ep9_rc_val_pred.csv` | evidence-tagger candidate spans | `afa7a9862ef116e46729ea8b488fc9b9807353a9f69d5ed310d40df0c750d4b7` |
| `artifacts/evidence_runs_ff11/evidence_lr2_rc_test_candidates.csv` | evidence-tagger candidate spans | `6b36f52889eb57d1cfe1032b933ca7057852b2fc8f051421c3ea30015415d34a` |
| `artifacts/evidence_runs_ff11/evidence_lr2_rc_test_pred.csv` | evidence-tagger candidate spans | `a91ca71efc9e337b1ca4472f732364460de5071803e769e54341a95fb12926a4` |
| `artifacts/evidence_runs_ff11/evidence_lr2_rc_val_candidates.csv` | evidence-tagger candidate spans | `85fe4e72571af0248997b1da71370eec07d47c5b3b5313c2b58c7a6a1484b920` |
| `artifacts/evidence_runs_ff11/evidence_lr2_rc_val_pred.csv` | evidence-tagger candidate spans | `4cbe97dc669c8a5c88db9f56b5807d09c6314c4c8d01c0dcf5ae66c9fc0c7814` |
| `artifacts/evidence_runs_ff11/evidence_s14_rc_test_candidates.csv` | evidence-tagger candidate spans | `77d0925247cc2d28034802ae7d33012bda015b33623cfe207ac436ea891f2df3` |
| `artifacts/evidence_runs_ff11/evidence_s14_rc_test_pred.csv` | evidence-tagger candidate spans | `4f401f1b78818eb0dbd807de8430239722e246bc9b0cb25dc30df930e157deed` |
| `artifacts/evidence_runs_ff11/evidence_s14_rc_val_candidates.csv` | evidence-tagger candidate spans | `bbe37010d92faf9d7fa85379fb56a611d258f1f2ced1381d0c4404b150ca50ca` |
| `artifacts/evidence_runs_ff11/evidence_s14_rc_val_pred.csv` | evidence-tagger candidate spans | `bcea3284808c3de7915de6cb5e40a6a02c9893f8d81daab7e8bfe4cdd03ddea2` |
| `artifacts/evidence_runs_ff11/evidence_s15_rc_test_candidates.csv` | evidence-tagger candidate spans | `d46cf5172f6b2f7f8ed4f90805a87662a73e45d3d32e4a3a9ccf913733ffbbe4` |
| `artifacts/evidence_runs_ff11/evidence_s15_rc_test_pred.csv` | evidence-tagger candidate spans | `4fe21967046bc38d70234a327c7b4b9448968415557db410a0ed7616aedf94a9` |
| `artifacts/evidence_runs_ff11/evidence_s15_rc_val_candidates.csv` | evidence-tagger candidate spans | `c929cc9d7601f34832356dca6b69a4d8f51c0f17ae523fdd79d73ba2e0fd2f2b` |
| `artifacts/evidence_runs_ff11/evidence_s15_rc_val_pred.csv` | evidence-tagger candidate spans | `07117bd53754f4abaa58f2547839f463a3772e7bc0ca1ff27946197306d96dcc` |
| `artifacts/evidence_runs_ff11/evidence_s16_rc_test_candidates.csv` | evidence-tagger candidate spans | `33009d3749741c8ade5817a8a70454f71253af4c70b099df682e5411f162eede` |
| `artifacts/evidence_runs_ff11/evidence_s16_rc_test_pred.csv` | evidence-tagger candidate spans | `3ddfcf83fcc97a1d147f34ab3361fe6b05d7ce7175fde0cf69fc94902d7d10c9` |
| `artifacts/evidence_runs_ff11/evidence_s16_rc_val_candidates.csv` | evidence-tagger candidate spans | `6faa726469f6c7b0455ad249d32fc7a30342fd8f2986aa0429f170fa7271f4b1` |
| `artifacts/evidence_runs_ff11/evidence_s16_rc_val_pred.csv` | evidence-tagger candidate spans | `550d3290c5d53648013dd0c8450989f6e8b64f332a14c930fcc4b90ffcc59971` |
| `artifacts/evidence_runs_ff11/evidence_s17_rc_test_candidates.csv` | evidence-tagger candidate spans | `86e366e2c07faa6b66b22ce7051874e8efe6120cddc66564fd04166e19c9b069` |
| `artifacts/evidence_runs_ff11/evidence_s17_rc_test_pred.csv` | evidence-tagger candidate spans | `3d92da1319bd83638905ec38567b1307e63aed9ca9ed4b6c7d547fbe877bb68c` |
| `artifacts/evidence_runs_ff11/evidence_s17_rc_val_candidates.csv` | evidence-tagger candidate spans | `6de94d733ddab2f08815d84842b1492069f7d36493f65b1c59fef59e1c1c9dc6` |
| `artifacts/evidence_runs_ff11/evidence_s17_rc_val_pred.csv` | evidence-tagger candidate spans | `6db2cd49df8d274a76a0f9fb6ba48ca9cff3370f0a96a6750cc008ea5cbf0b6e` |
| `artifacts/evidence_runs_ff11/evidence_v1_rc_test_candidates.csv` | evidence-tagger candidate spans | `826db57954f7354a8790319caf707b2d14b09f032bef7898fc2b59f420e910e5` |
| `artifacts/evidence_runs_ff11/evidence_v1_rc_test_pred.csv` | evidence-tagger candidate spans | `1019ffea3656891a2e2b22dac286538b6ccbd88656fd4abb3d2611cd2a3c0373` |
| `artifacts/evidence_runs_ff11/evidence_v1_rc_val_candidates.csv` | evidence-tagger candidate spans | `8e3b7cdc5d46ba5ffac4cb8ead71d0b6700b9680ba84aadb6a9a05e56acb4ef5` |
| `artifacts/evidence_runs_ff11/evidence_v1_rc_val_pred.csv` | evidence-tagger candidate spans | `41876173bdd33bc478879ce01ca4fba4ab2255c54e33dc5ddf562a59c992eb45` |

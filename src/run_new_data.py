"""Score a NEW test set end to end, in the champion configuration.

    python src/run_new_data.py --test your_test.csv --out submissions/NIDHI_new.csv
    python src/run_new_data.py --test your_test.csv --dry-run    # print, run nothing
    python src/run_new_data.py --test your_test.csv --skip-route # skip the ~20h stage

Input  : a CSV with data/test.csv's header -- row_id,anon_user_id,post_id,post
Output : row_id,risk_level,evidence,factors

Nothing is refitted on the new rows. Every threshold, weight, member and seed
is the champion's; only the posts change. The organizers are checking whether
the solution generalises, so a run that retuned itself on their data would
answer the wrong question.

This calls only scripts that already exist in this package. It exists as one
entry point because four details are easy to get wrong and fail QUIETLY,
producing a plausible but wrong column rather than an error:

  1. FACTOR_DEFS_JSON, FACTOR_ASPECTS_JSON, PROTECTIVE_TEMPLATE,
     CONFUSABLE_JSON and FACTOR_SUBSET are read at INFERENCE time. Inherited
     from a login shell they silently change what the factor arms predict.
  2. The `prot` arm was trained with PROTECTIVE_TEMPLATE=1 and every other arm
     with it off. The wrong setting yields plausible garbage.
  3. The 11 factor arms come in three families with three loaders and three
     on-disk layouts. One loader cannot read another's checkpoints.
  4. The evidence taggers must be conditioned on the risk column THIS run
     produces. Conditioning them on a stored submission binds them to our 378
     row_ids, which do not exist in the new data.

Requires a GPU: this is 62 model loads over the input posts.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RC7 = ("evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,"
       "evidence_s16_rc,evidence_s17_rc,evidence_v1_rc")
ENC_ARMS = ["mnli", "dmnli", "cmine", "prot", "d4", "ltail", "cbnsynapi"]
LLM_ARMS = ["qnli", "mnli7", "lnli"]
ALL_ARMS = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli",
            "ltail", "cbnsynapi", "friendfac"]
RISK_SEEDS = "runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15"
NLI_RUNS = "runs/risk_nli,runs/risk_nli_s2"

# (1) read at inference; inherited values silently change the predictions
CORRUPTING = ["FACTOR_DEFS_JSON", "FACTOR_ASPECTS_JSON", "PROTECTIVE_TEMPLATE",
              "CONFUSABLE_JSON", "FACTOR_SUBSET"]


class Runner:
    def __init__(self, dry):
        self.dry = dry
        self.n = 0

    def __call__(self, argv, stage, env=None):
        """Run one stage. Any non-zero exit stops the pipeline."""
        self.n += 1
        shown = " ".join(argv[1:])
        print(f"\n--- [{self.n}] {stage}\n    python {shown}", flush=True)
        if self.dry:
            return
        e = dict(os.environ)
        for k in CORRUPTING:
            e.pop(k, None)
        if env:
            e.update(env)
        r = subprocess.run([sys.executable] + argv[1:], cwd=ROOT, env=e)
        if r.returncode != 0:
            sys.exit(f"\nFAILED at stage: {stage}\n"
                     f"(exit {r.returncode}) -- nothing after this ran.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="CSV of posts to score")
    ap.add_argument("--out", default="submissions/NIDHI_new.csv")
    ap.add_argument("--workdir", default="artifacts/new_run")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command in order and run nothing")
    ap.add_argument("--skip-route", action="store_true",
                    help="skip the ~20h Qwen2.5-72B routing stage; the "
                         "submission still builds but the risk column ships "
                         "without attempt routing and scores lower")
    ap.add_argument("--qwen", default=os.environ.get(
        "QWEN", "Qwen/Qwen2.5-72B-Instruct"),
        help="the routing model: a local directory, or a Hub id to download "
             "(~145 GB). Compute nodes are often offline, so point this at a "
             "local copy: --qwen /path/to/Qwen2.5-72B-Instruct, or set QWEN. "
             "The submitted system used a local directory, not the Hub.")
    ap.add_argument("--cot-batch", default=None,
                    help="batch size for the routing stage; leave unset for "
                         "the probe's own default")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="resolve every pretrained model from the local "
                         "Hugging Face cache and never touch the network. "
                         "Use this on compute nodes without internet, where "
                         "the default is 5 slow retries per model before it "
                         "falls back to the cache anyway.")
    a = ap.parse_args()

    os.chdir(ROOT)
    if not os.path.isfile(a.test):
        sys.exit(f"no such file: {a.test}")
    D = a.workdir
    for d in (D, f"{D}/factor_runs", f"{D}/evidence_runs"):
        os.makedirs(d, exist_ok=True)

    if a.offline:
        # Cache-only. Without this, every stage that builds a tokenizer from
        # microsoft/deberta-v3-large tries the network first, retries 5 times
        # with backoff, and only then reads the cache it already had -- about
        # 25 wasted seconds per model load, and on some setups a hard failure
        # rather than a fallback.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    run = Runner(a.dry_run)
    print("=" * 62)
    print(f" scoring {a.test}")
    print(f" workspace {D}   ->   {a.out}")
    if a.skip_route:
        print(" NOTE: --skip-route, attempt routing DISABLED")
    if a.offline:
        print(" NOTE: --offline, pretrained models come from the local cache")
    print("=" * 62)

    if not a.skip_preflight:
        run(["py", "src/check_setup.py", "--test", a.test],
            "preflight: are all 62 checkpoints present?")

    # 1. The MentalRoBERTa tri-head is BOTH the second risk member and the
    #    friendfac factor arm, and one pass produces both. It reads the
    #    vendored encoder beside its checkpoints, so the GATED
    #    mental/mental-roberta-base repo is never contacted.
    friend = f"{D}/friend_risk_probs.csv"
    run(["py", "src/friend_factor_predict.py", "--test", a.test,
         "--tag", "friendfac", "--outdir", f"{D}/factor_runs",
         "--dump-risk", friend],
        "MentalRoBERTa tri-head: risk member + friendfac arm")

    # 2. risk
    run(["py", "src/ensemble_risk.py", "--test", a.test, "--cat-only",
         "--our-risk-run", RISK_SEEDS, "--friend-probs", friend,
         "--out", f"{D}/ensembled_risk.csv"],
        "risk seed ensemble (3 seeds + tri-head member)")

    run(["py", "src/dump_nli_probs.py", "--data", a.test,
         "--nli-run", NLI_RUNS, "--out", f"{D}/nli_risk_probs.csv"],
        "risk entailment arm")

    # 3. routing -- by far the slowest stage
    route = f"{D}/route.csv"
    if a.skip_route:
        run(["py", "src/_new_run_helpers.py", "noop-route",
             "--test", a.test, "--out", route],
            "routing SKIPPED (--skip-route): writing a no-op route file")
    else:
        # --load-4bit matches how the submitted system ran this stage; the
        # model is used as-is, prompted, so nothing here is fine-tuned.
        cot = ["py", "src/risk_cot_probe.py", "--model", a.qwen, "--soft",
               "--load-4bit", "--extra-csv", a.test, "--out", f"{D}/cot.csv"]
        if a.cot_batch:
            cot += ["--batch", str(a.cot_batch)]
        run(cot, "chain-of-thought routing (Qwen2.5-72B, ~20h)")
        if not a.dry_run:
            # --extra-csv writes the new rows to <out>.extra.csv
            src = (f"{D}/cot.csv.extra.csv"
                   if os.path.isfile(f"{D}/cot.csv.extra.csv")
                   else f"{D}/cot.csv")
            shutil.copyfile(src, route)

    # 4. the risk column, then the evidence taggers conditioned on IT
    run(["py", "src/_new_run_helpers.py", "risk-column", "--test", a.test,
         "--workdir", D, "--route", route],
        "risk column (0.70*ensemble + 0.30*entailment -> rules -> routing)")

    run(["py", "src/evidence_dump_all.py", "--runs", RC7, "--test", a.test,
         "--val", f"{D}/val_stub.csv", "--outdir", f"{D}/evidence_runs",
         "--risk-cond", "--test-risk", f"{D}/risk_col.csv",
         "--val-risk", f"{D}/val_stub_risk.csv"],
        "7 evidence taggers, conditioned on this run's own risk column")

    run(["py", "src/_new_run_helpers.py", "check-evidence", "--workdir", D],
        "refuse a partial evidence vote")

    # 5. the 11 factor arms, three families
    for arm in ENC_ARMS:
        runs = ",".join(f"runs/factors_{arm}_cv_f{f}" for f in range(5))
        # (2) prot alone was trained with the protective template ON
        env = {"PROTECTIVE_TEMPLATE": "1"} if arm == "prot" else None
        run(["py", "src/predict_factors_nli.py", "--test", a.test,
             "--nli-run", runs,
             "--dump-probs", f"{D}/factor_runs/{arm}_cv_test.npz",
             "--out", f"{D}/factor_runs/{arm}_labels.csv"],
            f"factor arm {arm} (encoder)"
            + (" with PROTECTIVE_TEMPLATE=1" if arm == "prot" else ""),
            env=env)

    for arm in LLM_ARMS:
        run(["py", "src/_new_run_helpers.py", "lora-arm", "--arm", arm,
             "--test", a.test, "--outdir", f"{D}/factor_runs"],
            f"factor arm {arm} (LoRA adapter on its recorded base model)")

    # friendfac was produced in stage 1, in the same pass as the risk member.

    run(["py", "src/factor_threshold_stabilise.py", "--arms", *ALL_ARMS,
         "--use", "tfit", "--match-oof-scale", "--cold-test", a.test,
         "--test-dir", f"{D}/factor_runs", "--build-test", f"{D}/factors.csv"],
        "factor decode, COLD (--cold-test refits nothing on the new rows)")

    # 6. assemble
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    run(["py", "src/build_submission.py", "--test", a.test,
         "--ensembled", f"{D}/ensembled_risk.csv",
         "--nli", f"{D}/nli_risk_probs.csv", "--route", route,
         "--factors", f"{D}/factors.csv", "--ev-dir", f"{D}/evidence_runs",
         "--reference", f"{D}/__no_reference__.csv", "--team", "NIDHI",
         "--out", a.out],
        "assemble all three columns")

    print("\n" + "=" * 62)
    if a.dry_run:
        print(f" DRY RUN: {run.n} stages listed, nothing executed.")
    else:
        print(f" wrote {a.out}")
        if a.skip_route:
            print(" NOTE: attempt routing was skipped (--skip-route).")
    print("=" * 62)


if __name__ == "__main__":
    main()

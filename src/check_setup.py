"""Preflight check. Run this BEFORE anything else.

Reports exactly which model checkpoints and data files are present and which
are missing, so a run fails here with a readable list instead of failing later
with a FileNotFoundError from inside pandas.

    python src/check_setup.py                 # check for a run on new posts
    python src/check_setup.py --test data/my_new_test.csv

Exit code 0 if everything needed is present, 1 otherwise.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACTOR_ARMS = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7",
               "lnli", "ltail", "cbnsynapi", "friendfac"]
EVIDENCE_TAGGERS = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
                    "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
                    "evidence_v1_rc"]
RISK_SEEDS = ["risk_v1", "risk_v1_s14", "risk_v1_s15"]
RISK_NLI = ["risk_nli", "risk_nli_s2"]


def has_weights(d):
    """A run directory counts as present only if it holds real weights.

    Two storage shapes, not one. The encoder arms save a state dict as
    model_final.pt or model.pt. The LoRA arms (qnli, mnli7, lnli) save a PEFT
    adapter into a lora/ subdirectory and have NO .pt at all -- checking only
    for .pt files reported all three as 0/5 folds and refused to start, on a
    machine where they were present and complete.
    """
    p = os.path.join(ROOT, "runs", d)
    if not os.path.isdir(p):
        return False
    if any(os.path.isfile(os.path.join(p, f))
           for f in ("model_final.pt", "model.pt")):
        return True
    lora = os.path.join(p, "lora")
    return any(os.path.isfile(os.path.join(lora, f))
               for f in ("adapter_model.safetensors", "adapter_model.bin"))


def report(title, items, kind="run"):
    print(f"\n{title}")
    ok, missing = [], []
    for it in items:
        (ok if has_weights(it) else missing).append(it)
    for it in ok:
        print(f"  present  runs/{it}")
    for it in missing:
        print(f"  MISSING  runs/{it}")
    print(f"  -> {len(ok)}/{len(items)} present")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.csv",
                    help="the CSV of posts to be scored")
    a = ap.parse_args()

    print("=" * 68)
    print("PREFLIGHT: what this system needs in order to score new posts")
    print("=" * 68)

    missing = []
    missing += report("Risk, 3 seeds:", RISK_SEEDS)
    missing += report("Risk, entailment arm:", RISK_NLI)
    missing += report("Evidence, 7 risk-conditioned taggers:", EVIDENCE_TAGGERS)

    # Three conventions are in use, so check all of them before reporting a
    # member missing:
    #   runs/factors_{arm}_cv_f0..f4   most arms
    #   runs/factors_{arm}_f0..f4      qnli, mnli7, lnli (LoRA arms)
    #   friendfac has no runs/ dir at all; it comes from src/friend_pipeline
    print("\nFactors, 11 members x 5 folds:")
    bad_members = []
    for arm in FACTOR_ARMS:
        cv = [f"factors_{arm}_cv_f{i}" for i in range(5)]
        lo = [f"factors_{arm}_f{i}" for i in range(5)]
        n_cv = sum(has_weights(f) for f in cv)
        n_lo = sum(has_weights(f) for f in lo)
        n, pat = (n_cv, "_cv_f0..f4") if n_cv >= n_lo else (n_lo, "_f0..f4")
        if arm == "friendfac" and n == 0:
            ok = os.path.isdir(os.path.join(ROOT, "src", "friend_pipeline"))
            print(f"  {'present ' if ok else 'MISSING '} friendfac"
                  f"  (from src/friend_pipeline, not runs/)")
            if not ok:
                bad_members.append(arm)
            continue
        flag = "present " if n == 5 else "MISSING "
        print(f"  {flag} runs/factors_{arm}{pat}   ({n}/5 folds)")
        if n != 5:
            bad_members.append(arm)
    missing += bad_members
    print(f"  -> {len(FACTOR_ARMS) - len(bad_members)}"
          f"/{len(FACTOR_ARMS)} members complete")

    print("\nData:")
    tp = os.path.join(ROOT, a.test)
    print(f"  {'present ' if os.path.isfile(tp) else 'MISSING '} {a.test}")
    if not os.path.isfile(tp):
        missing.append(a.test)

    print("\nExternal models, downloaded at first use, not checked here:")
    print("  microsoft/deberta-v3-large")
    print("  mental/mental-roberta-base            (gated, needs a Hub account)")
    print("  MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    print("  Qwen2.5-72B-Instruct                  (routing step only)")

    print("\n" + "=" * 68)
    if missing:
        print(f"NOT READY: {len(missing)} required items are missing.")
        print("The weights are too large to ship in the code package. See the")
        print("'Getting the model weights' section of README.md.")
        sys.exit(1)
    print("READY: every checkpoint this system needs is present.")
    sys.exit(0)


if __name__ == "__main__":
    main()

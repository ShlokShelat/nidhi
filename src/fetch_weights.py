"""Download the model checkpoints needed to score a NEW test set.

The submission package carries code plus the ~4 MB of model outputs needed to
rebuild the submitted predictions. Scoring different posts needs the models
themselves, which are about 100 GB and are distributed separately.

    python src/fetch_weights.py --repo <user>/nidhi-suicide-risk-weights
    python src/fetch_weights.py --repo ... --token hf_...   # if repo is private

Files land in runs/ and artifacts/friend_ckpt/ with the exact names the
pipeline expects. The download resumes, so an interrupted run can be repeated.
Afterwards `python src/check_setup.py` should print READY.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="Hugging Face repo id holding the checkpoints")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="read token; needed while the repo is private")
    ap.add_argument("--dest", default=ROOT,
                    help="where to unpack (default: the repository root)")
    a = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed. "
                 "Run: pip install -r requirements.txt")

    print(f"downloading {a.repo} -> {a.dest}")
    print("About 100 GB. This resumes if interrupted, so re-run on failure.\n")

    # hf_transfer is several times faster on a fast link and is a no-op if the
    # package is absent, so it is requested rather than required.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    snapshot_download(repo_id=a.repo, repo_type="model", local_dir=a.dest,
                      token=a.token, max_workers=8)

    # The LoRA factor arms (qnli, mnli7, lnli) are adapters. Their
    # run_config.json records the base model as an absolute path on the
    # cluster they were trained on, which does not exist here. upload_weights.py
    # resolved each of those paths to a Hub id at publish time -- that is done
    # there and not here because only the training machine can inspect the
    # directory the path points at. Apply the mapping.
    bases = os.path.join(a.dest, "lora_base_models.json")
    if os.path.isfile(bases):
        rewritten, unresolved = 0, []
        for arm, rec in json.load(open(bases)).items():
            hid = rec.get("hub_id") if isinstance(rec, dict) else None
            if not hid:
                unresolved.append(arm)
                continue
            cfg = os.path.join(a.dest, "runs", arm, "run_config.json")
            if not os.path.isfile(cfg):
                continue
            d = json.load(open(cfg))
            local = str(d.get("model", ""))
            if os.path.isabs(local) and not os.path.exists(local):
                d["model"] = hid
                json.dump(d, open(cfg, "w"), indent=2)
                rewritten += 1
        if rewritten:
            print(f"\nrewrote {rewritten} LoRA base-model paths to Hub ids")
        if unresolved:
            print(f"\nWARNING: no Hub id recorded for {unresolved}. Those "
                  f"factor arms will fail; set the `model` field of their "
                  f"runs/<arm>/run_config.json by hand.")

        # These two bases are GATED on Hugging Face. Say so now, not 20 hours
        # into a scoring run.
        gated = sorted({rec["hub_id"] for rec in json.load(open(bases)).values()
                        if isinstance(rec, dict)
                        and rec.get("hub_id", "").split("/")[0]
                        in ("meta-llama", "mistralai")})
        if gated:
            print("\nNOTE: 3 of the 11 factor arms are LoRA adapters over base "
                  "models that\nare GATED on Hugging Face. Accept their terms "
                  "while signed in, or those\narms cannot load:")
            for g in gated:
                print(f"  https://huggingface.co/{g}")

    print("\nDone. Now run:  python src/check_setup.py")


if __name__ == "__main__":
    main()

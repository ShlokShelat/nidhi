"""Publish the checkpoints the organizers need to score a NEW test set.

Run this ON THE MACHINE THAT HOLDS runs/ (the training cluster), inside
screen or tmux -- it moves about 100 GB and takes hours.

    export HF_TOKEN=hf_...        # a WRITE token
    python src/upload_weights.py --repo <your-hf-username>/nidhi-weights

The repo is created PRIVATE. These weights are derived from a clinical corpus
and can memorise training text, so they are not published openly by default;
give the organizers read access instead, or pass --public if they ask for it.

Staging uses hard links where possible, so it needs no extra disk. Only
inference files are published -- weights, run_config.json, tokenizer and LoRA
adapter directories. Training logs and per-run validation predictions are
excluded, because the latter are model output over corpus posts.
"""
import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENC_ARMS = ["mnli", "dmnli", "cmine", "prot", "d4", "ltail", "cbnsynapi"]
LLM_ARMS = ["qnli", "mnli7", "lnli"]
EVIDENCE = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
            "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
            "evidence_v1_rc"]
RISK = ["risk_v1", "risk_v1_s14", "risk_v1_s15", "risk_nli", "risk_nli_s2"]

# whitelist: everything inference needs, and nothing else
KEEP_FILES = ["model_final.pt", "model.pt", "run_config.json"]
KEEP_DIRS = ["tok", "lora"]


def hub_id(path):
    """Turn a local base-model path into the Hub id that names the same model.

    The LoRA arms record wherever their base model sat on THIS machine, which
    is one of three shapes, none of which survives the trip to another
    machine:

      /…/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<sha>
      /home/<user>/qwen/Qwen2.5-14B-Instruct
      (anything else a future run might record)

    Taking the basename is wrong for all of them: the first two yield a commit
    hash, the third drops the org. Returns (id, how_it_was_found) so the
    caller can print the mapping and a human can sanity-check it before 86 GB
    moves.
    """
    p = path.rstrip("/")

    # 1. the Hugging Face cache layout: models--<org>--<name>/snapshots/<sha>
    m = re.search(r"models--([^/]+)--([^/]+)[/\\]snapshots", p + "/snapshots")
    if m:
        return f"{m.group(1)}/{m.group(2)}", "hf-cache path"

    # 2. a plain directory: transformers records where it came from
    cfg = os.path.join(p, "config.json")
    if os.path.isfile(cfg):
        try:
            v = str(json.load(open(cfg)).get("_name_or_path", "")).strip()
            if v and "/" in v and not os.path.isabs(v):
                return v, "config.json _name_or_path"
        except Exception:
            pass

    # 3. the well-known families, by directory name
    base = os.path.basename(p)
    for prefix, org in (("Qwen", "Qwen"), ("Llama", "meta-llama"),
                        ("Meta-Llama", "meta-llama"),
                        ("Mistral", "mistralai"), ("deberta", "microsoft")):
        if base.startswith(prefix):
            return f"{org}/{base}", "known family"

    return base, "UNRESOLVED -- check this by hand"


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)          # same filesystem: no extra disk
    except OSError:
        shutil.copy2(src, dst)


def take(run, stage, missing):
    d = os.path.join(ROOT, "runs", run)
    if not os.path.isdir(d):
        missing.append(f"runs/{run}")
        return 0
    n = 0
    for f in KEEP_FILES:
        p = os.path.join(d, f)
        if os.path.isfile(p):
            link_or_copy(p, os.path.join(stage, "runs", run, f))
            n += 1
    for sub in KEEP_DIRS:
        s = os.path.join(d, sub)
        if os.path.isdir(s):
            for dirpath, _, files in os.walk(s):
                for f in files:
                    src = os.path.join(dirpath, f)
                    rel = os.path.relpath(src, d)
                    link_or_copy(src, os.path.join(stage, "runs", run, rel))
                    n += 1
    if n == 0:
        missing.append(f"runs/{run} (present but holds no weights)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="<user>/<repo> on the Hub")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--stage", default=os.path.join(ROOT, "_weights_upload"))
    ap.add_argument("--public", action="store_true",
                    help="create a PUBLIC repo (default is private)")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage and report, but upload nothing")
    a = ap.parse_args()

    # A token can come from three places, in this order: --token, the HF_TOKEN
    # environment variable, or the cached login written by `huggingface-cli
    # login`. Only the first two are visible here, so an earlier version
    # rejected a perfectly good cached login. Fall through to the cache and let
    # whoami() below be the real check.
    if not a.token and not a.dry_run:
        try:
            from huggingface_hub import get_token
            a.token = get_token()
        except Exception:
            a.token = None
        if not a.token:
            sys.exit("no Hugging Face credentials found. Either run\n"
                     "    huggingface-cli login\n"
                     "or pass --token hf_..., or set HF_TOKEN.")

    runs = RISK + EVIDENCE
    runs += [f"factors_{x}_cv_f{f}" for x in ENC_ARMS for f in range(5)]
    runs += [f"factors_{x}_f{f}" for x in LLM_ARMS for f in range(5)]

    print(f"staging {len(runs)} run directories into {a.stage}")
    missing, total = [], 0
    for r in runs:
        total += take(r, a.stage, missing)

    # the tri-head lives under artifacts/, not runs/, and carries the local
    # encoder copy that removes the gated-repo dependency
    fc = os.path.join(ROOT, "artifacts", "friend_ckpt")
    if os.path.isdir(fc):
        for dirpath, _, files in os.walk(fc):
            for f in files:
                src = os.path.join(dirpath, f)
                rel = os.path.relpath(src, os.path.join(ROOT, "artifacts"))
                link_or_copy(src, os.path.join(a.stage, "artifacts", rel))
                total += 1
    else:
        missing.append("artifacts/friend_ckpt")

    # A LoRA adapter is unusable without its base model, and the path recorded
    # at training time is absolute and specific to this machine. Recording the
    # names lets fetch_weights.py rewrite them to Hub ids on the far side --
    # this is the portability blocker, and writing it down is what removes it.
    bases = {}
    # `factors_*_f*` would also match factors_cv_final and factors_cv2_final,
    # recording two unrelated runs as LoRA arms. Enumerate the three arms
    # explicitly instead of pattern-matching directory names.
    for arm in [f"factors_{x}_f{f}" for x in LLM_ARMS for f in range(5)]:
        cfg = os.path.join(ROOT, "runs", arm, "run_config.json")
        if not os.path.isfile(cfg):
            continue
        try:
            local = str(json.load(open(cfg)).get("model", ""))
        except Exception as e:
            bases[arm] = {"local": f"UNREADABLE: {e}", "hub_id": "",
                          "resolved_by": "error"}
            continue
        hid, how = hub_id(local)
        bases[arm] = {"local": local, "hub_id": hid, "resolved_by": how}
    if bases:
        with open(os.path.join(a.stage, "lora_base_models.json"), "w") as f:
            json.dump(bases, f, indent=2)
        print(f"\nresolved {len(bases)} LoRA base models to Hub ids:")
        seen = {}
        for k, v in sorted(bases.items()):
            seen.setdefault((v["hub_id"], v["resolved_by"]), []).append(k)
        for (hid, how), arms in seen.items():
            print(f"  {hid:<38} [{how}]")
            print(f"      used by: {', '.join(arms)}")
        if any(v["resolved_by"].startswith("UNRESOLVED")
               for v in bases.values()):
            print("\n  WARNING: at least one base model could not be resolved "
                  "to a Hub id.\n  Those arms will fail on another machine. "
                  "Fix before uploading.")

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(a.stage) for f in fs)
    print(f"\nstaged {total} files, {size / 1e9:.1f} GB")

    if missing:
        print(f"\nREFUSING: {len(missing)} items are missing:")
        for m in missing:
            print(f"  {m}")
        sys.exit("A partial upload is worse than none -- the organizers would "
                 "get a model that loads but is not the one that was measured.")

    if a.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install 'huggingface_hub[hf_transfer]' first")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    api = HfApi(token=a.token)

    # Resolve the namespace rather than trusting one that was typed. Passing a
    # bare name is the normal case; a guessed username fails with a 403 only
    # AFTER staging, which is a slow way to learn you mistyped it.
    try:
        me = api.whoami()
        user = me.get("name", "")
    except Exception as e:
        sys.exit(f"could not identify the token's owner: {e}\n"
                 f"Run `huggingface-cli login`, or check the token is valid.")
    print(f"\nauthenticated as: {user}")

    repo = a.repo
    if "/" not in repo:
        repo = f"{user}/{repo}"
        print(f"  no namespace given, using {repo}")
    ns = repo.split("/")[0]
    if ns != user:
        orgs = [o.get("name") for o in me.get("orgs", [])]
        if ns not in orgs:
            sys.exit(
                f"\nthis token cannot write to the namespace '{ns}'.\n"
                f"  token owner : {user}\n"
                f"  its orgs    : {orgs or 'none'}\n"
                f"Use --repo {user}/{repo.split('/')[-1]} , or a namespace "
                f"listed above.")

    try:
        api.create_repo(repo, repo_type="model", private=not a.public,
                        exist_ok=True)
    except Exception as e:
        # A fine-grained token may be allowed to WRITE to a repo without being
        # allowed to CREATE one. That is a legitimate, tighter setup, so before
        # giving up check whether the repo already exists and is reachable.
        try:
            api.repo_info(repo, repo_type="model")
            print(f"  cannot create repos with this token, but {repo} already "
                  f"exists and is readable -- continuing")
        except Exception:
            sys.exit(
                f"\ncould not create or reach {repo}:\n  {e}\n\n"
                f"This token cannot create repositories. Two ways forward:\n"
                f"  1. Create a token of type WRITE at\n"
                f"     https://huggingface.co/settings/tokens\n"
                f"     then: huggingface-cli login\n"
                f"  2. Or create the repo by hand at\n"
                f"     https://huggingface.co/new  (name: "
                f"{repo.split('/')[-1]}, set it to Private)\n"
                f"     and give this fine-grained token write access to it,\n"
                f"     then re-run. Staging is already done and is re-used.")
    a.repo = repo
    print(f"\nuploading to {a.repo} "
          f"({'PUBLIC' if a.public else 'private'})...")

    # upload_large_folder arrived in huggingface_hub 0.25 and is the one to
    # use here: it shards the commit and resumes. transformers 4.46.3 only
    # requires >=0.23.2, so an older hub is entirely possible on a cluster --
    # and discovering that AFTER staging 100 GB would waste the whole run.
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=a.repo, folder_path=a.stage,
                                repo_type="model", num_workers=8)
    else:
        import huggingface_hub
        print(f"  huggingface_hub {huggingface_hub.__version__} has no "
              f"upload_large_folder; falling back to upload_folder.")
        print("  That is one large commit rather than many small ones, so it "
              "does NOT resume. If it fails, run:")
        print("      pip install -U 'huggingface_hub>=0.25'")
        print("  and re-run this script; staging is already done and is "
              "re-used.")
        api.upload_folder(repo_id=a.repo, folder_path=a.stage,
                          repo_type="model")

    print("\nDONE.")
    print("  1. On huggingface.co, give the organizers read access.")
    print(f"  2. In README.md, replace HF_REPO_PLACEHOLDER with: {a.repo}")


if __name__ == "__main__":
    main()

# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm can
# be rebuilt from source without the original project tree.
"""Central configuration. Everything tunable lives here."""
import os

# ---------------- paths ----------------
# VENDORING DEPTH. In the original project this file sat one level below the
# project root (<root>/src/config.py), so two dirname() calls reached the root.
# Here it sits two levels below (<root>/src/friend_pipeline/config.py), so it
# takes three. Getting this wrong does not raise: DATA_DIR would resolve to
# src/data, the .xlsx inputs would appear absent, and run_pipeline.py would exit
# reporting missing data files that are in fact present at the real root.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("FRIEND_DATA_DIR", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("FRIEND_OUT_DIR", os.path.join(ROOT, "outputs"))
TRAIN_XLSX = os.path.join(DATA_DIR, "train.xlsx")
LEADERBOARD_XLSX = os.path.join(DATA_DIR, "leaderboard.xlsx")

# ---------------- task constants ----------------
RISK_LEVELS = ["Indicator", "Ideation", "Behavior", "Attempt"]
FACTORS = [
    "mental health issues", "physical health/characteristic", "substance use",
    "hopelessness", "emotion dysregulation", "low self-esteem",
    "poor school performance", "low socio-economic status", "interpersonal violence",
    "prior self-harm or suicidal thought/attempt", "poor social support",
    "interpersonal difficulty", "dysfunctional family", "exposure to others' suicide",
    "stressful life event", "traumatic experience", "cognitive deficits",
    "suicide means (with access)", "sexual orientation related issues",
    "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
]

# ---------------- model ----------------
# Primary encoder. Pre-trained on Reddit mental-health text -> best domain fit.
# If the download fails, the code falls back to "roberta-base" automatically.
MODEL_NAME = "mental/mental-roberta-base"
FALLBACK_MODEL = "roberta-base"

MAX_LEN = 320          # tokens; covers the vast majority of posts (median ~50 tokens)
N_FOLDS = 5
EPOCHS = 8
LR_ENCODER = 2e-5
LR_HEADS = 1e-4
WARMUP_FRAC = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

# multi-task loss weights (risk / evidence-tagging / factors)
W_RISK, W_EVID, W_FACT = 1.0, 0.7, 1.2

# ---------------- inference / post-processing ----------------
EVID_THRESHOLD_GRID = [0.30, 0.40, 0.50, 0.60]  # tuned on out-of-fold preds
FACTOR_THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
MAX_SPAN_WORDS = 14     # safety cap: gold spans are short; the 3x-length rule
                        # kills over-long predictions, so we never emit long spans
MIN_SPAN_CHARS = 3

# ---------------- hardware auto-config ----------------
def hardware_profile():
    """Pick batch size / accumulation / checkpointing based on available VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            name = torch.cuda.get_device_name(0)
        else:
            return dict(device="cpu", batch=4, accum=4, fp16=False,
                        grad_ckpt=False, note="CPU (slow!)")
    except Exception:
        return dict(device="cpu", batch=4, accum=4, fp16=False,
                    grad_ckpt=False, note="CPU (slow!)")
    if vram_gb >= 20:
        p = dict(batch=32, accum=1, grad_ckpt=False)
    elif vram_gb >= 10:          # RTX A3000 12GB lands here
        p = dict(batch=16, accum=1, grad_ckpt=False)
    elif vram_gb >= 7:           # 8GB cards
        p = dict(batch=8, accum=2, grad_ckpt=False)
    else:                        # RTX A3000 6GB lands here
        p = dict(batch=4, accum=4, grad_ckpt=True)
    p.update(device="cuda", fp16=True, note=f"{name} ({vram_gb:.0f} GB)")
    return p

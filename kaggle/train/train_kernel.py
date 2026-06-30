
"""Kaggle script kernel: resume-train the mayalin LoRA headlessly.

Pushed via the Kaggle API (mirrors notebooks/train_lora_kaggle.ipynb), plus a
seed-checkpoint copy so training resumes from the 500-step LoRA via the repo's
``--network_weights`` warm-start. Attaches two datasets:
  * mayalin102   -> raw training images (under raw/)
  * mayalin-lora -> the 500-step seed checkpoint (lora.safetensors)
"""

import glob
import os
import shutil
import subprocess
import sys

# Reduce CUDA fragmentation — training OOMs on a 16GB T4 by a hair otherwise.
# Set before torch loads anywhere downstream (propagates to the trainer subprocess).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# === INJECTED BY ORCHESTRATOR (do not edit by hand) ===
CHARACTER = "mayalin"
STEPS = 500
HF_TOKEN = ""
# === END INJECTED ===

REPO_URL = "https://github.com/maivandangkhoa/ai-character-studio.git"
REPO_DIR = "/kaggle/working/ai-character-studio"

# FLUX.1-dev is gated. Prefer the injected token; fall back to a Kaggle Secret.
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF_TOKEN loaded from injected value.")
else:
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:  # noqa: BLE001
        print("WARN: could not read HF_TOKEN secret:", exc)

if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
subprocess.run([sys.executable, "scripts/setup.py"], check=True)

# A script kernel runs as /kaggle/src/script.py, so the repo dir isn't on the
# import path the way it is in a notebook — add it before importing `src`.
sys.path.insert(0, REPO_DIR)
from src.utils.paths import Paths  # noqa: E402

paths = Paths.load().ensure()

# Stage raw images (dataset mounts the raw/ folder under some /kaggle/input path).
cp = paths.character(CHARACTER).ensure()
staged = 0
for src in glob.glob("/kaggle/input/**/raw/*", recursive=True):
    if src.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        dst = cp.raw / os.path.basename(src)
        if not dst.exists():
            shutil.copyfile(src, dst)
            staged += 1
print(f"Staged {staged} raw images into {cp.raw}")

# Seed checkpoint so resume warm-starts from the 500-step LoRA. Name must contain
# "step<n>" for the repo's checkpoint discovery to pick it up.
lora_dir = paths.lora_dir(CHARACTER)
lora_dir.mkdir(parents=True, exist_ok=True)
seeds = [p for p in glob.glob("/kaggle/input/**/lora.safetensors", recursive=True) if "mayalin-lora" in p]
if seeds:
    seed_dst = lora_dir / "mayalin-step00000500.safetensors"
    if not seed_dst.exists():
        shutil.copyfile(seeds[0], seed_dst)
    print("Seed checkpoint ready:", seed_dst)
else:
    print("WARN: no mayalin-lora seed found; training will start fresh.")

# Download FLUX (training) weights, then train. resume auto-detects the seed.
subprocess.run([sys.executable, "scripts/download_models.py"], check=True)
subprocess.run(
    [sys.executable, "scripts/train.py", "--character", CHARACTER, "--steps", str(STEPS)],
    check=True,
)
print("Training done.")

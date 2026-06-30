"""Kaggle script kernel: generate images from the trained mayalin LoRA.

Driven by the VM web app via the Kaggle API. The orchestrator rewrites the
INJECTED block below (prompts/seed/character) before each `kaggle kernels push`.
The LoRA is read from the attached dataset `maivandangkhoa/mayalin-lora`.
Generated images are copied to /kaggle/working/images for easy output retrieval.
"""

import glob
import os
import shutil
import subprocess
import sys

# === INJECTED BY ORCHESTRATOR (do not edit by hand) ===
CHARACTER = "mayalin"
SEED = None
HF_TOKEN = ""
PROMPTS = [
    "portrait, soft studio lighting, looking at camera",
]
# === END INJECTED ===

REPO_URL = "https://github.com/maivandangkhoa/ai-character-studio.git"
REPO_DIR = "/kaggle/temp/repo"  # ephemeral: keep it out of the kernel output
LORA = "/kaggle/input/mayalin-lora/lora.safetensors"
IMAGES_OUT = "/kaggle/working/images"

# FLUX.1-dev is gated. Prefer the token injected by the orchestrator (from the
# VM's .env); fall back to a Kaggle Secret named HF_TOKEN if none was injected.
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
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
# Generation-only deps: newer diffusers/transformers that load a LoRA into a
# 4-bit NF4 FLUX transformer. requirements.txt's <4.50 transformers pin (for
# Florence-2 captioning, unused here) would pull an old, crashing diffusers.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-generate.txt"],
    check=True,
)

with open("prompts.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(PROMPTS))

if not os.path.exists(LORA):
    # Kaggle may mount the dataset under a different folder name than the slug.
    # Discover the LoRA anywhere under /kaggle/input instead of hardcoding it.
    matches = glob.glob("/kaggle/input/**/*.safetensors", recursive=True)
    if matches:
        LORA = matches[0]
        print("Discovered LoRA at:", LORA)
    else:
        print("No .safetensors under /kaggle/input. Tree:")
        for p in sorted(glob.glob("/kaggle/input/**/*", recursive=True)):
            print("  ", p)
        raise SystemExit("LoRA not found. Is the mayalin-lora dataset attached?")

cmd = [
    sys.executable,
    "scripts/generate.py",
    "--character",
    CHARACTER,
    "--prompts-file",
    "prompts.txt",
    "--lora",
    LORA,
]
if SEED is not None:
    cmd += ["--seed", str(SEED)]
subprocess.run(cmd, check=True)

# Collect images into a flat folder so the orchestrator can pull just these.
os.makedirs(IMAGES_OUT, exist_ok=True)
src_dir = f"/kaggle/working/AICharacter/outputs/generated/{CHARACTER}"
copied = 0
for path in glob.glob(os.path.join(src_dir, "*")):
    shutil.copy(path, IMAGES_OUT)
    copied += 1
print(f"Done: {copied} files in {IMAGES_OUT}")

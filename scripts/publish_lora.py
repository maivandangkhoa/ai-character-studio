"""Publish a trained LoRA to the Kaggle dataset that the generate kernel reads.

    python scripts/publish_lora.py /path/to/lora.safetensors

Creates ``maivandangkhoa/mayalin-lora`` on first run, then versions it on every
subsequent run. The generate kernel attaches this dataset as input.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

# The dataset-metadata.json shipped next to this repo defines the dataset id.
_META_DIR = Path(__file__).resolve().parent.parent / "kaggle" / "lora-dataset"


def _exists(dataset_id: str) -> bool:
    res = subprocess.run(
        ["kaggle", "datasets", "status", dataset_id],
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish lora.safetensors to Kaggle.")
    ap.add_argument("lora", help="Path to lora.safetensors")
    ap.add_argument("-m", "--message", default="update lora", help="Version note")
    args = ap.parse_args()

    lora = Path(args.lora)
    if not lora.exists():
        raise SystemExit(f"LoRA not found: {lora}")

    meta = _META_DIR / "dataset-metadata.json"
    if not meta.exists():
        raise SystemExit(f"Missing dataset metadata: {meta}")
    dataset_id = "maivandangkhoa/mayalin-lora"

    # Stage the single file alongside the metadata for the upload.
    shutil.copyfile(lora, _META_DIR / "lora.safetensors")
    # Upload the raw file (default dir-mode), NOT a zip: a zip-mode dataset mounts
    # as /kaggle/input/<slug>/<slug>.zip, so the kernel never finds lora.safetensors.
    if _exists(dataset_id):
        cmd = ["kaggle", "datasets", "version", "-p", str(_META_DIR), "-m", args.message]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(_META_DIR)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Published {lora.name} -> {dataset_id}")


if __name__ == "__main__":
    main()

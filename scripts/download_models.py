"""Download FLUX weights (transformer, AE, CLIP-L, T5-XXL) into the models cache.

FLUX.1-dev is gated on Hugging Face — export HF_TOKEN before running. Already
downloaded files are skipped.
"""

from __future__ import annotations

import os
import sys

import _bootstrap  # noqa: F401

from src.models.download import ensure_flux_assets
from src.utils.logging import get_logger
from src.utils.paths import Paths

logger = get_logger()


def main() -> None:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        logger.error("HF_TOKEN not set. FLUX.1-dev is gated; export your token first.")
        sys.exit(1)
    paths = Paths.load().ensure()
    assets = ensure_flux_assets(paths)
    logger.info("All FLUX assets present: %s", assets.all_exist())


if __name__ == "__main__":
    main()

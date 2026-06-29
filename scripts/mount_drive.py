"""Mount Google Drive (on Colab) and ensure the AICharacter/ tree exists.

Safe to run anywhere: outside Colab it just creates the local directory tree.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

from src.utils.env import Environment, detect_environment
from src.utils.logging import get_logger
from src.utils.paths import Paths

logger = get_logger()


def mount_drive() -> None:
    """Mount Drive when on Colab; no-op elsewhere."""
    if detect_environment() is Environment.COLAB:
        from google.colab import drive  # type: ignore

        logger.info("Mounting Google Drive at /content/drive ...")
        drive.mount("/content/drive")
    else:
        logger.info("Not on Colab; skipping Drive mount.")


def main() -> None:
    mount_drive()
    paths = Paths.load().ensure()
    logger.info("Storage root ready: %s", paths.root)
    for label, path in {
        "datasets": paths.datasets,
        "models": paths.models,
        "loras": paths.loras,
        "generated": paths.generated,
        "logs": paths.logs,
        "cache": paths.cache,
    }.items():
        logger.info("  %-10s %s", label, path)


if __name__ == "__main__":
    main()

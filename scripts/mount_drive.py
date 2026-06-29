"""Mount Google Drive (on Colab) and ensure the AICharacter/ tree exists.

Safe to run anywhere: outside Colab it just creates the local directory tree.
"""

from __future__ import annotations

import os

import _bootstrap  # noqa: F401  (sys.path side effect)

from src.utils.env import Environment, detect_environment
from src.utils.logging import get_logger
from src.utils.paths import Paths

logger = get_logger()

_MOUNT_POINT = "/content/drive"


def mount_drive() -> None:
    """Ensure Drive is mounted on Colab; no-op elsewhere or if already mounted.

    ``google.colab.drive.mount`` only works inside the notebook kernel, not in a
    subprocess (``!python ...``). The notebook mounts Drive in its own cell, so
    here we just verify and otherwise warn instead of crashing.
    """
    if detect_environment() is not Environment.COLAB:
        logger.info("Not on Colab; skipping Drive mount.")
        return
    if os.path.isdir(f"{_MOUNT_POINT}/MyDrive"):
        logger.info("Drive already mounted at %s.", _MOUNT_POINT)
        return
    try:
        from google.colab import drive  # type: ignore

        logger.info("Mounting Google Drive at %s ...", _MOUNT_POINT)
        drive.mount(_MOUNT_POINT)
    except Exception as exc:  # subprocess has no notebook kernel
        logger.warning(
            "Could not mount Drive from this process (%s). Mount it from a "
            "notebook cell: `from google.colab import drive; "
            "drive.mount('/content/drive')`, then re-run.",
            exc,
        )


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

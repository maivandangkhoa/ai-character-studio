"""Environment setup: install dependencies and report GPU status."""

from __future__ import annotations

import subprocess
import sys

import _bootstrap  # noqa: F401

from src.utils.config import project_root
from src.utils.env import detect_environment, gpu_info
from src.utils.logging import get_logger

logger = get_logger()


def install_requirements() -> None:
    req = project_root() / "requirements.txt"
    logger.info("Installing dependencies from %s ...", req)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)


def main() -> None:
    logger.info("Environment: %s", detect_environment().value)
    install_requirements()
    gpu = gpu_info()
    if gpu.available:
        logger.info("GPU: %s (%.1f GB) x%d", gpu.name, gpu.total_memory_gb, gpu.count)
    else:
        logger.warning("No GPU detected — training/generation will be very slow.")


if __name__ == "__main__":
    main()

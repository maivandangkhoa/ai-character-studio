"""Environment detection (Colab / RunPod / local) and GPU introspection.

Lets the same code run unchanged across Colab and RunPod, as required by the
spec ("Switch between Google Colab and RunPod without code changes").
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum


class Environment(str, Enum):
    COLAB = "colab"
    KAGGLE = "kaggle"
    RUNPOD = "runpod"
    LOCAL = "local"


def is_kaggle() -> bool:
    """True when running inside a Kaggle notebook kernel."""
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle")


def is_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return os.path.isdir("/content")


def is_runpod() -> bool:
    """True when running inside a RunPod container."""
    return bool(os.environ.get("RUNPOD_POD_ID")) or os.path.isdir("/workspace")


def detect_environment() -> Environment:
    """Detect the active execution environment."""
    if is_kaggle():
        return Environment.KAGGLE
    if is_colab():
        return Environment.COLAB
    if is_runpod():
        return Environment.RUNPOD
    return Environment.LOCAL


@dataclass(frozen=True)
class GPUInfo:
    available: bool
    name: str
    total_memory_gb: float
    count: int


def gpu_info() -> GPUInfo:
    """Return GPU details, gracefully degrading when torch/CUDA is absent."""
    try:
        import torch

        if not torch.cuda.is_available():
            return GPUInfo(False, "cpu", 0.0, 0)
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return GPUInfo(
            available=True,
            name=props.name,
            total_memory_gb=round(props.total_memory / (1024**3), 1),
            count=torch.cuda.device_count(),
        )
    except Exception:
        return GPUInfo(False, "unknown", 0.0, 0)


def has_command(name: str) -> bool:
    """True when an executable is on PATH."""
    return shutil.which(name) is not None

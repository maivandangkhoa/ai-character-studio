"""Checkpoint discovery and pruning for resume-safe training."""

from __future__ import annotations

import re
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger()

# sd-scripts names intermediate states like "<output>-step00001500.safetensors"
_STEP_RE = re.compile(r"step0*(\d+)", re.IGNORECASE)


def _step_of(path: Path) -> int:
    match = _STEP_RE.search(path.stem)
    return int(match.group(1)) if match else -1


def list_checkpoints(lora_dir: Path) -> list[Path]:
    """All intermediate step checkpoints in a character's LoRA dir, oldest first."""
    if not lora_dir.exists():
        return []
    found = [p for p in lora_dir.glob("*.safetensors") if _step_of(p) >= 0]
    return sorted(found, key=_step_of)


def latest_checkpoint(lora_dir: Path) -> Path | None:
    """Return the highest-step checkpoint, or None when training is fresh."""
    checkpoints = list_checkpoints(lora_dir)
    return checkpoints[-1] if checkpoints else None


def prune_checkpoints(lora_dir: Path, keep_last_n: int) -> int:
    """Delete older checkpoints beyond the most recent ``keep_last_n``.

    Returns the number of files removed. No-op when keep_last_n <= 0.
    """
    if keep_last_n <= 0:
        return 0
    checkpoints = list_checkpoints(lora_dir)
    stale = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []
    for path in stale:
        path.unlink(missing_ok=True)
        logger.info("Pruned old checkpoint %s", path.name)
    return len(stale)

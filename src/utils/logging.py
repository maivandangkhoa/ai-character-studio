"""Structured logging.

Per the spec, every operation logs start time, end time, GPU, training loss,
image count and elapsed time. ``StageTimer`` captures the timing/GPU envelope
and writes a JSON line per stage to the logs directory.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from src.utils.env import gpu_info

_CONFIGURED = False


def get_logger(name: str = "aics") -> logging.Logger:
    """Return a process-wide console logger (idempotent setup)."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED = True
    return logger


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageTimer:
    """Context manager that records a structured event for a pipeline stage.

    Usage::

        with StageTimer("train", log_dir, character="emily") as t:
            ...
            t.add(final_loss=0.12, image_count=30)
    """

    def __init__(self, stage: str, log_dir: Path | None = None, **fields: Any) -> None:
        self.stage = stage
        self.log_dir = log_dir
        self.fields: dict[str, Any] = dict(fields)
        self._start_perf = 0.0
        self.logger = get_logger()

    def add(self, **fields: Any) -> None:
        """Attach extra fields (e.g. final_loss, image_count) to the event."""
        self.fields.update(fields)

    def __enter__(self) -> "StageTimer":
        self._start_perf = time.perf_counter()
        gpu = gpu_info()
        self.fields.update(
            stage=self.stage,
            start_time=_utcnow(),
            gpu=gpu.name,
            gpu_memory_gb=gpu.total_memory_gb,
        )
        self.logger.info("▶ %s started (gpu=%s)", self.stage, gpu.name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        elapsed = round(time.perf_counter() - self._start_perf, 2)
        self.fields.update(
            end_time=_utcnow(),
            elapsed_seconds=elapsed,
            status="error" if exc_type else "ok",
        )
        if exc_type:
            self.fields["error"] = repr(exc)
            self.logger.error("✖ %s failed after %.2fs: %s", self.stage, elapsed, exc)
        else:
            self.logger.info("✔ %s finished in %.2fs", self.stage, elapsed)
        self._persist()
        return False  # never suppress exceptions

    def _persist(self) -> None:
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"{self.stage}.jsonl"
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.fields, ensure_ascii=False) + "\n")

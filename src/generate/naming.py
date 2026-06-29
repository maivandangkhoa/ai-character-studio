"""Output file naming: ``YYYYMMDD_HHMMSS_UUID`` with a JSON metadata sidecar."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path


def output_name(when: datetime, fmt: str = "png") -> str:
    """Return a timestamped, UUID-suffixed filename (collision-resistant)."""
    stamp = when.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{stamp}_{short}.{fmt}"


def metadata_path(image_path: Path) -> Path:
    """Sidecar JSON path for a generated image."""
    return image_path.with_suffix(".json")

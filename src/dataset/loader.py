"""Dataset discovery and image/caption pairing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.paths import CharacterPaths, Paths

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_RECOMMENDED = 20
MAX_RECOMMENDED = 40

logger = get_logger()


@dataclass(frozen=True)
class DatasetItem:
    """One image and its (optional) caption sidecar."""

    image: Path
    caption: Path

    @property
    def has_caption(self) -> bool:
        return self.caption.exists() and self.caption.stat().st_size > 0


def list_characters(paths: Paths) -> list[str]:
    """Return character names that have a dataset directory."""
    if not paths.datasets.exists():
        return []
    return sorted(p.name for p in paths.datasets.iterdir() if p.is_dir())


def _images_in(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_dataset(cp: CharacterPaths, source: str = "processed") -> list[DatasetItem]:
    """Pair images with caption files.

    ``source`` selects which subdir to read images from ("raw" or "processed").
    Captions are read from the ``captions`` directory by image stem.
    """
    image_dir = cp.raw if source == "raw" else cp.processed
    items = [
        DatasetItem(image=img, caption=cp.captions / f"{img.stem}.txt")
        for img in _images_in(image_dir)
    ]
    _warn_on_count(cp.name, len(items))
    return items


def _warn_on_count(name: str, count: int) -> None:
    if count == 0:
        logger.warning("Character '%s' has no images.", name)
    elif count < MIN_RECOMMENDED:
        logger.warning(
            "Character '%s' has %d images (recommended %d-%d).",
            name,
            count,
            MIN_RECOMMENDED,
            MAX_RECOMMENDED,
        )
    elif count > MAX_RECOMMENDED:
        logger.warning(
            "Character '%s' has %d images (> recommended %d).", name, count, MAX_RECOMMENDED
        )


def missing_captions(items: list[DatasetItem]) -> list[DatasetItem]:
    """Items that still need a caption."""
    return [it for it in items if not it.has_caption]

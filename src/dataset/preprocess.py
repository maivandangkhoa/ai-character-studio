"""Image preprocessing: resize/center-crop to target resolution, strip EXIF.

Idempotent and resume-safe: images already present in ``processed/`` are
skipped, so re-running never duplicates work.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from src.dataset.loader import IMAGE_EXTS
from src.utils.logging import get_logger
from src.utils.paths import CharacterPaths

logger = get_logger()


def _center_crop_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def process_image(src: Path, dst: Path, resolution: int) -> None:
    """Normalise one image: EXIF-orient, square center-crop, resize, save PNG."""
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = _center_crop_square(img)
        if img.size[0] != resolution:
            img = img.resize((resolution, resolution), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="PNG")


def preprocess_character(cp: CharacterPaths, resolution: int, force: bool = False) -> int:
    """Process every raw image into ``processed/`` as zero-padded PNGs.

    Returns the number of images written this run. Existing outputs are skipped
    unless ``force`` is set.
    """
    cp.ensure()
    raw_images = sorted(p for p in cp.raw.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    written = 0
    for idx, src in enumerate(raw_images, start=1):
        dst = cp.processed / f"{idx:04d}.png"
        if dst.exists() and not force:
            continue
        process_image(src, dst, resolution)
        written += 1
        logger.info("Processed %s -> %s", src.name, dst.name)
    logger.info(
        "Preprocess '%s': %d new, %d total raw images.", cp.name, written, len(raw_images)
    )
    return written

"""Emit the kohya/sd-scripts dataset config (TOML) from our character dataset.

sd-scripts consumes a TOML describing image directories, resolution and repeats.
We generate it on the fly so the user never hand-edits trainer config.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import CharacterPaths


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_dataset_toml(
    cp: CharacterPaths,
    resolution: int,
    repeats: int = 10,
    batch_size: int = 1,
) -> str:
    """Render the dataset TOML pointing at the processed images + captions.

    Captions live in a separate ``captions/`` dir; sd-scripts reads sidecar
    ``.txt`` files alongside images, so we point it at ``processed`` and rely on
    captions having been copied/written next to images. To keep both layouts
    working we pass ``caption_extension`` and the image dir.
    """
    image_dir = _toml_escape(str(cp.processed))
    return (
        "[general]\n"
        f"resolution = {resolution}\n"
        "enable_bucket = true\n"
        "min_bucket_reso = 512\n"
        "max_bucket_reso = 2048\n\n"
        "[[datasets]]\n"
        f"resolution = {resolution}\n"
        f"batch_size = {batch_size}\n\n"
        "  [[datasets.subsets]]\n"
        f'  image_dir = "{image_dir}"\n'
        f"  num_repeats = {repeats}\n"
        '  caption_extension = ".txt"\n'
    )


def write_dataset_toml(
    cp: CharacterPaths,
    out_dir: Path,
    resolution: int,
    repeats: int = 10,
    batch_size: int = 1,
) -> Path:
    """Write the dataset TOML and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    toml_path = out_dir / f"{cp.name}_dataset.toml"
    toml_path.write_text(
        build_dataset_toml(cp, resolution, repeats, batch_size), encoding="utf-8"
    )
    return toml_path

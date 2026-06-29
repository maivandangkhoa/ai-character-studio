"""Finalise a training run into the spec's LoRA output bundle.

Produces, per the spec, in the character LoRA directory:
  lora.safetensors · metadata.json · preview.png · training_log.json
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.train.checkpoint import list_checkpoints
from src.utils.logging import get_logger
from src.utils.schemas import TrainingConfig

logger = get_logger()


def _final_weights(lora_dir: Path, character: str) -> Path | None:
    """Locate the final LoRA: the named output, else the latest checkpoint."""
    named = lora_dir / f"{character}.safetensors"
    if named.exists():
        return named
    checkpoints = list_checkpoints(lora_dir)
    return checkpoints[-1] if checkpoints else None


def build_metadata(cfg: TrainingConfig, image_count: int) -> dict[str, object]:
    return {
        "character": cfg.character.capitalize(),
        "trigger_word": cfg.trigger_word,
        "base_model": cfg.base_model,
        "steps": cfg.steps,
        "resolution": cfg.resolution,
        "network_dim": cfg.network_dim,
        "network_alpha": cfg.network_alpha,
        "image_count": image_count,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def export_lora(
    lora_dir: Path,
    cfg: TrainingConfig,
    image_count: int,
    log_records: list[dict[str, object]] | None = None,
    preview_src: Path | None = None,
) -> dict[str, Path]:
    """Assemble the output bundle. Returns a map of artifact -> path."""
    lora_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    weights = _final_weights(lora_dir, cfg.character)
    lora_out = lora_dir / "lora.safetensors"
    if weights and weights.resolve() != lora_out.resolve():
        shutil.copyfile(weights, lora_out)
    if lora_out.exists():
        artifacts["lora"] = lora_out
    else:
        logger.warning("No LoRA weights found to export in %s", lora_dir)

    meta_path = lora_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(build_metadata(cfg, image_count), indent=4), encoding="utf-8"
    )
    artifacts["metadata"] = meta_path

    log_path = lora_dir / "training_log.json"
    log_path.write_text(json.dumps(log_records or [], indent=2), encoding="utf-8")
    artifacts["training_log"] = log_path

    if preview_src and preview_src.exists():
        preview_out = lora_dir / "preview.png"
        shutil.copyfile(preview_src, preview_out)
        artifacts["preview"] = preview_out

    logger.info("Exported %d artifacts to %s", len(artifacts), lora_dir)
    return artifacts

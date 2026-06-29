"""Auto-captioning with Florence-2.

Captions are written next to each image stem in ``captions/<stem>.txt`` with the
trigger word prepended. Resume-safe: images that already have a non-empty
caption are skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.dataset.loader import DatasetItem, load_dataset, missing_captions
from src.utils.logging import get_logger
from src.utils.paths import CharacterPaths

logger = get_logger()

_CAPTION_TASK = "<DETAILED_CAPTION>"

# transformers>=4.50 dropped these legacy generation attributes from
# PretrainedConfig (they live on GenerationConfig now), but Florence-2's remote
# configuration_florence2.py still reads ``self.forced_bos_token_id`` inside
# ``__init__``. Restore the legacy class-level defaults so the model loads.
_LEGACY_CONFIG_DEFAULTS = ("forced_bos_token_id", "forced_eos_token_id")


def _patch_florence2_config_compat() -> None:
    from transformers import PretrainedConfig

    for attr in _LEGACY_CONFIG_DEFAULTS:
        if not hasattr(PretrainedConfig, attr):
            setattr(PretrainedConfig, attr, None)


class Florence2Captioner:
    """Lazy wrapper around the Florence-2 image-captioning model."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        _patch_florence2_config_compat()
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        logger.info("Loading Florence-2 (%s) on %s ...", self.model_id, self._device)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def caption(self, image_path: Path) -> str:
        self._ensure_loaded()
        from PIL import Image

        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        inputs = self._processor(text=_CAPTION_TASK, images=image, return_tensors="pt").to(
            self._device
        )
        generated = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
        )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            text, task=_CAPTION_TASK, image_size=(image.width, image.height)
        )
        return str(parsed.get(_CAPTION_TASK, "")).strip()


def _write_caption(item: DatasetItem, trigger_word: str, body: str) -> None:
    item.caption.parent.mkdir(parents=True, exist_ok=True)
    text = f"{trigger_word}, {body}" if body else trigger_word
    item.caption.write_text(text, encoding="utf-8")


def caption_character(
    cp: CharacterPaths,
    trigger_word: str,
    model_id: str,
    force: bool = False,
) -> int:
    """Caption images lacking a caption. Returns count written."""
    cp.ensure()
    items = load_dataset(cp, source="processed")
    todo = items if force else missing_captions(items)
    if not todo:
        logger.info("All %d images for '%s' already captioned.", len(items), cp.name)
        return 0
    captioner = Florence2Captioner(model_id)
    for item in todo:
        body = captioner.caption(item.image)
        _write_caption(item, trigger_word, body)
        logger.info("Captioned %s", item.image.name)
    logger.info("Captioned %d images for '%s'.", len(todo), cp.name)
    return len(todo)

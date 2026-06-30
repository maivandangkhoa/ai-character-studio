"""FLUX + LoRA image generation.

Loads the FLUX pipeline once, applies a trained LoRA, then renders prompts to
disk with a JSON metadata sidecar each. Resume-safe batch generation never
overwrites existing files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.generate.naming import metadata_path, output_name
from src.utils.logging import get_logger
from src.utils.schemas import GenerationConfig

logger = get_logger()


@dataclass
class GenerationResult:
    image_path: Path
    prompt: str
    seed: int


class FluxGenerator:
    """Wraps a diffusers FLUX pipeline with an attached character LoRA."""

    def __init__(self, cfg: GenerationConfig, lora_path: Path, output_dir: Path) -> None:
        self.cfg = cfg
        self.lora_path = lora_path
        self.output_dir = output_dir
        self._pipe: Any = None
        self._lora_scale: float | None = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline

        dtype = torch.bfloat16
        cuda = torch.cuda.is_available()
        quantize = self.cfg.low_vram and cuda
        logger.info("Loading FLUX pipeline %s ...", self.cfg.base_model)

        if quantize:
            # bf16 FLUX (~24GB transformer) busts both Kaggle's ~29GB RAM and a
            # 16GB T4's VRAM under cpu-offload. Load the transformer in 4-bit NF4
            # (~6GB) so it fits — same fp8-for-training rationale, generation side.
            from diffusers import BitsAndBytesConfig, FluxTransformer2DModel

            logger.info("Quantizing transformer to 4-bit (NF4) for low-VRAM hosts.")
            transformer = FluxTransformer2DModel.from_pretrained(
                self.cfg.base_model,
                subfolder="transformer",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                ),
                torch_dtype=dtype,
            )
            pipe = FluxPipeline.from_pretrained(
                self.cfg.base_model, transformer=transformer, torch_dtype=dtype
            )
        else:
            pipe = FluxPipeline.from_pretrained(self.cfg.base_model, torch_dtype=dtype)

        if cuda:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cpu")

        logger.info("Attaching LoRA %s (weight=%.2f)", self.lora_path.name, self.cfg.lora_weight)
        # sd-scripts FLUX LoRAs are transformer-only. Drop any text-encoder keys
        # before loading: diffusers' TE path trips on the empty/CLIP-named rank
        # dict (list index out of range). lora_state_dict handles kohya->diffusers
        # key conversion; we keep only the transformer adapter.
        state_dict = pipe.lora_state_dict(str(self.lora_path))
        if isinstance(state_dict, tuple):
            state_dict = state_dict[0]
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("text_encoder")
        }
        pipe.load_lora_weights(state_dict)
        if quantize:
            # fuse_lora can't write into 4-bit weights; apply the scale per call.
            self._lora_scale = self.cfg.lora_weight
        else:
            pipe.fuse_lora(lora_scale=self.cfg.lora_weight)
            self._lora_scale = None
        self._pipe = pipe

    def _seed_generator(self, seed: int) -> Any:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.Generator(device=device).manual_seed(seed)

    def _resolve_seed(self, index: int) -> int:
        if self.cfg.seed is not None:
            return self.cfg.seed + index
        # Deterministic-but-varied fallback without Math.random-style nondeterminism.
        return int.from_bytes(output_name(datetime.now(timezone.utc)).encode()[:4], "big")

    def generate_one(self, payload: dict[str, Any], index: int = 0) -> GenerationResult:
        """Render a single prompt payload to disk with metadata."""
        self._load()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prompt = payload["prompt"]
        seed = self._resolve_seed(index)
        call_kwargs: dict[str, Any] = {}
        if self._lora_scale is not None:
            # LoRA wasn't fused (4-bit base): apply its weight at inference time.
            call_kwargs["joint_attention_kwargs"] = {"scale": self._lora_scale}
        image = self._pipe(
            prompt=prompt,
            width=self.cfg.width,
            height=self.cfg.height,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=self._seed_generator(seed),
            **call_kwargs,
        ).images[0]

        out_path = self.output_dir / output_name(
            datetime.now(timezone.utc), self.cfg.output_format
        )
        image.save(out_path)
        if self.cfg.save_metadata:
            self._write_metadata(out_path, payload, seed)
        logger.info("Generated %s", out_path.name)
        return GenerationResult(image_path=out_path, prompt=prompt, seed=seed)

    def _write_metadata(self, image_path: Path, payload: dict[str, Any], seed: int) -> None:
        meta = {
            **payload,
            "seed": seed,
            "lora": self.lora_path.name,
            "base_model": self.cfg.base_model,
            "steps": self.cfg.num_inference_steps,
            "guidance_scale": self.cfg.guidance_scale,
            "size": [self.cfg.width, self.cfg.height],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path(image_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def generate_batch(self, payloads: list[dict[str, Any]]) -> list[GenerationResult]:
        """Generate many prompts. ``skip_existing`` keeps runs idempotent."""
        results: list[GenerationResult] = []
        for index, payload in enumerate(payloads):
            results.append(self.generate_one(payload, index))
        logger.info("Generated %d images into %s", len(results), self.output_dir)
        return results

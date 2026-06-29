"""Typed configuration schemas built from the YAML config files.

Keeps the rest of the codebase fully typed while staying config-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils.config import deep_merge, env_override, load_config


def _trigger(character: str, configured: str | None) -> str:
    return configured or f"{character}_ai"


@dataclass(frozen=True)
class TrainingConfig:
    character: str
    trigger_word: str
    base_model: str
    resolution: int
    steps: int
    train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    network_dim: int
    network_alpha: int
    optimizer: str
    lr_scheduler: str
    mixed_precision: str
    seed: int
    save_every_n_steps: int
    keep_last_n_checkpoints: int
    resume: bool
    low_vram: bool
    fp8_base: bool
    cache_latents: bool
    cache_text_encoder_outputs: bool
    auto_caption: bool
    caption_model: str
    flux: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, character: str, overrides: dict[str, Any] | None = None) -> "TrainingConfig":
        cfg = env_override(load_config("training"))
        if overrides:
            cfg = deep_merge(cfg, overrides)
        return cls(
            character=character,
            trigger_word=_trigger(character, cfg.get("trigger_word")),
            base_model=cfg["base_model"],
            resolution=int(cfg["resolution"]),
            steps=int(cfg["steps"]),
            train_batch_size=int(cfg["train_batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            learning_rate=float(cfg["learning_rate"]),
            network_dim=int(cfg["network_dim"]),
            network_alpha=int(cfg["network_alpha"]),
            optimizer=str(cfg["optimizer"]),
            lr_scheduler=str(cfg["lr_scheduler"]),
            mixed_precision=str(cfg["mixed_precision"]),
            seed=int(cfg["seed"]),
            save_every_n_steps=int(cfg["save_every_n_steps"]),
            keep_last_n_checkpoints=int(cfg["keep_last_n_checkpoints"]),
            resume=bool(cfg["resume"]),
            low_vram=bool(cfg["low_vram"]),
            fp8_base=bool(cfg["fp8_base"]),
            cache_latents=bool(cfg["cache_latents"]),
            cache_text_encoder_outputs=bool(cfg["cache_text_encoder_outputs"]),
            auto_caption=bool(cfg["auto_caption"]),
            caption_model=str(cfg["caption_model"]),
            flux=dict(cfg.get("flux", {})),
        )


@dataclass(frozen=True)
class GenerationConfig:
    base_model: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int | None
    lora_weight: float
    count: int
    skip_existing: bool
    output_format: str
    save_metadata: bool
    negative_prompt: str

    @classmethod
    def load(cls, overrides: dict[str, Any] | None = None) -> "GenerationConfig":
        cfg = env_override(load_config("generation"))
        if overrides:
            cfg = deep_merge(cfg, overrides)
        seed = cfg.get("seed")
        return cls(
            base_model=cfg["base_model"],
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            num_inference_steps=int(cfg["num_inference_steps"]),
            guidance_scale=float(cfg["guidance_scale"]),
            seed=int(seed) if seed is not None else None,
            lora_weight=float(cfg["lora_weight"]),
            count=int(cfg["count"]),
            skip_existing=bool(cfg["skip_existing"]),
            output_format=str(cfg["output_format"]),
            save_metadata=bool(cfg["save_metadata"]),
            negative_prompt=str(cfg["negative_prompt"]),
        )

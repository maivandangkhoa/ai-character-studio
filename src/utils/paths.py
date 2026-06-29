"""Centralised, config-driven path resolution.

Single source of truth for where everything lives. The active root is chosen by
environment (Drive on Colab/RunPod, local dir otherwise) and may be overridden
with the ``AICS_DRIVE_ROOT`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import load_config, project_root
from src.utils.env import Environment, detect_environment

DRIVE_ROOT_ENV = "AICS_DRIVE_ROOT"


@dataclass(frozen=True)
class CharacterPaths:
    """Per-character dataset locations."""

    name: str
    root: Path
    raw: Path
    processed: Path
    captions: Path

    def ensure(self) -> "CharacterPaths":
        for d in (self.raw, self.processed, self.captions):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Paths:
    """Resolved top-level paths for the active environment."""

    root: Path
    datasets: Path
    models: Path
    loras: Path
    generated: Path
    logs: Path
    cache: Path
    sd_scripts_dir: Path
    sd_scripts_repo: str
    sd_scripts_branch: str

    @classmethod
    def load(cls, env: Environment | None = None) -> "Paths":
        cfg: dict[str, Any] = load_config("paths")
        env = env or detect_environment()
        root = cls._resolve_root(cfg, env)
        subs = cfg["subdirs"]
        sd_dir = cfg["sd_scripts_dir"]
        return cls(
            root=root,
            datasets=root / subs["datasets"],
            models=root / subs["models"],
            loras=root / subs["loras"],
            generated=root / subs["generated"],
            logs=root / subs["logs"],
            cache=root / subs["cache"],
            sd_scripts_dir=(root / sd_dir) if not Path(sd_dir).is_absolute() else Path(sd_dir),
            sd_scripts_repo=cfg["sd_scripts_repo"],
            sd_scripts_branch=cfg["sd_scripts_branch"],
        )

    @staticmethod
    def _resolve_root(cfg: dict[str, Any], env: Environment) -> Path:
        override = os.environ.get(DRIVE_ROOT_ENV)
        if override:
            return Path(override)
        if env in (Environment.COLAB, Environment.RUNPOD):
            return Path(cfg["drive_root"])
        local = cfg.get("local_root", ".")
        base = project_root() if local == "." else Path(local)
        return base

    def ensure(self) -> "Paths":
        """Create all top-level directories if missing."""
        for d in (self.datasets, self.models, self.loras, self.generated, self.logs, self.cache):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def character(self, name: str) -> CharacterPaths:
        base = self.datasets / name
        return CharacterPaths(
            name=name,
            root=base,
            raw=base / "raw",
            processed=base / "processed",
            captions=base / "captions",
        )

    def lora_dir(self, character: str) -> Path:
        return self.loras / character

    def generated_dir(self, character: str) -> Path:
        return self.generated / character

"""Download FLUX components needed by the kohya FLUX trainer.

Files are cached under the models directory on Drive and skipped if already
present (resume-safe, no re-downloads). FLUX.1-dev is gated: set HF_TOKEN (or
HUGGINGFACE_TOKEN) in the environment before running.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.paths import Paths

logger = get_logger()

# Single-file weights used by sd-scripts' FLUX trainer.
_FLUX_TRANSFORMER = ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors")
_FLUX_AE = ("black-forest-labs/FLUX.1-dev", "ae.safetensors")
_CLIP_L = ("comfyanonymous/flux_text_encoders", "clip_l.safetensors")
_T5XXL = ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors")


@dataclass(frozen=True)
class FluxAssets:
    """Resolved local paths to the FLUX weights required for training."""

    transformer: Path
    ae: Path
    clip_l: Path
    t5xxl: Path

    def all_exist(self) -> bool:
        return all(p.exists() for p in (self.transformer, self.ae, self.clip_l, self.t5xxl))


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _download(repo: str, filename: str, dest_dir: Path) -> Path:
    """Download a single file from the Hub into ``dest_dir`` (cached)."""
    from huggingface_hub import hf_hub_download

    target = dest_dir / filename
    if target.exists():
        logger.info("Cached: %s", target.name)
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s/%s ...", repo, filename)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(dest_dir),
        token=_hf_token(),
    )
    return Path(path)


def ensure_flux_assets(paths: Paths) -> FluxAssets:
    """Ensure all FLUX weights are present locally, downloading if needed."""
    models = paths.models
    transformer = _download(*_FLUX_TRANSFORMER, models)
    ae = _download(*_FLUX_AE, models)
    clip_l = _download(*_CLIP_L, models)
    t5xxl = _download(*_T5XXL, models)
    assets = FluxAssets(transformer=transformer, ae=ae, clip_l=clip_l, t5xxl=t5xxl)
    logger.info("FLUX assets ready in %s", models)
    return assets

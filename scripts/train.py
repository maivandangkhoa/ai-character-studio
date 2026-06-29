"""End-to-end training CLI.

    python scripts/train.py --character emily [--steps 4000] [--no-caption]

Pipeline: prepare dataset -> (auto-caption) -> download FLUX -> train (resume)
-> export LoRA bundle. Every stage is resume-safe.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.dataset.caption import caption_character
from src.dataset.loader import load_dataset
from src.dataset.preprocess import preprocess_character
from src.models.download import ensure_flux_assets
from src.train.checkpoint import prune_checkpoints
from src.train.export import export_lora
from src.train.kohya_runner import KohyaRunner
from src.utils.logging import StageTimer, get_logger
from src.utils.paths import Paths
from src.utils.schemas import TrainingConfig

logger = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a character LoRA on FLUX.")
    p.add_argument("--character", required=True, help="Character/dataset name.")
    p.add_argument("--steps", type=int, default=None, help="Override max train steps.")
    p.add_argument("--trigger-word", default=None, help="Override trigger word.")
    p.add_argument("--no-caption", action="store_true", help="Skip auto-captioning.")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints.")
    return p.parse_args()


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    if args.steps is not None:
        out["steps"] = args.steps
    if args.trigger_word is not None:
        out["trigger_word"] = args.trigger_word
    if args.no_caption:
        out["auto_caption"] = False
    if args.no_resume:
        out["resume"] = False
    return out


def main() -> None:
    args = parse_args()
    paths = Paths.load().ensure()
    cfg = TrainingConfig.load(args.character, _overrides(args))
    cp = paths.character(cfg.character).ensure()
    log_dir = paths.logs / cfg.character

    with StageTimer("prepare_dataset", log_dir, character=cfg.character) as t:
        written = preprocess_character(cp, cfg.resolution)
        t.add(images_processed=written)

    if cfg.auto_caption:
        with StageTimer("caption", log_dir, character=cfg.character) as t:
            captioned = caption_character(cp, cfg.trigger_word, cfg.caption_model)
            t.add(images_captioned=captioned)

    with StageTimer("download_models", log_dir, character=cfg.character):
        assets = ensure_flux_assets(paths)

    image_count = len(load_dataset(cp, source="processed"))
    with StageTimer("train", log_dir, character=cfg.character, image_count=image_count) as t:
        runner = KohyaRunner(paths, cfg, assets)
        code = runner.run()
        t.add(return_code=code, steps=cfg.steps)
        if code != 0:
            raise RuntimeError(f"Trainer exited with code {code}")

    with StageTimer("export", log_dir, character=cfg.character) as t:
        prune_checkpoints(paths.lora_dir(cfg.character), cfg.keep_last_n_checkpoints)
        artifacts = export_lora(paths.lora_dir(cfg.character), cfg, image_count)
        t.add(artifacts=list(artifacts.keys()))

    logger.info("✔ Training complete for '%s'.", cfg.character)


if __name__ == "__main__":
    main()

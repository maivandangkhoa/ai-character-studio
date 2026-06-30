"""End-to-end generation CLI.

    python scripts/generate.py --character emily --count 20

Builds randomised prompts, loads FLUX + the character LoRA, and renders images
with metadata. Resume-safe: only generates the shortfall below --count when the
output dir already holds images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.generate.pipeline import FluxGenerator
from src.prompts.builder import PromptBuilder, prompts_to_payloads
from src.utils.logging import StageTimer, get_logger
from src.utils.paths import Paths
from src.utils.schemas import GenerationConfig, TrainingConfig

logger = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images from a character LoRA.")
    p.add_argument("--character", required=True, help="Character name (LoRA must exist).")
    p.add_argument("--count", type=int, default=None, help="Target number of images.")
    p.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility.")
    p.add_argument("--lora", default=None, help="Explicit LoRA path (defaults to exported one).")
    p.add_argument(
        "--prompts-file",
        default=None,
        help="Text file, one prompt per line. Overrides random prompts; the "
        "trigger word is prepended to each line.",
    )
    return p.parse_args()


def _existing_images(output_dir: Path, fmt: str) -> int:
    return len(list(output_dir.glob(f"*.{fmt}"))) if output_dir.exists() else 0


def main() -> None:
    args = parse_args()
    paths = Paths.load().ensure()
    overrides = {"count": args.count} if args.count is not None else {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = GenerationConfig.load(overrides)

    train_cfg = TrainingConfig.load(args.character)
    lora_path = Path(args.lora) if args.lora else paths.lora_dir(args.character) / "lora.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA not found: {lora_path}. Train the character first.")

    output_dir = paths.generated_dir(args.character)
    log_dir = paths.logs / args.character
    already = 0

    if args.prompts_file:
        lines = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        payloads = prompts_to_payloads(train_cfg.trigger_word, lines, cfg.negative_prompt)
        if not payloads:
            logger.info("No prompts found in %s. Nothing to do.", args.prompts_file)
            return
        logger.info("Generating %d image(s) from custom prompts.", len(payloads))
    else:
        target = cfg.count
        already = _existing_images(output_dir, cfg.output_format) if cfg.skip_existing else 0
        remaining = max(0, target - already)
        if remaining == 0:
            logger.info("Already have %d/%d images for '%s'. Nothing to do.", already, target, args.character)
            return
        builder = PromptBuilder(train_cfg.trigger_word, cfg.negative_prompt, seed=args.seed)
        payloads = builder.to_payloads(remaining)

    with StageTimer("generate", log_dir, character=args.character) as t:
        generator = FluxGenerator(cfg, lora_path, output_dir)
        results = generator.generate_batch(payloads)
        t.add(image_count=len(results), total=already + len(results))

    logger.info("✔ Generated %d new images (%d total) for '%s'.", len(results), already + len(results), args.character)


if __name__ == "__main__":
    main()

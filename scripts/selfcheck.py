"""GPU-free smoke test of the wiring: config, schemas, prompts, dataset TOML.

Validates that everything loads and renders without importing torch/diffusers.
Exits non-zero on the first failure.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from src.generate.naming import metadata_path, output_name
from src.prompts.builder import PromptBuilder
from src.train.dataset_config import build_dataset_toml
from src.utils.logging import get_logger
from src.utils.paths import Paths
from src.utils.schemas import GenerationConfig, TrainingConfig

logger = get_logger()


def main() -> None:
    from datetime import datetime, timezone

    paths = Paths.load()
    logger.info("Resolved root: %s", paths.root)

    tcfg = TrainingConfig.load("emily")
    assert tcfg.trigger_word == "emily_ai", tcfg.trigger_word
    logger.info("TrainingConfig OK: %d steps, trigger=%s", tcfg.steps, tcfg.trigger_word)

    gcfg = GenerationConfig.load({"count": 3})
    assert gcfg.count == 3
    logger.info("GenerationConfig OK: %dx%d, count=%d", gcfg.width, gcfg.height, gcfg.count)

    builder = PromptBuilder(tcfg.trigger_word, gcfg.negative_prompt, seed=1)
    payloads = builder.to_payloads(2)
    assert len(payloads) == 2 and tcfg.trigger_word in payloads[0]["prompt"]
    logger.info("Prompt sample: %s", payloads[0]["prompt"])

    cp = paths.character("emily")
    toml = build_dataset_toml(cp, tcfg.resolution)
    assert "[[datasets]]" in toml and str(cp.processed) in toml
    logger.info("Dataset TOML rendered (%d chars)", len(toml))

    name = output_name(datetime.now(timezone.utc), gcfg.output_format)
    assert name.endswith(".png") and metadata_path(__import__("pathlib").Path(name)).suffix == ".json"
    logger.info("Output naming OK: %s", name)

    logger.info("✅ selfcheck passed.")


if __name__ == "__main__":
    main()

"""Randomised prompt generator.

Builds varied prompts from editable word banks while keeping the character
trigger word fixed for identity consistency. A seed makes batches reproducible.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from src.prompts.template import PromptTemplate
from src.utils.config import load_yaml, project_root

_DEFAULT_BANK = project_root() / "src" / "prompts" / "wordbanks.yaml"


class PromptBuilder:
    """Assembles random :class:`PromptTemplate` instances from word banks."""

    def __init__(
        self,
        trigger_word: str,
        negative: str = "blurry, low quality, deformed",
        banks_path: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self.trigger_word = trigger_word
        self.negative = negative
        self.banks: dict[str, list[str]] = load_yaml(banks_path or _DEFAULT_BANK)
        self._rng = random.Random(seed)

    def _pick(self, key: str) -> str:
        values = self.banks.get(key) or [""]
        return self._rng.choice(values)

    def build(self) -> PromptTemplate:
        """Generate one randomised prompt template."""
        scene = (
            f"wearing {self._pick('clothing')}, {self._pick('pose')} "
            f"in {self._pick('country')}, {self._pick('weather')}, "
            f"{self._pick('season')}, {self._pick('time_of_day')}, "
            f"{self._pick('emotion')} expression"
        )
        return PromptTemplate(
            character=self.trigger_word,
            scene=scene,
            camera=f"{self._pick('camera')} {self._pick('lens')}",
            style=self._pick("style"),
            negative=self.negative,
        )

    def build_many(self, count: int) -> list[PromptTemplate]:
        return [self.build() for _ in range(count)]

    def to_payloads(self, count: int) -> list[dict[str, Any]]:
        """Return render-ready dicts for the generation pipeline."""
        return [t.as_dict() for t in self.build_many(count)]


def prompts_to_payloads(
    trigger_word: str,
    prompts: list[str],
    negative: str = "blurry, low quality, deformed",
) -> list[dict[str, Any]]:
    """Turn user-supplied prompt lines into generation payloads.

    The trigger word is prepended to each line to keep the character identity,
    so the user writes only the scene they want ("on a beach at sunset").
    """
    payloads: list[dict[str, Any]] = []
    for line in prompts:
        scene = line.strip()
        if not scene:
            continue
        payloads.append(
            {
                "character": trigger_word,
                "scene": scene,
                "prompt": f"{trigger_word}, {scene}",
                "negative": negative,
            }
        )
    return payloads

"""Structured prompt template.

Mirrors the spec's template fields (Character / Scene / Camera / Lighting /
Style / Negative) and renders them into a single FLUX-friendly prompt string.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptTemplate:
    character: str
    scene: str
    camera: str = "50mm"
    lighting: str = "natural light"
    style: str = "cinematic, photorealistic"
    extra: list[str] = field(default_factory=list)
    negative: str = "blurry, low quality, deformed"

    def render(self) -> str:
        """Render a comma-joined positive prompt string."""
        parts = [
            self.character,
            self.scene,
            f"shot on {self.camera}",
            self.lighting,
            self.style,
            *self.extra,
        ]
        return ", ".join(p for p in parts if p)

    def as_dict(self) -> dict[str, object]:
        return {
            "character": self.character,
            "scene": self.scene,
            "camera": self.camera,
            "lighting": self.lighting,
            "style": self.style,
            "extra": list(self.extra),
            "negative": self.negative,
            "prompt": self.render(),
        }

"""YAML configuration loading, merging and validation.

Config-driven by design: every tunable lives in ``config/*.yaml`` and is loaded
through these helpers. No module hardcodes values that belong in config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

CONFIG_DIR_ENV = "AICS_CONFIG_DIR"


def project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Directory holding the YAML config files."""
    override = os.environ.get(CONFIG_DIR_ENV)
    return Path(override) if override else project_root() / "config"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict, raising clear errors on failure."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {p}")
    return data


def load_config(name: str) -> dict[str, Any]:
    """Load a named config (e.g. ``training``) from the config directory."""
    stem = name[:-5] if name.endswith(".yaml") else name
    return load_yaml(config_dir() / f"{stem}.yaml")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating inputs."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def require(cfg: dict[str, Any], key: str, expected: type[T] | None = None) -> T:
    """Fetch a required key, validating presence and (optionally) type."""
    if key not in cfg or cfg[key] is None:
        raise KeyError(f"Missing required config key: '{key}'")
    value = cfg[key]
    if expected is not None and not isinstance(value, expected):
        raise TypeError(
            f"Config key '{key}' must be {expected.__name__}, got {type(value).__name__}"
        )
    return value  # type: ignore[return-value]


def env_override(cfg: dict[str, Any], prefix: str = "AICS_") -> dict[str, Any]:
    """Overlay top-level scalar config keys from environment variables.

    Example: ``AICS_STEPS=6000`` overrides ``cfg['steps']``. Values are parsed
    as YAML so ints/floats/bools are coerced correctly.
    """
    result = dict(cfg)
    for env_key, raw in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        key = env_key[len(prefix) :].lower()
        if key in result:
            result[key] = yaml.safe_load(raw)
    return result

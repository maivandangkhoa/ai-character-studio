"""Tiny JSON-backed job store for the generation web app.

Single-user, so a flat JSON file is plenty. Each job tracks the prompts, the
Kaggle kernel run state, and the resulting image filenames.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    prompts: list[str]
    seed: int | None = None
    status: str = "queued"  # queued | running | done | error
    message: str = ""
    images: list[str] = field(default_factory=list)
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    """Persists jobs to ``<data_dir>/jobs.json`` and serves image files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._file = data_dir / "jobs.json"
        self._jobs: dict[str, Job] = {}
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            self._jobs = {jid: Job(**data) for jid, data in raw.items()}

    def _flush(self) -> None:
        payload = {jid: job.as_dict() for jid, job in self._jobs.items()}
        self._file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save(self, job: Job) -> None:
        with _LOCK:
            self._jobs[job.id] = job
            self._flush()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def job_images_dir(self, job_id: str) -> Path:
        d = self.images_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

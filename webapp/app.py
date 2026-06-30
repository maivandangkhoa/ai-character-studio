"""FastAPI web UI: type prompts, get images generated on Kaggle.

Run on the VM:  uvicorn webapp.app:app --host 0.0.0.0 --port 8000
Requires the Kaggle CLI authenticated (~/.kaggle/kaggle.json) and the
`maivandangkhoa/mayalin-lora` dataset published (see scripts/publish_lora.py).
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from webapp import orchestrator
from webapp.store import Job, JobStore

_DATA_DIR = Path(os.environ.get("AICS_WEBAPP_DATA", Path(__file__).parent / "data"))
_DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Mayalin Studio")
store = JobStore(_DATA_DIR)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _run_job(job: Job, character: str) -> None:
    """Background worker: push to Kaggle, wait, pull images."""
    try:
        job.status = "running"
        job.message = "Pushing to Kaggle (cold start ~20-40 min)..."
        store.save(job)

        work_dir = _DATA_DIR / "push" / job.id
        orchestrator.push(job.prompts, job.seed, character, work_dir)
        orchestrator.poll()
        images = orchestrator.pull_images(store.job_images_dir(job.id))

        job.images = images
        job.status = "done" if images else "error"
        job.message = f"{len(images)} image(s)" if images else "No images returned."
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.message = str(exc)
    store.save(job)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html", {"request": request, "jobs": store.all()}
    )


@app.post("/submit")
def submit(prompts: str = Form(...), seed: str = Form(""), character: str = Form("mayalin")):
    lines = [ln.strip() for ln in prompts.splitlines() if ln.strip()]
    job = Job(
        id=uuid.uuid4().hex[:12],
        prompts=lines,
        seed=int(seed) if seed.strip() else None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store.save(job)
    if lines:
        threading.Thread(target=_run_job, args=(job, character), daemon=True).start()
    return RedirectResponse("/", status_code=303)


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = store.get(job_id)
    return job.as_dict() if job else {"error": "not found"}


@app.get("/images/{job_id}/{filename}")
def image(job_id: str, filename: str):
    path = store.job_images_dir(job_id) / filename
    if not path.exists():
        return {"error": "not found"}
    return FileResponse(path)

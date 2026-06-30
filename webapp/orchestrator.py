"""Drive Kaggle batch generation from the VM via the Kaggle CLI.

Per request: inject the prompts into a copy of the generate kernel, push it,
poll until the run finishes, then pull the produced images. The Kaggle CLI must
be installed and authenticated (``~/.kaggle/kaggle.json``) on the VM.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KERNEL_SRC = _REPO_ROOT / "kaggle" / "generate"
_KERNEL_ID = "maivandangkhoa/mayalin-generate"
_INJECT_RE = re.compile(
    r"# === INJECTED BY ORCHESTRATOR.*?# === END INJECTED ===",
    re.DOTALL,
)


class KaggleError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _hf_token() -> str:
    """HF token for the gated FLUX download: environment first, then repo .env.

    Injected into the pushed (private) kernel so generation does not depend on a
    Kaggle Secret being attached to the kernel.
    """
    tok = os.environ.get("HUGGING_FACE_KEY") or os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    env_file = _REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(("HUGGING_FACE_KEY=", "HF_TOKEN=")):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def _inject(prompts: list[str], seed: int | None, character: str) -> str:
    lines = ",\n    ".join(repr(p) for p in prompts)
    return (
        "# === INJECTED BY ORCHESTRATOR (do not edit by hand) ===\n"
        f"CHARACTER = {character!r}\n"
        f"SEED = {seed!r}\n"
        f"HF_TOKEN = {_hf_token()!r}\n"
        f"PROMPTS = [\n    {lines},\n]\n"
        "# === END INJECTED ==="
    )


def _build_push_dir(dest: Path, prompts: list[str], seed: int | None, character: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_KERNEL_SRC / "kernel-metadata.json", dest / "kernel-metadata.json")
    src = (_KERNEL_SRC / "generate_kernel.py").read_text(encoding="utf-8")
    patched = _INJECT_RE.sub(_inject(prompts, seed, character), src, count=1)
    (dest / "generate_kernel.py").write_text(patched, encoding="utf-8")


def _force_t4_accelerator() -> None:
    """Make `kernels_push` request a T4 instead of Kaggle's default P100.

    Kaggle's PyTorch build dropped Pascal (sm_60), so a P100 session can't run
    CUDA at all. `kaggle kernels push` has no metadata knob for the accelerator
    type (it only sets ``enable_gpu``), so we set ``machine_shape`` on the SDK
    request directly. Patched once, in-process (hence the API push below instead
    of the CLI subprocess).
    """
    from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient

    if getattr(KernelsApiClient, "_force_t4", False):
        return
    _orig = KernelsApiClient.save_kernel

    def _save(self, request=None):  # noqa: ANN001
        if request is not None and getattr(request, "enable_gpu", False):
            request.machine_shape = "NvidiaTeslaT4"
        return _orig(self, request)

    KernelsApiClient.save_kernel = _save
    KernelsApiClient._force_t4 = True


def push(prompts: list[str], seed: int | None, character: str, work_dir: Path) -> None:
    """Inject prompts and trigger a kernel run on a T4 (not the default P100)."""
    _build_push_dir(work_dir, prompts, seed, character)
    import kaggle

    _force_t4_accelerator()
    resp = kaggle.api.kernels_push(str(work_dir))
    err = getattr(resp, "error", None)
    if err:
        raise KaggleError(f"kernels push failed: {err}")


def poll(timeout_s: int = 3600, interval_s: int = 30) -> str:
    """Block until the kernel run completes. Returns the final status string."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        res = _run(["kaggle", "kernels", "status", _KERNEL_ID])
        out = (res.stdout + res.stderr).lower()
        if "complete" in out:
            return "complete"
        if "error" in out or "cancel" in out:
            raise KaggleError(f"kernel run failed: {res.stdout or res.stderr}")
        time.sleep(interval_s)
    raise KaggleError("timed out waiting for kernel run")


def pull_images(dest: Path) -> list[str]:
    """Download kernel output and return the saved image filenames."""
    dest.mkdir(parents=True, exist_ok=True)
    res = _run(["kaggle", "kernels", "output", _KERNEL_ID, "-p", str(dest)])
    if res.returncode != 0:
        raise KaggleError(f"kernels output failed: {res.stderr or res.stdout}")
    names: list[str] = []
    for img in dest.rglob("*"):
        if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and img.is_file():
            if img.parent != dest:
                target = dest / img.name
                shutil.copyfile(img, target)
            names.append(img.name)
    return sorted(set(names))

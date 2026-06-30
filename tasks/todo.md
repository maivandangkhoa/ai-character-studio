# AI Character Studio — Implementation Plan

> Spec: `AI Character Studio.md` v1.0
> Decisions: Trainer = wrap **kohya_ss/sd-scripts** (FLUX branch) · Scope = **full scaffold + both pipelines** · Captioning = **Florence-2**
> Constraints (from global rules): typed everywhere, config-driven (no hardcoded paths), no globals, every module < 300 LOC, PEP8, resume-safe.

---

## Phase 0 — Repo scaffold & foundation
- [ ] Create folder tree per spec (`src/`, `config/`, `notebooks/`, `scripts/`, `datasets/`, `outputs/`, `docs/`)
- [ ] `pyproject.toml` (Python 3.11) + `requirements.txt` (torch, diffusers, transformers, peft, accelerate, pyyaml, pillow, etc.)
- [ ] `.gitignore` (datasets/raw, outputs, models, caches, `.safetensors`)
- [ ] `src/utils/config.py` — load/merge/validate YAML, env-aware path resolution (Colab vs RunPod vs local)
- [ ] `src/utils/paths.py` — single source of truth for all paths, driven by config (no hardcoding)
- [ ] `src/utils/logging.py` — structured logger capturing start/end time, GPU, loss, image count, elapsed
- [ ] `src/utils/env.py` — detect environment (Colab/RunPod/local), GPU info, Drive availability

## Phase 1 — Configuration layer
- [ ] `config/training.yaml` — base model (FLUX.1-dev), steps, resolution, batch, LR, save-every-N, trigger word, paths
- [ ] `config/generation.yaml` — sampler, steps, guidance, output dir, naming, negative defaults
- [ ] `config/paths.yaml` (or section) — Drive root `AICharacter/`, datasets/models/loras/outputs/cache
- [ ] Schema validation with clear errors on missing/invalid keys

## Phase 2 — Dataset pipeline (`src/dataset/`)
- [ ] `loader.py` — discover characters, pair image+`.txt`, validate counts (warn if <20 / >40)
- [ ] `preprocess.py` — resize/center-crop to ≥1024, strip EXIF, normalize format → `processed/`
- [ ] `caption.py` — Florence-2 auto-caption (optional), inject trigger word, write `captions/*.txt`
- [ ] Idempotent / resume-safe: skip already-processed images, no duplicate outputs

## Phase 3 — Training pipeline (`src/train/`)
- [ ] `kohya_runner.py` — clone/locate sd-scripts FLUX branch, build CLI args from `training.yaml`
- [ ] `dataset_config.py` — emit kohya dataset config (toml) from our character dataset
- [ ] `checkpoint.py` — resume from latest checkpoint automatically; save every N steps
- [ ] `export.py` — produce `lora.safetensors`, `metadata.json`, `preview.png`, `training_log.json`
- [ ] `scripts/train.py` — thin CLI entry: `python scripts/train.py --character emily`

## Phase 4 — Generation pipeline (`src/generate/`)
- [ ] `pipeline.py` — load FLUX + LoRA via diffusers, generate, save image + metadata JSON
- [ ] `naming.py` — `YYYYMMDD_HHMMSS_UUID.png` + sidecar JSON
- [ ] Resume-safe batch generation (skip existing, no dup outputs)
- [ ] `scripts/generate.py` — CLI: `python scripts/generate.py --character emily --count 20`

## Phase 5 — Prompt system (`src/prompts/`)
- [ ] `template.py` — structured template (Character/Scene/Camera/Lighting/Style/Negative)
- [ ] `builder.py` — random clothing/weather/country/pose/camera/lens/emotion/time/season
- [ ] `wordbanks/` — YAML word lists feeding the randomizer (config-driven, editable)

## Phase 6 — Setup & Drive scripts (`scripts/`)
- [ ] `setup.py` — install deps, verify GPU/CUDA
- [ ] `download_models.py` — fetch FLUX.1-dev (+ Florence-2) into Drive cache, skip if present
- [ ] `mount_drive.py` — mount Google Drive, ensure `AICharacter/` tree exists

## Phase 7 — Colab notebooks (`notebooks/`)
- [ ] `train_lora.ipynb` — mount Drive → clone repo → install → download model → train (resume) → export → push to Drive, shutdown cleanly
- [ ] `generate.ipynb` — mount → load FLUX+LoRA → build prompts → batch generate → save to Drive

## Phase 8 — Docs & verification
- [ ] `README.md` + `docs/` (quickstart, Colab guide, config reference)
- [ ] Dry-run smoke test of dataset → (mock) train args → generate path resolution locally
- [ ] Update `MEMORY.md` (architecture, key paths) and `tasks/lessons.md`

---

## Open questions / risks
- FLUX.1-dev is gated on HuggingFace → need HF token handling (env/Drive secret).
- kohya FLUX branch + Colab VRAM (T4/A100) → may need fp8/low-VRAM flags; validate on target GPU tier.
- Florence-2 download size vs Colab disk → cache to Drive.

## Phase 9 — Web UI + Kaggle-backed generation (2026-06-30)
Goal: a web UI on the VM where the user types prompts and gets images. Training
runs once on Kaggle → LoRA pulled to VM. Generation runs on Kaggle batch
(user chose free/slow), driven by the VM via the Kaggle API.
- [x] `src/prompts/builder.py` — `prompts_to_payloads()` for user-supplied prompts
- [x] `scripts/generate.py` — `--prompts-file` (custom prompts, trigger auto-prepended)
- [x] `kaggle/generate/generate_kernel.py` — script kernel: clone repo, load LoRA from dataset, run generate
- [x] `kaggle/generate/kernel-metadata.json` — kernel id `maivandangkhoa/mayalin-generate`, GPU+internet, lora dataset
- [x] `scripts/publish_lora.py` — push pulled LoRA to Kaggle Dataset `maivandangkhoa/mayalin-lora`
- [x] `webapp/` — FastAPI: submit prompts → background job (push kernel → poll → pull images) → gallery
- [x] `webapp/README.md` — VM setup (kaggle token, run uvicorn)
- Risks: generation on T4 not verified (FLUX diffusers ~24GB, same OOM risk as training — may need fp8); 20-40 min cold-start per batch; Kaggle 30h/week quota.

## Review (completed 2026-06-29)
All 9 phases implemented in one pass. Every checkbox above is done.

**Verified (GPU-free):**
- `scripts/selfcheck.py` passes — config load, typed schemas, prompt builder, dataset TOML, output naming all OK.
- Functional test of `preprocess_character` + `load_dataset`: produces 1024×1024 PNGs, correct image/caption pairing, resume-safe (re-run writes 0).
- Verified on python3.12 venv (system python is 3.9 and can't run the 3.11+ syntax).

**Standards met:** typed throughout, config-driven (no hardcoded paths/values), no globals,
every module < 300 LOC, lazy torch/diffusers imports, resume-safe stages.

**NOT yet validated (needs GPU + HF_TOKEN):** real FLUX training run, Florence-2 captioning,
diffusers generation. These run on Colab/RunPod. CLI wiring is tested; the heavy ML path is not.

**Follow-ups / risks:**
- Confirm sd-scripts branch `sd3` still hosts `flux_train_network.py` (pin a commit for reproducibility).
- Tune `--blocks_to_swap` / fp8 flags per GPU tier (T4 vs A100).
- Set the real repo URL in both notebooks (`REPO_URL` placeholder).

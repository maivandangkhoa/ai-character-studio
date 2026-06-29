# AI Character Studio

Automated AI character **LoRA training** and **image generation** on **FLUX.1-dev**,
designed to run primarily on **Google Colab + Google Drive** and managed by Claude Code.

> One-click pipeline: Character Photos → Train LoRA → Generate Dataset → Generate Images → Publish

## Features

- Train a character LoRA on FLUX via the kohya `sd-scripts` FLUX trainer.
- Identity consistency through a fixed trigger word per character.
- Florence-2 auto-captioning when captions are missing.
- Randomised prompt generation (clothing, weather, country, pose, camera, lens, emotion, time, season).
- Fully config-driven (`config/*.yaml`) — **no hardcoded paths or values**.
- Resume-safe at every stage; identical re-runs never duplicate work.
- Runs unchanged on Colab / RunPod / local (environment auto-detected).

## Layout

```
config/        training.yaml · generation.yaml · paths.yaml
src/
  utils/       config, paths, env, logging, schemas
  dataset/     loader, preprocess, caption (Florence-2)
  models/      FLUX weight downloader
  train/       kohya runner, dataset config, checkpoints, export
  generate/    FLUX+LoRA pipeline, output naming
  prompts/     template, builder, wordbanks.yaml
scripts/       setup · mount_drive · download_models · train · generate · selfcheck
notebooks/     train_lora.ipynb · generate.ipynb
```

## Quickstart (Colab)

1. Open `notebooks/train_lora.ipynb` in Colab, set `CHARACTER` and `HF_TOKEN`.
2. Upload 20–40 images to `AICharacter/datasets/<character>/raw/` on Drive.
3. Run all cells: install → download FLUX → train → export → shutdown.
4. Open `notebooks/generate.ipynb`, set `CHARACTER` and `COUNT`, run all cells.

## Quickstart (local / RunPod)

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx                       # FLUX.1-dev is gated
python scripts/mount_drive.py                # creates the storage tree
python scripts/download_models.py            # downloads FLUX weights
python scripts/train.py --character emily --steps 4000
python scripts/generate.py --character emily --count 20
```

Verify the wiring without a GPU:

```bash
python scripts/selfcheck.py
```

## Configuration

Edit `config/training.yaml` and `config/generation.yaml`. Any top-level scalar can
also be overridden via environment variables (e.g. `AICS_STEPS=6000`) or CLI flags.
Storage location is set in `config/paths.yaml` or via `AICS_DRIVE_ROOT`.

See [docs/quickstart.md](docs/quickstart.md) and [docs/configuration.md](docs/configuration.md).

# Quickstart

## Prerequisites
- Hugging Face account with access to `black-forest-labs/FLUX.1-dev` (gated — click "Agree").
- A Hugging Face token: https://huggingface.co/settings/tokens → export as `HF_TOKEN`.
- GPU: Colab T4/L4/A100 or RunPod equivalent. T4 works with `low_vram: true`.

## 1. Prepare a dataset
Put 20–40 images of one character under:

```
AICharacter/datasets/<character>/raw/
```

Guidelines (from the spec): multiple angles, varied expressions and lighting,
consistent hairstyle, no heavy filters, no text overlay, ≥1024px preferred.

Captions are optional — Florence-2 generates them automatically. To caption
manually, drop `<stem>.txt` files in `datasets/<character>/captions/`.

## 2. Train
**Colab:** run `notebooks/train_lora.ipynb` top to bottom.

**CLI:**
```bash
export HF_TOKEN=hf_xxx
python scripts/train.py --character emily
```
Output bundle lands in `outputs/lora/<character>/`:
`lora.safetensors`, `metadata.json`, `preview.png`, `training_log.json`.

Training auto-resumes from the latest checkpoint if interrupted.

## 3. Generate
**Colab:** run `notebooks/generate.ipynb`.

**CLI:**
```bash
python scripts/generate.py --character emily --count 20
```
Images + JSON sidecars are written to `outputs/generated/<character>/` named
`YYYYMMDD_HHMMSS_UUID.png`. Re-running only tops up to `--count`.

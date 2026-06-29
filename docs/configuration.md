# Configuration Reference

All behaviour is config-driven. Precedence (lowest → highest):
**YAML file → `AICS_*` env var → CLI flag.**

## paths.yaml
| Key | Meaning |
|-----|---------|
| `drive_root` | Storage root on Colab/RunPod (default Drive mount). |
| `local_root` | Root when running locally (`.` = repo root). |
| `subdirs.*` | Names of datasets/models/loras/generated/logs/cache dirs. |
| `sd_scripts_repo` / `sd_scripts_branch` | kohya trainer source. |

Override the root anywhere with `AICS_DRIVE_ROOT=/path`.

## training.yaml
| Key | Meaning |
|-----|---------|
| `base_model` | FLUX base, default `black-forest-labs/FLUX.1-dev`. |
| `trigger_word` | Identity token; `null` → `<character>_ai`. |
| `resolution` / `steps` | Image size and total train steps. |
| `network_dim` / `network_alpha` | LoRA rank / alpha. |
| `learning_rate`, `optimizer`, `lr_scheduler` | Optimisation. |
| `save_every_n_steps`, `keep_last_n_checkpoints`, `resume` | Checkpointing. |
| `low_vram`, `fp8_base`, `cache_latents`, `cache_text_encoder_outputs` | Memory. |
| `auto_caption`, `caption_model` | Florence-2 captioning. |

## generation.yaml
| Key | Meaning |
|-----|---------|
| `width` / `height` / `num_inference_steps` / `guidance_scale` | Sampling. |
| `seed` | `null` → varied per image; integer → reproducible. |
| `lora_weight` | LoRA fuse scale. |
| `count` | Default images per run. |
| `skip_existing` | Resume-safe top-up to `count`. |
| `negative_prompt` | Stored in metadata/templates. |

## Examples
```bash
AICS_STEPS=6000 python scripts/train.py --character emily
python scripts/train.py --character emily --steps 6000 --no-caption
python scripts/generate.py --character emily --count 50 --seed 1234
```

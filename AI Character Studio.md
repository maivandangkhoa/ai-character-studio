AI Character Studio

Version: 1.0

---

# Vision

Build a fully automated AI character generation platform capable of:

- Training a LoRA from a custom character
- Keeping identity consistency
- Generating unlimited images
- Running primarily on Google Colab
- Managed entirely by Claude Code

Long-term goal:

One-click pipeline:

Character Photos
      ↓
Train LoRA
      ↓
Generate Dataset
      ↓
Generate Images
      ↓
Publish

---

# Objectives

The project should allow users to:

✓ Create one or multiple AI characters

✓ Train LoRA models

✓ Generate highly consistent images

✓ Reuse the same LoRA indefinitely

✓ Switch between Google Colab and RunPod without code changes

---

# Tech Stack

Python 3.11

PyTorch

Diffusers

Transformers

PEFT

Accelerate

ComfyUI (optional)

Google Colab

Google Drive

GitHub

Claude Code

---

# Repository Structure

project/

    docs/

    notebooks/

        train_lora.ipynb

        generate.ipynb

    src/

        train/

        generate/

        dataset/

        prompts/

        models/

        utils/

    config/

        training.yaml

        generation.yaml

    datasets/

        character_name/

            raw/

            processed/

            captions/

    outputs/

        lora/

        generated/

        logs/

    scripts/

        setup.py

        download_models.py

        mount_drive.py

        train.py

        generate.py

---

# Character Dataset

Each character owns an independent dataset.

Example:

datasets/

    emily/

        raw/

        processed/

        captions/

Every image should have a matching caption.

Example

0001.jpg

0001.txt

---

# Dataset Recommendation

20-40 images

Requirements

Multiple angles

Different expressions

Different lighting

Consistent hairstyle

No heavy filters

No text overlay

Resolution above 1024px preferred

---

# Training Pipeline

Step 1

Mount Google Drive

↓

Clone repository

↓

Install dependencies

↓

Download FLUX model

↓

Prepare dataset

↓

Auto-caption images (optional)

↓

Train LoRA

↓

Save checkpoints

↓

Export final LoRA

↓

Push metadata to Drive

---

# LoRA Output

Each training creates

lora.safetensors

metadata.json

preview.png

training_log.json

---

# Metadata Example

{
    "character":"Emily",
    "trigger_word":"emily_ai",
    "base_model":"FLUX.1-dev",
    "steps":4000,
    "resolution":1024
}

---

# Generation Pipeline

Load FLUX

↓

Load LoRA

↓

Read prompt template

↓

Generate image

↓

Save image

↓

Save metadata

↓

Repeat

---

# Prompt Template

Character:

emily_ai

Scene:

walking in Tokyo

Camera:

50mm

Lighting:

golden hour

Style:

cinematic

Negative:

blurry

low quality

deformed

---

# Prompt Generator

Prompt Builder should support:

Random clothing

Random weather

Random country

Random pose

Random camera

Random lens

Random emotion

Random time

Random season

---

# Output Naming

YYYYMMDD_HHMMSS_UUID.png

Metadata stored as JSON.

---

# Configuration

training.yaml

generation.yaml

No hardcoded values.

Everything configurable.

---

# Google Drive Structure

AICharacter/

    datasets/

    models/

    loras/

    outputs/

    cache/

---

# Colab Workflow

Notebook must:

Mount Drive

↓

Clone latest repository

↓

Install missing packages

↓

Resume previous checkpoint automatically

↓

Continue if interrupted

↓

Save every N steps

↓

Shutdown cleanly

---

# Error Recovery

Every stage should support resume.

No duplicated outputs.

No manual intervention.

---

# Coding Rules

Use typing everywhere.

No global variables.

Config-driven.

Reusable modules only.

Every module under 300 lines.

PEP8 compliant.

---

# Logging

Every operation logs:

start time

end time

GPU

training loss

image count

elapsed time

---

# Claude Code Responsibilities

Claude should be able to:

Create notebooks

Update notebooks

Generate Python code

Fix bugs

Refactor

Generate prompt templates

Create datasets

Generate README

Generate documentation

Never hardcode paths.

Always use configuration.

---

# Future Features

Video generation

ControlNet

IPAdapter

Face swap

Pose control

Character chatting

Automatic social posting

Web dashboard

REST API

Queue system

RunPod support

Vast.ai support

Multi-character support

Training scheduler

Cloud deployment

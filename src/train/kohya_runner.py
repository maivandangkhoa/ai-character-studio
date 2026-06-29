"""Orchestrate FLUX LoRA training via kohya_ss/sd-scripts.

Our code owns config, dataset prep, captions, checkpoints and Drive layout;
sd-scripts performs the actual training. We build the ``flux_train_network.py``
command from :class:`TrainingConfig` so nothing is hardcoded.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from src.dataset.loader import load_dataset
from src.models.download import FluxAssets
from src.train.checkpoint import latest_checkpoint
from src.train.dataset_config import write_dataset_toml
from src.utils.logging import get_logger
from src.utils.paths import CharacterPaths, Paths
from src.utils.schemas import TrainingConfig

logger = get_logger()


class KohyaRunner:
    """Builds and runs the sd-scripts FLUX training command."""

    def __init__(self, paths: Paths, cfg: TrainingConfig, assets: FluxAssets) -> None:
        self.paths = paths
        self.cfg = cfg
        self.assets = assets
        self.cp: CharacterPaths = paths.character(cfg.character)
        self.lora_dir = paths.lora_dir(cfg.character)

    def ensure_sd_scripts(self) -> None:
        """Clone sd-scripts (FLUX branch) and install its deps if not present."""
        sd_dir = self.paths.sd_scripts_dir
        if (sd_dir / "flux_train_network.py").exists():
            return
        if sd_dir.exists():
            # Leftover from an interrupted/partial clone (common on Drive sync).
            # Remove it so the clone below starts from a clean directory.
            logger.warning("Removing incomplete sd-scripts at %s ...", sd_dir)
            shutil.rmtree(sd_dir, ignore_errors=True)
        sd_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning sd-scripts (%s) ...", self.paths.sd_scripts_branch)
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                self.paths.sd_scripts_branch,
                "--depth",
                "1",
                self.paths.sd_scripts_repo,
                str(sd_dir),
            ],
            check=True,
        )
        self._install_sd_scripts_deps(sd_dir)

    @staticmethod
    def _install_sd_scripts_deps(sd_dir) -> None:
        """Install sd-scripts' own requirements (excluding its editable self-line)."""
        req = sd_dir / "requirements.txt"
        if not req.exists():
            return
        logger.info("Installing sd-scripts requirements ...")
        lines = [
            ln.strip()
            for ln in req.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#") and ln.strip() != "."
        ]
        if lines:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *lines], check=True)

    def _sync_captions(self) -> int:
        """Copy captions next to processed images (sd-scripts reads sidecars)."""
        copied = 0
        for item in load_dataset(self.cp, source="processed"):
            if item.has_caption:
                dst = self.cp.processed / f"{item.image.stem}.txt"
                if not dst.exists():
                    shutil.copyfile(item.caption, dst)
                    copied += 1
        return copied

    def build_command(self) -> list[str]:
        """Construct the accelerate launch command for flux_train_network.py."""
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        toml = write_dataset_toml(
            self.cp,
            out_dir=self.lora_dir,
            resolution=self.cfg.resolution,
            batch_size=self.cfg.train_batch_size,
        )
        script = self.paths.sd_scripts_dir / "flux_train_network.py"
        cmd: list[str] = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_cpu_threads_per_process",
            "2",
            str(script),
            "--pretrained_model_name_or_path",
            str(self.assets.transformer),
            "--ae",
            str(self.assets.ae),
            "--clip_l",
            str(self.assets.clip_l),
            "--t5xxl",
            str(self.assets.t5xxl),
            "--dataset_config",
            str(toml),
            "--output_dir",
            str(self.lora_dir),
            "--output_name",
            self.cfg.character,
            "--network_module",
            "networks.lora_flux",
            "--network_dim",
            str(self.cfg.network_dim),
            "--network_alpha",
            str(self.cfg.network_alpha),
            "--learning_rate",
            str(self.cfg.learning_rate),
            "--optimizer_type",
            self.cfg.optimizer,
            "--lr_scheduler",
            self.cfg.lr_scheduler,
            "--max_train_steps",
            str(self.cfg.steps),
            "--save_every_n_steps",
            str(self.cfg.save_every_n_steps),
            "--mixed_precision",
            self.cfg.mixed_precision,
            "--save_precision",
            self.cfg.mixed_precision,
            "--seed",
            str(self.cfg.seed),
            "--gradient_accumulation_steps",
            str(self.cfg.gradient_accumulation_steps),
            "--guidance_scale",
            "1.0",
            "--timestep_sampling",
            "shift",
            "--model_prediction_type",
            "raw",
            "--loss_type",
            "l2",
        ]
        cmd += self._memory_flags()
        cmd += self._resume_flags()
        return cmd

    def _memory_flags(self) -> list[str]:
        flags: list[str] = ["--sdpa", "--gradient_checkpointing"]
        if self.cfg.cache_latents:
            flags += ["--cache_latents", "--cache_latents_to_disk"]
        if self.cfg.cache_text_encoder_outputs:
            flags += ["--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk"]
        if self.cfg.fp8_base:
            flags.append("--fp8_base")
        if self.cfg.low_vram:
            flags += ["--blocks_to_swap", "8"]
        return flags

    def _resume_flags(self) -> list[str]:
        if not self.cfg.resume:
            return []
        ckpt = latest_checkpoint(self.lora_dir)
        if ckpt is None:
            return []
        logger.info("Resuming from %s", ckpt.name)
        return ["--network_weights", str(ckpt)]

    def run(self) -> int:
        """Prepare and execute training. Returns the subprocess return code."""
        self.ensure_sd_scripts()
        synced = self._sync_captions()
        logger.info("Synced %d captions next to images.", synced)
        cmd = self.build_command()
        logger.info("Launching trainer: %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(self.paths.sd_scripts_dir))
        return proc.returncode

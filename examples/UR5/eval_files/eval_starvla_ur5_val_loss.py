#!/usr/bin/env python3
"""Evaluate one StarVLA UR5 checkpoint on the independent validation split.

This script intentionally performs validation only:
  * loads the complete checkpoint (VLM/LoRA + action model),
  * does not call backward(), optimizer.step(), or checkpoint saving,
  * computes Flow-Matching action loss on ``starvla_ur5_val``, and
  * upserts one row into a CSV file for checkpoint comparison.

Run it through ``accelerate launch`` with the same DeepSpeed launcher config
used for training. Evaluate multiple checkpoints one process at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List

# The trainer initializes W&B when prepare_training() is called. Disabled mode
# keeps validation local unless the caller explicitly overrides WANDB_MODE.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.train_starvla import (
    VLATrainer,
    accelerator,
    prepare_data,
    prepare_validation_data,
    setup_directories,
    setup_optimizer_and_scheduler,
)
from starVLA.training.trainer_utils.config_tracker import wrap_config


DEFAULT_CONFIG = (
    "examples/UR5/train_files/"
    "starvla_qwengroot_ur5_4090.yaml"
)
DEFAULT_RESULTS = (
    "results/ur5_val_loss/checkpoint_val_losses.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a StarVLA UR5 checkpoint on the validation split."
        )
    )
    parser.add_argument(
        "--config_yaml",
        default=DEFAULT_CONFIG,
        help="UR5 training YAML containing val_data_mix.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint .pt file to load and evaluate.",
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=50,
        help=(
            "Number of validation batches. Use 0 to evaluate the full "
            "validation dataloader."
        ),
    )
    parser.add_argument(
        "--results_csv",
        default=DEFAULT_RESULTS,
        help="CSV file used to collect checkpoint validation losses.",
    )
    return parser.parse_args()


def checkpoint_step(checkpoint: Path) -> int:
    match = re.search(r"steps_(\d+)_pytorch_model\.pt$", checkpoint.name)
    return int(match.group(1)) if match else -1


def safe_run_id(checkpoint: Path) -> str:
    parent = checkpoint.parent.name
    stem = checkpoint.stem
    raw_name = f"val_only_{parent}_{stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name)


def upsert_result(results_path: Path, row: Dict[str, str]) -> None:
    """Insert or replace a result identified by its absolute checkpoint path."""
    fieldnames = [
        "checkpoint_name",
        "checkpoint_path",
        "step",
        "eval_batches",
        "val_action_dit_loss",
    ]
    rows: List[Dict[str, str]] = []

    if results_path.is_file():
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    rows = [
        existing
        for existing in rows
        if existing.get("checkpoint_path") != row["checkpoint_path"]
    ]
    rows.append(row)
    rows.sort(
        key=lambda item: (
            int(item.get("step", "-1")),
            item.get("checkpoint_name", ""),
        )
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = results_path.with_suffix(results_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, results_path)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_yaml).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    results_path = Path(args.results_csv).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.eval_batches < 0:
        raise ValueError("--eval_batches must be 0 or a positive integer.")

    cfg = OmegaConf.load(config_path)
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = str(config_path)

    # Use a small validation runtime directory. No model checkpoint is written.
    cfg.run_root_dir = "./results/ur5_val_loss/runtime"
    cfg.run_id = safe_run_id(checkpoint)

    # None means load the complete saved state. In particular, do not use the
    # training-time reload_modules="action_model" setting for evaluation.
    cfg.trainer.pretrained_checkpoint = str(checkpoint)
    cfg.trainer.reload_modules = None
    cfg.trainer.is_resume = False
    cfg.trainer.eval_batches = (
        args.eval_batches if args.eval_batches > 0 else 1_000_000_000
    )

    cfg = wrap_config(cfg)
    output_dir = setup_directories(cfg)
    model = build_framework(cfg)

    # VLATrainer uses the train loader to construct its normal distributed
    # state. No training batch is consumed by this script.
    train_dataloader = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
    )
    val_dataloader = prepare_validation_data(
        cfg=cfg,
        accelerator=accelerator,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(
        model=model,
        cfg=cfg,
    )

    trainer = VLATrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=train_dataloader,
        vla_val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    metrics = trainer.eval_action_model({})

    if accelerator.is_main_process:
        if "val_action_dit_loss" not in metrics:
            raise RuntimeError("Validation did not produce val_action_dit_loss.")

        row = {
            "checkpoint_name": checkpoint.name,
            "checkpoint_path": str(checkpoint),
            "step": str(checkpoint_step(checkpoint)),
            "eval_batches": (
                str(args.eval_batches) if args.eval_batches > 0 else "all"
            ),
            "val_action_dit_loss": f"{metrics['val_action_dit_loss']:.10g}",
        }
        upsert_result(results_path, row)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        print(f"Validation results: {results_path}")

    if wandb.run is not None:
        wandb.finish()
    accelerator.wait_for_everyone()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

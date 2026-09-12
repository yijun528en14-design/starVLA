#!/usr/bin/env python3
"""Compare StarVLA predictions with normalized GT actions on training samples.

Run this file with `accelerate launch` using the same DeepSpeed config used for
training.  It deliberately compares in normalized training space, before the
ROS adapter, so adapter/frame/unit errors cannot affect the metrics.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_ddim_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output_csv",
        default="results/ur5_train_action_compare/per_sample.csv",
    )
    return parser.parse_args()


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def cosine_xyz(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = prediction[..., :3]
    gt = target[..., :3]
    numerator = np.sum(pred * gt, axis=-1)
    denominator = np.linalg.norm(pred, axis=-1) * np.linalg.norm(gt, axis=-1)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > 1e-8
    result[valid] = numerator[valid] / denominator[valid]
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")

    config_path = Path(args.config_yaml).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    cfg = OmegaConf.load(config_path)
    cfg = apply_config_compat(cfg)
    cfg.trainer.pretrained_checkpoint = str(checkpoint_path)
    cfg.trainer.reload_modules = None
    cfg.trainer.is_resume = False
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.run_root_dir = "./results/ur5_train_action_compare/runtime"
    cfg.run_id = checkpoint_path.stem
    cfg.wandb_project = "starvla_ur5_action_compare"
    cfg = wrap_config(cfg)

    output_dir = setup_directories(cfg)
    model = build_framework(cfg)
    train_loader = prepare_data(cfg, accelerator, output_dir)
    val_loader = prepare_validation_data(cfg, accelerator)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=train_loader,
        vla_val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()

    model_for_inference = accelerator.unwrap_model(trainer.model)
    model_for_inference.eval()
    iterator = iter(trainer.vla_train_dataloader)

    all_predictions = []
    all_targets = []
    sample_rows = []
    collected = 0

    with torch.no_grad():
        while collected < args.num_samples:
            try:
                examples = next(iterator)
            except StopIteration:
                iterator = iter(trainer.vla_train_dataloader)
                examples = next(iterator)

            torch.manual_seed(args.seed + collected)
            np.random.seed(args.seed + collected)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + collected)

            result = model_for_inference.predict_action(
                examples=examples,
                use_ddim=True,
                num_ddim_steps=args.num_ddim_steps,
            )
            predictions = to_numpy(result["normalized_actions"])

            horizon = predictions.shape[-2]
            targets = np.stack(
                [to_numpy(example["action"])[-horizon:] for example in examples]
            )
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Prediction shape {predictions.shape} != GT shape {targets.shape}"
                )

            for batch_index in range(len(predictions)):
                pred = predictions[batch_index]
                gt = targets[batch_index]
                xyz_mae = float(np.mean(np.abs(pred[:, :3] - gt[:, :3])))
                cosines = cosine_xyz(pred[None], gt[None]).reshape(-1)
                valid_cosines = cosines[np.isfinite(cosines)]
                cosine_mean = (
                    float(np.mean(valid_cosines)) if len(valid_cosines) else float("nan")
                )
                sign_mask = np.abs(gt[:, :3]) > 1e-5
                sign_accuracy = (
                    float(np.mean(np.sign(pred[:, :3][sign_mask]) == np.sign(gt[:, :3][sign_mask])))
                    if np.any(sign_mask)
                    else float("nan")
                )
                sample_rows.append(
                    {
                        "sample": collected,
                        "xyz_mae_normalized": xyz_mae,
                        "xyz_cosine": cosine_mean,
                        "xyz_sign_accuracy": sign_accuracy,
                        "gripper_mae_normalized": float(
                            np.mean(np.abs(pred[:, 6] - gt[:, 6]))
                        ),
                    }
                )
                all_predictions.append(pred)
                all_targets.append(gt)
                collected += 1
                if collected >= args.num_samples:
                    break

    predictions = np.stack(all_predictions)
    targets = np.stack(all_targets)
    axis_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    per_axis_mae = np.mean(np.abs(predictions - targets), axis=(0, 1))

    gt_gripper_min = float(targets[..., 6].min())
    gt_gripper_max = float(targets[..., 6].max())
    gripper_threshold = 0.5 * (gt_gripper_min + gt_gripper_max)
    gripper_accuracy = float(
        np.mean(
            (predictions[..., 6] >= gripper_threshold)
            == (targets[..., 6] >= gripper_threshold)
        )
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "num_samples": int(len(predictions)),
        "comparison_space": "normalized training action space",
        "xyz_mae_normalized": float(
            np.mean(np.abs(predictions[..., :3] - targets[..., :3]))
        ),
        "xyz_cosine": float(np.nanmean(cosine_xyz(predictions, targets))),
        "per_axis_mae_normalized": {
            name: float(value) for name, value in zip(axis_names, per_axis_mae)
        },
        "gripper_gt_min": gt_gripper_min,
        "gripper_gt_max": gt_gripper_max,
        "gripper_threshold_for_report": gripper_threshold,
        "gripper_classification_accuracy": gripper_accuracy,
        "first_gt_action": targets[0, 0].tolist(),
        "first_predicted_action": predictions[0, 0].tolist(),
    }

    if accelerator.is_main_process:
        output_csv = Path(args.output_csv).expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)
        summary_path = output_csv.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Per-sample CSV: {output_csv}")
        print(f"Summary JSON: {summary_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wandb


EPS = 1e-8


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < EPS or nb < EPS:
        return np.nan

    return float(np.dot(a, b) / (na * nb))


def sign_equal(a, b, threshold=1e-5):
    """
    小於 threshold 視為 zero。
    """
    def s(x):
        if x > threshold:
            return 1
        if x < -threshold:
            return -1
        return 0

    return int(s(a) == s(b))


def evaluate_sample(gt, pred):
    """
    gt/pred:
        [16, 7]

    action:
        0 x
        1 y
        2 z
        3 roll
        4 pitch
        5 yaw
        6 gripper
    """

    assert gt.shape == pred.shape
    assert gt.ndim == 2
    assert gt.shape[1] == 7

    err = pred - gt

    # --------------------------------------------------
    # First step
    # --------------------------------------------------

    first_gt_xyz = gt[0, :3]
    first_pred_xyz = pred[0, :3]

    # --------------------------------------------------
    # Whole chunk displacement
    # --------------------------------------------------

    gt_sum_xyz = gt[:, :3].sum(axis=0)
    pred_sum_xyz = pred[:, :3].sum(axis=0)

    endpoint_error = np.linalg.norm(
        pred_sum_xyz - gt_sum_xyz
    )

    return {
        # first-step XYZ
        "first/x_mae": abs(err[0, 0]),
        "first/y_mae": abs(err[0, 1]),
        "first/z_mae": abs(err[0, 2]),

        "first/x_sign_correct":
            sign_equal(gt[0, 0], pred[0, 0]),

        "first/y_sign_correct":
            sign_equal(gt[0, 1], pred[0, 1]),

        "first/z_sign_correct":
            sign_equal(gt[0, 2], pred[0, 2]),

        "first/xyz_cosine":
            cosine(first_gt_xyz, first_pred_xyz),

        # full chunk axis error
        "chunk/x_mae":
            np.mean(np.abs(err[:, 0])),

        "chunk/y_mae":
            np.mean(np.abs(err[:, 1])),

        "chunk/z_mae":
            np.mean(np.abs(err[:, 2])),

        # rotation
        "chunk/roll_mae":
            np.mean(np.abs(err[:, 3])),

        "chunk/pitch_mae":
            np.mean(np.abs(err[:, 4])),

        "chunk/yaw_mae":
            np.mean(np.abs(err[:, 5])),

        # gripper
        "chunk/gripper_mae":
            np.mean(np.abs(err[:, 6])),

        # accumulated translation
        "gt/sum_dx": gt_sum_xyz[0],
        "gt/sum_dy": gt_sum_xyz[1],
        "gt/sum_dz": gt_sum_xyz[2],

        "pred/sum_dx": pred_sum_xyz[0],
        "pred/sum_dy": pred_sum_xyz[1],
        "pred/sum_dz": pred_sum_xyz[2],

        "chunk/x_direction_correct":
            sign_equal(gt_sum_xyz[0],
                       pred_sum_xyz[0]),

        "chunk/y_direction_correct":
            sign_equal(gt_sum_xyz[1],
                       pred_sum_xyz[1]),

        "chunk/z_direction_correct":
            sign_equal(gt_sum_xyz[2],
                       pred_sum_xyz[2]),

        "chunk/xyz_cosine":
            cosine(gt_sum_xyz, pred_sum_xyz),

        "chunk/endpoint_error":
            endpoint_error,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="NPZ containing gt_actions and pred_actions"
    )

    parser.add_argument(
        "--project",
        default="starvla-ur5-eval"
    )

    parser.add_argument(
        "--name",
        default="steps15000-train"
    )

    args = parser.parse_args()

    d = np.load(args.input)

    gt_all = d["gt_actions"]
    pred_all = d["pred_actions"]

    print("GT:", gt_all.shape)
    print("Pred:", pred_all.shape)

    assert gt_all.shape == pred_all.shape
    assert gt_all.shape[1:] == (16, 7)

    run = wandb.init(
        project=args.project,
        name=args.name,
        config={
            "samples": len(gt_all),
            "action_chunk": 16,
            "action_dim": 7,
            "input_file": args.input,
        },
    )

    rows = []

    # step-wise error
    step_xyz_mae = np.zeros(16)
    step_x_mae = np.zeros(16)
    step_y_mae = np.zeros(16)
    step_z_mae = np.zeros(16)

    for i, (gt, pred) in enumerate(
        zip(gt_all, pred_all)
    ):

        metrics = evaluate_sample(gt, pred)

        metrics["sample"] = i
        rows.append(metrics)

        step_x_mae += np.abs(
            pred[:, 0] - gt[:, 0]
        )

        step_y_mae += np.abs(
            pred[:, 1] - gt[:, 1]
        )

        step_z_mae += np.abs(
            pred[:, 2] - gt[:, 2]
        )

        step_xyz_mae += np.linalg.norm(
            pred[:, :3] - gt[:, :3],
            axis=1,
        )

    n = len(gt_all)

    step_x_mae /= n
    step_y_mae /= n
    step_z_mae /= n
    step_xyz_mae /= n

    df = pd.DataFrame(rows)

    # --------------------------------------------------
    # Overall
    # --------------------------------------------------

    summary = {

        "overall/first_x_mae":
            df["first/x_mae"].mean(),

        "overall/first_y_mae":
            df["first/y_mae"].mean(),

        "overall/first_z_mae":
            df["first/z_mae"].mean(),

        "overall/first_x_sign_acc":
            df["first/x_sign_correct"].mean(),

        "overall/first_y_sign_acc":
            df["first/y_sign_correct"].mean(),

        "overall/first_z_sign_acc":
            df["first/z_sign_correct"].mean(),

        "overall/first_xyz_cosine":
            df["first/xyz_cosine"].mean(),

        "overall/chunk_x_mae":
            df["chunk/x_mae"].mean(),

        "overall/chunk_y_mae":
            df["chunk/y_mae"].mean(),

        "overall/chunk_z_mae":
            df["chunk/z_mae"].mean(),

        "overall/chunk_x_direction_acc":
            df["chunk/x_direction_correct"].mean(),

        "overall/chunk_y_direction_acc":
            df["chunk/y_direction_correct"].mean(),

        "overall/chunk_z_direction_acc":
            df["chunk/z_direction_correct"].mean(),

        "overall/chunk_xyz_cosine":
            df["chunk/xyz_cosine"].mean(),

        "overall/endpoint_error":
            df["chunk/endpoint_error"].mean(),

        "overall/roll_mae":
            df["chunk/roll_mae"].mean(),

        "overall/pitch_mae":
            df["chunk/pitch_mae"].mean(),

        "overall/yaw_mae":
            df["chunk/yaw_mae"].mean(),

        "overall/gripper_mae":
            df["chunk/gripper_mae"].mean(),
    }

    wandb.log(summary)

    # --------------------------------------------------
    # Per-step W&B table
    # --------------------------------------------------

    step_table = wandb.Table(
        columns=[
            "step",
            "x_mae",
            "y_mae",
            "z_mae",
            "xyz_mae",
        ]
    )

    for i in range(16):

        step_table.add_data(
            i,
            float(step_x_mae[i]),
            float(step_y_mae[i]),
            float(step_z_mae[i]),
            float(step_xyz_mae[i]),
        )

    wandb.log({
        "tables/per_step_error": step_table
    })

    # --------------------------------------------------
    # Every sample
    # --------------------------------------------------

    sample_table = wandb.Table(
        dataframe=df
    )

    wandb.log({
        "tables/sample_predictions": sample_table
    })

    # Save local CSV too
    output_csv = (
        Path(args.input).with_suffix(
            ".metrics.csv"
        )
    )

    df.to_csv(output_csv, index=False)

    print("\n==============================")
    print("StarVLA Action Evaluation")
    print("==============================")

    for k, v in summary.items():
        print(f"{k:35s}: {v:.6f}")

    print("\nCSV:", output_csv)

    wandb.finish()


if __name__ == "__main__":
    main()

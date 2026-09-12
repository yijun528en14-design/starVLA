#!/usr/bin/env python3
"""Plot exact-frame StarVLA server-vs-GT diagnostics from evaluator CSV."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--transition-frame", type=int, default=26)
    return parser.parse_args()


def mark_transition(axis, frame):
    axis.axvline(frame, color="tab:red", linestyle="--", linewidth=1.5,
                 label=f"GT gripper closes (frame {frame})")
    axis.grid(True, alpha=0.25)


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("frame")
    required = {
        "frame", "xyz_mae_m", "xyz_cosine", "xyz_sign_accuracy",
        "gripper_mae", "gt_gripper_first", "pred_gripper_first",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"CSV missing columns: {sorted(missing)}")

    frames = df["frame"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(frames, df["xyz_mae_m"] * 1000.0, marker="o", markersize=3)
    mark_transition(axis, args.transition_frame)
    axis.set_title("XYZ action-chunk MAE by input frame")
    axis.set_xlabel("Input frame")
    axis.set_ylabel("Mean absolute error (mm)")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    axis.plot(frames, df["xyz_cosine"], marker="o", markersize=3)
    axis.axhline(0.0, color="black", linewidth=0.8)
    mark_transition(axis, args.transition_frame)
    axis.set_ylim(-1.05, 1.05)
    axis.set_title("XYZ direction cosine by input frame")
    axis.set_xlabel("Input frame")
    axis.set_ylabel("Cosine similarity")
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    axis.plot(frames, df["xyz_sign_accuracy"] * 100.0, marker="o", markersize=3)
    mark_transition(axis, args.transition_frame)
    axis.set_ylim(0, 105)
    axis.set_title("XYZ sign accuracy by input frame")
    axis.set_xlabel("Input frame")
    axis.set_ylabel("Correct signs (%)")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    axis.step(frames, df["gt_gripper_first"], where="mid", linewidth=2.2,
              label="Ground truth first action")
    axis.plot(frames, df["pred_gripper_first"], marker="o", markersize=4,
              label="Predicted first action")
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1.2,
                 label="Open/close threshold")
    mark_transition(axis, args.transition_frame)
    axis.set_ylim(-0.15, 1.15)
    axis.set_title("First-step gripper command")
    axis.set_xlabel("Input frame")
    axis.set_ylabel("Gripper action (1=open, 0=close)")
    axis.legend(fontsize=8)

    diagnostics_path = output_dir / "fixed_episode_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_path = csv_path.with_suffix(".summary.json")
    translation_path = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        per_axis = summary["per_axis_mae"]
        labels = ["X", "Y", "Z"]
        values = np.array([per_axis["x"], per_axis["y"], per_axis["z"]]) * 1000.0
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        bars = axis.bar(labels, values)
        axis.set_title("Translation MAE by axis")
        axis.set_xlabel("Action dimension")
        axis.set_ylabel("Mean absolute error (mm)")
        axis.grid(True, axis="y", alpha=0.25)
        axis.bar_label(bars, labels=[f"{value:.3f} mm" for value in values], padding=3)
        translation_path = output_dir / "translation_axis_mae_mm.png"
        fig.savefig(translation_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Charts saved to: {output_dir}")
    print(f"  {diagnostics_path}")
    if translation_path is not None:
        print(f"  {translation_path}")


if __name__ == "__main__":
    main()

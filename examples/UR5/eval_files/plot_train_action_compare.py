#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    summary_path = csv_path.with_suffix(".summary.json")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    summary = json.loads(summary_path.read_text())

    # 每個樣本的評估結果
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(
        df["sample"],
        df["xyz_mae_normalized"],
        marker="o",
        markersize=3,
    )
    axes[0, 0].set_title("XYZ MAE per Sample")
    axes[0, 0].set_xlabel("Training Sample")
    axes[0, 0].set_ylabel("Normalized MAE")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(
        df["sample"],
        df["xyz_cosine"],
        marker="o",
        markersize=3,
    )
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_ylim(-1.05, 1.05)
    axes[0, 1].set_title("XYZ Direction Cosine Similarity")
    axes[0, 1].set_xlabel("Training Sample")
    axes[0, 1].set_ylabel("Cosine similarity")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(
        df["sample"],
        df["xyz_sign_accuracy"],
        marker="o",
        markersize=3,
    )
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("XYZ Sign Accuracy")
    axes[1, 0].set_xlabel("Training Sample")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(
        df["sample"],
        df["gripper_mae_normalized"],
        marker="o",
        markersize=3,
    )
    axes[1, 1].set_title("Gripper MAE per Sample")
    axes[1, 1].set_xlabel("Training Sample")
    axes[1, 1].set_ylabel("Normalized MAE")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "per_sample_metrics.png",
        dpi=200,
    )
    plt.close(fig)

    # 各 action 軸 MAE
    per_axis = summary["per_axis_mae_normalized"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        per_axis.keys(),
        per_axis.values(),
    )
    ax.bar_label(bars, fmt="%.4f")
    ax.set_title("Per-axis Action MAE")
    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("Normalized MAE")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "per_axis_mae.png",
        dpi=200,
    )
    plt.close(fig)

    print(f"Charts saved to: {output_dir}")
    print(f"  {output_dir / 'per_sample_metrics.png'}")
    print(f"  {output_dir / 'per_axis_mae.png'}")


if __name__ == "__main__":
    main()

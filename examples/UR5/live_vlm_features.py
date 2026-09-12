#!/usr/bin/env python3

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_grid_size(token_count):
    side = int(round(math.sqrt(token_count)))

    if side * side == token_count:
        return side, side

    for height in range(
        int(math.sqrt(token_count)),
        0,
        -1,
    ):
        if token_count % height == 0:
            return height, token_count // height

    return 1, token_count


def load_tokens(path):
    with np.load(path, allow_pickle=False) as data:
        tokens = np.asarray(
            data["visual_tokens"],
            dtype=np.float32,
        )

    if tokens.ndim != 2:
        raise ValueError(
            f"Expected [N,D], received {tokens.shape}"
        )

    return tokens


def calculate_pca(tokens):
    centered = tokens - tokens.mean(
        axis=0,
        keepdims=True,
    )

    _, _, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    return centered @ vt[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/tmp/starvla_vlm_live/latest.npz"
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()
    input_path = args.input.expanduser()

    print("Waiting for:", input_path)

    while not input_path.is_file():
        time.sleep(args.interval)

    plt.ion()

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19, 5.5),
    )

    last_timestamp = None
    previous_pca = None
    frame_index = 0

    while plt.fignum_exists(figure.number):
        try:
            timestamp = input_path.stat().st_mtime_ns
        except FileNotFoundError:
            plt.pause(args.interval)
            continue

        if timestamp == last_timestamp:
            plt.pause(args.interval)
            continue

        try:
            tokens = load_tokens(input_path)
        except (
            OSError,
            EOFError,
            ValueError,
        ) as error:
            print("Retry:", error)
            plt.pause(args.interval)
            continue

        last_timestamp = timestamp
        frame_index += 1

        token_count, hidden_dim = tokens.shape
        grid_h, grid_w = get_grid_size(token_count)

        feature_norm = np.linalg.norm(
            tokens,
            axis=1,
        ).reshape(grid_h, grid_w)

        pca = calculate_pca(tokens)

        # 避免PCA正負號在相鄰影格間隨機翻轉。
        if (
            previous_pca is not None
            and previous_pca.shape == pca.shape
            and np.dot(previous_pca, pca) < 0
        ):
            pca = -pca

        previous_pca = pca.copy()
        pca_grid = pca.reshape(grid_h, grid_w)

        lengths = np.linalg.norm(
            tokens,
            axis=1,
            keepdims=True,
        )

        normalized = tokens / np.maximum(
            lengths,
            1e-8,
        )

        cosine = normalized @ normalized.T

        for axis in axes:
            axis.clear()

        axes[0].imshow(
            feature_norm,
            cmap="viridis",
            interpolation="nearest",
        )
        axes[0].set_title("Feature norm")
        axes[0].set_xlabel("Token-grid X")
        axes[0].set_ylabel("Token-grid Y")

        axes[1].imshow(
            pca_grid,
            cmap="coolwarm",
            interpolation="nearest",
        )
        axes[1].set_title("PCA component 1")
        axes[1].set_xlabel("Token-grid X")
        axes[1].set_ylabel("Token-grid Y")

        axes[2].imshow(
            cosine,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[2].set_title("Token cosine similarity")
        axes[2].set_xlabel("Token index")
        axes[2].set_ylabel("Token index")

        figure.suptitle(
            f"QwenGR00T live VLM features | "
            f"frame={frame_index} | "
            f"tokens={token_count} | "
            f"hidden={hidden_dim}"
        )

        figure.tight_layout()
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        plt.pause(0.001)


if __name__ == "__main__":
    main()

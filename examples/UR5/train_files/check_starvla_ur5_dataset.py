#!/usr/bin/env python3
"""Fail-fast checks for the converted StarVLA UR5 datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def check_dataset(path: Path, expected_episodes: int) -> None:
    info = read_json(path / "meta/info.json")
    modality = read_json(path / "meta/modality.json")
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == expected_episodes
    assert count_jsonl(path / "meta/episodes.jsonl") == expected_episodes
    assert list(modality["video"]) == [
        "primary_image",
        "primary_depth",
    ]
    assert list(modality["state"]) == [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "gripper",
        "rel_target_x",
        "rel_target_y",
        "rel_target_z",
    ]
    assert list(modality["action"]) == ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    parquet_paths = sorted(
        path.glob(
            "data/chunk-*/episode_*.parquet"
        )
    )

    primary_video_paths = sorted(
        path.glob(
            "videos/chunk-*/"
            "observation.images.primary_image/"
            "episode_*.mp4"
        )
    )

    primary_depth_video_paths = sorted(
        path.glob(
            "videos/chunk-*/"
            "observation.images.primary_depth/"
            "episode_*.mp4"
        )
    )

    # wrist_video_paths = sorted(
    #     path.glob(
    #         "videos/chunk-*/"
    #         "observation.images.wrist_image/"
    #         "episode_*.mp4"
    #     )
    # )

    assert len(parquet_paths) == expected_episodes

    assert (
        len(primary_video_paths)
        == expected_episodes
    )

    assert (
        len(primary_depth_video_paths)
        == expected_episodes
    )

    # assert (
    #     len(wrist_video_paths)
    #     == expected_episodes
    # )
    assert "observation.images.primary_image" in info["features"]
    assert "observation.images.primary_depth" in info["features"]
    # assert "observation.images.wrist_image" in info["features"]
    total_frames = 0
    action_parts = []
    state_parts = []
    geometry_parts = []
    for parquet_path in parquet_paths:
        table = pq.read_table(
            parquet_path,
            columns=[
                "action",
                "observation.state",
                "frame_index",
                "task_index",
            ]
        )
        
        action = np.asarray(
            table["action"].to_pylist(),
            dtype=np.float32,
        )

        state = np.asarray(
            table["observation.state"].to_pylist(),
            dtype=np.float32,
        )

        assert action.ndim == 2
        assert action.shape[1] == 7

        assert state.ndim == 2
        assert state.shape[1] == 10

        assert np.isfinite(action).all()
        assert np.isfinite(state).all()

        eef_state = state[:, :7]
        explicit_geometry = state[:, 7:10]

        assert eef_state.shape[1] == 7
        assert explicit_geometry.shape[1] == 3
        assert np.isfinite(explicit_geometry).all()

        assert np.isin(action[:, 6], [0.0, 1.0]).all()
        assert np.isin(eef_state[:, 6], [0.0, 1.0]).all()
        total_frames += len(table)
        action_parts.append(action)
        state_parts.append(state)
    assert total_frames == info["total_frames"]
    action = np.concatenate(action_parts, axis=0)
    state = np.concatenate(state_parts, axis=0)
    motion = action[:, :6]
    print(f"[OK] {path.name}: episodes={expected_episodes}, frames={total_frames}")
    print(f"     motion min={motion.min(axis=0)}")
    print(f"     motion max={motion.max(axis=0)}")
    print(f"     gripper open ratio={action[:, 6].mean():.4f}")
    print(f"     state position min={state[:, :3].min(axis=0)}")
    print(f"     state position max={state[:, :3].max(axis=0)}")
    print(f"     state gripper open ratio={state[:, 6].mean():.4f}")
    
    explicit_geometry = state[:, 7:10]

    print(
        "     explicit geometry min=",
        explicit_geometry.min(axis=0),
    )

    print(
        "     explicit geometry max=",
        explicit_geometry.max(axis=0),
    )

    print(
        "     explicit geometry mean=",
        explicit_geometry.mean(axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=(
            Path.home()
            / "ros_ur_driver"
            / "dataset_VLA_sim"
            / "starvla_ur5_front_rgbd_pick_600_geometry_v8"
        ),
    )
    args = parser.parse_args()
    expected = {
        "starvla_ur5_train": 480,
        "starvla_ur5_val": 60,
        "starvla_ur5_test": 60,
    }
    for name, count in expected.items():
        check_dataset(args.root.expanduser().resolve() / name, count)
    print("[PASS] Dataset structure, dimensions, finite values, and split counts are valid.")


if __name__ == "__main__":
    main()

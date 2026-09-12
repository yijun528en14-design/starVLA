#!/usr/bin/env python3
"""Compare deployed StarVLA predictions with GT for exact LeRobot frames."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=30)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6678)
    p.add_argument("--instruction", default="Pick up the plug and insert it into the hole.")
    p.add_argument("--unnorm-key", default="new_embodiment")
    p.add_argument("--num-ddim-steps", type=int, default=20)
    p.add_argument("--image-width", type=int, default=224)
    p.add_argument("--image-height", type=int, default=224)
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def find_episode_file(root: Path, kind: str, episode: int, suffix: str) -> Path:
    matches = sorted((root / kind).rglob(f"episode_{episode:06d}.{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {kind} file for episode {episode}, found {matches}"
        )
    return matches[0]


def padded_chunk(actions: np.ndarray, index: int, horizon: int) -> np.ndarray:
    chunk = actions[index : index + horizon]
    if len(chunk) < horizon:
        chunk = np.concatenate(
            [chunk, np.repeat(chunk[-1][None], horizon - len(chunk), axis=0)], axis=0
        )
    return chunk


def xyz_cosine(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_xyz, gt_xyz = pred[:, :3], gt[:, :3]
    denom = np.linalg.norm(pred_xyz, axis=1) * np.linalg.norm(gt_xyz, axis=1)
    valid = denom > 1e-10
    if not np.any(valid):
        return float("nan")
    values = np.sum(pred_xyz[valid] * gt_xyz[valid], axis=1) / denom[valid]
    return float(np.mean(values))


def main():
    args = parse_args()
    root = Path(args.dataset).expanduser().resolve()
    repo = Path(os.environ.get("STARVLA_ROOT", "~/vla_workspace/starVLA")).expanduser()
    sys.path.insert(0, str(repo))
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    parquet = find_episode_file(root, "data", args.episode, "parquet")
    video = find_episode_file(root, "videos", args.episode, "mp4")
    table = pd.read_parquet(parquet).sort_values("frame_index").reset_index(drop=True)
    frames = table["frame_index"].to_numpy(dtype=np.int64)
    actions = np.stack([np.asarray(x, dtype=np.float32) for x in table["action"]])
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions [N,7], got {actions.shape}")

    selected = np.flatnonzero(
        (frames >= args.start_frame) & (frames <= args.end_frame)
    )
    if not len(selected):
        raise ValueError(f"No frames in range {args.start_frame}..{args.end_frame}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    print("Server metadata:", client.get_server_metadata())
    print(f"Episode {args.episode}, frames {frames[selected[0]]}..{frames[selected[-1]]}")

    rows, predictions, targets = [], [], []
    for sequence_index in selected:
        frame = int(frames[sequence_index])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read video frame {frame}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb, (args.image_width, args.image_height), interpolation=cv2.INTER_AREA
        )
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        request = {
            "examples": [{"image": [rgb], "lang": args.instruction}],
            "do_sample": False,
            "use_ddim": True,
            "num_ddim_steps": args.num_ddim_steps,
            "unnorm_key": args.unnorm_key,
        }
        response = client.predict_action(request)
        pred = np.asarray(response["data"]["actions"][0], dtype=np.float32)
        gt = padded_chunk(actions, sequence_index, args.horizon)
        if pred.shape != gt.shape:
            raise ValueError(f"frame {frame}: prediction {pred.shape} != GT {gt.shape}")

        sign_mask = np.abs(gt[:, :3]) > 1e-5
        sign_accuracy = (
            float(np.mean(np.sign(pred[:, :3][sign_mask]) == np.sign(gt[:, :3][sign_mask])))
            if np.any(sign_mask)
            else float("nan")
        )
        row = {
            "episode": args.episode,
            "frame": frame,
            "xyz_mae_m": float(np.mean(np.abs(pred[:, :3] - gt[:, :3]))),
            "xyz_cosine": xyz_cosine(pred, gt),
            "xyz_sign_accuracy": sign_accuracy,
            "gripper_mae": float(np.mean(np.abs(pred[:, 6] - gt[:, 6]))),
            "gt_gripper_first": float(gt[0, 6]),
            "pred_gripper_first": float(pred[0, 6]),
        }
        rows.append(row)
        predictions.append(pred)
        targets.append(gt)
        print(json.dumps(row, ensure_ascii=False))

    capture.release()
    predictions = np.stack(predictions)
    targets = np.stack(targets)
    axis_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    axis_mae = np.mean(np.abs(predictions - targets), axis=(0, 1))
    summary = {
        "dataset": str(root),
        "episode": args.episode,
        "frame_range": [int(frames[selected[0]]), int(frames[selected[-1]])],
        "num_frames": len(rows),
        "instruction": args.instruction,
        "unnorm_key": args.unnorm_key,
        "comparison_space": "server output / raw dataset action units",
        "xyz_mae_m": float(np.mean(np.abs(predictions[..., :3] - targets[..., :3]))),
        "xyz_cosine": xyz_cosine(predictions.reshape(-1, 7), targets.reshape(-1, 7)),
        "per_axis_mae": dict(zip(axis_names, map(float, axis_mae))),
        "gripper_classification_accuracy": float(
            np.mean((predictions[..., 6] >= 0.5) == (targets[..., 6] >= 0.5))
        ),
    }
    output = Path(args.output_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("CSV:", output)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()

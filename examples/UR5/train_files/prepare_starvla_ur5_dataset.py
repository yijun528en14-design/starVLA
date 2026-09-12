#!/usr/bin/env python3
"""Convert UR5 Gazebo demonstrations to StarVLA's LeRobot-v2.1 layout.

Model contract produced by this converter:
  input : one front RGB frame + one full natural-language instruction
          + current robot state [x, y, z, Rx, Ry, Rz, gripper]
  target: [dx, dy, dz, dRx, dRy, dRz, gripper] in base_link coordinates

Rotation targets are relative rotation vectors in radians.  Gripper is binary:
1=open, 0=closed.  Episodes are kept intact; phase/subtask/force/depth/ground
truth fields are deliberately not copied into the training features.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
STATE_NAMES = [
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
CHUNKS_SIZE = 1000


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_quaternion_xyzw(q: Any) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"invalid quaternion: {value.tolist()}")
    return value / norm


def quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def relative_rotvec(q_now_xyzw: Any, q_next_xyzw: Any) -> np.ndarray:
    q_now = normalize_quaternion_xyzw(q_now_xyzw)
    q_next = normalize_quaternion_xyzw(q_next_xyzw)
    q_conjugate = np.asarray([-q_now[0], -q_now[1], -q_now[2], q_now[3]])
    relative = normalize_quaternion_xyzw(quaternion_multiply_xyzw(q_conjugate, q_next))
    if relative[3] < 0.0:
        relative = -relative
    vector_norm = float(np.linalg.norm(relative[:3]))
    if vector_norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(relative[3], -1.0, 1.0)))
    return (relative[:3] / vector_norm * angle).astype(np.float32)


def absolute_rotvec(q_xyzw: Any) -> np.ndarray:
    """Represent the current tool orientation as an identity-relative rotvec."""
    return relative_rotvec(
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        q_xyzw,
    )


def pose_from_step(step: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pose = step["tool0_pose"]
    position_value = pose["position"]
    orientation_value = pose.get("quaternion", pose.get("orientation"))
    if isinstance(position_value, dict):
        position = [position_value[key] for key in ("x", "y", "z")]
    else:
        position = position_value
    if isinstance(orientation_value, dict):
        quaternion = [orientation_value[key] for key in ("x", "y", "z", "w")]
    else:
        quaternion = orientation_value
    position_array = np.asarray(position, dtype=np.float64).reshape(3)
    if not np.isfinite(position_array).all():
        raise ValueError("non-finite tool position")
    return position_array, normalize_quaternion_xyzw(quaternion)


def gripper_from_step(step: dict[str, Any], threshold: float) -> float:
    command = step.get("gripper_command", {}).get("positions")
    if command:
        return 1.0 if float(command[0]) > threshold else 0.0
    state = step.get("gripper_state", {}).get("position")
    if isinstance(state, list):
        state = state[0]
    if state is None:
        raise KeyError("missing gripper_command.positions and gripper_state.position")
    return 1.0 if float(state) > threshold else 0.0

# ============================================================
# V4E: Pick-only phase-aware geometry target
# ============================================================

BAD_PHASE_PREFIXES = (
    "reset_",
    "failure_",
    "setup_",
    "pre_episode_",
)


def xyz_target(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise KeyError(f"missing target: {name}")

    if isinstance(value, dict):
        value = [
            value["x"],
            value["y"],
            value["z"],
        ]

    target = np.asarray(
        value,
        dtype=np.float32,
    ).reshape(3)

    if not np.isfinite(target).all():
        raise ValueError(
            f"invalid {name}: {target}"
        )

    return target


def pick_geometry_target(
    step: dict[str, Any],
    episode_info: dict[str, Any],
) -> np.ndarray:
    """
    Return the current phase-specific Cartesian geometry target.

    approach / align:
        target = plug_above_target

    descend / close / hold:
        target = plug_grasp_target

    lift:
        target = plug_above_target
    """

    phase = str(
        step.get("phase", "")
    ).strip().lower()

    if phase in (
        "approach_plug",
        "align_above_plug",
        "lift_plug",
    ):
        target_value = episode_info.get(
            "plug_above_target"
        )

        if target_value is None:
            raise KeyError(
                "missing plug_above_target"
            )

        return xyz_target(
            target_value,
            "plug_above_target",
        )

    if phase in (
        "descend_to_grasp",
        "close_gripper",
        "hold_plug",
    ):
        target_value = episode_info.get(
            "plug_grasp_target"
        )

        if target_value is None:
            target_value = episode_info.get(
                "plug_position"
            )

        return xyz_target(
            target_value,
            "plug_grasp_target",
        )

    raise ValueError(
        f"Unsupported Pick phase for geometry target: {phase}"
    )

DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 1.20
VISIBLE_BIAS_X = 0.006990
VISIBLE_BIAS_Y = 0.003296
PLUG_TO_GRASP_Y = 0.004000
GRASP_OFFSET_Z = 0.254188
PREGRASP_EXTRA_Z = 0.055000
ANCHOR_JUMP_THRESHOLD_M = 0.010

PREGRASP_RAW_PHASES = {
    "approach_plug",
    "align_above_plug",
}

def raw_phase_to_target_phase(raw_phase: str) -> str:
    raw_phase = str(raw_phase).strip().lower()

    if raw_phase in (
        "approach_plug",
        "align_above_plug",
        "lift_plug",
    ):
        return "before_grasp"

    if raw_phase in (
        "descend_to_grasp",
        "close_gripper",
        "hold_plug",
    ):
        return "grasp"

    raise ValueError(
        f"Unsupported raw phase: {raw_phase}"
    )

def visible_to_task_target(
    visible_xyz: np.ndarray,
    phase: str,
) -> np.ndarray:

    plug_x = visible_xyz[0] - VISIBLE_BIAS_X
    plug_y = visible_xyz[1] - VISIBLE_BIAS_Y

    target_x = plug_x
    target_y = plug_y + PLUG_TO_GRASP_Y
    target_z = visible_xyz[2] + GRASP_OFFSET_Z

    if phase == "before_grasp":
        target_z += PREGRASP_EXTRA_Z

    return np.asarray(
        [target_x, target_y, target_z],
        dtype=np.float32,
    )

def backproject_pixel(
    u,
    v,
    depth_m,
    fx,
    fy,
    cx,
    cy,
):
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m

    return np.asarray(
        [x, y, z],
        dtype=np.float64,
    )

def optical_to_camera_link(p_optical):
    x, y, z = p_optical

    return np.asarray(
        [z, -x, -y],
        dtype=np.float64,
    )

def transform_point(T, p):
    p_h = np.concatenate([p, [1.0]])
    return (T @ p_h)[:3]

def robust_bbox_visible_xyz(
    bbox,
    depth_path: Path,
    camera_calibration,
) -> np.ndarray:

    depth_mm = cv2.imread(
        str(depth_path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth_mm is None:
        raise FileNotFoundError(
            depth_path
        )

    if depth_mm.ndim != 2:
        raise ValueError(
            f"Expected depth HxW, "
            f"got {depth_mm.shape}"
        )

    depth_m = (
        depth_mm.astype(np.float64)
        / 1000.0
    )

    h, w = depth_m.shape

    x1, y1, x2, y2 = [
        float(v)
        for v in bbox
    ]

    # Qwen bbox: normalized 0..1000
    x1 = int(
        np.clip(
            round(x1 / 1000.0 * w),
            0,
            w - 1,
        )
    )

    x2 = int(
        np.clip(
            round(x2 / 1000.0 * w),
            0,
            w - 1,
        )
    )

    y1 = int(
        np.clip(
            round(y1 / 1000.0 * h),
            0,
            h - 1,
        )
    )

    y2 = int(
        np.clip(
            round(y2 / 1000.0 * h),
            0,
            h - 1,
        )
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bbox: {bbox}"
        )

    # central 50% ROI
    bw = x2 - x1
    bh = y2 - y1

    rx1 = int(
        round(x1 + 0.25 * bw)
    )
    rx2 = int(
        round(x2 - 0.25 * bw)
    )

    ry1 = int(
        round(y1 + 0.25 * bh)
    )
    ry2 = int(
        round(y2 - 0.25 * bh)
    )

    fx = camera_calibration["fx"]
    fy = camera_calibration["fy"]
    cx = camera_calibration["cx"]
    cy = camera_calibration["cy"]

    T = camera_calibration[
        "T_base_camera_link"
    ]


    def collect_points(
        px1,
        py1,
        px2,
        py2,
    ):

        collected = []

        for v in range(py1, py2):
            for u in range(px1, px2):

                z = depth_m[v, u]

                if (
                    not np.isfinite(z)
                    or z <= 0.0
                ):
                    continue

                p_optical = backproject_pixel(
                    u,
                    v,
                    z,
                    fx,
                    fy,
                    cx,
                    cy,
                )

                p_camera_link = (
                    optical_to_camera_link(
                        p_optical
                    )
                )

                p_base = transform_point(
                    T,
                    p_camera_link,
                )

                # remove table plane
                if p_base[2] <= 0.103:
                    continue

                collected.append(
                    p_base
                )

        return collected


    # ------------------------------------------------------------
    # Pass 1:
    # use central 50% ROI first
    # ------------------------------------------------------------

    points_base = collect_points(
        rx1,
        ry1,
        rx2,
        ry2,
    )

    # ------------------------------------------------------------
    # Pass 2:
    # if central ROI contains no valid object 3D points,
    # fall back to the full Qwen bbox
    # ------------------------------------------------------------

    if not points_base:
        points_base = collect_points(
            x1,
            y1,
            x2,
            y2,
        )


    # ------------------------------------------------------------
    # Still empty:
    # the full bbox itself has no valid object 3D points
    # ------------------------------------------------------------

    if not points_base:
        raise ValueError(
            f"No valid 3D points "
            f"inside bbox {bbox}"
        )

    points = np.asarray(
        points_base,
        dtype=np.float64,
    )

    center = np.median(
        points,
        axis=0,
    )

    distances = np.linalg.norm(
        points - center,
        axis=1,
    )

    distance_median = np.median(
        distances
    )

    mad = np.median(
        np.abs(
            distances
            - distance_median
        )
    )

    if mad > 1e-9:
        keep = (
            np.abs(
                distances
                - distance_median
            )
            <= 3.5 * mad
        )

        points = points[keep]

    if len(points) == 0:
        raise ValueError(
            "All bbox depth points "
            "were rejected by MAD"
        )

    visible_xyz = np.median(
        points,
        axis=0,
    )

    return visible_xyz.astype(
        np.float32
    )

def depth_to_gray_rgb(depth_path: Path) -> np.ndarray:
    depth_mm = cv2.imread(
        str(depth_path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth_mm is None:
        raise FileNotFoundError(depth_path)

    if depth_mm.ndim != 2:
        raise ValueError(
            f"Expected single-channel depth: "
            f"{depth_path}, shape={depth_mm.shape}"
        )

    depth_m = depth_mm.astype(np.float32) / 1000.0

    valid = (
        np.isfinite(depth_m)
        & (depth_m > 0.0)
    )

    clipped = np.clip(
        depth_m,
        DEPTH_MIN_M,
        DEPTH_MAX_M,
    )

    normalized = 1.0 - (
        (clipped - DEPTH_MIN_M)
        / (DEPTH_MAX_M - DEPTH_MIN_M)
    )

    gray = np.clip(
        normalized * 255.0,
        0,
        255,
    ).astype(np.uint8)

    gray[~valid] = 0

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2RGB,
    )


def make_rgbd_composite(
    rgb_path: Path,
    depth_path: Path,
) -> np.ndarray:

    rgb_bgr = cv2.imread(
        str(rgb_path),
        cv2.IMREAD_COLOR,
    )

    if rgb_bgr is None:
        raise FileNotFoundError(rgb_path)

    rgb = cv2.cvtColor(
        rgb_bgr,
        cv2.COLOR_BGR2RGB,
    )

    depth_rgb = depth_to_gray_rgb(
        depth_path
    )

    if depth_rgb.shape[:2] != rgb.shape[:2]:
        depth_rgb = cv2.resize(
            depth_rgb,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return np.hstack(
        [rgb, depth_rgb]
    )


def image_paths_for_step(
    episode_dir: Path,
    step: dict[str, Any],
    frame_index: int,
) -> tuple[Path, Path]:

    images = step.get("images", {})

    primary_rgb_relative = images.get(
        "static_rgb",
        f"camera/{frame_index:06d}.png",
    )

    primary_depth_relative = images.get(
        "static_depth"
    )

    if primary_depth_relative is None:
        raise KeyError(
            f"Missing images.static_depth "
            f"at frame {frame_index}. "
            f"Available keys: {list(images.keys())}"
        )

    primary_rgb_path = (
        episode_dir / primary_rgb_relative
    )

    primary_depth_path = (
        episode_dir / primary_depth_relative
    )

    if not primary_rgb_path.is_file():
        raise FileNotFoundError(
            primary_rgb_path
        )

    if not primary_depth_path.is_file():
        raise FileNotFoundError(
            primary_depth_path
        )

    return (
        primary_rgb_path,
        primary_depth_path,
    )


def encode_rgb_video(
    rgb_paths: list[Path],
    output_path: Path,
    fps: int,
) -> tuple[int, int]:

    if not rgb_paths:
        raise ValueError(
            "empty RGB frame list"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_bgr = cv2.imread(
        str(rgb_paths[0]),
        cv2.IMREAD_COLOR,
    )

    if first_bgr is None:
        raise FileNotFoundError(
            rgb_paths[0]
        )

    first_rgb = cv2.cvtColor(
        first_bgr,
        cv2.COLOR_BGR2RGB,
    )

    height, width = first_rgb.shape[:2]

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",

        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",

        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",

        str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
    )

    try:
        for rgb_path in rgb_paths:

            frame_bgr = cv2.imread(
                str(rgb_path),
                cv2.IMREAD_COLOR,
            )

            if frame_bgr is None:
                raise FileNotFoundError(
                    rgb_path
                )

            frame = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

            if frame.shape[:2] != (
                height,
                width,
            ):
                raise ValueError(
                    f"Inconsistent RGB frame size: "
                    f"{frame.shape[:2]} != "
                    f"{(height, width)}"
                )

            process.stdin.write(
                np.ascontiguousarray(
                    frame,
                    dtype=np.uint8,
                ).tobytes()
            )

        process.stdin.close()
        process.stdin = None

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with code "
                f"{return_code}: "
                f"{output_path}"
            )

    except Exception:
        if process.stdin is not None:
            process.stdin.close()

        process.kill()
        process.wait()
        raise

    return height, width

def encode_depth_video(
    depth_paths: list[Path],
    output_path: Path,
    fps: int,
) -> tuple[int, int]:

    if not depth_paths:
        raise ValueError(
            "empty depth frame list"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_frame = depth_to_gray_rgb(
        depth_paths[0]
    )

    height, width = first_frame.shape[:2]

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",

        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",

        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",

        str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
    )

    try:
        for depth_path in depth_paths:

            frame = depth_to_gray_rgb(
                depth_path
            )

            if frame.shape[:2] != (
                height,
                width,
            ):
                raise ValueError(
                    f"Inconsistent depth frame size: "
                    f"{frame.shape[:2]} != "
                    f"{(height, width)}"
                )

            process.stdin.write(
                np.ascontiguousarray(
                    frame,
                    dtype=np.uint8,
                ).tobytes()
            )

        process.stdin.close()
        process.stdin = None

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with code "
                f"{return_code}: "
                f"{output_path}"
            )

    except Exception:
        if process.stdin is not None:
            process.stdin.close()

        process.kill()
        process.wait()
        raise

    return height, width


def load_manifest(subset_root: Path, split_name: str) -> list[Path]:
    manifest = subset_root / "splits" / f"{split_name}.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing split manifest: {manifest}")
    episodes: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        entry = raw_line.strip()
        if not entry:
            continue
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = subset_root / candidate.name
        if not candidate.is_dir():
            raise FileNotFoundError(f"manifest episode does not exist: {candidate}")
        episodes.append(candidate)
    if not episodes:
        raise ValueError(f"empty manifest: {manifest}")
    return episodes

def _filter_pick_steps_for_v7(
    episode_dir: Path,
) -> list[dict[str, Any]]:
    steps = read_jsonl(
        episode_dir / "steps.jsonl"
    )

    filtered_steps = []

    for step in steps:
        subtask = str(
            step.get("subtask", "")
        ).strip().lower()

        phase = str(
            step.get("phase", "")
        ).strip().lower()

        if subtask != "pick":
            continue

        if phase.startswith(
            BAD_PHASE_PREFIXES
        ):
            continue

        filtered_steps.append(step)

    if not filtered_steps:
        raise ValueError(
            f"{episode_dir}: no valid Pick frames"
        )

    return filtered_steps[::2]

def _load_qwen_grounding_model(
    config_yaml: Path,
):
    import torch
    from omegaconf import OmegaConf
    from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T

    config_yaml = (
        config_yaml
        .expanduser()
        .resolve()
    )

    cfg = OmegaConf.load(
        str(config_yaml)
    )

    model = Qwen_GR00T(cfg)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    return model

def _validate_grounding_result(
    grounding: dict[str, Any],
    cache_key: str,
) -> dict[str, Any]:

    if not isinstance(
        grounding,
        dict,
    ):
        raise ValueError(
            f"Invalid grounding result: {cache_key}"
        )

    bbox = grounding.get(
        "primary_bbox"
    )

    phase = str(
        grounding.get(
            "phase",
            "",
        )
    ).strip().lower()

    if bbox is None:
        raise ValueError(
            f"Missing bbox: {cache_key}"
        )

    bbox_array = np.asarray(
        bbox,
        dtype=np.float64,
    ).reshape(-1)

    if bbox_array.size != 4:
        raise ValueError(
            f"Invalid bbox: {bbox}"
        )

    if not np.isfinite(bbox_array).all():
        raise ValueError(
            f"Non-finite bbox {bbox}: {cache_key}"
        )

    if (
        np.any(bbox_array < 0.0)
        or np.any(bbox_array > 1000.0)
    ):
        raise ValueError(
            f"Bbox outside 0..1000 "
            f"{bbox}: {cache_key}"
        )

    x1, y1, x2, y2 = bbox_array.tolist()

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Degenerate bbox "
            f"{bbox}: {cache_key}"
        )

    if phase not in (
        "before_grasp",
        "grasp",
    ):
        raise ValueError(
            f"Invalid phase {phase!r}: {cache_key}"
        )

    return {
        "target": str(
            grounding.get(
                "target",
                "unknown",
            )
        ),
        "phase": phase,
        "primary_bbox": [
            float(v)
            for v in bbox_array.tolist()
        ],
        "raw_output": str(
            grounding.get(
                "raw_output",
                "",
            )
        ),
    }

def build_or_update_grounding_cache(
    manifests,
    cache_path: Path,
    config_yaml: Path,
    instruction: str,
):

    cache_path = (
        cache_path
        .expanduser()
        .resolve()
    )

    if cache_path.is_file():
        grounding_cache = read_json(
            cache_path
        )
    else:
        grounding_cache = {}

    pending = []

    for split_name in (
        "train",
        "val",
        "test",
    ):
        for episode_dir in manifests[
            split_name
        ]:
            retained_steps = (
                _filter_pick_steps_for_v7(
                    episode_dir
                )
            )

            for retained_index, step in enumerate(
                retained_steps
            ):
                (
                    primary_rgb_path,
                    primary_depth_path,
                ) = image_paths_for_step(
                    episode_dir,
                    step,
                    retained_index,
                )

                images = step.get(
                    "images",
                    {},
                )

                primary_rgb_relative = (
                    images.get(
                        "static_rgb"
                    )
                )

                if primary_rgb_relative is None:
                    raise KeyError(
                        f"{episode_dir.name}: "
                        f"missing static_rgb"
                    )

                cache_key = (
                    f"{episode_dir.name}/"
                    f"{primary_rgb_relative}"
                )

                if cache_key in grounding_cache:
                    continue

                pending.append(
                    (
                        cache_key,
                        primary_rgb_path,
                        primary_depth_path,
                    )
                )

    if not pending:
        return grounding_cache

    model = _load_qwen_grounding_model(
        config_yaml
    )

    batch_size = 4

    total_pending = len(
        pending
    )

    for batch_start in range(
        0,
        total_pending,
        batch_size,
    ):
        batch_items = pending[
            batch_start:
            batch_start + batch_size
        ]

        examples = []
        cache_keys = []

        for (
            cache_key,
            primary_rgb_path,
            primary_depth_path,
        ) in batch_items:

            rgb_image = Image.open(
                primary_rgb_path
            ).convert("RGB")

            depth_image = Image.fromarray(
                depth_to_gray_rgb(
                    primary_depth_path
                )
            )

            examples.append(
                {
                    "image": [
                        rgb_image,
                        depth_image,
                    ],
                    "lang": instruction,
                }
            )

            cache_keys.append(
                cache_key
            )

        results = model.predict_grounding(
            examples,
            max_new_tokens=64,
        )

        if len(results) != len(cache_keys):
            raise RuntimeError(
                f"Grounding batch size mismatch: "
                f"{len(results)} != "
                f"{len(cache_keys)}"
            )

        for local_index, (
            cache_key,
            result,
        ) in enumerate(
            zip(
                cache_keys,
                results,
            )
        ):
            result = (
                _validate_grounding_result(
                    result,
                    cache_key,
                )
            )

            grounding_cache[
                cache_key
            ] = result

            global_index = (
                batch_start
                + local_index
                + 1
            )

            print(
                f"[GROUNDING] "
                f"{global_index}/"
                f"{total_pending} "
                f"{cache_key}"
            )

        # 每個 batch 做完就存一次
        write_json(
            cache_path,
            grounding_cache,
        )

    return grounding_cache

def inspect_episode(
    episode_dir: Path,
    default_instruction: str,
    gripper_threshold: float,
    camera_calibration,
    grounding_cache,
) -> dict[str, Any]:
    summary_path = episode_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)

        success = summary.get(
            "task_success",
            summary.get("success"),
        )

        if success is False:
            raise ValueError(
                "episode is marked unsuccessful"
            )
    steps = read_jsonl(
        episode_dir / "steps.jsonl"
    )

    # ==========================================================
    # V4E Pick-only dataset
    #
    # 1. Keep Pick subtask only.
    # 2. Remove non-training phases.
    # 3. Temporal downsample raw ~20 Hz -> training ~10 Hz.
    #
    # IMPORTANT:
    # Downsample BEFORE state/action construction so that
    # action(t) is recomputed between retained frames.
    # ==========================================================

    filtered_steps = []

    for step in steps:
        subtask = str(
            step.get("subtask", "")
        ).strip().lower()

        phase = str(
            step.get("phase", "")
        ).strip().lower()

        if subtask != "pick":
            continue

        if phase.startswith(
            BAD_PHASE_PREFIXES
        ):
            continue

        filtered_steps.append(step)

    if not filtered_steps:
        raise ValueError(
            f"{episode_dir}: no valid Pick frames"
        )

    # ----------------------------------------------------------
    # V4E temporal stride
    # raw recorder ~20 Hz
    # training dataset ~10 Hz
    # ----------------------------------------------------------

    steps = filtered_steps[::2]

    episode_info = steps[0].get(
        "episode_info",
        {},
    )

    if not isinstance(episode_info, dict):
        raise ValueError(
            f"{episode_dir}: invalid episode_info"
        )

    if len(steps) < 17:
        raise ValueError(
            f"pick-only has only {len(steps)} frames; "
            "at least 17 are required"
        )

    instruction = (
        "Move the robot end-effector toward the orange plug using small Cartesian "
        "corrections, continuously adjust the motion according to the relative "
        "position between the end-effector and the orange plug, align the gripper "
        "with the orange plug, and grasp it securely."
    )

    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    grippers: list[float] = []
    states: list[list[float]] = []
    primary_rgb_paths: list[Path] = []
    primary_depth_paths: list[Path] = []
    # wrist_rgb_paths: list[Path] = []
    # wrist_depth_paths: list[Path] = []
    pregrasp_anchor = None
    locked_anchor = None
    anchor_locked = False
    for frame_index, step in enumerate(steps):
        position, quaternion = pose_from_step(step)

        positions.append(position)
        quaternions.append(quaternion)
        grippers.append(
            gripper_from_step(
                step,
                gripper_threshold,
            )
        )

        (
            primary_rgb_path,
            primary_depth_path,
        ) = image_paths_for_step(
            episode_dir,
            step,
            frame_index,
        )

        images = step.get("images", {})

        primary_rgb_relative = images.get(
            "static_rgb"
        )

        if primary_rgb_relative is None:
            raise KeyError(
                f"{episode_dir.name}: "
                f"missing images.static_rgb "
                f"at retained frame {frame_index}"
            )

        cache_key = (
            f"{episode_dir.name}/"
            f"{primary_rgb_relative}"
        )

        if cache_key not in grounding_cache:
            raise KeyError(
                f"Missing grounding cache: "
                f"{cache_key}"
            )

        grounding = grounding_cache[
            cache_key
        ]

        bbox = grounding.get(
            "primary_bbox"
        )

        if bbox is None:
            raise ValueError(
                f"Missing bbox for {cache_key}"
            )

        raw_phase = str(
            step.get("phase", "")
        ).strip().lower()

        phase = raw_phase_to_target_phase(
            raw_phase
        )

        raw_visible_xyz = (
            robust_bbox_visible_xyz(
                bbox=bbox,
                depth_path=primary_depth_path,
                camera_calibration=(
                    camera_calibration
                ),
            )
        )

        raw_visible_xyz = np.asarray(
            raw_visible_xyz,
            dtype=np.float32,
        )

        # ==========================================================
        # V8 anchor logic
        #
        # approach / align:
        #   causal 10 mm consistency update
        #
        # first non-pregrasp phase:
        #   lock the last accepted anchor
        #
        # descend / close / hold / lift:
        #   reuse locked anchor
        # ==========================================================

        if raw_phase in PREGRASP_RAW_PHASES and not anchor_locked:

            if pregrasp_anchor is None:
                pregrasp_anchor = raw_visible_xyz.copy()

            else:
                jump_m = float(
                    np.linalg.norm(
                        raw_visible_xyz
                        - pregrasp_anchor
                    )
                )

                if jump_m <= ANCHOR_JUMP_THRESHOLD_M:
                    pregrasp_anchor = (
                        raw_visible_xyz.copy()
                    )

            visible_xyz = (
                pregrasp_anchor.copy()
            )

        else:

            if not anchor_locked:

                if pregrasp_anchor is None:
                    pregrasp_anchor = (
                        raw_visible_xyz.copy()
                    )

                locked_anchor = (
                    pregrasp_anchor.copy()
                )

                anchor_locked = True

            visible_xyz = (
                locked_anchor.copy()
            )

        target_position = (
            visible_to_task_target(
                visible_xyz,
                phase,
            )
        )

        explicit_geo_rel_xyz = (
            target_position
            - position.astype(np.float32)
        )

        state = np.concatenate(
            [
                position.astype(np.float32),
                absolute_rotvec(quaternion),
                [grippers[-1]],
                explicit_geo_rel_xyz.astype(
                    np.float32
                ),
            ]
        ).astype(np.float32)

        if not np.isfinite(state).all():
            raise ValueError(
                f"non-finite state "
                f"at frame {frame_index}"
            )

        states.append(
            state.tolist()
        )

        primary_rgb_paths.append(
            primary_rgb_path
        )

        primary_depth_paths.append(
            primary_depth_path
        )

        # wrist_rgb_paths.append(
        #     wrist_rgb_path
        # )
        # wrist_depth_paths.append(
        #     wrist_depth_path
        # )
    actions: list[list[float]] = []
    for frame_index in range(len(steps)):
        if frame_index + 1 < len(steps):
            translation = (positions[frame_index + 1] - positions[frame_index])
            rotation = relative_rotvec(quaternions[frame_index], quaternions[frame_index + 1])
            gripper = grippers[frame_index + 1]
        else:
            translation = np.zeros(3, dtype=np.float32)
            rotation = np.zeros(3, dtype=np.float32)
            gripper = grippers[frame_index]
        action = np.concatenate([translation, rotation, [gripper]]).astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError(f"non-finite action at frame {frame_index}")
        actions.append(action.tolist())
    return {
        "instruction": instruction,
        "steps": steps,

        "primary_rgb_paths":
            primary_rgb_paths,

        "primary_depth_paths":
            primary_depth_paths,

        # "wrist_rgb_paths":
        #     wrist_rgb_paths,

        # "wrist_depth_paths":
        #     wrist_depth_paths,

        "states": states,
        "actions": actions,
    }


def video_feature(height: int, width: int, fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": fps,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def scalar_feature(dtype: str) -> dict[str, Any]:
    return {"dtype": dtype, "shape": [1], "names": None}


def write_dataset(
    split_name,
    episode_dirs,
    output_dir,
    fps,
    default_instruction,
    gripper_threshold,
    camera_calibration,
    grounding_cache,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"output already exists: {output_dir}\n"
            "Move it aside explicitly before rebuilding; this script will not delete data."
        )
    output_dir.mkdir(parents=True)
    modality = {
        "state": {
            "x": {
                "start": 0,
                "end": 1,
            },
            "y": {
                "start": 1,
                "end": 2,
            },
            "z": {
                "start": 2,
                "end": 3,
            },
            "roll": {
                "start": 3,
                "end": 4,
            },
            "pitch": {
                "start": 4,
                "end": 5,
            },
            "yaw": {
                "start": 5,
                "end": 6,
            },
            "gripper": {
                "start": 6,
                "end": 7,
            },

            "rel_target_x": {
                "start": 7,
                "end": 8,
            },
            "rel_target_y": {
                "start": 8,
                "end": 9,
            },
            "rel_target_z": {
                "start": 9,
                "end": 10,
            },
        },
        "action": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {
            "primary_image": {
                "original_key":
                    "observation.images.primary_image"
            },

            "primary_depth": {
                "original_key":
                    "observation.images.primary_depth"
            },

            # "wrist_image": {
            #     "original_key":
            #         "observation.images.wrist_image"
            # },
        },

        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index"
            }
        },
    }
    write_json(output_dir / "meta" / "modality.json", modality)

    instruction_to_task: dict[str, int] = {}
    episode_metadata: list[dict[str, Any]] = []
    all_actions: list[list[float]] = []
    all_states: list[list[float]] = []
    global_index = 0
    video_hw: tuple[int, int] | None = None

    for episode_index, episode_dir in enumerate(episode_dirs):
        item = inspect_episode(
            episode_dir,
            default_instruction,
            gripper_threshold,
            camera_calibration,
            grounding_cache,
        )
        instruction = item["instruction"]
        if instruction not in instruction_to_task:
            instruction_to_task[instruction] = len(instruction_to_task)
        task_index = instruction_to_task[instruction]
        actions = item["actions"]
        states = item["states"]
        frame_count = len(actions)
        if len(states) != frame_count:
            raise ValueError(
                f"state/action length mismatch: {len(states)} != {frame_count}"
            )
        episode_chunk = episode_index // CHUNKS_SIZE
        chunk_name = f"chunk-{episode_chunk:03d}"
        data_dir = output_dir / "data" / chunk_name
        primary_video_dir = (
            output_dir
            / "videos"
            / chunk_name
            / "observation.images.primary_image"
        )

        primary_depth_video_dir = (
            output_dir
            / "videos"
            / chunk_name
            / "observation.images.primary_depth"
        )

        # wrist_video_dir = (
        #     output_dir
        #     / "videos"
        #     / chunk_name
        #     / "observation.images.wrist_image"
        # )

        data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        primary_video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        primary_depth_video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # wrist_video_dir.mkdir(
        #     parents=True,
        #     exist_ok=True,
        # )

        table = pa.Table.from_arrays(
            [
                pa.array(actions, type=pa.list_(pa.float32(), 7)),
                pa.array(
                    states,
                    type=pa.list_(
                        pa.float32(),
                        10,
                    ),
                ),
                pa.array([frame / fps for frame in range(frame_count)], type=pa.float32()),
                pa.array(range(frame_count), type=pa.int64()),
                pa.array([episode_index] * frame_count, type=pa.int64()),
                pa.array(range(global_index, global_index + frame_count), type=pa.int64()),
                pa.array([task_index] * frame_count, type=pa.int64()),
            ],
            names=[
                "action",
                "observation.state",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ]
        )
        pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet", compression="zstd")
        primary_hw = encode_rgb_video(
            item["primary_rgb_paths"],
            primary_video_dir
            / f"episode_{episode_index:06d}.mp4",
            fps,
        )

        depth_hw = encode_depth_video(
            item["primary_depth_paths"],
            primary_depth_video_dir
            / f"episode_{episode_index:06d}.mp4",
            fps,
        )

        if primary_hw != depth_hw:
            raise ValueError(
                f"RGB/depth size mismatch: "
                f"{primary_hw} != {depth_hw}"
            )

        video_hw = primary_hw

        episode_metadata.append(
            {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": frame_count,
                "source_episode": episode_dir.name,
            }
        )
        all_actions.extend(actions)
        all_states.extend(states)
        global_index += frame_count
        print(f"[{split_name}] {episode_index + 1:03d}/{len(episode_dirs):03d} "
              f"{episode_dir.name}: {frame_count} frames")

    if video_hw is None:
        raise RuntimeError("no episode was converted")
    task_rows = [
        {"task_index": index, "task": instruction}
        for instruction, index in sorted(instruction_to_task.items(), key=lambda pair: pair[1])
    ]
    write_jsonl(output_dir / "meta" / "tasks.jsonl", task_rows)
    write_jsonl(output_dir / "meta" / "episodes.jsonl", episode_metadata)

    action_array = np.asarray(all_actions, dtype=np.float32)
    state_array = np.asarray(all_states, dtype=np.float32)
    stats = {
        "action": {
            "min": action_array.min(axis=0).tolist(),
            "max": action_array.max(axis=0).tolist(),
            "mean": action_array.mean(axis=0).tolist(),
            "std": np.maximum(action_array.std(axis=0), 1e-6).tolist(),
            "count": int(action_array.shape[0]),
        },
        "observation.state": {
            "min": state_array.min(axis=0).tolist(),
            "max": state_array.max(axis=0).tolist(),
            "mean": state_array.mean(axis=0).tolist(),
            "std": np.maximum(state_array.std(axis=0), 1e-6).tolist(),
            "count": int(state_array.shape[0]),
        },
    }
    write_json(output_dir / "meta" / "stats.json", stats)

    height, width = video_hw
    info = {
        "codebase_version": "v2.1",
        "robot_type": "ur5_rg2_starvla",
        "total_episodes": len(episode_metadata),
        "total_frames": global_index,
        "total_tasks": len(task_rows),
        "total_videos": len(episode_metadata) * 2,
        "total_chunks": max(1, math.ceil(len(episode_metadata) / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [7], "names": ACTION_NAMES},
            "observation.state": {
                "dtype": "float32",
                "shape": [10],
                "names": [
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
                ],
            },
            "observation.images.primary_image":
                video_feature(
                    height,
                    width,
                    fps,
                ),

            "observation.images.primary_depth":
                video_feature(
                    height,
                    width,
                    fps,
                ),

            # "observation.images.wrist_image":
            #     video_feature(
            #         height,
            #         width,
            #         fps,
            #     ),
            "timestamp": scalar_feature("float32"),
            "frame_index": scalar_feature("int64"),
            "episode_index": scalar_feature("int64"),
            "index": scalar_feature("int64"),
            "task_index": scalar_feature("int64"),
        },
    }
    write_json(output_dir / "meta" / "info.json", info)
    print(f"[DONE] {split_name}: {len(episode_metadata)} episodes, {global_index} frames -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=(
            Path.home()
            / "ros_ur_driver"
            / "dataset_VLA_sim"
        ),
    )

    parser.add_argument(
        "--camera-json",
        type=Path,
        default=(
            Path.home()
            / "ros_ur_driver"
            / "src"
            / "moveit_config"
            / "config"
            / "sim_camera_calibration_summary.json"
        ),
    )
    parser.add_argument(
        "--grounding-cache",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--grounding-config",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "starvla_qwengroot_ur5_4090.yaml"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            Path.home()
            / "ros_ur_driver"
            / "dataset_VLA_sim"
            / "starvla_ur5_front_rgbd_pick_600_geometry_v8"
        ),
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gripper-threshold", type=float, default=0.0)
    parser.add_argument(
        "--instruction",
        default=(
            "Move the robot end-effector toward the orange plug using small Cartesian "
            "corrections, continuously adjust the motion according to the relative "
            "position between the end-effector and the orange plug, align the gripper "
            "with the orange plug, and grasp it securely."
        ),
        help="Fallback only; per-step subtask_instruction takes priority.",
    )
    args = parser.parse_args()
    args.subset_root = (
        args.subset_root
        .expanduser()
        .resolve()
    )

    args.camera_json = (
        args.camera_json
        .expanduser()
        .resolve()
    )

    args.grounding_config = (
        args.grounding_config
        .expanduser()
        .resolve()
    )

    args.output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if args.grounding_cache is None:
        args.grounding_cache = (
            args.subset_root
            / "v7_grounding_cache.json"
        )
    else:
        args.grounding_cache = (
            args.grounding_cache
            .expanduser()
            .resolve()
        )
    camera_data = read_json(
        args.camera_json
    )

    static_camera = camera_data[
        "static_camera"
    ]

    intrinsics = static_camera[
        "intrinsics"
    ]

    extrinsics = static_camera[
        "extrinsics"
    ]

    camera_calibration = {
        "fx": float(
            intrinsics["fx"]
        ),
        "fy": float(
            intrinsics["fy"]
        ),
        "cx": float(
            intrinsics["cx"]
        ),
        "cy": float(
            intrinsics["cy"]
        ),
        "T_base_camera_link": np.asarray(
            extrinsics[
                "T_base_camera_link"
            ],
            dtype=np.float64,
        ).reshape(4, 4),
    }
    if args.fps <= 0:
        raise ValueError(
            "--fps must be positive"
        )

    manifests = {
        name: load_manifest(
            args.subset_root,
            name,
        )
        for name in (
            "train",
            "val",
            "test",
        )
    }

    identities = {
        name: {
            path.resolve()
            for path in paths
        }
        for name, paths
        in manifests.items()
    }

    if (
        identities["train"]
        & identities["val"]
        or identities["train"]
        & identities["test"]
        or identities["val"]
        & identities["test"]
    ):
        raise ValueError(
            "train/val/test manifests overlap"
        )

    grounding_cache = (
        build_or_update_grounding_cache(
            manifests=manifests,
            cache_path=args.grounding_cache,
            config_yaml=args.grounding_config,
            instruction=args.instruction,
        )
    )
    output_names = {
        "train": "starvla_ur5_train",
        "val": "starvla_ur5_val",
        "test": "starvla_ur5_test",
    }
    for split_name in ("train", "val", "test"):
        write_dataset(
            split_name,
            manifests[split_name],
            args.output_root
            / output_names[split_name],
            args.fps,
            args.instruction,
            args.gripper_threshold,
            camera_calibration,
            grounding_cache,
        )


if __name__ == "__main__":
    main()

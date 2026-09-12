#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# StarVLA UR5 Pick-only pipeline
#   1) Activate StarVLA environment
#   2) Build Pick-only dataset (if it does not already exist)
#   3) Validate dataset
#   4) Start training
#
# Usage:
#   bash run_starvla_pick_only.sh
#
# Optional overrides:
#   MAX_TRAIN_STEPS=10000 SAVE_INTERVAL=1000 bash run_starvla_pick_only.sh
#   FORCE_REBUILD=1 bash run_starvla_pick_only.sh
#   SKIP_TRAIN=1 bash run_starvla_pick_only.sh
# ============================================================

STARVLA_ROOT="${STARVLA_ROOT:-$HOME/vla_workspace/starVLA}"
TRAIN_DIR="$STARVLA_ROOT/examples/UR5/train_files"

PREPARE_SCRIPT="$TRAIN_DIR/prepare_starvla_ur5_dataset.py"
CHECK_SCRIPT="$TRAIN_DIR/check_starvla_ur5_dataset.py"
TRAIN_SCRIPT="$TRAIN_DIR/run_ur5_qwengroot_smoke.sh"

DATASET_ROOT="${DATASET_ROOT:-$HOME/ros_ur_driver/dataset_VLA_sim/starvla_ur5_front_rgbd_pick_600_geometry_v8}"
GROUNDING_CACHE="${GROUNDING_CACHE:-$HOME/ros_ur_driver/dataset_VLA_sim/v7_grounding_cache.json}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-12000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
RUN_ID="${RUN_ID:-ur5_front_rgbd_pick_600_geometry_v8}"

FORCE_REBUILD="${FORCE_REBUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

echo "============================================================"
echo " StarVLA UR5 Pick-only"
echo "============================================================"
echo "STARVLA_ROOT    : $STARVLA_ROOT"
echo "DATASET_ROOT    : $DATASET_ROOT"
echo "GROUNDING_CACHE : $GROUNDING_CACHE"
echo "RUN_ID          : $RUN_ID"
echo "MAX_TRAIN_STEPS : $MAX_TRAIN_STEPS"
echo "SAVE_INTERVAL   : $SAVE_INTERVAL"
echo "============================================================"

cd "$STARVLA_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "[ERROR] Missing virtual environment: $STARVLA_ROOT/.venv"
    exit 1
fi

source .venv/bin/activate

for f in "$PREPARE_SCRIPT" "$CHECK_SCRIPT" "$TRAIN_SCRIPT"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Missing file: $f"
        exit 1
    fi
done

# ------------------------------------------------------------
# 1. Prepare dataset
# ------------------------------------------------------------
if [[ -d "$DATASET_ROOT" ]]; then
    if [[ "$FORCE_REBUILD" == "1" ]]; then
        echo
        echo "[1/3] FORCE_REBUILD=1 -> removing old Pick-only dataset..."
        rm -rf "$DATASET_ROOT"

        echo "[1/3] Building Pick-only dataset..."
        python3 "$PREPARE_SCRIPT" \
            --output-root "$DATASET_ROOT" \
            --grounding-cache "$GROUNDING_CACHE"
    else
        echo
        echo "[1/3] Dataset already exists -> skip conversion."
        echo "      Use FORCE_REBUILD=1 if you really want to rebuild it."
    fi
else
    echo
    echo "[1/3] Building Pick-only dataset..."
    python3 "$PREPARE_SCRIPT" \
        --output-root "$DATASET_ROOT" \
        --grounding-cache "$GROUNDING_CACHE"
fi

# ------------------------------------------------------------
# 2. Validate dataset
# ------------------------------------------------------------
echo
echo "[2/3] Checking dataset..."
python3 "$CHECK_SCRIPT" \
    --root "$DATASET_ROOT"

echo
echo "[INFO] Training task:"
cat "$DATASET_ROOT/starvla_ur5_train/meta/tasks.jsonl"

# ------------------------------------------------------------
# 3. Train
# ------------------------------------------------------------
if [[ "$SKIP_TRAIN" == "1" ]]; then
    echo
    echo "[3/3] SKIP_TRAIN=1 -> dataset prepared and checked only."
    exit 0
fi

echo
echo "[3/3] Starting training..."
echo

RUN_ID="$RUN_ID" \
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
bash "$TRAIN_SCRIPT"

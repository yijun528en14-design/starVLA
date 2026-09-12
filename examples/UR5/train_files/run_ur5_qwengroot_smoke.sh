#!/usr/bin/env bash
set -euo pipefail

STARVLA_ROOT="${STARVLA_ROOT:-$HOME/vla_workspace/starVLA}"
CONFIG_YAML="${CONFIG_YAML:-$STARVLA_ROOT/examples/UR5/train_files/starvla_qwengroot_ur5_4090.yaml}"
RUN_ID="${RUN_ID:-ur5_front_rgbd_pick_600_geometry_v8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-12000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"

cd "$STARVLA_ROOT"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export STARVLA_GRAD_ACC_STEPS="${STARVLA_GRAD_ACC_STEPS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"

test -f "$CONFIG_YAML"
test -f playground/Pretrained_models/Qwen3-VL-4B-Instruct/config.json
test -f playground/Pretrained_models/StarVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_ur5_4090.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG_YAML" \
  --run_id "$RUN_ID" \
  --trainer.max_train_steps "$MAX_TRAIN_STEPS" \
  --trainer.save_interval "$SAVE_INTERVAL"

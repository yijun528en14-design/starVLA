# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import time
import sys
from pathlib import Path
import json
import re

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import os
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TDefaultConfig:
    """QwenGR00T framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
            "lora": {
                "enabled": False,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "gradient_checkpointing": True,
            },
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        lora_cfg = self.config.framework.qwenvl.get(
            "lora",
            {},
        )

        if lora_cfg and bool(
            lora_cfg.get("enabled", False)
        ):
            qwen_model = self.qwen_vl_interface.model

            peft_config = LoraConfig(
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(
                    lora_cfg.get("lora_alpha", 16)
                ),
                lora_dropout=float(
                    lora_cfg.get(
                        "lora_dropout",
                        0.05,
                    )
                ),
                target_modules=list(
                    lora_cfg.get(
                        "target_modules",
                        ["q_proj", "v_proj"],
                    )
                ),
                bias="none",
            )

            self.qwen_vl_interface.model = (
                get_peft_model(
                    qwen_model,
                    peft_config,
                )
            )

            if bool(
                lora_cfg.get(
                    "gradient_checkpointing",
                    True,
                )
            ):
                peft_model = (
                    self.qwen_vl_interface.model
                )

                if hasattr(
                    peft_model,
                    "enable_input_require_grads",
                ):
                    peft_model.enable_input_require_grads()

                if hasattr(
                    peft_model,
                    "gradient_checkpointing_enable",
                ):
                    peft_model.gradient_checkpointing_enable()

                if hasattr(peft_model, "config"):
                    peft_model.config.use_cache = False

            self.qwen_vl_interface.model.print_trainable_parameters()
            
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # ============================================================
        # V6-A: Camera calibration context for frozen VLM
        # State7 remains OUTSIDE the VLM.
        # ============================================================

        camera_cfg = self.config.framework.get(
            "camera_context",
            {},
        )

        self.camera_context_enabled = bool(
            camera_cfg.get("enabled", False)
        )

        self.camera_calibration = None

        if self.camera_context_enabled:

            calibration_path = Path(
                str(
                    camera_cfg.get(
                        "calibration_path",
                        "",
                    )
                )
            ).expanduser()

            if not calibration_path.is_file():
                raise FileNotFoundError(
                    f"Camera calibration not found: "
                    f"{calibration_path}"
                )

            with calibration_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                calibration_data = json.load(f)

            static_camera = calibration_data[
                "static_camera"
            ]

            intrinsics = static_camera[
                "intrinsics"
            ]

            extrinsics = static_camera[
                "extrinsics"
            ]

            self.camera_calibration = {
                "width": int(
                    intrinsics["width"]
                ),
                "height": int(
                    intrinsics["height"]
                ),
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

            print(
                "\n"
                "========================================\n"
                "CAMERA CONTEXT ENABLED\n"
                "========================================"
            )

            print(
                "Calibration:",
                calibration_path,
            )

            print(
                "fx fy cx cy:",
                self.camera_calibration["fx"],
                self.camera_calibration["fy"],
                self.camera_calibration["cx"],
                self.camera_calibration["cy"],
            )

            print(
                "T_base_camera_link:\n",
                self.camera_calibration[
                    "T_base_camera_link"
                ],
            )

            print(
                "========================================\n"
            )

    def _add_camera_context(
        self,
        instruction: str,
    ) -> str:

        if not self.camera_context_enabled:
            return instruction

        if self.camera_calibration is None:
            raise RuntimeError(
                "camera_context enabled but "
                "camera_calibration is missing"
            )

        c = self.camera_calibration

        T = c[
            "T_base_camera_link"
        ]

        T_text = "\n".join(
            "  [" +
            ", ".join(
                f"{float(v):.6f}"
                for v in row
            )
            + "]"
            for row in T
        )

        return f"""
{instruction}

Camera geometry context:

The first image is the front RGB image.

The second image is the aligned front depth visualization.
The depth image represents metric depth using a fixed mapping:
near points are brighter and far points are darker.

Static camera image size:
width = {c["width"]}
height = {c["height"]}

Static camera intrinsics:
fx = {c["fx"]:.6f}
fy = {c["fy"]:.6f}
cx = {c["cx"]:.6f}
cy = {c["cy"]:.6f}

Static camera extrinsic matrix
T_base_camera_link:
{T_text}

Use the RGB image to identify the relevant robot and task objects.
Use the aligned depth observation together with the camera
intrinsics and extrinsics to preserve spatial-position and
relative-position information in the visual representation.
""".strip()

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        for i, images in enumerate(batch_images):
            if len(images) != 2:
                raise ValueError(
                    f"RGB-D expects exactly 2 images "
                    f"[front_rgb, front_depth], "
                    f"sample {i} got {len(images)}"
                )
        instructions = [
            self._add_camera_context(
                example["lang"]
            )
            for example in examples
        ]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        full_state = (
            [example["state"] for example in examples]
            if "state" in examples[0]
            else None
        )

        state = None
        explicit_geometry = None

        if full_state is not None:
            full_state = np.array(full_state)

            if full_state.shape[-1] != 10:
                raise ValueError(
                    f"Expected State10 during training, got {full_state.shape}"
                )

            state = full_state[..., :7]
            explicit_geometry = full_state[..., 7:10]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

            input_ids = qwen_inputs["input_ids"]

            image_token_id = getattr(
                self.qwen_vl_interface.model.config,
                "image_token_id",
                None,
            )

            if image_token_id is None:
                image_token_id = (
                    self.qwen_vl_interface.processor.tokenizer
                    .convert_tokens_to_ids("<|image_pad|>")
                )

            visual_token_mask = (
                input_ids == image_token_id
            )
            # ========================================================
            # V6-A:
            # Camera K/T is provided as text context to Frozen Qwen.
            #
            # Geometry fusion must therefore be allowed to read
            # BOTH image tokens and valid text tokens.
            #
            # State7 remains completely outside the VLM.
            # ========================================================

            if (
                self.camera_context_enabled
                and backbone_attention_mask is not None
            ):
                geometry_token_mask = (
                    backbone_attention_mask.bool()
                )
            else:
                geometry_token_mask = (
                    visual_token_mask
                )

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            geometry_token_mask_repeated = (
                geometry_token_mask.repeat(
                    repeated_diffusion_steps,
                    1,
                )
            )
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            explicit_geometry_repeated = None

            if explicit_geometry is not None:
                explicit_geometry = torch.tensor(
                    np.array(explicit_geometry),
                    device=last_hidden.device,
                    dtype=last_hidden.dtype,
                )

                explicit_geometry_repeated = (
                    explicit_geometry.repeat(
                        repeated_diffusion_steps,
                        1,
                        1,
                    )
                )

            losses = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                explicit_geometry=explicit_geometry_repeated,
                encoder_attention_mask=backbone_attention_mask,
                visual_token_mask=geometry_token_mask_repeated,
            )

        return {
            "action_loss":
                losses["total_loss"],

            "action_dit_loss":
                losses["flow_loss"],

            "geometry_loss":
                losses["geometry_loss"],

            "control_loss":
                losses["control_loss"],

            "metric_loss":
                losses["metric_loss"],
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [
            self._add_camera_context(
                example["lang"]
            )
            for example in examples
        ]

        full_state = (
            [example["state"] for example in examples]
            if "state" in examples[0]
            else None
        )

        state = None
        explicit_geometry = None

        if full_state is not None:
            full_state = np.array(full_state)

            if full_state.shape[-1] != 10:
                raise ValueError(
                    f"Expected State10, got {full_state.shape}"
                )

            state = full_state[..., :7]
            explicit_geometry = full_state[..., 7:10]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

            input_ids = qwen_inputs["input_ids"]

            image_token_id = getattr(
                self.qwen_vl_interface.model.config,
                "image_token_id",
                None,
            )

            if image_token_id is None:
                image_token_id = (
                    self.qwen_vl_interface.processor.tokenizer
                    .convert_tokens_to_ids("<|image_pad|>")
                )

            visual_token_mask = (
                input_ids == image_token_id
            )
            # ========================================================
            # V6-A inference:
            # allow Geometry fusion to read camera K/T text tokens
            # together with RGB-D visual tokens.
            # ========================================================

            if (
                self.camera_context_enabled
                and backbone_attention_mask is not None
            ):
                geometry_token_mask = (
                    backbone_attention_mask.bool()
                )
            else:
                geometry_token_mask = (
                    visual_token_mask
                )

        dump_path = os.environ.get(
            "STARVLA_VLM_FEATURE_DUMP",
            "",
        )

        if dump_path:
            input_ids = qwen_inputs[
                "input_ids"
            ]

            image_token_id = getattr(
                self.qwen_vl_interface.model.config,
                "image_token_id",
                None,
            )

            if image_token_id is None:
                image_token_id = (
                    self.qwen_vl_interface
                    .processor.tokenizer
                    .convert_tokens_to_ids(
                        "<|image_pad|>"
                    )
                )

            image_mask = (
                input_ids == image_token_id
            )

            visual_tokens = (
                last_hidden[0][image_mask[0]]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            output_path = Path(
                dump_path
            ).expanduser()

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_data = {
                "last_hidden": (
                    last_hidden[0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                ),
                "visual_tokens": visual_tokens,
                "input_ids": (
                    input_ids[0]
                    .detach()
                    .cpu()
                    .numpy()
                ),
                "attention_mask": (
                    qwen_inputs[
                        "attention_mask"
                    ][0]
                    .detach()
                    .cpu()
                    .numpy()
                ),
            }

            if (
                "image_grid_thw"
                in qwen_inputs
            ):
                save_data[
                    "image_grid_thw"
                ] = (
                    qwen_inputs[
                        "image_grid_thw"
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

            output_path = Path(
                dump_path
            ).expanduser()

            # 把環境變數當成資料夾使用
            output_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            timestamp = time.time_ns()

            save_path = (
                output_path /
                f"vlm_feature_{timestamp}.npz"
            )

            np.savez_compressed(
                save_path,
                **save_data,
            )

            print(
                "\n===== VLM feature dump ====="
            )
            print(
                "last_hidden:",
                tuple(last_hidden.shape),
            )
            print(
                "visual tokens:",
                visual_tokens.shape,
            )
            print(
                "saved:",
                save_path,
            )

            print(
                "\n===== VLM feature dump ====="
            )
            print(
                "last_hidden:",
                tuple(last_hidden.shape),
            )
            print(
                "image token id:",
                image_token_id,
            )
            print(
                "visual tokens:",
                visual_tokens.shape,
            )
            print(
                "saved:",
                output_path,
            )

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        explicit_geometry = (
            torch.from_numpy(
                np.array(explicit_geometry)
            ).to(
                last_hidden.device,
                dtype=last_hidden.dtype,
            )
            if explicit_geometry is not None
            else None
        )

        # Optional deterministic action sampling for diagnosis.
        # The Flow-Matching action head normally starts from torch.randn,
        # so repeated inference can produce different action chunks.
        action_seed = os.environ.get(
            "STARVLA_ACTION_SEED",
            "",
        )

        if action_seed:
            seed = int(action_seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        debug_mode = os.environ.get("STARVLA_HIDDEN_DEBUG", "normal")

        if debug_mode == "zero":
            last_hidden = torch.zeros_like(last_hidden)

        elif debug_mode == "override":
            override_path = os.environ.get("STARVLA_HIDDEN_OVERRIDE", "")

            if not override_path:
                raise RuntimeError(
                    "STARVLA_HIDDEN_OVERRIDE is empty"
                )

            d = np.load(override_path)

            x = d["last_hidden"]

            x = torch.from_numpy(x).to(
                device=last_hidden.device,
                dtype=last_hidden.dtype,
            )

            # 如果存的是 [seq, dim]
            # 而目前模型需要 [1, seq, dim]
            if x.ndim == 2:
                x = x.unsqueeze(0)

            if x.shape != last_hidden.shape:
                raise RuntimeError(
                    f"Hidden shape mismatch: "
                    f"override={x.shape}, "
                    f"current={last_hidden.shape}"
                )

            last_hidden = x

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions, pred_geometry = (
                self.action_model.predict_action(
                    last_hidden,
                    state,
                    explicit_geometry=explicit_geometry,
                    encoder_attention_mask=backbone_attention_mask,
                    visual_token_mask=geometry_token_mask,
                )
            ) # (B, chunk_len, action_dim)

        normalized_actions = (
            pred_actions
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if pred_geometry is not None:
            pred_geometry_np = (
                pred_geometry
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        else:
            pred_geometry_np = None

        print(
            "\n"
            "========================================\n"
            "V3 GEOMETRY DEBUG\n"
            "========================================\n"
            f"Pred rel_xyz: {pred_geometry_np}\n"
            "========================================\n"
        )

        return {
            "normalized_actions": normalized_actions,
            "pred_geometry": pred_geometry_np,
        }

    @torch.inference_mode()
    def predict_grounding(
        self,
        examples: List[dict],
        max_new_tokens: int = 64,
    ) -> List[dict]:

        if not isinstance(examples, list):
            examples = [examples]

        model = self.qwen_vl_interface.model
        processor = self.qwen_vl_interface.processor

        # Batch generation 建議 left padding
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"

        conversations = []

        for example in examples:
            raw_images = example["image"]

            rgb_only_images = []

            for image in raw_images:
                image_np = np.asarray(image)

                rgb_only_images.append(
                    np.ascontiguousarray(
                        image_np,
                        dtype=np.uint8,
                    )
                )

            images = to_pil_preserve(
                rgb_only_images
            )

            if not isinstance(
                images,
                (list, tuple),
            ):
                images = [images]

            instruction = str(
                example["lang"]
            )

            content = []

            # image[0] = RGB
            if len(images) >= 1:
                content.append(
                    {
                        "type": "text",
                        "text": "PRIMARY CAMERA:",
                    }
                )

                content.append(
                    {
                        "type": "image",
                        "image": images[0],
                    }
                )

            # image[1] = Depth visualization
            if len(images) >= 2:
                content.append(
                    {
                        "type": "text",
                        "text": "PRIMARY DEPTH:",
                    }
                )

                content.append(
                    {
                        "type": "image",
                        "image": images[1],
                    }
                )

            grounding_prompt = f"""
    Robot task:
    {instruction}

    Based only on the CURRENT camera images and the robot task,
    identify the single visual object that is currently most important
    for the robot to interact with NEXT in order to make progress.

    Infer the current task progress from the images.
    Do not assume or use a manually provided task phase.

    The phase must be exactly one of:
    "before_grasp"
    "grasp"

    Return ONLY one valid JSON object in exactly this form:

    {{
    "target": "object name",
    "phase": "before_grasp",
    "primary_bbox": [x1, y1, x2, y2]
    }}

    Bounding box coordinates must use normalized coordinates
    from 0 to 1000.

    Do not output markdown.
    Do not output explanations.
    """.strip()

            content.append(
                {
                    "type": "text",
                    "text": grounding_prompt,
                }
            )

            conversations.append(
                [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            )

        # ============================================================
        # IMPORTANT:
        # 整個 examples batch 一次 processor
        # ============================================================

        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        inputs = inputs.to(
            model.device
        )

        # ============================================================
        # 一次 generate 整個 batch
        # ============================================================

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        input_length = (
            inputs.input_ids.shape[1]
        )

        generated_ids_trimmed = (
            generated_ids[
                :,
                input_length:
            ]
        )

        output_texts = (
            processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

        results = []

        for output_text in output_texts:

            parsed = None

            match = re.search(
                r"\{[\s\S]*?\}",
                output_text,
            )

            if match is not None:
                try:
                    parsed = json.loads(
                        match.group(0)
                    )
                except json.JSONDecodeError:
                    parsed = None

            if not isinstance(
                parsed,
                dict,
            ):
                parsed = {
                    "target": "unknown",
                    "phase": "unknown",
                    "primary_bbox": None,
                }

            parsed[
                "raw_output"
            ] = output_text

            results.append(
                parsed
            )

        return results

if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")

"""UR5 + RG2 data registry for StarVLA QwenGR00T."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


# ============================================================
# Original dual-camera config
# primary RGB + wrist RGB + State7
# ============================================================
class UR5PickInsertDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]

    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]

    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]

    language_keys = [
        "annotation.human.action.task_description"
    ]

    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        motion_keys = self.action_keys[:6]
        state_motion_keys = self.state_keys[:6]
        tensor_keys = self.state_keys + self.action_keys

        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(
                    apply_to=tensor_keys
                ),
                StateActionTransform(
                    apply_to=tensor_keys,
                    normalization_modes={
                        key: "min_max"
                        for key in state_motion_keys + motion_keys
                    },
                ),
            ]
        )


# ============================================================
# New experiment:
# fixed front RGB only + State7 + Pick-only
# ============================================================
class UR5FrontRGBPickDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = [
        "video.primary_image",
    ]

    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
        "state.rel_target_x",
        "state.rel_target_y",
        "state.rel_target_z",
    ]

    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]

    language_keys = [
        "annotation.human.action.task_description"
    ]

    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        motion_keys = self.action_keys[:6]

        # ============================================================
        # V4H:
        # Keep EEF XYZ in raw meters for metric-consistency loss.
        #
        # Only normalize orientation state.
        # ============================================================

        state_motion_keys = [
            "state.roll",
            "state.pitch",
            "state.yaw",
        ]

        tensor_keys = (
            self.state_keys
            + self.action_keys
        )

        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(
                    apply_to=tensor_keys,
                ),

                StateActionTransform(
                    apply_to=(
                        self.state_keys
                        + self.action_keys
                    ),
                    normalization_modes={
                        key: "min_max"
                        for key in (
                            state_motion_keys
                            + motion_keys
                        )
                    },
                ),
            ]
        )

class UR5FrontRGBDPickDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = [
        "video.primary_image",
        "video.primary_depth",
    ]

    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
        "state.rel_target_x",
        "state.rel_target_y",
        "state.rel_target_z",
    ]

    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]

    language_keys = [
        "annotation.human.action.task_description"
    ]

    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        motion_keys = self.action_keys[:6]

        state_motion_keys = [
            "state.roll",
            "state.pitch",
            "state.yaw",
        ]

        tensor_keys = (
            self.state_keys
            + self.action_keys
        )

        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(
                    apply_to=tensor_keys,
                ),

                StateActionTransform(
                    apply_to=(
                        self.state_keys
                        + self.action_keys
                    ),
                    normalization_modes={
                        key: "min_max"
                        for key in (
                            state_motion_keys
                            + motion_keys
                        )
                    },
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "ur5_pick_insert":
        UR5PickInsertDataConfig(),

    "ur5_front_rgb_pick":
        UR5FrontRGBPickDataConfig(),

    "ur5_front_rgbd_pick":
        UR5FrontRGBDPickDataConfig(),
}


ROBOT_TYPE_TO_EMBODIMENT_TAG = {}


DATASET_NAMED_MIXTURES = {
    # ========================================================
    # Original dataset
    # ========================================================
    "ur5_pick_insert_train": [
        (
            "starvla_ur5_train",
            1.0,
            "ur5_pick_insert",
        )
    ],

    "ur5_pick_insert_val": [
        (
            "starvla_ur5_val",
            1.0,
            "ur5_pick_insert",
        )
    ],

    "ur5_pick_insert_test": [
        (
            "starvla_ur5_test",
            1.0,
            "ur5_pick_insert",
        )
    ],

    # ========================================================
    # New front-RGB-only Pick dataset
    # ========================================================
    "ur5_front_rgb_pick_train": [
        (
            "starvla_ur5_train",
            1.0,
            "ur5_front_rgb_pick",
        )
    ],

    "ur5_front_rgb_pick_val": [
        (
            "starvla_ur5_val",
            1.0,
            "ur5_front_rgb_pick",
        )
    ],

    "ur5_front_rgb_pick_test": [
        (
            "starvla_ur5_test",
            1.0,
            "ur5_front_rgb_pick",
        )
    ],

    # ========================================================
    # New front-RGBD Pick dataset
    # ========================================================

    "ur5_front_rgbd_pick_train": [
        (
            "starvla_ur5_train",
            1.0,
            "ur5_front_rgbd_pick",
        )
    ],

    "ur5_front_rgbd_pick_val": [
        (
            "starvla_ur5_val",
            1.0,
            "ur5_front_rgbd_pick",
        )
    ],

    "ur5_front_rgbd_pick_test": [
        (
            "starvla_ur5_test",
            1.0,
            "ur5_front_rgbd_pick",
        )
    ],

    # ========================================================
    # Existing multitask datasets
    # ========================================================
    "ur5_pick_align_insert_train": [
        (
            "starvla_ur5_pick_train",
            1.0,
            "ur5_pick_insert",
        ),
        (
            "starvla_ur5_align_insert_train",
            1.0,
            "ur5_pick_insert",
        ),
    ],

    "ur5_pick_align_insert_val": [
        (
            "starvla_ur5_pick_val",
            1.0,
            "ur5_pick_insert",
        ),
        (
            "starvla_ur5_align_insert_val",
            1.0,
            "ur5_pick_insert",
        ),
    ],

    "ur5_pick_align_insert_test": [
        (
            "starvla_ur5_pick_test",
            1.0,
            "ur5_pick_insert",
        ),
        (
            "starvla_ur5_align_insert_test",
            1.0,
            "ur5_pick_insert",
        ),
    ],
}
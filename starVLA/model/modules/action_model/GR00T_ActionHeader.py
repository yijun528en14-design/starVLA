# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT


from dataclasses import dataclass, field

import os

import torch
# ============================================================
# Offline diagnostic debug cache
# ============================================================

DEBUG_LAST_PRED_REL_XYZ = None

# V4G-A:
# expose auxiliary Control Head prediction during inference
DEBUG_LAST_PRED_CONTROL_XYZ = None
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        # ------------------------------------------------------------------
        # DiT architecture selection
        #   action_model_type: "DiT-B" | "DiT-L"
        #     DiT-B → input_embedding_dim=768,  heads=12, head_dim=64
        #     DiT-L → input_embedding_dim=1536, heads=32, head_dim=48
        #   diffusion_model_cfg overrides/extends the base DiT shape.
        #   In particular, diffusion_model_cfg.cross_attention_dim MUST be
        #   set by the framework to match the VLM hidden size BEFORE calling
        #   get_action_model(), e.g.:
        #       cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim
        #           = vlm.model.config.hidden_size
        # ------------------------------------------------------------------
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)

        # ------------------------------------------------------------------
        # Action horizon (chunk length sent to the DiT)
        #   Single source of truth: `action_horizon` (e.g. 8).
        #   Legacy YAMLs that only provide `future_action_window_size` are
        #   normalised to `action_horizon` upstream by
        #   `share_tools.apply_config_compat`, so this code never touches
        #   the legacy alias.
        # ------------------------------------------------------------------
        self.action_horizon = int(config.action_horizon)

        # ------------------------------------------------------------------
        # Action / state dimensions
        #   action_dim: DoF of the robot action (e.g. 7 for 6-DoF + gripper)
        #   state_dim:  proprioception dimension; set to 0/None to disable
        #               the state_encoder branch entirely.
        # ------------------------------------------------------------------
        self.action_dim = config.action_dim

        # ------------------------------------------------------------------
        # Inference denoising steps
        #   num_inference_timesteps: Euler steps during predict_action().
        #   Typically 4–10; fewer = faster but less accurate.
        # ------------------------------------------------------------------
        self.num_inference_timesteps = config.num_inference_timesteps

        # ------------------------------------------------------------------
        # hidden_size: intermediate MLP width for state_encoder / action_decoder.
        #   Decoupled from input_embedding_dim so you can use a smaller hidden
        #   for the MLP without changing the DiT latent size.
        # ------------------------------------------------------------------
        self.hidden_size = config.hidden_size

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )
        # ============================================================
        # V3: explicit vision-state geometry fusion
        # ============================================================

        self.vlm_dim = int(
            config.diffusion_model_cfg.get(
                "cross_attention_dim",
                2048,
            )
        )

        self.geometry_dim = int(
            getattr(
                config,
                "geometry_dim",
                3,
            )
        )

        # 把 VLM pooled feature 投影到 DiT latent dimension
        self.visual_projector = nn.Sequential(
            nn.Linear(
                self.vlm_dim,
                self.input_embedding_dim,
            ),
            nn.SiLU(),
        )

        # Vision + proprio fusion
        self.vision_state_fusion = nn.Sequential(
            nn.Linear(
                self.input_embedding_dim * 2,
                self.hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_size,
                self.input_embedding_dim,
            ),
        )
        # ============================================================
        # V4A: state-query visual cross attention
        # ============================================================

        self.visual_kv_projector = nn.Linear(
            self.vlm_dim,
            self.input_embedding_dim,
        )

        self.geometry_cross_attn = nn.MultiheadAttention(
            embed_dim=self.input_embedding_dim,
            num_heads=8,
            batch_first=True,
        )

        self.geometry_attn_norm = nn.LayerNorm(
            self.input_embedding_dim
        )

        self.geometry_ffn = nn.Sequential(
            nn.Linear(
                self.input_embedding_dim,
                self.input_embedding_dim * 4,
            ),
            nn.SiLU(),
            nn.Linear(
                self.input_embedding_dim * 4,
                self.input_embedding_dim,
            ),
        )

        self.geometry_ffn_norm = nn.LayerNorm(
            self.input_embedding_dim
        )

        # ============================================================
        # V4B: Explicit State + Visual Geometry Fusion
        #
        # Cross-attention 已經產生 state-conditioned visual feature，
        # 這裡再把原始 state feature 明確 concat 回來，
        # 避免 EEF metric information 只存在於
        # state_query + attn_out 的 normalized residual 中。
        # ============================================================

        self.geometry_state_fusion = nn.Sequential(
            nn.Linear(
                self.input_embedding_dim * 2,
                self.hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_size,
                self.input_embedding_dim,
            ),
        )

        self.geometry_state_fusion_norm = nn.LayerNorm(
            self.input_embedding_dim
        )

        # ============================================================
        # V4G-A:
        # Direct EEF XYZ metric pathway
        # ============================================================

        self.eef_xyz_projector = nn.Sequential(
            nn.Linear(
                3,
                self.hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_size,
                self.input_embedding_dim,
            ),
        )

        # Old geometry feature
        # +
        # direct EEF XYZ feature
        # ->
        # metric-aware geometry feature
        self.metric_geometry_fusion = nn.Sequential(
            nn.Linear(
                self.input_embedding_dim * 2,
                self.hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_size,
                self.input_embedding_dim,
            ),
        )

        self.metric_geometry_fusion_norm = nn.LayerNorm(
            self.input_embedding_dim
        )
        self.explicit_geometry_encoder = nn.Sequential(
            nn.Linear(
                self.geometry_dim,
                self.hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_size,
                self.input_embedding_dim,
            ),
        )

        self.hybrid_geometry_fusion = nn.Sequential(
            nn.Linear(
                self.input_embedding_dim * 2,
                self.hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_size,
                self.input_embedding_dim,
            ),
        )

        self.hybrid_geometry_fusion_norm = nn.LayerNorm(
            self.input_embedding_dim
        )

        # ============================================================
        # Geometry auxiliary head
        # ============================================================

        self.geometry_head = nn.Sequential(
            nn.Linear(
                self.input_embedding_dim,
                256,
            ),
            nn.SiLU(),
            nn.Linear(
                256,
                self.geometry_dim,
            ),
        )

        self.geometry_loss_weight = float(
            getattr(
                config,
                "geometry_loss_weight",
                0.3,
            )
        )

        # ============================================================
        # V4G-A:
        # Auxiliary Step0 XYZ Control Head
        #
        # Important:
        # This head reads ONLY geometry_feature.
        # It does not create a new shortcut into Action DiT.
        # ============================================================

        self.control_head = nn.Sequential(
            nn.Linear(
                self.input_embedding_dim,
                256,
            ),
            nn.SiLU(),
            nn.Linear(
                256,
                3,
            ),
        )

        self.control_loss_weight = float(
            getattr(
                config,
                "control_loss_weight",
                0.3,
            )
        )

        # ============================================================
        # V4H:
        # Metric geometry consistency loss
        # ============================================================

        self.metric_loss_weight = float(
            getattr(
                config,
                "metric_loss_weight",
                0.3,
            )
        )

        # Cartesian perturbation magnitude in meters.
        self.metric_delta_m = float(
            getattr(
                config,
                "metric_delta_m",
                0.01,
            )
        )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # ------------------------------------------------------------------
        # future_tokens: learnable query tokens prepended before the action
        #   sequence so the DiT has dedicated "planning" slots.
        #   num_target_vision_tokens controls how many such tokens are added.
        # ------------------------------------------------------------------
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # ------------------------------------------------------------------
        # Positional embedding over the action sequence
        #   add_pos_embed: whether to add sinusoidal-style learned PE
        #   max_seq_len:   max supported action sequence length
        # ------------------------------------------------------------------
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # ------------------------------------------------------------------
        # Flow-matching noise schedule (Beta distribution)
        #   noise_beta_alpha / noise_beta_beta: Beta(α, β) shape params.
        #   noise_s: upper-clip of the sampled value so t ∈ [0, noise_s].
        #   num_timestep_buckets: discretise continuous t into N buckets for
        #     the timestep encoder inside DiT.
        # ------------------------------------------------------------------
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _build_geometry_feature(
        self,
        vl_embs,
        state_features,
        visual_token_mask,
    ):
        if visual_token_mask is None:
            raise ValueError(
                "visual_token_mask is required for V4A geometry fusion"
            )

        if visual_token_mask.shape != vl_embs.shape[:2]:
            raise ValueError(
                "visual_token_mask shape mismatch: "
                f"mask={visual_token_mask.shape}, "
                f"vl_embs={vl_embs.shape[:2]}"
            )

        mask = visual_token_mask.bool()

        # 每個 sample 至少要有一個 image token
        if not mask.any(dim=1).all():
            raise ValueError(
                "At least one sample contains no visual token"
            )

        # [B, L, VLM_dim]
        # ->
        # [B, L, D]
        visual_tokens = self.visual_kv_projector(
            vl_embs
        )

        # state_features:
        # [B, state_seq, D]
        #
        # 目前通常 state_seq = 1
        # 轉成一個 EEF query
        state_query = state_features.mean(
            dim=1,
            keepdim=True,
        )

        # PyTorch MultiheadAttention:
        # key_padding_mask True = ignore
        key_padding_mask = ~mask

        attn_out, attn_weights = self.geometry_cross_attn(
            query=state_query,
            key=visual_tokens,
            value=visual_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        # ============================================================
        # DEBUG: save Geometry Cross-Attention weights
        #
        # attn_weights:
        # [B, num_heads, query_len, seq_len]
        #
        # 現在 query_len 通常 = 1
        # ============================================================

        geometry_attention = attn_weights.mean(dim=1)
        # [B, 1, L]

        geometry_attention = geometry_attention.squeeze(1)
        # [B, L]

        self.debug_geometry_attention = (
            geometry_attention
            .detach()
            .cpu()
        )
        self.debug_visual_token_mask = (
            visual_token_mask
            .detach()
            .cpu()
        )

        if os.environ.get(
            "STARVLA_GEOMETRY_ATTN_DEBUG",
            "0"
        ) == "1":

            print("\n===== GEOMETRY ATTENTION DEBUG =====")

            print(
                "vl_embs shape:",
                tuple(vl_embs.shape)
            )

            print(
                "visual_token_mask shape:",
                tuple(visual_token_mask.shape)
            )

            print(
                "visual token count:",
                visual_token_mask.sum(dim=1).tolist()
            )

            print(
                "attn_weights shape:",
                tuple(attn_weights.shape)
            )

            print(
                "geometry_attention shape:",
                tuple(geometry_attention.shape)
            )

            image_attention = (
                geometry_attention[0][visual_token_mask[0].bool()]
            )

            print(
                "image_attention shape:",
                tuple(image_attention.shape)
            )

            print(
                "image attention sum:",
                float(image_attention.sum().item())
            )

            print("====================================\n")

        x = self.geometry_attn_norm(
            state_query + attn_out
        )

        x = self.geometry_ffn_norm(
            x + self.geometry_ffn(x)
        )

        # ============================================================
        # V4B: Explicit State + Visual Geometry Fusion
        # ============================================================

        # x:
        #   [B, 1, D]
        #   cross-attention 後的 visual/state geometry representation
        #
        # state_query:
        #   [B, 1, D]
        #   EEF7 經 state_encoder 後的 representation
        #
        # 再次 explicit concat，讓 Geometry Head 可以直接使用
        # EEF information，而不只是依賴 attention residual。

        geometry_fusion_input = torch.cat(
            [
                x,
                state_query,
            ],
            dim=-1,
        )
        # [B, 1, 2D]

        geometry_fused = self.geometry_state_fusion(
            geometry_fusion_input
        )
        # [B, 1, D]

        # Residual 保留原本 V4A geometry representation，
        # 避免新 fusion 一開始完全破壞舊表示。
        geometry_fused = self.geometry_state_fusion_norm(
            x + geometry_fused
        )

        # [B, 1, D] -> [B, D]
        geometry_feature = geometry_fused.squeeze(1)

        return {
            "state_feature": state_query.squeeze(1),
            "state_vlm_feature": x.squeeze(1),
            "geometry_feature": geometry_feature,
        }

    def _fuse_explicit_geometry(
        self,
        learned_geometry_feature,
        explicit_geometry,
    ):
        if explicit_geometry is None:
            raise ValueError(
                "explicit_geometry is required"
            )

        if explicit_geometry.ndim == 3:
            explicit_geometry = (
                explicit_geometry.mean(dim=1)
            )

        explicit_feature = (
            self.explicit_geometry_encoder(
                explicit_geometry
            )
        )

        hybrid_delta = (
            self.hybrid_geometry_fusion(
                torch.cat(
                    [
                        learned_geometry_feature,
                        explicit_feature,
                    ],
                    dim=-1,
                )
            )
        )

        return self.hybrid_geometry_fusion_norm(
            learned_geometry_feature
            + hybrid_delta
        )

    def forward(
        self,
        vl_embs,
        actions,
        state=None,
        explicit_geometry=None,
        geometry_target=None,
        encoder_attention_mask=None,
        visual_token_mask=None,
    ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, action_dim)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # embed state
        state_features = (
            self.state_encoder(state)
            if state is not None
            else None
        )
            
        geometry_feature = None
        pred_geometry = None

        if state_features is None:
            raise ValueError(
                "State7 is required for geometry-conditioned Action DiT"
            )

        geometry_outputs = self._build_geometry_feature(
            vl_embs=vl_embs,
            state_features=state_features,
            visual_token_mask=visual_token_mask,
        )

        base_geometry_feature = geometry_outputs[
            "geometry_feature"
        ]

        # ============================================================
        # V4G-A:
        # Explicit EEF XYZ pathway
        #
        # state:
        #   [B, 1, 7]
        #
        # eef_xyz:
        #   [B, 1, 3]
        # ============================================================

        eef_xyz = state[..., :3]

        eef_xyz_feature = self.eef_xyz_projector(
            eef_xyz
        )

        eef_xyz_feature = eef_xyz_feature.mean(
            dim=1
        )

        # [B,D] + [B,D] -> [B,2D]
        metric_fusion_input = torch.cat(
            [
                base_geometry_feature,
                eef_xyz_feature,
            ],
            dim=-1,
        )

        metric_geometry_feature = (
            self.metric_geometry_fusion(
                metric_fusion_input
            )
        )

        geometry_feature = (
            self.metric_geometry_fusion_norm(
                base_geometry_feature
                + metric_geometry_feature
            )
        )
        geometry_feature = (
            self._fuse_explicit_geometry(
                geometry_feature,
                explicit_geometry,
            )
        )

        pred_geometry = self.geometry_head(
            geometry_feature
        )

        pred_control_xyz = self.control_head(
            geometry_feature
        )

        # ============================================================
        # V4H-B:
        # Axis-isolated metric perturbation
        #
        # Each sample perturbs ONLY ONE Cartesian axis.
        #
        # Example:
        #   sample 0 -> X only
        #   sample 1 -> Y only
        #   sample 2 -> Z only
        #
        # This forces the Geometry representation to learn:
        #
        #   d(rel_xyz) / d(eef_xyz) ≈ -I
        #
        # instead of an entangled XYZ response.
        # ============================================================

        B = state.shape[0]

        metric_delta = torch.zeros(
            B,
            1,
            3,
            device=state.device,
            dtype=state.dtype,
        )

        # Randomly choose ONE axis per sample.
        metric_axis = torch.randint(
            low=0,
            high=3,
            size=(B,),
            device=state.device,
        )

        # Random signed perturbation magnitude.
        metric_value = torch.empty(
            B,
            device=state.device,
            dtype=state.dtype,
        ).uniform_(
            -self.metric_delta_m,
            self.metric_delta_m,
        )

        batch_idx = torch.arange(
            B,
            device=state.device,
        )

        metric_delta[
            batch_idx,
            0,
            metric_axis,
        ] = metric_value

        perturbed_state = state.clone()

        perturbed_state[..., :3] = (
            perturbed_state[..., :3]
            + metric_delta
        )

        # ------------------------------------------------------------
        # Recompute State feature for perturbed EEF
        # ------------------------------------------------------------

        perturbed_state_features = self.state_encoder(
            perturbed_state
        )

        perturbed_geometry_outputs = (
            self._build_geometry_feature(
                vl_embs=vl_embs,
                state_features=perturbed_state_features,
                visual_token_mask=visual_token_mask,
            )
        )

        perturbed_base_geometry_feature = (
            perturbed_geometry_outputs[
                "geometry_feature"
            ]
        )

        perturbed_eef_xyz = (
            perturbed_state[..., :3]
        )

        perturbed_eef_xyz_feature = (
            self.eef_xyz_projector(
                perturbed_eef_xyz
            )
        )

        perturbed_eef_xyz_feature = (
            perturbed_eef_xyz_feature.mean(
                dim=1
            )
        )

        perturbed_metric_fusion_input = (
            torch.cat(
                [
                    perturbed_base_geometry_feature,
                    perturbed_eef_xyz_feature,
                ],
                dim=-1,
            )
        )

        perturbed_metric_geometry_feature = (
            self.metric_geometry_fusion(
                perturbed_metric_fusion_input
            )
        )

        perturbed_geometry_feature = (
            self.metric_geometry_fusion_norm(
                perturbed_base_geometry_feature
                + perturbed_metric_geometry_feature
            )
        )

        pred_geometry_perturbed = (
            self.geometry_head(
                perturbed_geometry_feature
            )
        )

        # ============================================================
        # Desired:
        #
        # pred_rel(state + delta)
        # -
        # pred_rel(state)
        # =
        # -delta
        # ============================================================

        metric_pred_delta = (
            pred_geometry_perturbed
            - pred_geometry
        )

        metric_target_delta = (
            -metric_delta[:, 0, :]
        )

        # ============================================================
        # V4H-B:
        # Normalize metric loss by perturbation scale.
        #
        # metric_delta_m = 0.01 means 10 mm.
        #
        # Without normalization:
        #     MSE is only around 1e-5 ~ 1e-4
        #
        # After normalization:
        #     ideal Cartesian gain is around -1
        #     zero response gives an O(1) loss.
        # ============================================================

        metric_scale = max(
            self.metric_delta_m,
            1e-6,
        )

        metric_pred_delta_normalized = (
            metric_pred_delta
            / metric_scale
        )

        metric_target_delta_normalized = (
            metric_target_delta
            / metric_scale
        )

        metric_loss = F.mse_loss(
            metric_pred_delta_normalized,
            metric_target_delta_normalized,
        )

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = (
            self.future_tokens.weight
            .unsqueeze(0)
            .expand(
                vl_embs.shape[0],
                -1,
                -1,
            )
        )

        geometry_token = geometry_feature.unsqueeze(1)

        sa_embs = torch.cat(
            (
                geometry_token,
                future_tokens,
                action_features,
            ),
            dim=1,
        )

        dit_encoder_context = torch.zeros_like(vl_embs)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=dit_encoder_context,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        flow_loss = (
            (pred_actions - velocity) ** 2
        ).mean()

        geometry_loss = torch.zeros(
            (),
            device=flow_loss.device,
            dtype=flow_loss.dtype,
        )

        if (
            pred_geometry is not None
            and geometry_target is not None
        ):
            # geometry_target expected [B, 1, 3] or [B, 3]
            if geometry_target.ndim == 3:
                geometry_target_for_loss = (
                    geometry_target[:, 0, :]
                )
            else:
                geometry_target_for_loss = (
                    geometry_target
                )

            geometry_loss = F.mse_loss(
                pred_geometry,
                geometry_target_for_loss,
            )

        # ============================================================
        # V4G-A:
        # Step0 XYZ auxiliary control loss
        #
        # actions are already the normalized training actions.
        # pred_control_xyz therefore learns in the same action space.
        # ============================================================

        control_target_xyz = actions[
            :,
            0,
            :3,
        ]

        control_loss = F.mse_loss(
            pred_control_xyz,
            control_target_xyz,
        )

        total_loss = (
            flow_loss
            + self.geometry_loss_weight
            * geometry_loss
            + self.control_loss_weight
            * control_loss
            + self.metric_loss_weight
            * metric_loss
        )

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "geometry_loss": geometry_loss,
            "control_loss": control_loss,
            "metric_loss": metric_loss,
        }

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs,
        state=None,
        explicit_geometry=None,
        encoder_attention_mask=None,
        visual_token_mask=None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_dim_mode = os.environ.get(
            "STARVLA_STATE_DIM_MODE",
            "normal",
        ).strip().lower()

        if state is not None:
            state_for_encoder = state.clone()

            dim_map = {
                "zero_x": 0,
                "zero_y": 1,
                "zero_z": 2,
                "zero_roll": 3,
                "zero_pitch": 4,
                "zero_yaw": 5,
                "zero_gripper": 6,
            }

            if state_dim_mode in dim_map:
                idx = dim_map[state_dim_mode]
                state_for_encoder[..., idx] = 0.0

            elif state_dim_mode == "all_zero":
                state_for_encoder = torch.zeros_like(
                    state_for_encoder
                )
        else:
            state_for_encoder = None

        state_features = (
            self.state_encoder(state_for_encoder)
            if state_for_encoder is not None
            else None
        )
        # ============================================================
        # V3: vision-state fusion
        # ============================================================

        # vl_embs: [B, L, VLM_dim]
        geometry_feature = None
        pred_geometry = None

        if state_features is None:
            raise ValueError(
                "State7 is required for geometry-conditioned Action DiT"
            )

        geometry_outputs = self._build_geometry_feature(
            vl_embs=vl_embs,
            state_features=state_features,
            visual_token_mask=visual_token_mask,
        )

        base_geometry_feature = geometry_outputs[
            "geometry_feature"
        ]

        eef_xyz = state_for_encoder[..., :3]

        eef_xyz_feature = self.eef_xyz_projector(
            eef_xyz
        )

        eef_xyz_feature = eef_xyz_feature.mean(
            dim=1
        )

        metric_fusion_input = torch.cat(
            [
                base_geometry_feature,
                eef_xyz_feature,
            ],
            dim=-1,
        )

        metric_geometry_feature = (
            self.metric_geometry_fusion(
                metric_fusion_input
            )
        )

        geometry_feature = (
            self.metric_geometry_fusion_norm(
                base_geometry_feature
                + metric_geometry_feature
            )
        )
        geometry_feature = (
            self._fuse_explicit_geometry(
                geometry_feature,
                explicit_geometry,
            )
        )

        pred_geometry = self.geometry_head(
            geometry_feature
        )

        # ============================================================
        # V4G-A:
        # Auxiliary Control Head inference
        #
        # IMPORTANT:
        # This reads exactly the same geometry_feature used during
        # training by control_loss.
        # ============================================================

        pred_control_xyz = self.control_head(
            geometry_feature
        )

        # ============================================================
        # Geometry debug
        # ============================================================

        self.debug_pred_rel_xyz = (
            pred_geometry
            .detach()
            .float()
            .cpu()
        )

        global DEBUG_LAST_PRED_REL_XYZ

        DEBUG_LAST_PRED_REL_XYZ = (
            pred_geometry
            .detach()
            .float()
            .cpu()
        )

        # ============================================================
        # Control Head debug
        # ============================================================

        self.debug_pred_control_xyz = (
            pred_control_xyz
            .detach()
            .float()
            .cpu()
        )

        global DEBUG_LAST_PRED_CONTROL_XYZ

        DEBUG_LAST_PRED_CONTROL_XYZ = (
            pred_control_xyz
            .detach()
            .float()
            .cpu()
        )

        state_debug = (
            os.environ.get(
                "STARVLA_STATE_DEBUG",
                "0",
            ) == "1"
        )

        def _rms(x):
            x = x.float()
            return float(
                torch.sqrt(
                    torch.mean(x * x)
                ).item()
            )

        if state_debug and state_features is not None:
            print("\n===== ACTION DIT STATE DEBUG =====")

            print(
                "state input shape:",
                tuple(state.shape),
            )

            print(
                "state input norm:",
                float(
                    torch.norm(
                        state.float()
                    ).item()
                ),
            )

            print(
                "state_features shape:",
                tuple(state_features.shape),
            )

            print(
                "state_features norm:",
                float(
                    torch.norm(
                        state_features.float()
                    ).item()
                ),
            )

            print(
                "state_features RMS:",
                _rms(state_features),
            )

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)

            if state_debug and t == 0:
                print(
                    "action_features shape:",
                    tuple(action_features.shape),
                )

                print(
                    "action_features norm:",
                    float(
                        torch.norm(
                            action_features.float()
                        ).item()
                    ),
                )

                print(
                    "action_features RMS:",
                    _rms(action_features),
                )

                print(
                    "vlm_features shape:",
                    tuple(vl_embs.shape),
                )

                print(
                    "vlm_features norm:",
                    float(
                        torch.norm(
                            vl_embs.float()
                        ).item()
                    ),
                )

                print(
                    "vlm_features RMS:",
                    _rms(vl_embs),
                )

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)

            geometry_token = geometry_feature.unsqueeze(1)

            sa_embs = torch.cat(
                (
                    geometry_token,
                    future_tokens,
                    action_features,
                ),
                dim=1,
            )

            # Run model forward.
            # ============================================================
            # V4A OFFLINE ABLATION:
            # zero Raw VLM shortcut before Action DiT
            #
            # IMPORTANT:
            # geometry_feature 已經在上面使用原始 vl_embs 建立，
            # 所以這裡只關閉 Raw VLM -> Action DiT shortcut。
            # ============================================================

            dit_encoder_context = torch.zeros_like(vl_embs)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=dit_encoder_context,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions, pred_geometry

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(full_config=config)


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass

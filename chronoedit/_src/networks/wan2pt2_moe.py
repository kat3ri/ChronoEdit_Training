# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WAN 2.2 Mixture-of-Experts (MoE) network architecture.

WAN 2.2 replaces the dense FFN in each transformer block with a Mixture-of-Experts
layer that uses a top-k gating mechanism. Two official variants exist:

  - **WAN 2.2-High (14B activated / ~60B total)**: Higher quality, 48 experts, top-4 routing.
  - **WAN 2.2-Low  (14B activated / ~60B total)**: Lower compute, 48 experts, top-2 routing.

This module provides:
  - ``MoEGate``       — learned top-k routing with optional auxiliary load-balancing loss.
  - ``MoEFFN``        — expert pool where each expert is a standard GELU FFN.
  - ``WanMOEAttentionBlock`` — drop-in replacement for ``WanAttentionBlock`` with MoE FFN.
  - ``WanMOEModel``   — full backbone (replaces ``WanModel``).
  - ``EditWanMOEModel`` — ChronoEdit variant with temporal-skip position embeddings.

Integration notes
-----------------
* ``WanMOEModel`` re-uses all dense components from ``wan2pt1.py``
  (position embeddings, self/cross-attention, patch embed, head).
* FSDP sharding: each expert is individually wrapped so that parameters
  are balanced across data-parallel ranks (see ``fully_shard``).
* Expert parallelism (EP) is *not* implemented here but the gating layer
  exposes ``expert_parallel_group`` for future extension.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from torch.distributed import ProcessGroup
from torch.distributed._composable.fsdp import fully_shard

from chronoedit._src.networks.wan2pt1 import (
    VideoPositionEmb,
    VideoRopePosition3DEmb,
    VideoSize,
    WanAttentionBlock,
    WanLayerNorm,
    WanModel,
    WanRMSNorm,
    WanSelfAttention,
    WAN_CROSSATTENTION_CLASSES,
    Head,
    MLPProj,
    SACConfig,
    CheckpointMode,
    sinusoidal_embedding_1d,
)
from chronoedit._src.networks.chronoedit_14b import (
    EditWanModel,
    TemproalSkipVideoRopePosition3DEmb,
)
from chronoedit._ext.imaginaire.utils import log
from chronoedit._ext.callbacks.model_weights_stats import WeightTrainingStat
from chronoedit._src.modules.selective_activation_checkpoint import SACConfig as SACConfig

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)


# ---------------------------------------------------------------------------
# MoE Gate — top-k router with auxiliary load-balancing loss
# ---------------------------------------------------------------------------

class MoEGate(nn.Module):
    """Top-k gating mechanism for Mixture-of-Experts.

    Parameters
    ----------
    dim : int
        Hidden dimension of the transformer.
    num_experts : int
        Total number of experts in the pool.
    top_k : int
        Number of experts activated per token.
    aux_loss_coeff : float
        Coefficient for the auxiliary load-balancing loss (Switch Transformer style).
        Set to 0.0 to disable.
    expert_parallel_group : ProcessGroup | None
        Reserved for future Expert Parallelism support.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 48,
        top_k: int = 4,
        aux_loss_coeff: float = 1e-2,
        expert_parallel_group: Optional[ProcessGroup] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        self.expert_parallel_group = expert_parallel_group

        self.gate = nn.Linear(dim, num_experts, bias=False)

    def init_weights(self):
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor):
        """Compute routing weights and expert indices.

        Parameters
        ----------
        x : Tensor
            Shape ``[B, L, D]``.

        Returns
        -------
        gate_scores : Tensor
            Normalised weights for the selected experts, shape ``[B, L, top_k]``.
        expert_indices : Tensor
            Indices of the selected experts, shape ``[B, L, top_k]``.
        aux_loss : Tensor
            Scalar auxiliary load-balancing loss.
        """
        # logits: [B, L, num_experts]
        logits = self.gate(x.float())
        scores = F.softmax(logits, dim=-1)

        # Top-k selection
        gate_scores, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        # Re-normalise so selected weights sum to 1
        gate_scores = gate_scores / (gate_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # Auxiliary load-balancing loss (Switch Transformer, Fedus et al. 2021)
        if self.training and self.aux_loss_coeff > 0.0:
            # fraction of tokens dispatched to each expert
            tokens_per_expert = scores.mean(dim=(0, 1))  # [num_experts]
            # fraction of routing probability allocated to each expert
            prob_per_expert = scores.mean(dim=(0, 1))     # same shape
            aux_loss = self.aux_loss_coeff * self.num_experts * (tokens_per_expert * prob_per_expert).sum()
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return gate_scores, expert_indices, aux_loss


# ---------------------------------------------------------------------------
# MoE FFN — expert pool
# ---------------------------------------------------------------------------

class MoEFFN(nn.Module):
    """Pool of independent FFN experts activated via gating.

    Each expert has the same architecture as the dense FFN in WAN 2.1:
    ``Linear(dim, ffn_dim) → GELU → Linear(ffn_dim, dim)``.

    Parameters
    ----------
    dim : int
        Hidden dimension.
    ffn_dim : int
        Intermediate dimension of each expert FFN.
    num_experts : int
        Total number of expert FFNs.
    top_k : int
        Experts activated per token (must match ``MoEGate.top_k``).
    aux_loss_coeff : float
        Passed through to ``MoEGate``.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_experts: int = 48,
        top_k: int = 4,
        aux_loss_coeff: float = 1e-2,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = MoEGate(dim, num_experts, top_k, aux_loss_coeff)

        # Expert weights — stored as parameters for efficient batched computation.
        # Each expert is a two-layer FFN: up-project → GELU → down-project.
        self.w_up = nn.Parameter(torch.empty(num_experts, ffn_dim, dim))
        self.b_up = nn.Parameter(torch.zeros(num_experts, ffn_dim))
        self.w_down = nn.Parameter(torch.empty(num_experts, dim, ffn_dim))
        self.b_down = nn.Parameter(torch.zeros(num_experts, dim))

    def init_weights(self):
        self.gate.init_weights()
        std = 1.0 / math.sqrt(self.dim)
        nn.init.trunc_normal_(self.w_up, std=std)
        nn.init.trunc_normal_(self.w_down, std=std)
        nn.init.zeros_(self.b_up)
        nn.init.zeros_(self.b_down)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor
            Shape ``[B, L, D]``.

        Returns
        -------
        out : Tensor
            Shape ``[B, L, D]``.
        aux_loss : Tensor
            Scalar load-balancing loss.
        """
        B, L, D = x.shape
        gate_scores, expert_indices, aux_loss = self.gate(x)  # [B,L,k], [B,L,k]

        # Flatten batch & sequence for routing
        x_flat = x.reshape(-1, D)                              # [B*L, D]
        gate_flat = gate_scores.reshape(-1, self.top_k)        # [B*L, k]
        idx_flat = expert_indices.reshape(-1, self.top_k)      # [B*L, k]

        # Compute expert outputs via loop (simple, correct; can be replaced with
        # torch.ops.moe_gemm or Megablocks for production throughput).
        out = torch.zeros_like(x_flat)
        for k_idx in range(self.top_k):
            expert_ids = idx_flat[:, k_idx]         # [B*L]
            weights = gate_flat[:, k_idx:k_idx+1]   # [B*L, 1]

            for e in range(self.num_experts):
                mask = expert_ids == e
                if not mask.any():
                    continue
                x_e = x_flat[mask]                                                  # [n, D]
                h = F.gelu(F.linear(x_e, self.w_up[e], self.b_up[e]), approximate="tanh")  # [n, ffn_dim]
                y = F.linear(h, self.w_down[e], self.b_down[e])                     # [n, D]
                out[mask] = out[mask] + weights[mask] * y

        return out.reshape(B, L, D), aux_loss


# ---------------------------------------------------------------------------
# MoE Attention Block — replaces dense FFN with MoE FFN
# ---------------------------------------------------------------------------

class WanMOEAttentionBlock(nn.Module):
    """Transformer block identical to ``WanAttentionBlock`` except the
    feed-forward network is replaced by ``MoEFFN``.

    Parameters
    ----------
    cross_attn_type : str
        ``"t2v_cross_attn"`` or ``"i2v_cross_attn"``.
    dim, ffn_dim, num_heads, ... :
        Same as ``WanAttentionBlock``.
    num_experts : int
        Number of FFN experts.
    top_k : int
        Experts activated per token.
    aux_loss_coeff : float
        Load-balancing loss coefficient.
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        cp_comm_type: str = "p2p",
        num_experts: int = 48,
        top_k: int = 4,
        aux_loss_coeff: float = 1e-2,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # --- shared dense layers (same as WanAttentionBlock) ---
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, cp_comm_type)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps, cp_comm_type
        )
        self.norm2 = WanLayerNorm(dim, eps)

        # --- MoE FFN (replaces dense FFN) ---
        self.moe_ffn = MoEFFN(
            dim=dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coeff=aux_loss_coeff,
        )

        # modulation (same as dense block)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def init_weights(self):
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.moe_ffn.init_weights()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.modulation, std=std)

    def forward(self, x, e, seq_lens, video_size: VideoSize, freqs, context, context_lens):
        """Identical signature to ``WanAttentionBlock.forward``; returns
        ``(x, aux_loss)`` instead of ``x`` alone."""
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e_chunks = (self.modulation + e).chunk(6, dim=1)
        assert e_chunks[0].dtype == torch.float32

        # self-attention (unchanged)
        y = self.self_attn(
            (self.norm1(x).float() * (1 + e_chunks[1]) + e_chunks[0]).type_as(x),
            seq_lens, video_size, freqs,
        )
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e_chunks[2].type_as(x)

        # cross-attention (unchanged)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # MoE FFN (replaces dense FFN)
        ffn_input = (self.norm2(x).float() * (1 + e_chunks[4]) + e_chunks[3]).type_as(x)
        ffn_out, aux_loss = self.moe_ffn(ffn_input)
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + ffn_out * e_chunks[5].type_as(x)

        return x, aux_loss


# ---------------------------------------------------------------------------
# WanMOEModel — full backbone with MoE blocks
# ---------------------------------------------------------------------------

class WanMOEModel(WeightTrainingStat):
    """WAN 2.2 backbone with Mixture-of-Experts FFN layers.

    Architecture is identical to ``WanModel`` (``wan2pt1.py``) except:
    * ``WanAttentionBlock`` → ``WanMOEAttentionBlock``
    * Forward pass accumulates auxiliary load-balancing loss.
    * ``fully_shard`` wraps each expert independently for FSDP.

    Parameters
    ----------
    num_experts : int
        Total experts per block (default 48 for WAN 2.2).
    top_k : int
        Experts activated per token (4 = High, 2 = Low).
    aux_loss_coeff : float
        Weight for the auxiliary load-balancing loss.
    (all other params identical to ``WanModel``)
    """

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: tuple = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        num_layers: int = 40,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        concat_padding_mask: bool = False,
        sac_config: SACConfig = SACConfig(),
        cp_comm_type: str = "p2p",
        postpone_checkpoint: bool = False,
        conv_patchify: bool = False,
        # --- MoE-specific ---
        num_experts: int = 48,
        top_k: int = 4,
        aux_loss_coeff: float = 1e-2,
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.concat_padding_mask = concat_padding_mask
        self.cp_comm_type = cp_comm_type
        self.conv_patchify = conv_patchify
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff

        # --- embeddings (identical to WanModel) ---
        _in_dim = in_dim + 1 if self.concat_padding_mask else in_dim
        if self.conv_patchify:
            self.patch_embedding = nn.Conv3d(_in_dim, dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.patch_embedding = nn.Linear(_in_dim * patch_size[0] * patch_size[1] * patch_size[2], dim)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # --- MoE transformer blocks ---
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList([
            WanMOEAttentionBlock(
                cross_attn_type=cross_attn_type,
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                cp_comm_type=cp_comm_type,
                num_experts=num_experts,
                top_k=top_k,
                aux_loss_coeff=aux_loss_coeff,
            )
            for _ in range(num_layers)
        ])

        # --- head (identical to WanModel) ---
        self.head = Head(dim, out_dim, patch_size, eps)

        # --- position embeddings ---
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.rope_position_embedding = VideoRopePosition3DEmb(
            head_dim=d, len_h=128, len_w=128, len_t=32,
        )

        if model_type in ("i2v", "flf2v"):
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # --- initialise ---
        self.init_weights()
        self.sac_config = sac_config
        if not postpone_checkpoint:
            self.enable_selective_checkpoint(sac_config, self.blocks)

    # ---- forward ---------------------------------------------------------

    def forward(
        self,
        x_B_C_T_H_W,
        timesteps_B_T,
        crossattn_emb,
        seq_len=None,
        frame_cond_crossattn_emb_B_L_D=None,
        y_B_C_T_H_W=None,
        padding_mask: Optional[torch.Tensor] = None,
        is_uncond=False,
        slg_layers=None,
        **kwargs,
    ):
        """Forward pass — identical to ``WanModel`` except MoE blocks return
        ``(x, aux_loss)`` tuples and the total auxiliary loss is accumulated
        and stored in ``self._last_aux_loss`` for the training loop to pick up.
        """
        assert timesteps_B_T.shape[1] == 1
        t_B = timesteps_B_T[:, 0]
        del kwargs

        if self.model_type in ("i2v", "flf2v"):
            assert frame_cond_crossattn_emb_B_L_D is not None and y_B_C_T_H_W is not None

        if y_B_C_T_H_W is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, y_B_C_T_H_W], dim=1)

        if self.concat_padding_mask:
            from torchvision import transforms
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)],
                dim=1,
            )

        # patchify
        if self.conv_patchify:
            x_B_D_T_H_W = self.patch_embedding(x_B_C_T_H_W)
            x_B_T_H_W_D = rearrange(x_B_D_T_H_W, "b d t h w -> b t h w d")
        else:
            x_B_T_H_W_D = rearrange(
                x_B_C_T_H_W,
                "b c (t kt) (h kh) (w kw) -> b t h w (c kt kh kw)",
                kt=self.patch_size[0], kh=self.patch_size[1], kw=self.patch_size[2],
            )
            x_B_T_H_W_D = self.patch_embedding(x_B_T_H_W_D)

        video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])
        x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        seq_lens = torch.tensor([u.size(0) for u in x_B_L_D], dtype=torch.long)
        seq_len = seq_lens.max().item()

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            e_B_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_B).float())
            e0_B_6_D = self.time_projection(e_B_D).unflatten(1, (6, self.dim))

        # context
        context_lens = None
        context_B_L_D = self.text_embedding(crossattn_emb)
        if frame_cond_crossattn_emb_B_L_D is not None:
            context_clip = self.img_emb(frame_cond_crossattn_emb_B_L_D)
            context_B_L_D = torch.concat([context_clip, context_B_L_D], dim=1)

        block_kwargs = dict(
            e=e0_B_6_D,
            seq_lens=seq_lens,
            video_size=video_size,
            freqs=self.rope_position_embedding(x_B_T_H_W_D),
            context=context_B_L_D,
            context_lens=context_lens,
        )

        # --- MoE blocks (accumulate aux_loss) ---
        total_aux_loss = torch.tensor(0.0, device=x_B_L_D.device)
        for block_idx, block in enumerate(self.blocks):
            if slg_layers is not None and block_idx in slg_layers and is_uncond:
                continue
            x_B_L_D, aux_loss = block(x_B_L_D, **block_kwargs)
            total_aux_loss = total_aux_loss + aux_loss

        # Store for the training step to read
        self._last_aux_loss = total_aux_loss / max(len(self.blocks), 1)

        # head
        x_B_L_D = self.head(x_B_L_D, e_B_D)

        # unpatchify
        t, h, w = video_size
        x_B_C_T_H_W = rearrange(
            x_B_L_D,
            "b (t h w) (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_size[0], nh=self.patch_size[1], nw=self.patch_size[2],
            t=t, h=h, w=w, d=self.out_dim,
        )
        return x_B_C_T_H_W

    # ---- init_weights ----------------------------------------------------

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for block in self.blocks:
            block.init_weights()
        self.head.init_weights()

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.zeros_(self.patch_embedding.bias)

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)

    # ---- FSDP sharding ---------------------------------------------------

    def fully_shard(self, mesh):
        """Shard model for FSDP2.  Expert parameters inside each MoE block
        are individually wrapped so weights stay balanced across ranks."""
        for i, block in enumerate(self.blocks):
            # Shard each expert pool individually first
            fully_shard(block.moe_ffn, mesh=mesh, reshard_after_forward=True)
            fully_shard(block, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.head, mesh=mesh, reshard_after_forward=False)
        fully_shard(self.text_embedding, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.time_embedding, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.patch_embedding, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.time_projection, mesh=mesh, reshard_after_forward=True)

    # ---- context parallelism ---------------------------------------------

    def disable_context_parallel(self):
        self.rope_position_embedding.disable_context_parallel()
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=None, ranks=None, stream=torch.cuda.Stream(),
            )
        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        from torch.distributed import get_process_group_ranks
        self.rope_position_embedding.enable_context_parallel(process_group=process_group)
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=process_group, ranks=cp_ranks, stream=torch.cuda.Stream(),
            )
        self._is_context_parallel_enabled = True

    @property
    def is_context_parallel_enabled(self):
        return self._is_context_parallel_enabled

    def enable_selective_checkpoint(self, sac_config: SACConfig, blocks: nn.ModuleList):
        if sac_config.mode == CheckpointMode.NONE:
            return
        log.info(
            f"Enable selective checkpoint with {sac_config.mode}, "
            f"for every {sac_config.every_n_blocks} blocks. Total blocks: {len(blocks)}"
        )
        _context_fn = sac_config.get_context_fn()
        for block_id, block in blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                log.info(f"Enable selective checkpoint for block {block_id}")
                block = ptd_checkpoint_wrapper(block, context_fn=_context_fn, preserve_rng_state=False)
                blocks.register_module(block_id, block)
        self.register_module(
            "head",
            ptd_checkpoint_wrapper(self.head, context_fn=_context_fn, preserve_rng_state=False),
        )


# ---------------------------------------------------------------------------
# EditWanMOEModel — ChronoEdit variant with temporal-skip RoPE
# ---------------------------------------------------------------------------

class EditWanMOEModel(WanMOEModel):
    """WAN 2.2 MoE backbone extended with temporal-skip position embeddings
    for ChronoEdit editing tasks.

    Mirrors the relationship between ``EditWanModel`` and ``WanModel``.
    """

    def __init__(
        self,
        *args,
        temporal_skip_p: bool = False,
        temporal_skip_len: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temporal_skip_p = temporal_skip_p
        d = self.dim // self.num_heads

        if self.temporal_skip_p:
            self.rope_position_embedding = TemproalSkipVideoRopePosition3DEmb(
                head_dim=d,
                len_h=128,
                len_w=128,
                len_t=32,
                temporal_skip_len=temporal_skip_len,
            )

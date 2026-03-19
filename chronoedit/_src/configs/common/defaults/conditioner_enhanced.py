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
Dual-encoder and enhanced conditioner configurations for ChronoEdit.

This module provides conditioner configs that combine:
  - **SigLIP** instead of CLIP for image conditioning
  - **Dual text streams** (primary + secondary) for richer semantic understanding
  - **Gemma 3** as an alternative text encoder (drop-in for UMT5)

Configs registered:
  - ``i2v_conditioner_siglip``             — SigLIP + single text (UMT5/Gemma3)
  - ``i2v_conditioner_dual_text``          — CLIP + dual text
  - ``i2v_conditioner_dual_text_siglip``   — SigLIP + dual text (full upgrade)

Network changes required:
  - SigLIP configs need ``img_dim=1152`` in the network
  - Dual-text configs need ``secondary_text_dim=4096`` and ``use_dual_text=True``
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from chronoedit._ext.imaginaire.lazy_config import LazyCall as L
from chronoedit._ext.imaginaire.lazy_config import LazyDict
from chronoedit._src.modules.conditioner import (
    BaseCondition,
    GeneralConditioner,
    ReMapkey,
    T2VCondition,
    TextAttr,
    TextAttrEmptyStringDrop,
)
from chronoedit._src.models.wan_i2v_model import WAN2PT1_I2V_COND_LATENT_KEY
from chronoedit._src.modules.clip import Wan2pt1CLIPEmb
from chronoedit._src.modules.siglip import SigLIPEmb
from chronoedit._src.utils.context_parallel import broadcast_split_tensor


# ---------------------------------------------------------------------------
# Dual-encoder condition dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualTextImg2VidCondition(T2VCondition):
    """Condition with dual text streams and image/video conditioning.

    Fields
    ------
    crossattn_emb : Tensor | None
        Primary text embeddings (e.g., UMT5 or Gemma 3), shape ``[B, L1, D1]``.
    crossattn_emb_secondary : Tensor | None
        Secondary text embeddings (e.g., second encoder), shape ``[B, L2, D2]``.
    frame_cond_crossattn_emb_B_L_D : Tensor | None
        CLIP / SigLIP image embeddings, shape ``[B, L, D_img]``.
    y_B_C_T_H_W : Tensor | None
        Conditional video latents + mask.
    """

    crossattn_emb_secondary: Optional[torch.Tensor] = None
    frame_cond_crossattn_emb_B_L_D: Optional[torch.Tensor] = None
    y_B_C_T_H_W: Optional[torch.Tensor] = None

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> BaseCondition:
        if self.is_broadcasted:
            return self

        y_B_C_T_H_W = self.y_B_C_T_H_W
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["y_B_C_T_H_W"] = None
        new_condition = T2VCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )
        kwargs = new_condition.to_dict(skip_underscore=False)
        if process_group is not None:
            y_B_C_T_H_W = broadcast_split_tensor(y_B_C_T_H_W, seq_dim=2, process_group=process_group)
        kwargs["y_B_C_T_H_W"] = y_B_C_T_H_W
        return type(self)(**kwargs)


class DualTextImg2VidConditioner(GeneralConditioner):
    """Conditioner that produces ``DualTextImg2VidCondition`` from dual text + image embedders."""

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> DualTextImg2VidCondition:
        output = super()._forward(batch, override_dropout_rate)
        return DualTextImg2VidCondition(**output)


# ---------------------------------------------------------------------------
# Secondary text attribute (for the second text encoder stream)
# ---------------------------------------------------------------------------


class SecondaryTextAttr(TextAttr):
    """Text attribute that outputs to ``crossattn_emb_secondary`` instead of ``crossattn_emb``."""

    def forward(self, token: torch.Tensor):
        return {"crossattn_emb_secondary": token}

    def details(self) -> str:
        return "Output key: [crossattn_emb_secondary]"


class SecondaryTextAttrEmptyStringDrop(TextAttrEmptyStringDrop):
    """Text attribute with empty-string dropout for the secondary stream."""

    def forward(self, token: torch.Tensor):
        return {"crossattn_emb_secondary": token}

    def details(self) -> str:
        return "Output key: [crossattn_emb_secondary]"


# ---------------------------------------------------------------------------
# Config 1: SigLIP + single text encoder (UMT5 or Gemma3)
# ---------------------------------------------------------------------------

SigLIPConditionerConfig: LazyDict = L(DualTextImg2VidConditioner)(
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    wanclip=L(SigLIPEmb)(
        input_key=["images", "video", WAN2PT1_I2V_COND_LATENT_KEY],
        dropout_rate=0.0,
        num_token=257,
        dtype="bfloat16",
        model_name="google/siglip-so400m-patch14-384",
        image_size=384,
    ),
)

SigLIPConditionerEmptyStringDropConfig: LazyDict = L(DualTextImg2VidConditioner)(
    text=L(TextAttrEmptyStringDrop)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    wanclip=L(SigLIPEmb)(
        input_key=["images", "video", WAN2PT1_I2V_COND_LATENT_KEY],
        dropout_rate=0.0,
        num_token=257,
        dtype="bfloat16",
        model_name="google/siglip-so400m-patch14-384",
        image_size=384,
    ),
)


# ---------------------------------------------------------------------------
# Config 2: CLIP + dual text encoders
# ---------------------------------------------------------------------------

DualTextCLIPConditionerConfig: LazyDict = L(DualTextImg2VidConditioner)(
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    text_secondary=L(SecondaryTextAttr)(
        input_key=["secondary_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    wanclip=L(Wan2pt1CLIPEmb)(
        input_key=["images", "video", WAN2PT1_I2V_COND_LATENT_KEY],
        dropout_rate=0.0,
        dtype="bfloat16",
    ),
)


# ---------------------------------------------------------------------------
# Config 3: SigLIP + dual text encoders (full upgrade)
# ---------------------------------------------------------------------------

DualTextSigLIPConditionerConfig: LazyDict = L(DualTextImg2VidConditioner)(
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    text_secondary=L(SecondaryTextAttr)(
        input_key=["secondary_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    wanclip=L(SigLIPEmb)(
        input_key=["images", "video", WAN2PT1_I2V_COND_LATENT_KEY],
        dropout_rate=0.0,
        num_token=257,
        dtype="bfloat16",
        model_name="google/siglip-so400m-patch14-384",
        image_size=384,
    ),
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_enhanced_conditioner():
    cs = ConfigStore.instance()

    # SigLIP-based (single text)
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="i2v_conditioner_siglip",
        node=SigLIPConditionerConfig,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="i2v_conditioner_siglip_empty_string_drop",
        node=SigLIPConditionerEmptyStringDropConfig,
    )

    # Dual text + CLIP
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="i2v_conditioner_dual_text",
        node=DualTextCLIPConditionerConfig,
    )

    # Dual text + SigLIP (full upgrade)
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="i2v_conditioner_dual_text_siglip",
        node=DualTextSigLIPConditionerConfig,
    )

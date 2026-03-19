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
SigLIP vision encoder for ChronoEdit.

Replaces the CLIP ViT-H/14 (1280-dim, 257 tokens) with SigLIP SO400M
(1152-dim, 729 tokens for 384px or 257 tokens for 224px).

SigLIP (Sigmoid Loss for Language-Image Pre-training) provides better
image-text alignment than CLIP, particularly for fine-grained edit tasks.

Key difference from CLIP:
  - Output dim: 1152 vs 1280 → requires ``img_dim`` parameter in network config
  - No CLS token: all tokens are patch tokens (SigLIP uses global average pooling
    for classification, but we use all patch tokens for conditioning)
  - Better semantic alignment via sigmoid cross-entropy loss (vs softmax in CLIP)

Integration points:
  - ``SigLIPModel`` — loads and wraps a pretrained SigLIP vision encoder
  - ``SigLIPEmb`` — drop-in replacement for ``Wan2pt1CLIPEmb`` in conditioner configs
  - Network configs need ``img_dim=1152`` (default was 1280 for CLIP)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from chronoedit._ext.imaginaire.utils import distributed, log
from chronoedit._src.modules.conditioner import AbstractEmbModel

__all__ = ["SigLIPModel", "SigLIPEmb"]


class SigLIPModel:
    """Wrapper around a SigLIP vision encoder for frame conditioning.

    Loads a pretrained SigLIP model (default: SO400M/14@384) via
    ``transformers`` and provides the ``visual()`` method that returns
    patch-level features.

    Parameters
    ----------
    dtype : torch.dtype
        Precision for inference.
    device : str
        Target device.
    model_name : str
        HuggingFace model identifier for SigLIP.
    image_size : int
        Expected input resolution.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        model_name: str = "google/siglip-so400m-patch14-384",
        image_size: int = 384,
    ):
        self.dtype = dtype
        self.device = device
        self.image_size = image_size
        self.model_name = model_name

        try:
            from transformers import SiglipVisionModel, SiglipImageProcessor
        except ImportError:
            raise ImportError(
                "transformers>=4.45.0 is required for SigLIP support. "
                "Install with: pip install transformers>=4.45.0"
            )

        log.info(f"Loading SigLIP vision model from {model_name}")
        self.vision_model = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device).eval()
        self.vision_model.requires_grad_(False)

        self.processor = SiglipImageProcessor.from_pretrained(model_name)

        # Determine output dimensions from the model config
        self.hidden_size = self.vision_model.config.hidden_size  # 1152 for SO400M
        self.num_patches = (image_size // self.vision_model.config.patch_size) ** 2

        log.info(
            f"SigLIP ready: hidden_size={self.hidden_size}, "
            f"num_patches={self.num_patches}, image_size={image_size}"
        )

        # Normalization transform matching SigLIP preprocessing
        mean = self.processor.image_mean
        std = self.processor.image_std
        self.normalize = T.Normalize(mean=mean, std=std)

    @torch.inference_mode()
    def visual(self, images_B_C_H_W: torch.Tensor, return_cls: bool = False) -> torch.Tensor:
        """Extract patch-level visual features from images.

        Parameters
        ----------
        images_B_C_H_W : torch.Tensor
            Input images in ``[-1, 1]`` range, shape ``[B, 3, H, W]``.
        return_cls : bool
            If True, prepend a mean-pooled CLS-like token (SigLIP has no native CLS).

        Returns
        -------
        torch.Tensor
            Shape ``[B, num_patches(+1 if return_cls), hidden_size]``.
        """
        # Resize to model's expected resolution
        size = (self.image_size,) * 2
        images = F.interpolate(images_B_C_H_W, size=size, mode="bicubic", align_corners=False)

        # Normalize: input is [-1, 1], convert to [0, 1] then apply ImageNet norm
        images = self.normalize(images.mul_(0.5).add_(0.5))

        with torch.amp.autocast("cuda", dtype=self.dtype):
            outputs = self.vision_model(pixel_values=images)
            # SigLIP returns last_hidden_state: [B, num_patches, hidden_size]
            patch_features = outputs.last_hidden_state  # [B, num_patches, 1152]

        if return_cls:
            # Create a CLS-like token via mean pooling (SigLIP has no native CLS)
            cls_token = patch_features.mean(dim=1, keepdim=True)  # [B, 1, 1152]
            return torch.cat([cls_token, patch_features], dim=1)   # [B, num_patches+1, 1152]

        return patch_features


class SigLIPEmb(AbstractEmbModel):
    """SigLIP-based frame conditioning embedder.

    Drop-in replacement for ``Wan2pt1CLIPEmb`` in conditioner configs.
    Produces ``frame_cond_crossattn_emb_B_L_D`` and ``y_B_C_T_H_W``
    with the same dictionary keys so the rest of the pipeline is unchanged.

    Key differences from Wan2pt1CLIPEmb:
      - Output dim is 1152 (SigLIP SO400M) instead of 1280 (CLIP ViT-H/14)
      - Network's ``img_emb = MLPProj(img_dim, dim)`` needs ``img_dim=1152``
      - Token count depends on image_size: 729 for 384px, 256 for 224px
        (+ 1 if using synthetic CLS token)

    Parameters
    ----------
    input_key : list[str]
        Keys to extract from batch: ``["images", "video", <latent_key>]``.
    dropout_rate : float
        Dropout rate for classifier-free guidance.
    num_token : int
        Number of tokens in the output embedding (e.g., 730 for 384px + CLS).
        Set to 257 for backward compat with CLIP token count (uses CLS + 256 patches at 224px).
    dtype : str
        Precision string.
    model_name : str
        HuggingFace SigLIP model identifier.
    image_size : int
        Input resolution for the SigLIP model.
    """

    def __init__(
        self,
        input_key: List[str],
        dropout_rate: Optional[float] = 0.0,
        num_token: int = 257,
        dtype: str = "bfloat16",
        model_name: str = "google/siglip-so400m-patch14-384",
        image_size: int = 384,
    ):
        super().__init__()
        self.num_token = num_token
        self.image_size = image_size
        self.model_name = model_name

        self.siglip_model = SigLIPModel(
            dtype={
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[dtype],
            model_name=model_name,
            image_size=image_size,
        )
        self.model_dim = self.siglip_model.hidden_size  # 1152

        self._input_key = input_key
        self._output_key = None
        self._dropout_rate = dropout_rate
        self.dtype_torch = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

    def random_dropout_input(
        self,
        in_tensor: Optional[torch.Tensor] = None,
        dropout_rate: Optional[float] = None,
        key: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        if in_tensor is None:
            return None
        return super().random_dropout_input(in_tensor, dropout_rate, key)

    def forward(
        self,
        image_tensor: Optional[torch.Tensor] = None,
        video_tensor: Optional[torch.Tensor] = None,
        media_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract SigLIP embeddings from input frames.

        Same interface as ``Wan2pt1CLIPEmb.forward()``.

        Returns
        -------
        dict
            ``frame_cond_crossattn_emb_B_L_D``: [B, num_token, model_dim]
            ``y_B_C_T_H_W``: concatenated mask + media_latents
        """
        b, _, latent_f, latent_h, latent_w = media_latents.shape
        mask = torch.zeros(b, 4, latent_f, latent_h, latent_w).type_as(media_latents).to(self.dtype_torch)

        if image_tensor is not None:
            # Image-only: zero embeddings (same as CLIP path)
            context_B_L_D = torch.zeros(b, self.num_token, self.model_dim).type_as(media_latents).to(self.dtype_torch)
        else:
            # Video: extract features from first frame
            first_frame_B_C_H_W = video_tensor[:, :, 0, :, :]
            with torch.no_grad():
                # Use return_cls=True to get CLS + patches
                context_B_L_D = self.siglip_model.visual(
                    first_frame_B_C_H_W, return_cls=True
                ).to(self.dtype_torch)

            # Truncate or pad to num_token
            if context_B_L_D.shape[1] > self.num_token:
                context_B_L_D = context_B_L_D[:, : self.num_token, :]
            elif context_B_L_D.shape[1] < self.num_token:
                pad = torch.zeros(
                    b,
                    self.num_token - context_B_L_D.shape[1],
                    self.model_dim,
                    device=context_B_L_D.device,
                    dtype=self.dtype_torch,
                )
                context_B_L_D = torch.cat([context_B_L_D, pad], dim=1)

            mask[:, :, :1] = 1.0

        y = torch.concat([mask, media_latents.to(self.dtype_torch)], dim=1)

        return {"frame_cond_crossattn_emb_B_L_D": context_B_L_D, "y_B_C_T_H_W": y}

    def details(self) -> str:
        output_key = ["frame_cond_crossattn_emb_B_L_D", "y_B_C_T_H_W"]
        return (
            f"Input key: {self.input_key} \n\tOutput key: {output_key}"
            f"\n\tModel: {self.model_name} ({self.model_dim}D, {self.num_token} tokens)"
        )

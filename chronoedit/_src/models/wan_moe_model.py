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
WAN 2.2 MoE diffusion model classes.

Provides:
  - ``WANMOEDiffusionModel``    — T2V MoE model (extends WANDiffusionModel).
  - ``I2V_Edit_Wan2pt2MoeModel`` — ChronoEdit MoE editing model.

The key difference from the WAN 2.1 models is that the training step adds
the auxiliary load-balancing loss produced by the MoE gate to the flow
matching objective.
"""

from __future__ import annotations

from typing import Dict, Tuple

import attrs
import torch
from einops import rearrange
from torch import Tensor

from chronoedit._src.models.wan_t2v_model import (
    DataType,
    T2VCondition,
    T2VModelConfig,
    WANDiffusionModel,
)
from chronoedit._src.utils.misc import sync_timer
from chronoedit._src.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from chronoedit._ext.imaginaire.utils import misc, log

WAN2PT1_I2V_COND_LATENT_KEY = "i2v_WAN2PT1_cond_latents"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@attrs.define(slots=False)
class MoET2VModelConfig(T2VModelConfig):
    """Extended config for MoE models — adds an ``aux_loss_weight`` knob
    that controls how much the load-balancing loss contributes to the total."""

    aux_loss_weight: float = 1.0


@attrs.define(slots=False)
class MoEEditModelConfig(MoET2VModelConfig):
    """MoE config for the ChronoEdit editing model."""

    is_video_prior: bool = False


# ---------------------------------------------------------------------------
# WANMOEDiffusionModel — T2V MoE
# ---------------------------------------------------------------------------

class WANMOEDiffusionModel(WANDiffusionModel):
    """Text-to-Video diffusion model backed by a WAN 2.2 MoE backbone.

    Inherits all training infrastructure from ``WANDiffusionModel`` and only
    overrides ``training_step`` to incorporate the auxiliary load-balancing
    loss produced by the MoE gating mechanism.
    """

    def __init__(self, config: MoET2VModelConfig):
        super().__init__(config)
        self._aux_loss_weight = getattr(config, "aux_loss_weight", 1.0)

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Same as ``WANDiffusionModel.training_step`` but adds MoE auxiliary loss."""
        self._update_train_stats(data_batch)

        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.flow_matching_kwargs)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.flow_matching_kwargs)
        t_B = rearrange(t_B, "b -> b 1")

        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )

        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.flow_matching_kwargs)
        sigmas = self.rectified_flow.get_sigmas(timesteps, self.flow_matching_kwargs)
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas
        )

        vt_pred_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps.to(**self.tensor_kwargs),
            **condition.to_dict(),
        )

        # Flow matching loss
        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.flow_matching_kwargs)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2,
            dim=list(range(1, vt_pred_B_C_T_H_W.dim())),
        )
        fm_loss = torch.mean(time_weights_B * per_instance_loss)

        # MoE auxiliary loss (stored by the network's forward pass)
        aux_loss = getattr(self.net, "_last_aux_loss", torch.tensor(0.0, device=fm_loss.device))
        total_loss = fm_loss + self._aux_loss_weight * aux_loss

        output_batch = {
            "edm_loss": fm_loss,
            "moe_aux_loss": aux_loss,
            "total_loss": total_loss,
        }
        return output_batch, total_loss


# ---------------------------------------------------------------------------
# I2V_Edit_Wan2pt2MoeModel — ChronoEdit MoE editing model
# ---------------------------------------------------------------------------

class I2V_Edit_Wan2pt2MoeModel(WANMOEDiffusionModel):
    """ChronoEdit editing model backed by WAN 2.2 MoE.

    Mirrors ``I2V_Edit_Wan2pt1Model`` but inherits from ``WANMOEDiffusionModel``
    so that the auxiliary loss is included during training.
    """

    def __init__(self, config: MoEEditModelConfig):
        super().__init__(config)

    # ----- data & conditioning (same as I2V_Edit_Wan2pt1Model) -----

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, T2VCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]

        # Build 5-frame edit sequence: [first_frame, last_frame × 4]
        last_frame = raw_state[:, :, -1:, :, :]
        last_frame_repeated = last_frame.repeat(1, 1, 4, 1, 1)
        raw_state_edit = torch.cat([raw_state[:, :, :1, :, :], last_frame_repeated], dim=2)
        latent_state = self.encode(raw_state_edit).contiguous().float()

        is_video_prior = data_batch.get("is_video_prior", False)
        if is_video_prior:
            raw_state_video = raw_state[:, :, :-1, :, :]
            latent_state_video = self.encode(raw_state_video).contiguous().float()
            latent_state = torch.cat([latent_state_video, latent_state[:, :, 1:, :, :]], dim=2)
            raw_state = torch.cat([raw_state_video, last_frame_repeated], dim=2)
        else:
            raw_state = raw_state_edit

        if WAN2PT1_I2V_COND_LATENT_KEY not in data_batch:
            conditional_content = torch.zeros_like(raw_state).to(**self.tensor_kwargs)
            if not is_image_batch:
                conditional_content[:, :, 0] = raw_state[:, :, 0]
            data_batch[WAN2PT1_I2V_COND_LATENT_KEY] = self.encode(conditional_content).contiguous()

        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        return raw_state, latent_state, condition

    # ----- sampling (same as I2V_Edit_Wan2pt1Model) -----

    @sync_timer("WANMOEDiffusionModel: generate_samples_from_batch")
    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)
        timesteps = self.sample_scheduler.timesteps

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        latents = noise

        if self.net.is_context_parallel_enabled:
            latents = broadcast_split_tensor(latents, seq_dim=2, process_group=self.get_context_parallel_group())

        with sync_timer(f"WANMOEDiffusionModel: generate_samples_from_batch: {num_steps} diffusion_steps"):
            for _, t in enumerate(timesteps):
                latent_model_input = latents
                timestep = torch.stack([t])
                velocity_field_pred = x0_fn(latent_model_input, timestep.unsqueeze(0))
                temp_x0 = self.sample_scheduler.step(
                    velocity_field_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                    return_dict=False, generator=seed_g,
                )[0]
                latents = temp_x0.squeeze(0)

                # Drop-video-step for temporal reasoning
                if (
                    "drop_video_step" in kwargs
                    and kwargs["drop_video_step"] is not None
                    and t == timesteps[kwargs["drop_video_step"]]
                ):
                    if self.net.is_context_parallel_enabled:
                        latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

                    latents = latents[:, :, [0, -1]]

                    if self.net.is_context_parallel_enabled:
                        latents = broadcast_split_tensor(
                            latents, seq_dim=2, process_group=self.get_context_parallel_group()
                        )

                    if self.sample_scheduler.model_outputs:
                        for i in range(len(self.sample_scheduler.model_outputs)):
                            if latents.shape[-3] != self.sample_scheduler.model_outputs[i].shape[-3]:
                                if self.net.is_context_parallel_enabled:
                                    self.sample_scheduler.model_outputs[i] = cat_outputs_cp(
                                        self.sample_scheduler.model_outputs[i],
                                        seq_dim=3, cp_group=self.get_context_parallel_group(),
                                    )
                                self.sample_scheduler.model_outputs[i] = self.sample_scheduler.model_outputs[i][:, :, :, [0, -1]]
                                if self.net.is_context_parallel_enabled:
                                    self.sample_scheduler.model_outputs[i] = broadcast_split_tensor(
                                        self.sample_scheduler.model_outputs[i],
                                        seq_dim=3, process_group=self.get_context_parallel_group(),
                                    )

                        if self.sample_scheduler.last_sample is not None:
                            self.sample_scheduler.last_sample = latents

                    data_batch["i2v_WAN2PT1_cond_latents"] = data_batch["i2v_WAN2PT1_cond_latents"][:, :, [0, -1]]
                    data_batch["video"] = data_batch["video"][:, :, [0, -1]]
                    data_batch["is_video_prior"] = False

                    log.info(
                        f"rank {torch.distributed.get_rank()}: "
                        f"{data_batch['video'].shape}, "
                        f"{data_batch['i2v_WAN2PT1_cond_latents'].shape}, "
                        f"{latents.shape}, {data_batch['is_video_prior']}"
                    )

                    x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        return latents

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
WAN 2.2 MoE network configurations for ChronoEdit.

Registers two network variants:
  - ``wan2pt2_moe_14B_high_edit``: 48 experts, top-4 routing (High quality).
  - ``wan2pt2_moe_14B_low_edit``:  48 experts, top-2 routing (Lower compute).
"""

from hydra.core.config_store import ConfigStore

from chronoedit._ext.imaginaire.lazy_config import LazyCall as L
from chronoedit._ext.imaginaire.lazy_config import LazyDict
from chronoedit._src.modules.selective_activation_checkpoint import SACConfig
from chronoedit._src.networks.wan2pt2_moe import EditWanMOEModel


# WAN 2.2 MoE High — 48 experts, top-4 routing
WAN2PT2_MOE_14B_HIGH_EDIT: LazyDict = L(EditWanMOEModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="block_wise"),
    postpone_checkpoint=False,
    # MoE-specific
    num_experts=48,
    top_k=4,
    aux_loss_coeff=1e-2,
)

# WAN 2.2 MoE Low — 48 experts, top-2 routing (lower compute)
WAN2PT2_MOE_14B_LOW_EDIT: LazyDict = L(EditWanMOEModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="block_wise"),
    postpone_checkpoint=False,
    # MoE-specific
    num_experts=48,
    top_k=2,
    aux_loss_coeff=1e-2,
)


def edit_register_net_moe():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="wan2pt2_moe_14B_high_edit", node=WAN2PT2_MOE_14B_HIGH_EDIT)
    cs.store(group="net", package="model.config.net", name="wan2pt2_moe_14B_low_edit", node=WAN2PT2_MOE_14B_LOW_EDIT)

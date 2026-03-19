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
Experiment configs for ChronoEdit fine-tuning on WAN 2.2 MoE.

Defines two experiments:
  - ``edit_moe_14B_high_skip_pe8``  — WAN 2.2 High (48 experts, top-4).
  - ``edit_moe_14B_low_skip_pe8``   — WAN 2.2 Low  (48 experts, top-2).

Usage:
  torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \\
      --config=chronoedit/_src/configs/chronoedit/config.py \\
      -- experiment="edit_moe_14B_high_skip_pe8"
"""

from hydra.core.config_store import ConfigStore
from chronoedit._ext.imaginaire.lazy_config import LazyDict

cs = ConfigStore.instance()


# ---------------------------------------------------------------------------
# WAN 2.2 MoE HIGH — 48 experts, top-4 routing
# ---------------------------------------------------------------------------

CHRONO_EDIT_MOE_14B_HIGH: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/wan2pt1_i2v_14B_res480p_16fps",
            {"override /data_train": "mock_video"},
            {"override /model": "fsdp_wan2pt2_moe_edit"},
            {"override /net": "wan2pt2_moe_14B_high_edit"},
            {"override /conditioner": "i2v_conditioner_empty_string_drop"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling_edit",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        upload_reproducible_setup=False,
        model=dict(
            config=dict(
                shift=5,
                train_time_weight="uniform",
                aux_loss_weight=1.0,
                net=dict(
                    temporal_skip_p=True,
                    temporal_skip_len=8,
                    num_experts=48,
                    top_k=4,
                    aux_loss_coeff=1e-2,
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=1e-3,
        ),
        checkpoint=dict(
            save_iter=500,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
            # TODO: Update with actual WAN 2.2 MoE checkpoint path when available
            load_path="checkpoints/Wan2.2-MoE-I2V-14B.dcp",
            load_training_state=False,
            strict_resume=False,
        ),
        job=dict(
            group="chronoedit_moe",
            name="edit_moe_14B_high_skip_pe8",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        trainer=dict(
            timestamp_seed=True,
            max_iter=500000,
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                    guidance=[5],
                ),
            ),
        ),
        dataloader_train=dict(),
    )
)


# ---------------------------------------------------------------------------
# WAN 2.2 MoE LOW — 48 experts, top-2 routing
# ---------------------------------------------------------------------------

CHRONO_EDIT_MOE_14B_LOW: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/edit_moe_14B_high_skip_pe8",
            {"override /net": "wan2pt2_moe_14B_low_edit"},
            "_self_",
        ],
        model=dict(
            config=dict(
                net=dict(
                    top_k=2,
                ),
            ),
        ),
        optimizer=dict(
            lr=2e-5,
        ),
        job=dict(
            group="chronoedit_moe",
            name="edit_moe_14B_low_skip_pe8",
        ),
    )
)


# ---------------------------------------------------------------------------
# Register experiments
# ---------------------------------------------------------------------------

cs.store(
    group="experiment",
    package="_global_",
    name="edit_moe_14B_high_skip_pe8",
    node=CHRONO_EDIT_MOE_14B_HIGH,
)

cs.store(
    group="experiment",
    package="_global_",
    name="edit_moe_14B_low_skip_pe8",
    node=CHRONO_EDIT_MOE_14B_LOW,
)

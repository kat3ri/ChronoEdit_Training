# 🏗️ Architecture Deep Dive: Training Pipeline, Model Generalizability & T2V Module

This document provides a comprehensive technical summary of how ChronoEdit's full model training is organized, answers whether the configuration system supports arbitrary base models beyond WAN, clarifies support for image vs. video vs. edit-based models, and explains the role of the T2V (Text-to-Video) module within an edit-based environment.

---

## Table of Contents

1. [Full Model Training Setup](#1-full-model-training-setup)
2. [Configuration System & Base Model Generalizability](#2-configuration-system--base-model-generalizability)
3. [Video, Image, and Edit Model Support](#3-video-image-and-edit-model-support)
4. [The T2V Module: Purpose and Role in Editing](#4-the-t2v-module-purpose-and-role-in-editing)
5. [Component Coupling Summary](#5-component-coupling-summary)

---

## 1. Full Model Training Setup

### Overview

ChronoEdit's training infrastructure is inherited from [Cosmos 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5/) and built on the **Imaginaire** framework (`chronoedit/_ext/imaginaire/`). It uses [Hydra](https://hydra.cc/) for configuration management, supports distributed training via **DDP** and **FSDP2**, and uses **rectified flow matching** as the diffusion training objective.

### Training Entry Point

Training is launched via `scripts/train.py` using `torchrun`:

```bash
torchrun --nproc_per_node=4 --master_port=12341 \
  -m scripts.train \
  --config=chronoedit/_src/configs/chronoedit/config.py \
  -- experiment="edit_14B_skip_pe8_mock"
```

The `scripts/train.py` script:
1. Parses the Hydra config (resolving all defaults and experiment overrides).
2. Initializes distributed training (NCCL backend, process groups for data/context parallelism).
3. Instantiates the model from config (including tokenizer, conditioner, network, EMA).
4. Optionally loads a pretrained checkpoint (DCP or consolidated `.pth`).
5. Delegates to `ImaginaireTrainer.train(model, dataloader_train, dataloader_val)`.

### Training Loop (Single Iteration)

Inside `WANDiffusionModel.training_step()` (defined in `chronoedit/_src/models/wan_t2v_model.py`):

```
1. Fetch data batch → normalize video pixels to [-1, 1]
2. Encode video/images through the VAE tokenizer → latent tensor x₀
3. Build condition via conditioner (T5 text embeddings, optional CLIP image features, fps, padding masks)
4. Sample random timestep t and noise ε
5. Interpolate: x_t = (1 - σ(t)) · x₀ + σ(t) · ε   (rectified flow)
6. Forward pass: v_pred = net(x_t, t, condition)
7. Compute loss: MSE(v_pred, v_target) weighted by time-dependent weighting
8. Backward pass + optimizer step + EMA update
```

### Model Class Hierarchy

```
ImaginaireModel                              (chronoedit/_ext/imaginaire/model.py)
  └── WANDiffusionModel                      (chronoedit/_src/models/wan_t2v_model.py)
        ├── I2VWan2pt1Model                  (chronoedit/_src/models/wan_i2v_model.py)
        └── I2V_Edit_Wan2pt1Model            (chronoedit/_src/models/chronoedit_14b_edit_model.py)
```

- **`WANDiffusionModel`** — The base T2V diffusion model. Handles tokenizer setup, rectified flow training, EMA, FSDP sharding, LoRA injection, and inference sampling.
- **`I2VWan2pt1Model`** — Extends T2V for image-to-video by conditioning on the first frame (encoded as a latent and concatenated).
- **`I2V_Edit_Wan2pt1Model`** — The ChronoEdit editing model. Takes an input image (first frame) and an edited image (last frame), repeats the last frame 4× to form a 5-frame sequence `[first, last, last, last, last]`, and trains the model to denoise the editing trajectory between them.

### Key Training Components

| Component | Location | Role |
|-----------|----------|------|
| **Tokenizer (VAE)** | `chronoedit/_src/tokenizers/wan2pt1.py` | Encodes video/images to 16-channel latent space |
| **Conditioner** | `chronoedit/_src/modules/conditioner.py` | Produces text (T5/UMT5) and image (CLIP) conditioning |
| **Network (DiT)** | `chronoedit/_src/networks/wan2pt1.py`, `chronoedit_14b.py` | 40-layer transformer with 3D RoPE, cross-attention |
| **Scheduler** | `chronoedit/_src/schedulers/rectified_flow.py` | Flow matching objective with configurable time weighting |
| **EMA** | `chronoedit/_ext/imaginaire/utils/ema.py` | Power-function EMA for model weight averaging |
| **Checkpointer** | `chronoedit/_src/checkpointer/dcp.py` | Distributed checkpoint save/load (DCP format) |

---

## 2. Configuration System & Base Model Generalizability

### Config Architecture

The configuration system uses Hydra's `ConfigStore` with composable defaults. Each training config (in `chronoedit/_src/configs/`) defines a set of swappable component groups:

```python
# From chronoedit/_src/configs/chronoedit/config.py
defaults: [
    "_self_",
    {"data_train": "mock"},           # Dataset configuration
    {"optimizer": "adamw"},            # Optimizer
    {"scheduler": "lambdalinear"},     # LR scheduler
    {"model": "ddp"},                  # Model wrapper (DDP/FSDP)
    {"net": None},                     # Network architecture
    {"conditioner": "i2v_conditioner"},# Conditioning strategy
    {"ema": "power"},                  # EMA configuration
    {"tokenizer": "wan2pt1_tokenizer"},# VAE tokenizer
    {"checkpoint": "local"},           # Checkpoint strategy
    {"experiment": None},              # Experiment-specific overrides
]
```

### Is It Set Up to Train on Any Base Model?

**The config system is designed to be modular, but the current implementations are specifically built for WAN 2.1 models.** Here is the nuanced answer:

#### What IS Configurable (Model-Agnostic Components)

These components use abstract interfaces and can be swapped without modifying model code:

- **Tokenizer**: Defined via the `BaseVAE` abstract class (`chronoedit/_src/tokenizers/base_vae.py`). Any VAE implementing `encode()`, `decode()`, `spatial_compression_factor`, and `latent_ch` can be plugged in via config.
- **Conditioner**: Defined via `BaseCondition` / `T2VCondition` dataclasses (`chronoedit/_src/modules/conditioner.py`). The conditioner is a composable pipeline of text/image embedding modules.
- **Scheduler (Flow Matching)**: `RectifiedFlow` (`chronoedit/_src/schedulers/rectified_flow.py`) is architecture-agnostic — it only requires a callable velocity field.
- **Optimizer, EMA, Checkpointer, Data Loaders**: All fully generic.

#### What IS NOT Easily Swappable (WAN-Coupled Components)

- **Network Architecture**: The transformer backbone (`WanModel` in `chronoedit/_src/networks/wan2pt1.py` and `EditWanModel` in `chronoedit/_src/networks/chronoedit_14b.py`) is WAN-specific. There is no `BaseNetwork` abstraction. The architecture hard-codes:
  - `VideoRopePosition3DEmb` (3D rotary position embeddings specific to WAN's H/W/T factorization)
  - `WanSelfAttention` / `WanI2VCrossAttention` (attention patterns tied to WAN's token structure)
  - Modulation via 6-parameter shift vectors
  - TransformerEngine integration for mixed-precision attention
- **Model Classes**: `WANDiffusionModel`, `I2VWan2pt1Model`, and `I2V_Edit_Wan2pt1Model` contain WAN-specific logic such as the specific latent conditioning format (`WAN2PT1_I2V_COND_LATENT_KEY`), frame repetition strategies, and `drop_video_step` mid-diffusion frame dropping.

#### What Would Be Needed for a Non-WAN Base Model

To train ChronoEdit's editing approach on a different base model (e.g., CogVideoX, Flux, or a custom DiT), you would need to:
1. Implement a new network class (analogous to `WanModel`) with the target architecture's attention, position embedding, and modulation patterns.
2. Implement a corresponding tokenizer if the target model uses a different VAE.
3. Register the new network and tokenizer as Hydra config groups.
4. The model-level code (`WANDiffusionModel`) is partially reusable — the flow matching training loop, EMA logic, and sampling infrastructure are generic, but the data conditioning and latent manipulation in `get_data_and_condition()` may need adaptation.

**In summary**: The config *structure* supports pluggable base models, but the current *implementations* are WAN-specific. Adapting to a non-WAN model requires new network and potentially tokenizer implementations, not just config changes.

---

## 3. Video, Image, and Edit Model Support

### Video Models

ChronoEdit is fundamentally built on **video diffusion models**. The core insight from the [paper](https://arxiv.org/abs/2510.04290) is that image editing can be reframed as video generation — the input image is the first frame and the edited image is the last frame, with the model learning the temporal trajectory between them.

All three model variants operate on video tensors of shape `[B, C, T, H, W]`:

| Model | Input Frames | Purpose |
|-------|-------------|---------|
| T2V (`WANDiffusionModel`) | Text → T frames | Text-to-video generation |
| I2V (`I2VWan2pt1Model`) | 1 image → T frames | Image-to-video generation |
| Edit (`I2V_Edit_Wan2pt1Model`) | 2 images → 5 frames | Image editing via video trajectory |

### Image Support

The training pipeline natively supports **mixed image-video training**. This is handled through:

1. **`is_image_batch()` detection** in `WANDiffusionModel`: Checks if the batch contains single-frame data and routes it accordingly.
2. **`_augment_image_dim_inplace()`**: Adds a temporal dimension to image batches so they can be processed by the same video pipeline.
3. **Joint dataloaders** (`chronoedit/_src/datasets/joint_dataloader.py`): Mix image and video datasets with configurable ratios.
4. **Mock data configs** provide both `mock_image` and `mock_video` data sources.

The conditioner's `DataType` enum (`DataType.IMAGE` vs. `DataType.VIDEO`) adjusts padding masks and fps metadata for image vs. video batches.

### Edit Model Specifics

The editing model (`I2V_Edit_Wan2pt1Model`) processes edit pairs as follows:

```python
# From chronoedit/_src/models/chronoedit_14b_edit_model.py
last_frame = raw_state[:, :, -1:, :, :]           # Extract edited (last) frame
last_frame_repeated = last_frame.repeat(1, 1, 4, 1, 1)  # Repeat 4 times
raw_state_edit = torch.cat([                        # Build 5-frame sequence
    raw_state[:, :, :1, :, :],                      # [original_frame,
    last_frame_repeated                              #  edited×4]
], dim=2)
```

This 5-frame construction gives the model a strong signal about the desired edit outcome while maintaining temporal consistency.

### Could It Work with Pure Image Models?

**Not directly.** The architecture relies on:
- 3D position embeddings (temporal + spatial)
- Temporal attention patterns in the transformer
- Video-specific VAE encoding/decoding with temporal compression
- Frame-conditioning logic (first/last frame handling)

A pure image diffusion model (without temporal dimensions) would require significant architectural changes. However, the system can process **single images** through the video pipeline by treating them as 1-frame videos, which is already supported for mixed training.

---

## 4. The T2V Module: Purpose and Role in Editing

### What Is the T2V Module?

The T2V (Text-to-Video) module (`chronoedit/_src/configs/t2v_wan/`, `chronoedit/_src/models/wan_t2v_model.py`) is the **foundational base model** upon which both I2V and ChronoEdit editing are built. It is not an editing module itself — it is the pretrained video generation backbone.

### Architecture Relationship

```
T2V Base (WANDiffusionModel)
  ↓  inherits
I2V Extension (I2VWan2pt1Model)        ← adds first-frame conditioning
  ↓  sibling (also inherits from T2V)
Edit Model (I2V_Edit_Wan2pt1Model)      ← adds edit-pair frame handling
```

### Why T2V Exists in an Edit Codebase

The T2V module serves three critical purposes:

#### 1. **Pretrained Weight Source**

ChronoEdit's training starts from WAN 2.1 pretrained T2V or I2V checkpoints. The experiment configs make this explicit:

```python
# From chronoedit/_src/configs/chronoedit/experiment/wan2pt1/base.py
checkpoint=dict(
    load_path="checkpoints/Wan2.1-I2V-14B-720P.dcp",  # Start from I2V weights
    load_training_state=False,                          # Only load model weights
    strict_resume=False,                                # Allow architecture differences
)
```

The T2V configs (`t2v_wan/experiment/`) define the original pretrained model experiments, providing the foundation checkpoints that the edit model fine-tunes from.

#### 2. **Shared Training Infrastructure**

The ChronoEdit config directly imports and reuses T2V/I2V registration functions:

```python
# From chronoedit/_src/configs/chronoedit/config.py
from chronoedit._src.configs.i2v_wan.config import (
    register_optimizer, register_scheduler, register_model,
    register_callbacks, register_net, register_conditioner,
    register_ema, register_tokenizer, register_checkpoint,
    register_ckpt_type
)
```

This means all the T2V and I2V experiment configurations, network architectures, and model variants are available as building blocks when configuring edit training. The ChronoEdit config also imports T2V and I2V experiments:

```python
import_all_modules_from_package("chronoedit._src.configs.t2v_wan.experiment", reload=True)
import_all_modules_from_package("chronoedit._src.configs.i2v_wan.experiment", reload=True)
```

#### 3. **Base Class for Edit Model Code**

The `WANDiffusionModel` (T2V model class) provides all the core training machinery that the edit model inherits:

- **Flow matching training loop** (`training_step()`, `get_data_and_condition()`)
- **Rectified flow scheduling** (noise interpolation, velocity prediction)
- **VAE encoding/decoding** (`encode()`, `decode()`)
- **Sampling infrastructure** (`generate_samples_from_batch()`, `get_x0_fn_from_batch()`)
- **FSDP/DDP distributed training** (sharding, context parallelism)
- **LoRA fine-tuning** (adapter injection, weight loading)
- **EMA management** (exponential moving average of weights)

The edit model only overrides two methods:
- `get_data_and_condition()` — to handle edit-pair frame construction (first + last frame)
- `generate_samples_from_batch()` — to support `drop_video_step` (mid-diffusion frame dropping for temporal reasoning)

### T2V Network vs. Edit Network

The T2V network (`WanModel`) and edit network (`EditWanModel`) share the same transformer architecture. The key difference is in position embeddings:

| Component | T2V (`WanModel`) | Edit (`EditWanModel`) |
|-----------|-------------------|----------------------|
| Position Embedding | `VideoRopePosition3DEmb` | `TemproalSkipVideoRopePosition3DEmb` |
| Temporal Handling | Sequential frames | Temporal skip (configurable `temporal_skip_len`) |
| Input Channels | 16 (T2V) or 36 (I2V) | 36 (always I2V-style with image conditioning) |

The `TemproalSkipVideoRopePosition3DEmb` (note: class name reflects original source spelling) allows the edit model to assign position embeddings that reflect that the edit frames are not temporally adjacent but represent a "skip" — the first and last frames of a conceptual longer video. This is controlled by `temporal_skip_len` (default 8 in the edit experiments), giving the model information about the temporal gap between input and output frames.

---

## 5. Component Coupling Summary

| Component | Abstraction Layer | WAN Coupling | Swappable via Config? |
|-----------|------------------|-------------|----------------------|
| Tokenizer (VAE) | `BaseVAE` interface | Implementation-specific | ✅ Yes |
| Conditioner | `BaseCondition` dataclass | Low | ✅ Yes |
| Network (DiT) | None (no base class) | **High** | ⚠️ Requires new implementation |
| Flow Scheduler | Generic `RectifiedFlow` | None | ✅ Yes |
| Optimizer | PyTorch standard | None | ✅ Yes |
| EMA | Generic updater | None | ✅ Yes |
| Checkpointer | DCP wrapper | None | ✅ Yes |
| Data Pipeline | Generic loaders | None | ✅ Yes |
| **Overall** | | **Moderate** | **Partially** |

### Key Takeaways

1. **The config system is modular by design**, using Hydra's composable groups to swap components. However, the network layer lacks abstraction, making base model changes a code-level effort rather than a config-level one.

2. **Only video-based models are supported** — the training pipeline requires temporal dimensions (3D position embeddings, temporal attention, video VAE). Single images are handled by treating them as 1-frame videos within the existing video pipeline.

3. **The T2V module is the foundation, not a separate feature** — it provides the pretrained weights, the training infrastructure (flow matching, distributed training, EMA), and the base model class that the edit model extends. Without T2V, there would be no ChronoEdit.

4. **Extending to non-WAN models is feasible but requires implementation work** — primarily a new network architecture class and potentially a new tokenizer. The training loop, scheduler, optimizer, and data pipeline are all reusable.

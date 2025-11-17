#!/bin/bash

PYTHONPATH=$(pwd) accelerate launch /workspace/ChronoEdit_Training/scripts/train_diffsynth.py \
    --dataset_base_path /workspace/dataset/example_dataset \
    --dataset_metadata_path /workspace/dataset/example_dataset/metadata.csv \
    --height 1024 \
    --width 1024 \
    --num_frames 2 \
    --dataset_repeat 1 \
    --model_paths '[["checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00001-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00002-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00003-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00004-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00005-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00006-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00007-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00008-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00009-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00010-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00011-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00012-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00013-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00014-of-00014.safetensors"]]' \
    --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-720P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-720P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-720P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "./models/train/ChronoEdit-14B_lora" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --extra_inputs "input_image"  \
    --use_gradient_checkpointing_offload
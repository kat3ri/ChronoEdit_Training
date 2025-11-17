import os
import time
import torch
import numpy as np
import tempfile
from diffusers import AutoencoderKLWan
from diffusers.utils import load_image
from diffusers.schedulers import UniPCMultistepScheduler
from transformers import CLIPVisionModel
from chronoedit_diffusers.pipeline_chronoedit import ChronoEditPipeline
from chronoedit_diffusers.transformer_chronoedit import ChronoEditTransformer3DModel
from PIL import Image
from huggingface_hub import hf_hub_download
from prompt_enhancer import load_model, enhance_prompt
import argparse

def calculate_dimensions(image, mod_value):
    target_area = 720 * 1280
    aspect_ratio = image.height / image.width
    calculated_height = round(np.sqrt(target_area * aspect_ratio)) // mod_value * mod_value
    calculated_width = round(np.sqrt(target_area / aspect_ratio)) // mod_value * mod_value
    return calculated_width, calculated_height

def run_inference(
    image_path,
    prompt,
    enable_temporal_reasoning=False,
    num_inference_steps=8,
    guidance_scale=1.0,
    shift=2.0,
    num_temporal_reasoning_steps=8,
    output_path="output.png"
):
    model_id = "nvidia/ChronoEdit-14B-Diffusers"
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id,
        subfolder="image_encoder",
        torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )
    transformer = ChronoEditTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    pipe = ChronoEditPipeline.from_pretrained(
        model_id,
        image_encoder=image_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    lora_path = hf_hub_download(repo_id=model_id, filename="lora/chronoedit_distill_lora.safetensors")
    if lora_path:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=2.0
        )
    pipe.to("cuda")

    prompt_enhancer_model = "Qwen/Qwen3-VL-8B-Instruct"
    prompt_model, processor = load_model(prompt_enhancer_model)
    prompt_model.to("cuda")
    cot_prompt = enhance_prompt(
        image_path,
        prompt,
        prompt_model,
        processor,
    )
    prompt_model.to("cpu")
    final_prompt = cot_prompt

    image = load_image(image_path)
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    width, height = calculate_dimensions(image, mod_value)
    image = image.resize((width, height))
    num_frames = 29 if enable_temporal_reasoning else 5

    output = pipe(
        image=image,
        prompt=final_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        enable_temporal_reasoning=enable_temporal_reasoning,
        num_temporal_reasoning_steps=num_temporal_reasoning_steps,
    ).frames[0]

    Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save(output_path)
    print(f"Saved output to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChronoEdit CLI")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--temporal", action="store_true", help="Enable temporal reasoning")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--shift", type=float, default=2.0, help="Shift value")
    parser.add_argument("--temporal-steps", type=int, default=8, help="Temporal reasoning steps")
    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        prompt=args.prompt,
        enable_temporal_reasoning=args.temporal,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        shift=args.shift,
        num_temporal_reasoning_steps=args.temporal_steps,
        output_path=args.output
    )
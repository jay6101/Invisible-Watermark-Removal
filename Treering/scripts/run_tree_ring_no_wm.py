#!/usr/bin/env python

"""
Generate NON-watermarked images for Tree Ring evaluation.
Same prompts, same SD model, same latents — but WITHOUT applying the Tree Ring watermark.
Outputs are saved directly in your project folder on Drive.
"""

import argparse
import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler


# -------------------------------
# Prompt loader
# -------------------------------
def load_prompts(prompt_file=None):
    if prompt_file is None:
        return [
            "a high quality photo of a dog in a park",
            "a futuristic city at sunset, concept art",
            "a fantasy castle on a mountain, dramatic lighting",
            "a close up portrait of a person, cinematic lighting",
        ]
    with open(prompt_file, "r", encoding="utf8") as f:
        lines = [l.strip() for l in f.readlines()]
    return [l for l in lines if l]


# -------------------------------
# Device picker
# -------------------------------
def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------
# Load model
# -------------------------------
def load_pipeline(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


# -------------------------------
# Prepare latents
# -------------------------------
def prepare_latents(pipe, batch, height, width, generator, device):
    # in SD v1.4, in_channels = 4
    return torch.randn(
        (batch, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float16 if device.type != "cpu" else torch.float32,
    )


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument(
        "--output_root",
        default="/content/drive/MyDrive/Treering_pipeline/outputs",
        help="Root output folder (use your Drive path here)"
    )
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    args = parser.parse_args()

    # Device
    device = get_best_device()
    print("Using device:", device)

    # Prompts
    prompts = load_prompts(args.prompt_file)

    # Load SD pipeline
    pipe = load_pipeline(args.model_id, device)

    # Output directory
    run_dir = os.path.join(args.output_root, args.run_name)
    out_dir = os.path.join(run_dir, "no_wm")
    os.makedirs(out_dir, exist_ok=True)
    print("Saving images to:", out_dir)

    total = args.end - args.start
    print(f"Generating {total} NON-watermarked samples")

    # Loop for image generation
    for idx in range(args.start, args.end):
        prompt = prompts[idx % len(prompts)]
        seed = args.seed + idx
        print(f"[{idx}] Prompt: {prompt}")

        gen = torch.Generator(device=device).manual_seed(seed)

        # Generate latents (no watermark applied here!)
        latents = prepare_latents(
            pipe, args.batch_size, args.height, args.width, gen, device
        )

        # Generate image
        out = pipe(
            prompt=[prompt] * args.batch_size,
            latents=latents,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        )
        images = out.images

        for b, img in enumerate(images):
            fname = f"idx{idx:05d}_b{b}.png"
            img.save(os.path.join(out_dir, fname))

    print("Done generating NON-watermarked images.")


if __name__ == "__main__":
    main()

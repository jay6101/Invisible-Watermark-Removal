#!/usr/bin/env python

"""
run_tree_ring_watermark.py

Generate watermarked images using Tree Ring watermarking and optionally
apply removal attacks (pixel restoration and horizontal shift).
MPS-friendly: Stable Diffusion runs on MPS, watermark FFT runs on CPU.
"""

import argparse
import os
import json
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler


# -------------------------------------------------------------------
# Prompt loader
# -------------------------------------------------------------------

def load_prompts(prompt_file: Optional[str]) -> List[str]:
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


# -------------------------------------------------------------------
# Tree Ring Watermarker (FFT on CPU)
# -------------------------------------------------------------------

def make_radius_map(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(h, device=device) - (h - 1) / 2.0
    xs = torch.arange(w, device=device) - (w - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.sqrt(xx * xx + yy * yy)


class TreeRingWatermarker:
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        w_channel: int = 3,
        w_pattern: str = "ring",
        w_radius: int = 16,
        key_seed: int = 12345,
        device: Optional[torch.device] = None,
    ):
        self.device = torch.device("cpu")   # Force watermarking to CPU
        self.channels = channels
        self.height = height
        self.width = width
        self.w_channel = w_channel
        self.w_pattern = w_pattern
        self.w_radius = w_radius

        r = make_radius_map(height, width, self.device)
        self.radius_map = r
        self.mask = r <= float(w_radius)

        self.key_rand = {}
        self.key_ring = {}
        self.r_int = torch.round(r).long()

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(key_seed)

    def _key_rand(self, ch: int) -> torch.Tensor:
        if ch in self.key_rand:
            return self.key_rand[ch]
        real = torch.randn_like(self.radius_map, generator=self.rng)
        imag = torch.randn_like(self.radius_map, generator=self.rng)
        key = torch.complex(real, imag) * self.mask
        self.key_rand[ch] = key
        return key

    def _key_ring(self, ch: int) -> torch.Tensor:
        if ch in self.key_ring:
            return self.key_ring[ch]

        key = torch.zeros_like(self.radius_map, dtype=torch.complex64)
        unique_radii = torch.unique(self.r_int[self.mask])
        for rad in unique_radii:
            ring_mask = (self.r_int == rad) & self.mask
            if not torch.any(ring_mask):
                continue
            real = torch.randn(1, generator=self.rng, device=self.device)
            imag = torch.randn(1, generator=self.rng, device=self.device)
            key[ring_mask] = torch.complex(real, imag)[0]

        self.key_ring[ch] = key
        return key

    def _apply_pattern(self, fft_x: torch.Tensor, ch: int) -> torch.Tensor:
        fft_x = fft_x.clone()
        if self.w_pattern == "zeros":
            fft_x[self.mask] = 0
        elif self.w_pattern == "rand":
            fft_x[self.mask] = self._key_rand(ch)[self.mask]
        elif self.w_pattern == "ring":
            fft_x[self.mask] = self._key_ring(ch)[self.mask]
        else:
            raise ValueError(f"Unknown pattern {self.w_pattern}")
        return fft_x

    def embed(self, latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        device = latents.device
        dtype = latents.dtype

        lat_fp32 = latents.to(torch.float32)
        out = lat_fp32.clone()

        if self.w_channel == -1:
            channels = list(range(c))
        else:
            channels = [self.w_channel]

        for bi in range(b):
            for ch in channels:
                x = out[bi, ch]

                x_cpu = x.detach().to("cpu")

                fft_x = torch.fft.fft2(x_cpu)
                fft_x = torch.fft.fftshift(fft_x)
                fft_x = self._apply_pattern(fft_x, ch)
                fft_x = torch.fft.ifftshift(fft_x)

                x_wm = torch.fft.ifft2(fft_x)
                out[bi, ch] = torch.real(x_wm).to(device)

        return out.to(dtype)


# -------------------------------------------------------------------
# Pixel attacks
# -------------------------------------------------------------------

def pixel_restoration(img: Image.Image, downscale_factor: float = 0.5) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(w * downscale_factor))
    new_h = max(1, int(h * downscale_factor))
    small = img.resize((new_w, new_h), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def horizontal_shift(img: Image.Image, shift_pixels: int = 32, mode: str = "wrap") -> Image.Image:
    arr = np.array(img)
    if mode == "wrap":
        shifted = np.roll(arr, shift_pixels, axis=1)
    else:
        shifted = np.zeros_like(arr)
        if shift_pixels >= 0:
            shifted[:, shift_pixels:] = arr[:, : arr.shape[1] - shift_pixels]
        else:
            k = -shift_pixels
            shifted[:, : arr.shape[1] - k] = arr[:, k:]
    return Image.fromarray(shifted)


def apply_attacks(img: Image.Image, attack_type: str, downscale: float, shift_pixels: int, shift_mode: str):
    if attack_type == "pixel_restoration":
        return pixel_restoration(img, downscale)
    if attack_type == "horizontal_shift":
        return horizontal_shift(img, shift_pixels, shift_mode)
    return horizontal_shift(pixel_restoration(img, downscale), shift_pixels, shift_mode)


# -------------------------------------------------------------------
# Pipeline + latents
# -------------------------------------------------------------------

def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pipeline(model_id: str, device: torch.device):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def prepare_latents(pipe, batch, height, width, generator, device):
    return torch.randn(
        (batch, pipe.unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )


def sample_images(pipe, prompts, latents, guidance, steps):
    device = pipe.device.type
    if device == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            out = pipe(prompt=prompts, latents=latents, guidance_scale=guidance, num_inference_steps=steps)
    else:
        out = pipe(prompt=prompts, latents=latents, guidance_scale=guidance, num_inference_steps=steps)
    return out.images


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompt_file", default=None)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=2)

    p.add_argument("--w_channel", type=int, default=3)
    p.add_argument("--w_pattern", choices=["zeros", "rand", "ring"], default="ring")
    p.add_argument("--w_radius", type=int, default=16)
    p.add_argument("--key_seed", type=int, default=12345)

    p.add_argument("--apply_attack", action="store_true")
    p.add_argument("--attack_type", choices=["pixel_restoration", "horizontal_shift", "both"], default="both")
    p.add_argument("--attack_downscale", type=float, default=0.5)
    p.add_argument("--attack_shift_pixels", type=int, default=32)
    p.add_argument("--attack_shift_mode", choices=["wrap", "black"], default="wrap")
    return p.parse_args()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    args = parse_args()

    run_dir = os.path.join(args.output_root, args.run_name)
    wm_dir = os.path.join(run_dir, "watermarked")
    atk_dir = os.path.join(run_dir, "attacked")
    os.makedirs(wm_dir, exist_ok=True)
    if args.apply_attack:
        os.makedirs(atk_dir, exist_ok=True)

    device = get_best_device()
    print("Using device:", device)

    prompts_list = load_prompts(args.prompt_file)
    pipe = load_pipeline(args.model_id, device)

    channels = pipe.unet.in_channels
    h_latent = args.height // 8
    w_latent = args.width // 8

    watermarker = TreeRingWatermarker(
        channels,
        h_latent,
        w_latent,
        args.w_channel,
        args.w_pattern,
        args.w_radius,
        args.key_seed,
        device=torch.device("cpu"),
    )

    total = args.end - args.start
    print(f"Generating {total} images")

    for idx in range(args.start, args.end):
        prompt = prompts_list[idx % len(prompts_list)]
        seed = args.seed + idx
        print(f"[{idx}] Prompt: {prompt}")

        gen = torch.Generator(device=device).manual_seed(seed)

        latents = prepare_latents(pipe, args.batch_size, args.height, args.width, gen, device)
        latents_wm = watermarker.embed(latents)

        images = sample_images(
            pipe,
            [prompt] * args.batch_size,
            latents_wm,
            args.guidance_scale,
            args.num_inference_steps,
        )

        for b, img in enumerate(images):
            base = f"idx{idx:05d}_b{b}.png"
            img.save(os.path.join(wm_dir, base))

            if args.apply_attack:
                atk = apply_attacks(
                    img,
                    args.attack_type,
                    args.attack_downscale,
                    args.attack_shift_pixels,
                    args.attack_shift_mode,
                )
                atk.save(os.path.join(atk_dir, base.replace(".png", "_attacked.png")))

    print("Done.")


if __name__ == "__main__":
    main()
    
# run command:
# !python scripts/run_tree_ring_watermark_wo_eval.py \
#   --run_name colab_100 \
#   --w_channel 3 \
#   --w_pattern ring \
#   --w_radius 16 \
#   --start 0 \
#   --end 100 \
#   --apply_attack \
#   --attack_type both
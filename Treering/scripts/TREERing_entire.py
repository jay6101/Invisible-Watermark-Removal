#!/usr/bin/env python

import argparse
import os
import math
from contextlib import nullcontext

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from diffusers import StableDiffusionPipeline, DDIMScheduler
from sklearn import metrics


# -------------------------
# Tree Ring Watermarker
# -------------------------

class TreeRingWatermarker(nn.Module):
    """
    Tree Ring watermarking in latent Fourier space.

    - Builds a ring-shaped mask in frequency domain
    - Builds a complex key pattern (gt_patch) on that ring
    - apply() embeds the key into chosen channel(s)
    - eval_metrics() computes a per-example MSE metric vs the key

    FFT device logic:
      - If main device is cuda: FFT and key live on cuda (fast)
      - If main device is mps: FFT and key live on cpu (MPS cannot handle complex FFT)
      - Else: FFT and key live on cpu
    """

    def __init__(
        self,
        height,
        width,
        w_channel=3,
        w_radius=8.0,
        w_pattern="ring",   # "zeros", "rand", "ring"
        key_seed=12345,
        main_device=torch.device("cpu"),
    ):
        super().__init__()

        # Decide where FFT happens
        if main_device.type == "cuda":
            self.fft_device = torch.device("cuda")
        elif main_device.type == "mps":
            print("Note: MPS detected, Tree Ring FFT will run on CPU.")
            self.fft_device = torch.device("cpu")
        else:
            self.fft_device = torch.device("cpu")

        self.h = height
        self.w = width
        self.w_channel = w_channel
        self.w_radius = w_radius
        self.w_pattern = w_pattern
        self.key_seed = key_seed

        # Radius map
        yy, xx = torch.meshgrid(
            torch.arange(self.h),
            torch.arange(self.w),
            indexing="ij"
        )
        cy = (self.h - 1) / 2.0
        cx = (self.w - 1) / 2.0
        rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        self.register_buffer("radius_map", rr)

        # Ring mask
        ring_width = 1.0
        mask = (rr >= (self.w_radius - ring_width)) & (rr <= (self.w_radius + ring_width))
        self.register_buffer("mask", mask)

        # Build key pattern in Fourier domain
        self._build_key()

    def _build_key(self):
        g = torch.Generator(device=self.fft_device)
        g.manual_seed(self.key_seed)

        key = torch.zeros(self.h, self.w, dtype=torch.complex64, device=self.fft_device)

        if self.w_pattern == "zeros":
            # Zero coefficients on the ring mask
            key[:] = 0.0 + 0.0j

        elif self.w_pattern == "rand":
            # Random complex values on the masked ring
            real = torch.randn(self.h, self.w, generator=g, device=self.fft_device)
            imag = torch.randn(self.h, self.w, generator=g, device=self.fft_device)
            key = (real + 1j * imag).to(torch.complex64)
            key[~self.mask.to(self.fft_device)] = 0.0 + 0.0j

        elif self.w_pattern == "ring":
            # One complex value for the whole masked ring
            real = torch.randn(1, generator=g, device=self.fft_device).item()
            imag = torch.randn(1, generator=g, device=self.fft_device).item()
            val = complex(real, imag)
            key = torch.zeros(self.h, self.w, dtype=torch.complex64, device=self.fft_device)
            key[self.mask.to(self.fft_device)] = torch.tensor(val, dtype=torch.complex64, device=self.fft_device)

        else:
            raise ValueError(f"Unknown w_pattern {self.w_pattern}")

        # Store as buffer on fft_device
        self.register_buffer("key_fft", key)

    @torch.no_grad()
    def apply(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Embed the watermark into latents.

        latents: [B, C, H, W] real tensor on any device.
        Returns: watermarked latents, same shape and device.
        """
        B, C, H, W = latents.shape
        if H != self.h or W != self.w:
            raise ValueError(f"Latent spatial size {H}x{W} does not match Watermarker {self.h}x{self.w}")

        orig_device = latents.device
        orig_dtype = latents.dtype

        # Move to FFT device and upcast to float32 for stability
        x = latents.to(self.fft_device, dtype=torch.float32)

        # FFT over spatial dims
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft = torch.fft.fftshift(x_fft)

        key = self.key_fft.to(self.fft_device)
        mask = self.mask.to(self.fft_device)

        # Channels to watermark
        if self.w_channel < 0:
            channels = list(range(C))
        else:
            if self.w_channel >= C:
                raise ValueError(f"w_channel {self.w_channel} out of range for C={C}")
            channels = [self.w_channel]

        for ch in channels:
            ch_fft = x_fft[:, ch, :, :]            # [B, H, W]
            key_b = key.unsqueeze(0).expand_as(ch_fft)  # [B, H, W]
            ch_fft[:, mask] = key_b[:, mask]
            x_fft[:, ch, :, :] = ch_fft

        x_fft = torch.fft.ifftshift(x_fft)
        x_w = torch.fft.ifft2(x_fft, dim=(-2, -1)).real

        # Back to original device and dtype
        return x_w.to(orig_device, dtype=orig_dtype)

    @torch.no_grad()
    def eval_metrics(self, latents_no_w: torch.Tensor, latents_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per example watermark metrics on the given pair of latent batches.

        Metric:
          - For chosen channel, compute FFT
          - Compare masked positions to key_fft (complex MSE)
          - Returns: (no_w_metric, w_metric), each [B]
        """
        if latents_no_w.shape != latents_w.shape:
            raise ValueError("latents_no_w and latents_w must have same shape")

        B, C, H, W = latents_no_w.shape
        if H != self.h or W != self.w:
            raise ValueError("Latent size mismatch in eval_metrics")

        if self.w_channel < 0 or self.w_channel >= C:
            ch = 0
        else:
            ch = self.w_channel

        # Move to FFT device for eval
        x_no = latents_no_w.to(self.fft_device, dtype=torch.float32)
        x_wm = latents_w.to(self.fft_device, dtype=torch.float32)

        # Select channel
        ch_no = x_no[:, ch, :, :]  # [B, H, W]
        ch_wm = x_wm[:, ch, :, :]  # [B, H, W]

        fft_no = torch.fft.fft2(ch_no, dim=(-2, -1))
        fft_wm = torch.fft.fft2(ch_wm, dim=(-2, -1))

        key = self.key_fft.to(self.fft_device)    # [H, W]
        mask = self.mask.to(self.fft_device)      # [H, W]

        # flatten spatial
        mask_flat = mask.view(-1)
        key_flat = key.view(-1)                   # [HW]

        fft_no_flat = fft_no.view(B, -1)          # [B, HW]
        fft_wm_flat = fft_wm.view(B, -1)          # [B, HW]

        key_b = key_flat.unsqueeze(0)             # [1, HW]

        diff_no = fft_no_flat - key_b
        diff_wm = fft_wm_flat - key_b

        diff_no_masked = diff_no[:, mask_flat]    # [B, Nmask]
        diff_wm_masked = diff_wm[:, mask_flat]    # [B, Nmask]

        mse_no = (diff_no_masked.abs() ** 2).mean(dim=1)   # [B]
        mse_wm = (diff_wm_masked.abs() ** 2).mean(dim=1)   # [B]

        # Return on CPU for easier numpy conversion
        return mse_no.cpu(), mse_wm.cpu()


# -------------------------
# Attacks
# -------------------------

def pixel_restoration_attack(img: Image.Image, downscale: int = 4) -> Image.Image:
    if downscale <= 1:
        return img
    w, h = img.size
    new_w = max(1, w // downscale)
    new_h = max(1, h // downscale)
    small = img.resize((new_w, new_h), resample=Image.BILINEAR)
    restored = small.resize((w, h), resample=Image.BILINEAR)
    return restored


def horizontal_shift_attack(img: Image.Image, shift_pixels: int = 16, mode: str = "wrap") -> Image.Image:
    shift_pixels = int(shift_pixels)
    if shift_pixels == 0:
        return img

    arr = np.array(img)
    h, w = arr.shape[:2]
    shift = shift_pixels % w

    if mode == "wrap":
        arr = np.roll(arr, shift=shift, axis=1)
    else:
        # black fill
        if shift > 0:
            arr[:, shift:, ...] = arr[:, :-shift, ...]
            arr[:, :shift, ...] = 0
        else:
            s = -shift
            arr[:, :-s, ...] = arr[:, s:, ...]
            arr[:, -s:, ...] = 0

    return Image.fromarray(arr)


# -------------------------
# SD Pipeline helpers
# -------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def load_pipeline(model_id: str, device: torch.device):
    if device.type in ["cuda", "mps"]:
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    pipe.unet.to(memory_format=torch.channels_last)

    return pipe


def read_prompts(prompt_file: str | None, start: int, end: int | None) -> list[str]:
    if prompt_file is None:
        base = [
            "a high quality photo of a dog in a park",
            "a beautiful landscape with mountains and a lake",
            "a futuristic city at sunset, concept art",
            "a close up portrait of a person, cinematic lighting",
        ]
        prompts = (base * 1000)[start:end]
    else:
        with open(prompt_file, "r", encoding="utf8") as f:
            lines = [l.strip() for l in f.readlines()]
        lines = [l for l in lines if l]
        if end is None or end > len(lines):
            end = len(lines)
        prompts = lines[start:end]
    return prompts


def make_output_dirs(root: str, run_name: str):
    base = os.path.join(root, run_name)
    wm_dir = os.path.join(base, "watermarked")
    atk_dir = os.path.join(base, "attacked")
    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(atk_dir, exist_ok=True)
    return base, wm_dir, atk_dir


def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


# -------------------------
# ROC summary
# -------------------------

def summarize_roc(no_w_metrics: list[float], w_metrics: list[float]):
    no_w_metrics = np.array(no_w_metrics)  # latents without watermark
    w_metrics = np.array(w_metrics)        # latents with watermark

    # Use negative metrics so higher score = more likely watermarked
    preds = np.concatenate([-no_w_metrics, -w_metrics], axis=0)
    labels = np.array([0] * len(no_w_metrics) + [1] * len(w_metrics))

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1.0 - (fpr + (1.0 - tpr)) / 2.0)

    if np.any(fpr < 0.01):
        idx = np.where(fpr < 0.01)[0][-1]
        tpr_at_1 = tpr[idx]
    else:
        tpr_at_1 = 0.0

    print("")
    print(f"Number of no watermark samples: {len(no_w_metrics)}")
    print(f"Number of watermark samples:    {len(w_metrics)}")
    print(f"AUC:       {auc:.4f}")
    print(f"Best acc:  {acc:.4f}")
    print(f"TPR@1%FPR: {tpr_at_1:.4f}")


# -------------------------
# Main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--output_root", type=str, default="outputs")
    p.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompt_file", type=str, default=None)

    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100)

    # watermark
    p.add_argument("--w_channel", type=int, default=3)
    p.add_argument("--w_pattern", type=str, choices=["zeros", "rand", "ring"], default="ring")
    p.add_argument("--w_radius", type=float, default=8.0)
    p.add_argument("--key_seed", type=int, default=12345)

    # attacks
    p.add_argument("--apply_attack", action="store_true")
    p.add_argument("--attack_type", type=str, choices=["pixel_restoration", "horizontal_shift", "both"], default="both")
    p.add_argument("--attack_downscale", type=int, default=4)
    p.add_argument("--attack_shift_pixels", type=int, default=16)
    p.add_argument("--attack_shift_mode", type=str, choices=["wrap", "black"], default="wrap")

    return p.parse_args()


def main():
    args = parse_args()

    device, device_type = get_device()
    print(f"Using device: {device_type}")

    pipe = load_pipeline(args.model_id, device=device)

    latent_h = args.height // 8
    latent_w = args.width // 8
    channels = pipe.unet.config.in_channels

    wm = TreeRingWatermarker(
        height=latent_h,
        width=latent_w,
        w_channel=args.w_channel,
        w_radius=args.w_radius,
        w_pattern=args.w_pattern,
        key_seed=args.key_seed,
        main_device=device,
    )

    base_dir, wm_dir, atk_dir = make_output_dirs(args.output_root, args.run_name)
    prompts = read_prompts(args.prompt_file, args.start, args.end)

    num_images = len(prompts)
    if num_images == 0:
        print("No prompts to process.")
        return

    g = torch.Generator(device=device).manual_seed(args.seed)

    if device_type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    no_w_metrics = []
    w_metrics = []

    num_batches = math.ceil(num_images / args.batch_size)
    print(f"Generating {num_images} images in {num_batches} batches")
    print(f"Outputs will be in: {base_dir}")

    global_idx = 0

    for b in range(num_batches):
        batch_prompts = prompts[b * args.batch_size:(b + 1) * args.batch_size]
        if not batch_prompts:
            break
        current_bs = len(batch_prompts)

        # Sample base latents and scale by scheduler sigma
        if device.type in ["cuda", "mps"]:
            lat_dtype = torch.float16
        else:
            lat_dtype = torch.float32

        latents_no_w = torch.randn(
            current_bs,
            channels,
            latent_h,
            latent_w,
            generator=g,
            device=device,
            dtype=lat_dtype,
        )
        latents_no_w = latents_no_w * pipe.scheduler.init_noise_sigma

        # Apply watermark in Fourier space
        latents_w = wm.apply(latents_no_w.clone())

        # Latent level metrics
        batch_no_w_m, batch_w_m = wm.eval_metrics(latents_no_w, latents_w)
        no_w_metrics.extend(batch_no_w_m.tolist())
        w_metrics.extend(batch_w_m.tolist())

        # Generate images from watermarked latents
        with amp_ctx:
            with torch.no_grad():
                out = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    latents=latents_w,
                )
        images_w = out.images

        for i, (img_w, prompt) in enumerate(zip(images_w, batch_prompts)):
            idx = global_idx
            global_idx += 1

            fname = f"idx{idx:06d}.png"
            wm_path = os.path.join(wm_dir, fname)
            save_image(img_w, wm_path)

            if args.apply_attack:
                atk_img = img_w
                if args.attack_type in ["pixel_restoration", "both"]:
                    atk_img = pixel_restoration_attack(atk_img, args.attack_downscale)
                if args.attack_type in ["horizontal_shift", "both"]:
                    atk_img = horizontal_shift_attack(
                        atk_img,
                        args.attack_shift_pixels,
                        mode=args.attack_shift_mode,
                    )
                atk_path = os.path.join(atk_dir, fname.replace(".png", "_attacked.png"))
                save_image(atk_img, atk_path)

            if idx < 5:
                print(f"[{idx}] prompt: {prompt}")

    # Summarize evaluation metrics
    summarize_roc(no_w_metrics, w_metrics)


if __name__ == "__main__":
    main()

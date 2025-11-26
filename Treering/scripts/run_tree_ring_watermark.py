import argparse
import os
import math
from contextlib import nullcontext

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDIMScheduler
from sklearn import metrics


# -------------------------
# Tree-Ring Watermarker
# -------------------------

class TreeRingWatermarker(nn.Module):
    """
    Simple Tree-Ring style watermarker operating in Fourier space
    on the latent tensor.

    It builds:
      - a ring-shaped mask in frequency domain
      - a complex key pattern on that ring (gt_patch)

    apply() embeds the key into the chosen latent channel.
    """

    def __init__(
        self,
        height,
        width,
        w_channel=3,
        w_radius=8,
        w_pattern="ring",     # "zeros", "rand", "ring"
        key_seed=12345,
        device="cpu",
    ):
        super().__init__()
        self.h = height
        self.w = width
        self.device = torch.device(device)
        self.w_channel = w_channel
        self.w_radius = w_radius
        self.w_pattern = w_pattern
        self.key_seed = key_seed

        # build radius map and ring mask
        yy, xx = torch.meshgrid(
            torch.arange(self.h), torch.arange(self.w), indexing="ij"
        )
        cx = (self.w - 1) / 2.0
        cy = (self.h - 1) / 2.0
        rr = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # thin ring around radius w_radius (you can widen if wanted)
        ring_width = 1.0
        mask = (rr >= (self.w_radius - ring_width)) & (rr <= (self.w_radius + ring_width))

        self.register_buffer("radius_map", rr)
        self.register_buffer("mask", mask)

        # build key (complex pattern)
        self._build_key()

    def _build_key(self):
        g = torch.Generator(device=self.device)
        g.manual_seed(self.key_seed)

        # complex matrix same size as FFT
        key = torch.zeros(self.h, self.w, dtype=torch.complex64)

        if self.w_pattern == "zeros":
            # key is just zeros on the ring
            # (not very useful, but included to mirror repo options)
            key[:] = 0.0 + 0.0j

        elif self.w_pattern == "rand":
            # random complex values on the mask
            real = torch.randn(self.h, self.w, generator=g)
            imag = torch.randn(self.h, self.w, generator=g)
            key = (real + 1j * imag).to(torch.complex64)
            key[~self.mask] = 0.0 + 0.0j

        elif self.w_pattern == "ring":
            # one complex value per radius bucket, then broadcast along the ring
            # here we just use a constant complex value on the chosen ring
            real = torch.randn(1, generator=g).item()
            imag = torch.randn(1, generator=g).item()
            val = complex(real, imag)

            key = torch.zeros(self.h, self.w, dtype=torch.complex64)
            key[self.mask] = torch.tensor(val, dtype=torch.complex64)

        else:
            raise ValueError(f"Unknown w_pattern: {self.w_pattern}")

        self.register_buffer("key_fft", key)

    @torch.no_grad()
    def apply(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Embed the watermark key into the given latents.

        latents: [B, C, H, W] real tensor
        returns: watermarked latents of same shape
        """
        assert latents.shape[-2] == self.h and latents.shape[-1] == self.w, \
            "Latent spatial size mismatch for watermarker"

        B, C, H, W = latents.shape
        latents = latents.to(self.device)

        # FFT over spatial dims
        latents_fft = torch.fft.fft2(latents, dim=(-2, -1))

        # which channels to watermark
        if self.w_channel < 0:
            channels = list(range(C))
        else:
            if self.w_channel >= C:
                raise ValueError(f"w_channel {self.w_channel} out of range for C={C}")
            channels = [self.w_channel]

        # broadcast key to batch and channels
        key_fft = self.key_fft.to(self.device)  # [H, W]
        mask = self.mask.to(self.device)        # [H, W]

        for ch in channels:
            # latents_fft[:, ch, ...] is [B, H, W]
            ch_fft = latents_fft[:, ch, :, :]
            # replace values on mask with key value
            # we broadcast key_fft to [B, H, W]
            key_b = key_fft.unsqueeze(0).expand_as(ch_fft)
            ch_fft[:, mask] = key_b[:, mask]
            latents_fft[:, ch, :, :] = ch_fft

        # back to spatial
        watermarked = torch.fft.ifft2(latents_fft, dim=(-2, -1)).real
        return watermarked

    def get_gt_patch_and_mask(self):
        """
        Returns:
          - gt_patch: complex64 [H, W] pattern
          - mask: bool [H, W]
        """
        return self.key_fft, self.mask


# -------------------------
# Simple attacks
# -------------------------

def pixel_restoration_attack(pil_img: Image.Image, downscale: int) -> Image.Image:
    if downscale <= 1:
        return pil_img
    w, h = pil_img.size
    new_w = max(1, w // downscale)
    new_h = max(1, h // downscale)

    small = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
    restored = small.resize((w, h), resample=Image.BILINEAR)
    return restored


def horizontal_shift_attack(pil_img: Image.Image, shift_pixels: int, mode: str = "wrap") -> Image.Image:
    if shift_pixels == 0:
        return pil_img

    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    shift = shift_pixels % w

    if mode == "wrap":
        arr = np.roll(arr, shift=shift, axis=1)
    else:
        # fill vacated area with black
        if shift > 0:
            arr[:, shift:, ...] = arr[:, :-shift, ...]
            arr[:, :shift, ...] = 0
        else:
            s = -shift
            arr[:, :-s, ...] = arr[:, s:, ...]
            arr[:, -s:, ...] = 0

    return Image.fromarray(arr)


# -------------------------
# Pipeline loading
# -------------------------

def load_pipeline(model_id: str, device: torch.device, use_ddim: bool = True):
    dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if use_ddim:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)

    # small perf tweaks
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    # channels_last can help on mps/cuda
    pipe.unet.to(memory_format=torch.channels_last)

    return pipe


# -------------------------
# Sampling utils
# -------------------------

def read_prompts(prompt_file: str | None, start: int, end: int) -> list[str]:
    if prompt_file is None:
        # default: couple prompts repeated
        prompts = [
            "a high quality photo of a dog in a park",
            "a beautiful landscape with mountains and a lake",
        ] * 100
    else:
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        prompts = [l for l in lines if l]

    if end is None or end > len(prompts):
        end = len(prompts)
    return prompts[start:end]


def make_output_dirs(root: str, run_name: str):
    base = os.path.join(root, run_name)
    w_dir = os.path.join(base, "watermarked")
    a_dir = os.path.join(base, "attacked")
    os.makedirs(w_dir, exist_ok=True)
    os.makedirs(a_dir, exist_ok=True)
    return base, w_dir, a_dir


def save_pil_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


# -------------------------
# Evaluation
# -------------------------

@torch.no_grad()
def eval_watermark(
    reversed_latents_no_w: torch.Tensor,
    reversed_latents_w: torch.Tensor,
    watermarking_mask: torch.Tensor,
    gt_patch: torch.Tensor,
    w_channel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple evaluation metric inspired by the official repo.

    For each latent:
      - take FFT on chosen channel
      - compare masked region to gt_patch (MSE on complex values)
      - smaller value means "more like the watermark"

    Returns:
      no_w_metric: [B] tensor
      w_metric: [B] tensor
    """
    device = reversed_latents_no_w.device
    B, C, H, W = reversed_latents_no_w.shape

    if w_channel < 0 or w_channel >= C:
        raise ValueError(f"w_channel {w_channel} invalid for C={C}")

    # select channel
    lat_no = reversed_latents_no_w[:, w_channel, :, :]  # [B, H, W]
    lat_w = reversed_latents_w[:, w_channel, :, :]      # [B, H, W]

    fft_no = torch.fft.fft2(lat_no, dim=(-2, -1))       # [B, H, W]
    fft_w = torch.fft.fft2(lat_w, dim=(-2, -1))         # [B, H, W]

    mask = watermarking_mask.to(device).bool()          # [H, W]
    key = gt_patch.to(device)                           # [H, W] complex

    # flatten spatial dims
    mask_flat = mask.view(-1)                           # [HW]
    key_flat = key.view(-1)                             # [HW]

    fft_no_flat = fft_no.view(B, -1)                    # [B, HW]
    fft_w_flat = fft_w.view(B, -1)                      # [B, HW]

    key_flat = key_flat.unsqueeze(0)                    # [1, HW]

    diff_no = fft_no_flat - key_flat                    # [B, HW]
    diff_w = fft_w_flat - key_flat                      # [B, HW]

    # keep only masked positions
    diff_no_masked = diff_no[:, mask_flat]              # [B, Nmask]
    diff_w_masked = diff_w[:, mask_flat]                # [B, Nmask]

    # complex MSE per example
    mse_no = (diff_no_masked.abs() ** 2).mean(dim=1)    # [B]
    mse_w = (diff_w_masked.abs() ** 2).mean(dim=1)      # [B]

    return mse_no, mse_w


def compute_roc_and_print(no_w_metrics: list[float], w_metrics: list[float]):
    """
    Reproduce the GitHub style eval:

        preds = no_w_metrics + w_metrics
        labels = 0 for no watermark, 1 for watermark
        AUC, best accuracy, TPR at 1% FPR
    """
    # use negative metrics so that higher = more likely watermarked,
    # like in the original snippet (they did -no_w_metric etc.)
    preds = [-m for m in no_w_metrics] + [-m for m in w_metrics]
    labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    preds = np.array(preds)
    labels = np.array(labels)

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # best accuracy over all thresholds (same formula as snippet)
    acc = np.max(1.0 - (fpr + (1.0 - tpr)) / 2.0)

    # TPR at FPR < 1%
    if np.any(fpr < 0.01):
        idx = np.where(fpr < 0.01)[0][-1]
        tpr_at_1 = tpr[idx]
    else:
        tpr_at_1 = 0.0

    print(f"Number of no watermark samples: {len(no_w_metrics)}")
    print(f"Number of watermark samples:    {len(w_metrics)}")
    print(f"AUC:        {auc:.4f}")
    print(f"Best acc:   {acc:.4f}")
    print(f"TPR@1%FPR:  {tpr_at_1:.4f}")


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True,
                        help="Name for this run, used as output folder name.")
    parser.add_argument("--output_root", type=str, default="outputs",
                        help="Root folder for outputs.")
    parser.add_argument("--model_id", type=str,
                        default="CompVis/stable-diffusion-v1-4",
                        help="Hugging Face model id for Stable Diffusion.")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Text file with one prompt per line.")

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    # watermark params
    parser.add_argument("--w_channel", type=int, default=3,
                        help="Index of latent channel to watermark. -1 to watermark all.")
    parser.add_argument("--w_pattern", type=str, default="ring",
                        choices=["zeros", "rand", "ring"])
    parser.add_argument("--w_radius", type=float, default=8.0)
    parser.add_argument("--key_seed", type=int, default=12345)

    # attacks
    parser.add_argument("--apply_attack", action="store_true",
                        help="If set, apply attack to watermarked images.")
    parser.add_argument("--attack_type", type=str, default="both",
                        choices=["pixel_restoration", "horizontal_shift", "both"])
    parser.add_argument("--attack_downscale", type=int, default=4,
                        help="Downscale factor for pixel restoration.")
    parser.add_argument("--attack_shift_pixels", type=int, default=16,
                        help="Horizontal shift in pixels.")
    parser.add_argument("--attack_shift_mode", type=str, default="wrap",
                        choices=["wrap", "black"],
                        help="Wrap or black fill for horizontal shift.")

    args = parser.parse_args()

    # device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
    else:
        device = torch.device("cpu")
        device_type = "cpu"

    print(f"Using device: {device_type}")

    # pipeline
    pipe = load_pipeline(args.model_id, device=device, use_ddim=True)

    # compute latent spatial size from image size
    latent_h = args.height // 8
    latent_w = args.width // 8
    channels = pipe.unet.config.in_channels

    # watermarker over latent space
    watermarker = TreeRingWatermarker(
        height=latent_h,
        width=latent_w,
        w_channel=args.w_channel,
        w_radius=args.w_radius,
        w_pattern=args.w_pattern,
        key_seed=args.key_seed,
        device=device,
    ).to(device)

    gt_patch, watermark_mask = watermarker.get_gt_patch_and_mask()

    # output dirs
    base_dir, watermarked_dir, attacked_dir = make_output_dirs(
        args.output_root, args.run_name
    )

    # prompts
    prompts = read_prompts(args.prompt_file, args.start, args.end)

    # random generator for latents
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    # amp context (we avoid autocast on mps because it failed before)
    if device_type == "cuda":
        amp_ctx = torch.autocast(device_type=device_type, dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    no_w_metrics = []
    w_metrics = []

    num_images = len(prompts)
    num_batches = math.ceil(num_images / args.batch_size)

    print(f"Generating {num_images} prompts in {num_batches} batches into {base_dir}")

    idx_global = 0

    for b in range(num_batches):
        batch_prompts = prompts[b * args.batch_size:(b + 1) * args.batch_size]
        if not batch_prompts:
            break
        current_batch_size = len(batch_prompts)

        # sample no watermark latents
        latents_no_w = torch.randn(
            current_batch_size,
            channels,
            latent_h,
            latent_w,
            generator=g,
            device=device,
        )

        # watermarked latents
        latents_w = watermarker.apply(latents_no_w.clone())

        # eval on latents (no attack, clean setting)
        batch_no_w_metric, batch_w_metric = eval_watermark(
            reversed_latents_no_w=latents_no_w,
            reversed_latents_w=latents_w,
            watermarking_mask=watermark_mask,
            gt_patch=gt_patch,
            w_channel=args.w_channel if args.w_channel >= 0 else 0,  # if -1, just use channel 0 for eval
        )

        no_w_metrics.extend([m.item() for m in batch_no_w_metric])
        w_metrics.extend([m.item() for m in batch_w_metric])

        # SD expects latents in pipeline as starting point only when using latents directly,
        # but here we let the pipeline handle latents internally by giving a generator and seed.
        # To stay close to the official code, we pass latents explicitly.

        with amp_ctx:
            with torch.no_grad():
                # prepare latents in the format expected by diffusers
                # the pipeline interface uses latents as input when we set "latents" argument
                images_w = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    latents=latents_w,
                ).images

        # apply attacks and save
        for i, (img_w, prompt) in enumerate(zip(images_w, batch_prompts)):
            idx = idx_global
            idx_global += 1

            # save watermarked
            w_path = os.path.join(watermarked_dir, f"{idx:06d}.png")
            save_pil_image(img_w, w_path)

            # attacked
            if args.apply_attack:
                attacked_img = img_w

                if args.attack_type in ["pixel_restoration", "both"]:
                    attacked_img = pixel_restoration_attack(
                        attacked_img, args.attack_downscale
                    )

                if args.attack_type in ["horizontal_shift", "both"]:
                    attacked_img = horizontal_shift_attack(
                        attacked_img,
                        args.attack_shift_pixels,
                        mode=args.attack_shift_mode,
                    )

                a_path = os.path.join(attacked_dir, f"{idx:06d}.png")
                save_pil_image(attacked_img, a_path)

            if idx < 5:
                print(f"[{idx}] prompt: {prompt}")

    # final evaluation summary
    print("\nFinished generation. Running summary evaluation on latents.")
    compute_roc_and_print(no_w_metrics, w_metrics)


if __name__ == "__main__":
    main()

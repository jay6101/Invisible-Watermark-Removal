#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-Processing Pipeline for Watermark Removal

Implements test-time optimization and color/contrast transfer as described in the paper.
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers import AutoencoderKL
import lpips
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_img(path, device):
    """Load image and convert to tensor in [0, 1]"""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def to_m11(x):
    """Convert [0, 1] to [-1, 1]"""
    return x * 2 - 1


def to_01(x):
    """Convert [-1, 1] to [0, 1]"""
    return (x + 1) / 2


def gaussian(window_size, sigma):
    """Generate Gaussian window for SSIM"""
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create 2D Gaussian window for SSIM"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def compute_ssim(img1, img2, window_size=11, size_average=True):
    """Compute SSIM between two tensors in [-1, 1] range"""
    C1 = (0.01 * 2) ** 2
    C2 = (0.03 * 2) ** 2
    
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device).type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def test_time_optimize(x_r_init, x_w, vae_model_path, lpips_model, steps=10, lr=1e-4, device="cuda"):
    """
    Performs test-time VAE optimization per image (Algorithm 2 from paper).
    
    Args:
        x_r_init: Initial reconstruction from finetuned VAE [1,3,H,W] in [-1,1]
        x_w: Original watermarked image [1,3,H,W] in [-1,1]
        vae_model_path: Path to refiner VAE model
        lpips_model: LPIPS model for perceptual loss
        steps: Number of optimization steps
        lr: Learning rate
        device: Device to use
    
    Returns:
        x_opt: Optimized reconstruction [1,3,H,W] in [-1,1]
    """
    vae_refiner = AutoencoderKL.from_pretrained(
        vae_model_path,
        torch_dtype=torch.float32
    ).to(device)
    vae_refiner.train()
    
    optimizer = torch.optim.Adam(vae_refiner.parameters(), lr=lr)
    x_r = x_r_init.clone().detach()
    
    for t in range(steps):
        optimizer.zero_grad()
        
        posterior = vae_refiner.encode(x_r).latent_dist
        z = posterior.mean
        x_recon = vae_refiner.decode(z).sample
        
        # Paper Equation 2: MSE + LPIPS + 0.5*(1 - SSIM)
        mse_loss = F.mse_loss(x_recon, x_w)
        lpips_loss = lpips_model(x_recon, x_w).mean()
        ssim_val = compute_ssim(x_recon, x_w)
        ssim_loss = 0.5 * (1 - ssim_val)
        
        total_loss = mse_loss + lpips_loss + ssim_loss
        
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            x_r = x_recon.clone()
        
        if (t + 1) % 10 == 0:
            print(f"    Step [{t+1}/{steps}] MSE={mse_loss.item():.4f} "
                  f"LPIPS={lpips_loss.item():.4f} SSIM_Loss={ssim_loss.item():.4f}")
    
    vae_refiner.eval()
    with torch.no_grad():
        posterior = vae_refiner.encode(x_r).latent_dist
        z = posterior.mean
        x_opt = vae_refiner.decode(z).sample
    
    del vae_refiner
    torch.cuda.empty_cache()
    
    return x_opt


def color_contrast_transfer(x_opt, x_w):
    """
    Performs color and contrast transfer in CIELAB space.
    
    Args:
        x_opt: Optimized image [1,3,H,W] in [-1,1]
        x_w: Original watermarked image [1,3,H,W] in [-1,1]
    
    Returns:
        x_final: Final image [1,3,H,W] in [-1,1]
    """
    # Convert to [0,1] and numpy for skimage
    opt_np = to_01(x_opt).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    w_np = to_01(x_w).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    
    # Convert to CIELAB
    lab_opt = color.rgb2lab(opt_np)
    lab_w = color.rgb2lab(w_np)
    
    # Extract channels
    L_opt = lab_opt[:, :, 0]
    a_w = lab_w[:, :, 1]
    b_w = lab_w[:, :, 2]
    L_w = lab_w[:, :, 0]
    
    # Contrast transfer on luminance
    mu_c = L_opt.mean()
    sigma_c = L_opt.std()
    mu_w = L_w.mean()
    sigma_w = L_w.std()
    
    if sigma_c < 1e-6:
        sigma_c = 1e-6
    
    # Paper formula: L_final = (σ_w/σ_c)*(L_c - μ_c) + μ_w
    L_final = (sigma_w / sigma_c) * (L_opt - mu_c) + mu_w
    
    # Combine adjusted luminance with watermarked chrominance
    lab_final = np.stack([L_final, a_w, b_w], axis=2)
    
    # Convert back to RGB
    rgb_final = color.lab2rgb(lab_final)
    
    # Convert back to tensor in [-1, 1]
    x_final = torch.tensor(rgb_final).permute(2, 0, 1).unsqueeze(0).float().to(x_opt.device)
    x_final = to_m11(x_final)
    
    return x_final


def metric_psnr(a, b):
    """PSNR between two tensors in [-1, 1]"""
    a_np = to_01(a).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    b_np = to_01(b).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return peak_signal_noise_ratio(a_np, b_np, data_range=1.0)


def metric_ssim(a, b):
    """SSIM between two tensors in [-1, 1]"""
    a_np = to_01(a).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    b_np = to_01(b).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(a_np, b_np, channel_axis=2, data_range=1.0)


def metric_lpips(a, b, lpips_model):
    """LPIPS between two tensors in [-1, 1]"""
    return lpips_model(a, b).mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process VAE outputs with test-time optimization")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--finetuned_vae", type=str, required=True,
                        help="Path to fine-tuned VAE model")
    parser.add_argument("--refiner_vae", type=str, default="stabilityai/sdxl-vae",
                        help="Path to refiner VAE model")
    parser.add_argument("--output_dir", type=str, default="./postprocessed_outputs",
                        help="Directory to save final outputs")
    parser.add_argument("--results_csv", type=str, default="./postprocessing_metrics.csv",
                        help="Path to save metrics CSV")
    parser.add_argument("--test_time_steps", type=int, default=10,
                        help="Number of test-time optimization steps")
    parser.add_argument("--test_time_lr", type=float, default=1e-4,
                        help="Learning rate for test-time optimization")
    return parser.parse_args()


def main():
    args = parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print(f"\nLoading fine-tuned VAE from {args.finetuned_vae}...")
    vae_finetuned = AutoencoderKL.from_pretrained(
        args.finetuned_vae,
        torch_dtype=torch.float32
    ).to(DEVICE)
    vae_finetuned.eval()
    print("✓ Fine-tuned VAE loaded")
    
    print("\nLoading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
    print("✓ LPIPS model loaded")
    
    # Process all folders
    all_folders = sorted([d for d in os.listdir(args.data_root) if not d.startswith(".")])
    print(f"\nProcessing {len(all_folders)} folders...\n")
    
    records = []
    
    for folder in tqdm(all_folders, desc="Post-Processing"):
        fpath = os.path.join(args.data_root, folder)
        if not os.path.isdir(fpath):
            continue
        
        xw_path = os.path.join(fpath, f"{folder}_hidden.png")
        xi_path = os.path.join(fpath, f"{folder}_hidden_inv.png")
        
        if not (os.path.exists(xw_path) and os.path.exists(xi_path)):
            continue
        
        print(f"\n[{folder}]")
        
        # Load images
        x_w = to_m11(load_img(xw_path, DEVICE))
        x_i = to_m11(load_img(xi_path, DEVICE))
        
        # Step 1: Initial reconstruction
        print("  Step 1: Initial reconstruction (finetuned VAE)...")
        with torch.no_grad():
            posterior = vae_finetuned.encode(x_w).latent_dist
            z = posterior.mean
            x_r_init = vae_finetuned.decode(z).sample
        
        # Step 2: Test-time optimization
        print("  Step 2: Test-time optimization (refiner VAE)...")
        x_opt = test_time_optimize(x_r_init, x_w, args.refiner_vae, lpips_model,
                                   steps=args.test_time_steps, lr=args.test_time_lr, device=DEVICE)
        
        # Step 3: Color + Contrast transfer
        print("  Step 3: Color and contrast transfer (CIELAB)...")
        x_final = color_contrast_transfer(x_opt, x_w)
        
        # Save final output
        save_path = os.path.join(args.output_dir, f"{folder}_final.png")
        save_image(to_01(x_final), save_path)
        print(f"  Saved: {save_path}")
        
        # Compute metrics vs ground truth
        psnr_val = metric_psnr(x_final, x_i)
        ssim_val = metric_ssim(x_final, x_i)
        lpips_val = metric_lpips(x_final, x_i, lpips_model)
        
        print(f"  Metrics vs GT: PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}")
        
        records.append({
            "id": folder,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
        })
    
    # Save results
    df = pd.DataFrame(records)
    df.to_csv(args.results_csv, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.results_csv}")
    print(f"Output images saved to: {args.output_dir}")
    
    print("\n" + "="*60)
    print("AGGREGATED METRICS")
    print("="*60)
    print(f"PSNR: {df['psnr'].mean():.3f} dB")
    print(f"SSIM: {df['ssim'].mean():.4f}")
    print(f"LPIPS: {df['lpips'].mean():.4f}")
    print("="*60)


if __name__ == "__main__":
    main()


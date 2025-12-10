#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE Fine-tuning for StegaStamp Watermark Removal

This script fine-tunes a VAE to remove StegaStamp watermarks from images.
Based on Algorithm 1 from the paper.
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL


class WatermarkDataset(Dataset):
    """Dataset for paired watermarked and inverse watermarked images."""
    
    def __init__(self, root, transform=None):
        self.root = root
        self.tf = transform
        all_folders = sorted([d for d in os.listdir(root) if not d.startswith(".")])
        self.pairs = []
        
        for folder in all_folders:
            fpath = os.path.join(root, folder)
            if not os.path.isdir(fpath):
                continue
            xw = os.path.join(fpath, f"{folder}_hidden.png")
            xi = os.path.join(fpath, f"{folder}_hidden_inv.png")
            if os.path.exists(xw) and os.path.exists(xi):
                self.pairs.append((xw, xi))
        
        print(f"Loaded {len(self.pairs)} paired samples.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        xw_path, xi_path = self.pairs[idx]
        xw = Image.open(xw_path).convert("RGB")
        xi = Image.open(xi_path).convert("RGB")
        if self.tf:
            xw = self.tf(xw)
            xi = self.tf(xi)
        return xw, xi


def denorm(x):
    """Convert [-1, 1] to [0, 1]"""
    return ((x + 1) / 2).clamp(0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune VAE for watermark removal")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./vae_finetuned",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--vae_model", type=str, default="stabilityai/sdxl-vae",
                        help="Pre-trained VAE model to fine-tune")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of dataloader workers")
    return parser.parse_args()


def train_vae(args):
    """Main training function."""
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),  # range [0, 1]
    ])
    
    dataset = WatermarkDataset(args.data_root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                       num_workers=args.num_workers, pin_memory=True)
    
    # Load pre-trained VAE
    print(f"Loading VAE from {args.vae_model}...")
    vae = AutoencoderKL.from_pretrained(
        args.vae_model,
        torch_dtype=torch.float32
    ).to(DEVICE)
    vae.enable_slicing()
    vae.enable_tiling()
    vae = vae.to(memory_format=torch.channels_last)
    vae.train()
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        vae.train()
        total_loss = 0.0
        
        for step, (xw, xi) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"), start=1):
            xw, xi = xw.to(DEVICE, non_blocking=True), xi.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            xw = xw.to(memory_format=torch.channels_last)
            xi = xi.to(memory_format=torch.channels_last)
            
            # Scale to [-1, 1] (VAE expects this range)
            xw = xw * 2 - 1
            xi = xi * 2 - 1
            
            # Encode and decode
            posterior = vae.encode(xw).latent_dist
            z = posterior.mean
            x_inv_hat = vae.decode(z).sample
            loss = criterion(x_inv_hat, xi)
            
            if torch.isnan(loss):
                print("NaN loss detected – skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
            optimizer.step()
            
            total_loss += loss.item() * xw.size(0)
            
            if step % 100 == 0:
                print(f"[Epoch {epoch+1} | Step {step}] Loss = {loss.item():.6f}")
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Avg MSE Loss: {avg_loss:.6f}")
        
        # Save sample reconstructions
        with torch.no_grad():
            xw_vis, xi_vis = dataset[0]
            xw_vis = xw_vis.unsqueeze(0).to(DEVICE)
            xi_vis = xi_vis.unsqueeze(0).to(DEVICE)
            
            xw_vis = xw_vis.to(memory_format=torch.channels_last)
            xi_vis = xi_vis.to(memory_format=torch.channels_last)
            
            xw_vis = xw_vis * 2 - 1
            xi_vis = xi_vis * 2 - 1
            
            posterior_vis = vae.encode(xw_vis)
            z_vis = posterior_vis.latent_dist.sample()
            x_hat_vis = vae.decode(z_vis).sample
        
        save_image(
            torch.cat([denorm(xw_vis), denorm(x_hat_vis), denorm(xi_vis)], dim=0),
            os.path.join(args.output_dir, f"epoch{epoch+1}_recon.png"), 
            nrow=3
        )
        print(f"Saved reconstruction preview → epoch{epoch+1}_recon.png")
    
    # Make sure all model parameters are contiguous before saving
    for param in vae.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
    
    vae.save_pretrained(args.output_dir)
    print(f"\n✓ Fine-tuning complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train_vae(args)


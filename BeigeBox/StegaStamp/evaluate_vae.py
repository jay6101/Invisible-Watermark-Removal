#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE Evaluation with StegaStamp Decoding

Evaluates watermark removal by decoding 100-bit messages and computing Hamming distances.
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()


class InferenceDataset(Dataset):
    """Dataset for inference on watermarked images."""
    
    def __init__(self, root, transform=None):
        self.root = root
        self.tf = transform
        all_folders = sorted([d for d in os.listdir(root) if not d.startswith(".")])
        self.items = []

        for folder in all_folders:
            fpath = os.path.join(root, folder)
            if not os.path.isdir(fpath):
                continue

            xw_path = os.path.join(fpath, f"{folder}_hidden.png")
            secret_path = os.path.join(fpath, f"{folder}_secret.txt")

            if os.path.exists(xw_path) and os.path.exists(secret_path):
                self.items.append({
                    "id": folder,
                    "hidden": xw_path,
                    "secret": secret_path
                })

        print(f"Loaded {len(self.items)} items for evaluation.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        xw = Image.open(item["hidden"]).convert("RGB")
        if self.tf:
            xw = self.tf(xw)
        return xw, item["id"]

    def get_metadata(self):
        return self.items


def denorm(x):
    """Convert [-1, 1] to [0, 1]"""
    return ((x + 1) / 2).clamp(0, 1)


def load_decoder_handles(model_dir):
    """Load StegaStamp TensorFlow decoder model"""
    g = tf.Graph()
    sess = tf.compat.v1.Session(graph=g)
    meta = tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.SERVING], model_dir)
    sig = meta.signature_def["serving_default"]
    
    inp_image = g.get_tensor_by_name(sig.inputs["image"].name)
    inp_secret = g.get_tensor_by_name(sig.inputs["secret"].name)
    out_bits = g.get_tensor_by_name(sig.outputs["decoded"].name)
    
    return sess, {"image": inp_image, "secret": inp_secret}, out_bits


def preprocess_image_for_decoder(path):
    """Preprocess image for StegaStamp decoder: 400x400, RGB, [0,1]"""
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, (400, 400))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def decode_bits(sess, inputs_dict, out_tensor, img_path):
    """Decode 100-bit message from image using StegaStamp decoder"""
    if not os.path.isfile(img_path):
        return None
    
    img_arr = preprocess_image_for_decoder(img_path)
    dummy_secret = np.zeros((1, 100), dtype=np.float32)
    
    feed_dict = {
        inputs_dict["image"]: img_arr[np.newaxis, ...],
        inputs_dict["secret"]: dummy_secret
    }
    
    pred_bits = sess.run(out_tensor, feed_dict=feed_dict)
    pred_bits = np.squeeze(pred_bits)
    
    return pred_bits.astype(np.int64)


def read_secret_bits(txt_path):
    """Read 100-bit ground truth secret from text file"""
    with open(txt_path, "r") as f:
        secret_str = f.read().strip()
    
    # Extract only binary characters
    secret_str = "".join(ch for ch in secret_str if ch in "01")
    bits = np.fromiter((1 if c == '1' else 0 for c in secret_str), dtype=np.int64)
    
    assert bits.size == 100, f"Expected 100 bits, got {bits.size} from {txt_path}"
    return bits


def compute_hamming(pred_bits, true_bits):
    """Compute Hamming distance and accuracy"""
    if pred_bits is None or true_bits is None:
        return None, None
    
    L = min(len(pred_bits), len(true_bits))
    hamming_dist = int(np.sum(pred_bits[:L] != true_bits[:L]))
    accuracy = 1.0 - (hamming_dist / L)
    
    return hamming_dist, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAE watermark removal")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--vae_model", type=str, required=True,
                        help="Path to fine-tuned VAE model")
    parser.add_argument("--decoder_model", type=str, required=True,
                        help="Path to StegaStamp decoder model")
    parser.add_argument("--output_dir", type=str, default="./vae_outputs",
                        help="Directory to save VAE reconstructions")
    parser.add_argument("--results_csv", type=str, default="./evaluation_results.csv",
                        help="Path to save evaluation results CSV")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    return parser.parse_args()


def main():
    args = parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load fine-tuned VAE
    print(f"\nLoading fine-tuned VAE from {args.vae_model}...")
    vae = AutoencoderKL.from_pretrained(
        args.vae_model,
        torch_dtype=torch.float32
    ).to(DEVICE)
    vae.enable_slicing()
    vae.enable_tiling()
    vae = vae.to(memory_format=torch.channels_last)
    vae.eval()
    print("✓ VAE loaded successfully")
    
    # Setup dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    
    dataset = InferenceDataset(args.data_root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    metadata = dataset.get_metadata()
    
    # Run VAE reconstruction
    print("\nRunning VAE reconstruction...")
    with torch.no_grad():
        for xw_batch, ids_batch in tqdm(loader, desc="VAE Reconstruction"):
            xw_batch = xw_batch.to(DEVICE, non_blocking=True)
            xw_batch = xw_batch.to(memory_format=torch.channels_last)
            xw_batch = xw_batch * 2 - 1  # Scale to [-1, 1]
            
            posterior = vae.encode(xw_batch).latent_dist
            z = posterior.mean
            x_recon = vae.decode(z).sample
            x_recon = denorm(x_recon)
            
            for i in range(x_recon.size(0)):
                img_tensor = x_recon[i].cpu().float()
                img_id = ids_batch[i]
                save_path = os.path.join(args.output_dir, f"{img_id}_vae.png")
                img_pil = transforms.ToPILImage()(img_tensor)
                img_pil.save(save_path, 'PNG')
    
    print(f"✓ VAE reconstruction complete. Saved to {args.output_dir}")
    
    # Load StegaStamp decoder
    print(f"\nLoading StegaStamp decoder from {args.decoder_model}...")
    try:
        sess, inputs_dict, out_bits = load_decoder_handles(args.decoder_model)
        print("✓ StegaStamp decoder loaded successfully")
    except Exception as e:
        print(f"✗ Error loading decoder: {e}")
        return
    
    # Evaluate decoding and compute metrics
    print("\nEvaluating watermark removal...")
    records = []
    vae_files_found = 0
    
    for item in tqdm(metadata, desc="Decoding & Evaluation"):
        folder_id = item["id"]
        hidden_path = item["hidden"]
        secret_path = item["secret"]
        
        # Read ground truth secret
        true_bits = read_secret_bits(secret_path)
        
        # Decode original watermarked image
        pred_bits_orig = decode_bits(sess, inputs_dict, out_bits, hidden_path)
        ham_orig, acc_orig = compute_hamming(pred_bits_orig, true_bits)
        
        # Decode VAE reconstructed image
        vae_path = os.path.join(args.output_dir, f"{folder_id}_vae.png")
        pred_bits_vae = decode_bits(sess, inputs_dict, out_bits, vae_path)
        
        ham_vae, acc_vae = None, None
        if pred_bits_vae is not None:
            vae_files_found += 1
            ham_vae, acc_vae = compute_hamming(pred_bits_vae, true_bits)
        
        records.append({
            "id": folder_id,
            "hamming_original": ham_orig,
            "accuracy_original": acc_orig,
            "hamming_vae": ham_vae,
            "accuracy_vae": acc_vae
        })
    
    # Save results
    df = pd.DataFrame(records).sort_values("id")
    df.to_csv(args.results_csv, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.results_csv}")
    print(f"Total images evaluated: {len(records)}")
    print(f"VAE reconstructions found: {vae_files_found}/{len(records)}")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Original Watermarked Images:")
    print(f"  Mean Accuracy: {df['accuracy_original'].mean():.4f} ({df['accuracy_original'].mean()*100:.2f}%)")
    print(f"  Mean Hamming Distance: {df['hamming_original'].mean():.2f}/100 bits")
    
    print(f"\nVAE Reconstructed Images:")
    vae_acc_mean = df['accuracy_vae'].dropna().mean()
    vae_ham_mean = df['hamming_vae'].dropna().mean()
    print(f"  Mean Accuracy: {vae_acc_mean:.4f} ({vae_acc_mean*100:.2f}%)")
    print(f"  Mean Hamming Distance: {vae_ham_mean:.2f}/100 bits")
    
    detected_orig = (df['accuracy_original'] > 0.5).sum()
    detected_vae = (df['accuracy_vae'].dropna() > 0.5).sum()
    
    print(f"\n" + "="*70)
    print(f"WATERMARK DETECTION")
    print(f"="*70)
    print(f"Images with >50% bit accuracy (detected as watermarked):")
    print(f"  Original: {detected_orig}/{len(df)} ({100*detected_orig/len(df):.1f}%)")
    if vae_files_found > 0:
        print(f"  VAE Reconstructed: {detected_vae}/{vae_files_found} ({100*detected_vae/vae_files_found:.1f}%)")
        print(f"\nWatermark removal success: {100*(1 - detected_vae/vae_files_found):.1f}%")
    print("="*70)


if __name__ == "__main__":
    main()


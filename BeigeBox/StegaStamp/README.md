# StegaStamp Watermark Removal Using VAE

This repository contains tools for removing StegaStamp watermarks from images using Variational Autoencoders (VAE).

## Overview

The watermark removal process consists of three main stages:

1. **VAE Fine-tuning**: Train a VAE to reconstruct watermark-free images from watermarked inputs
2. **Evaluation**: Test the fine-tuned VAE and measure watermark removal effectiveness
3. **Post-processing**: Apply test-time optimization and color/contrast transfer for improved results

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

Your dataset should be organized as follows:

```
data_root/
├── 000000/
│   ├── 000000_hidden.png      # Watermarked image
│   ├── 000000_hidden_inv.png  # Ground truth (watermark-free)
│   └── 000000_secret.txt      # 100-bit secret (optional, for evaluation)
├── 000001/
│   ├── 000001_hidden.png
│   ├── 000001_hidden_inv.png
│   └── 000001_secret.txt
└── ...
```

## Usage

### 1. Train VAE

Fine-tune a VAE model to remove watermarks:

```bash
python train_vae.py \
  --data_root /path/to/dataset \
  --output_dir ./vae_finetuned \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-5
```

**Arguments:**
- `--data_root`: Path to dataset root directory
- `--output_dir`: Directory to save the fine-tuned model
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Training batch size (default: 8)
- `--lr`: Learning rate (default: 1e-5)
- `--vae_model`: Pre-trained VAE model (default: "stabilityai/sdxl-vae")

### 2. Evaluate VAE

Test watermark removal effectiveness using StegaStamp decoder:

```bash
python evaluate_vae.py \
  --data_root /path/to/dataset \
  --vae_model ./vae_finetuned \
  --decoder_model /path/to/stegastamp_decoder \
  --output_dir ./vae_outputs \
  --results_csv ./evaluation_results.csv
```

**Arguments:**
- `--data_root`: Path to dataset root directory
- `--vae_model`: Path to fine-tuned VAE model
- `--decoder_model`: Path to StegaStamp decoder model
- `--output_dir`: Directory to save VAE reconstructions
- `--results_csv`: Path to save evaluation results CSV
- `--batch_size`: Inference batch size (default: 16)

**Evaluation Metrics:**
- **Hamming Distance**: Number of differing bits between decoded and original secret
- **Accuracy**: Percentage of correctly decoded bits
- **Detection Rate**: Percentage of images with >50% bit accuracy

### 3. Post-process (Optional)

Apply test-time optimization and color/contrast transfer for improved results:

```bash
python postprocess_vae.py \
  --data_root /path/to/dataset \
  --finetuned_vae ./vae_finetuned \
  --refiner_vae stabilityai/sdxl-vae \
  --output_dir ./postprocessed_outputs \
  --results_csv ./postprocessing_metrics.csv \
  --test_time_steps 10
```

**Arguments:**
- `--data_root`: Path to dataset root directory
- `--finetuned_vae`: Path to fine-tuned VAE model
- `--refiner_vae`: Path to refiner VAE model (default: "stabilityai/sdxl-vae")
- `--output_dir`: Directory to save final outputs
- `--results_csv`: Path to save metrics CSV
- `--test_time_steps`: Number of test-time optimization steps (default: 10)
- `--test_time_lr`: Learning rate for test-time optimization (default: 1e-4)

**Quality Metrics:**
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)

## Notes

- GPU with at least 8GB VRAM is recommended
- Training time depends on dataset size 
- Post-processing is computationally intensive (test-time optimization per image)

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- StegaStamp decoder model (different conda env recommended, see README.md of StegaStamp)

## License

This code is provided for research purposes only.


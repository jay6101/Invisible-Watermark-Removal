# Tree-Ring Watermark: Horizontal Shift Attack Evaluation

This repository contains code for evaluating Tree-Ring watermarks under horizontal shift attacks.

Based on the paper: [Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](http://arxiv.org/abs/2305.20030)

## About

Tree-Ring Watermarking embeds watermarks in diffusion model outputs by modifying the initial noise array's Fourier transform. The watermark can be detected by inverting the diffusion process and checking for the pattern in the frequency domain.

This implementation evaluates the robustness of Tree-Ring watermarks against **horizontal shift attacks** (translate-and-restore).

## Dependencies

- PyTorch == 1.13.0
- transformers == 4.23.1
- diffusers == 0.11.1
- datasets

**Note:** Higher diffusers versions may not be compatible with the DDIM inversion code.

## Usage

### Horizontal Shift Attack

Run the watermark evaluation with horizontal shift attack:

```bash
python run_tree_ring_watermark.py \
  --run_name horizontal_shift_attack \
  --w_channel 3 \
  --w_pattern ring \
  --attack_shift 14 \
  --start 0 \
  --end 100 \
  --save_images \
  --image_out_dir ./outputs
```

**Key Arguments:**
- `--run_name`: Name for this experimental run
- `--w_channel`: Watermarked channel index (0-3, or -1 for all channels)
- `--w_pattern`: Watermark pattern type (`ring` recommended)
- `--attack_shift`: Number of pixels to shift horizontally (default: 40)
- `--start`, `--end`: Range of images to process
- `--save_images`: Save generated images to disk
- `--image_out_dir`: Output directory for images and results

### Output Structure

The script generates the following directory structure:

```
outputs/
└── {run_name}/
    ├── clean/                    # Non-watermarked images
    ├── watermarked/              # Watermarked images
    ├── attacked_clean/           # Shifted non-watermarked images
    ├── attacked_watermarked/     # Shifted watermarked images
    └── results.json              # Detection metrics and results
```

### Results Metrics

The `results.json` file contains:
- **AUC**: Area under ROC curve
- **TPR@1%FPR**: True positive rate at 1% false positive rate
- **TPR_on_attacked_at_thresh**: Detection rate on attacked watermarked images
- Per-sample detection flags for clean, watermarked, and attacked images

## Pre-computed Results

Pre-computed results for 100 and 1000 images are available at:
[https://drive.google.com/drive/folders/14u5MMsiTd70ywTCJc_QNICoiod_Qo6lP?usp=sharing](https://drive.google.com/drive/folders/14u5MMsiTd70ywTCJc_QNICoiod_Qo6lP?usp=sharing)

- `outputs_100/`: Results for 100 images
- `outputs_1000/`: Results for 1000 images

## Notes

- The horizontal shift attack translates images to the right and restores the left columns
- The script evaluates watermark detection on clean, watermarked, and attacked images
- Results are saved as JSON for further analysis

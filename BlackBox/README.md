# Invisible Watermark Removal — Black Box Track

This directory implements the cluster-specific black-box approach from the paper ["First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge"](https://arxiv.org/abs/2508.21072) for removing invisible watermarks from images. This is an unofficial implementation created since the official code is not publicly available.

## Available Resources

All datasets, models, and results are available on Google Drive:

**[Black Box Resources & Results](https://drive.google.com/drive/folders/1wKitdXSulUtoGOtZvQ7wxpyFKszRRUhD?usp=sharing)**

This folder contains:
- `BlackBox/`: Complete pipeline implementation and results
- `StegaStamp_100images/`: StegaStamp test dataset (100 images)
- `Treering_BeigeBox/`: Tree-Ring watermark evaluation outputs
- `VAE_finetuned_model/`: Fine-tuned VAE model weights
- `400x400_paired_data.tar.gz`: Training dataset (916.8 MB)
- `400x400_test_paired_data.zip`: Test dataset (288.9 MB)
- `clean_300_test_dataset.zip`: Clean test images (166.3 MB)
- `HF_SD2.1_prompts.csv`: Prompts for image generation

---

## Overview

### Cluster-Specific Removal Strategy

Images are assigned to one of four clusters based on observed watermark artifacts, with each cluster using a tailored removal technique:

| Cluster | Artifacts | Method | Implementation |
|:-------:|-----------|--------|----------------|
| **1** | No visible artifacts | Diffusion Refiner | `diffusion_pipeline.py` (strength = 0.16) |
| **2** | Boundary artifacts | VAE + Test-Time Optimization | `vae_pipeline.py` |
| **3** | Circular Fourier patterns | VAE + Test-Time Optimization | `vae_pipeline.py` |
| **4** | Square Fourier patterns | Diffusion + Horizontal Shift | `diffusion_pipeline.py` (strength = 0.04) + `treering_pipeline.py` (shift = 7px) |

### Repository Structure

```
BlackBox/
├── data/
│   ├── blackbox_images/          # Input images (0.png ... 299.png)
│   ├── blackbox_outputs/         # Final outputs per cluster
│   ├── clustered_images/         # Images grouped by cluster
│   ├── cluster_mapping.csv       # Filename → cluster_id mapping
│   └── HF_SD2.1_prompts.csv     # Image generation prompts
├── models/                        # VAE model weights & config
├── results/                       # Evaluation metrics (PSNR/SSIM/LPIPS/FID)
├── blackbox_pipeline.py          # Main pipeline (Algorithm 3)
├── diffusion_pipeline.py         # Diffusion-based removal
├── vae_pipeline.py              # VAE-based removal (Algorithm 2)
├── treering_pipeline.py         # Horizontal shift attack
├── build_cluster.py             # Create cluster folders
└── evaluate_blackbox_image_quality_metrics.py  # Compute metrics
```

---

## Setup

### Environment

**Requirements:**
- Python 3.10
- CUDA-capable GPU

**Installation:**
```bash
conda create -n watermark python=3.10
conda activate watermark
pip install -r requirements.txt
```

**Hugging Face Models:**

Login to Hugging Face (if models not cached):
```bash
huggingface-cli login
```

Required models:
- `stabilityai/stable-diffusion-xl-refiner-1.0` (image-to-image refiner)
- `stabilityai/sdxl-vae` (test-time VAE optimization)

**Optional - OpenAI API (for caption generation):**
```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

### Data Preparation

**1. Download Data**

Download datasets from [Google Drive](https://drive.google.com/drive/folders/1wKitdXSulUtoGOtZvQ7wxpyFKszRRUhD?usp=sharing) and extract to:
```
data/blackbox_images/         # Input images (0.png ... 299.png)
data/HF_SD2.1_prompts.csv    # Prompts for image generation
models/                       # VAE model weights
```

**2. Build Cluster Folders**

Organize images by cluster:
```bash
python build_cluster.py
```

This creates:
```
data/clustered_images/
├── cluster_1_no_artifacts/
├── cluster_2_boundary_artifacts/
├── cluster_3_circular_patterns/
└── cluster_4_square_patterns/
```

**3. (Optional) Generate Image Captions for Cluster 1**

You have two options for Cluster 1 captions:

1. **Use our pre generated captions **
   Download `cluster_mapping_with_captions.csv` from our shared drive and place it in the `data/` directory:

   Link: [cluster_mapping_with_captions.csv](https://drive.google.com/drive/folders/1GQB8-zycc7-A-BJqiUEnQ62at6LLfkfs) 

2. **Generate captions yourself using GPT** Run:
   ```bash
   python generate_cluster1_image_captions.py
   python cluster1_caption_mapping.py

This creates `data/cluster_mapping_with_captions.csv` used by the pipeline.

## Usage

### Main Pipeline (Algorithm 3)

Run the complete black box watermark removal pipeline:

```bash
python blackbox_pipeline.py
```

**What it does:**
1. Reads `data/cluster_mapping_with_captions.csv`
2. Routes each image to its cluster-specific pipeline
3. Applies the appropriate removal technique:
   - **Cluster 1**: Diffusion refiner (strength = 0.16)
   - **Clusters 2 & 3**: VAE + test-time optimization
   - **Cluster 4**: Mild diffusion (strength = 0.04) + horizontal shift (7px)
4. Saves outputs to `data/blackbox_outputs/cluster_k/`

**Output:**
```
data/blackbox_outputs/
├── cluster_1/<id>_refined.png
├── cluster_2/<id>_refined.png
├── cluster_3/<id>_refined.png
└── cluster_4/<id>_refined.png
```

**Note:** Existing outputs are skipped (restart-friendly).

### Pipeline Components

#### 1. Diffusion Refiner (`diffusion_pipeline.py`)

Uses `StableDiffusionXLImg2ImgPipeline` for watermark removal.

```python
remove_watermark_diffusion(
    image_path: str,
    save_path: Optional[str] = None,
    caption: str = " ",
    s: float = 0.16,  # Image-to-image strength
    T: int = 500,
) -> PIL.Image.Image
```

**Used in:** Clusters 1 and 4

#### 2. VAE + Test-Time Optimization (`vae_pipeline.py`)

Implements Algorithm 2 with fine-tuned VAE weights.

**Process:**
1. Encode watermarked image with fine-tuned VAE
2. Initialize fresh `stabilityai/sdxl-vae` refiner
3. Run test-time optimization with loss:
   ```
   MSE(D(E(x_r)), x_w) + LPIPS + 0.5 * (1 - SSIM)
   ```
4. Apply color/contrast transfer in CIELAB space

```python
remove_watermark_vae(
    image_path: str,
    save_path: Optional[str] = None,
    steps: int = 2,
    lr: float = 1e-4,
) -> torch.Tensor
```

**Used in:** Clusters 2 and 3

#### 3. Horizontal Shift Attack (`treering_pipeline.py`)

TreeRing-specific defense using horizontal translation.

```python
translate_and_restore_left_pil(
    original_img: Image.Image,
    diffused_img: Image.Image,
    shift: int = 7,
) -> Image.Image
```

**Process:**
1. Resize diffused image to match original
2. Shift image right by `shift` pixels
3. Restore leftmost `shift` columns from original

**Used in:** Cluster 4

### Evaluation

Compute image quality metrics:

```bash
python evaluate_blackbox_image_quality_metrics.py
```

**Metrics computed:**
- **Per-image:** PSNR, SSIM, LPIPS
- **Dataset-level:** FID (Inception V3), CLIP-FID (ViT-B/32)

**Output:**
- Per-cluster CSV: `results/blackbox_img_quality_metrics/cluster_k_metrics.csv`
- Console: Aggregate metrics for each cluster

---

## Additional Experiments

### StegaStamp Dataset Testing

Test diffusion pipeline on StegaStamp-style images with varying strength parameters:

```bash
python test_on_stegastamp100.py
```

**Output:** `data/cluster1_outputs/beigebox_test_output_s*/`

### Forward Diffusion Visualization

Generate noisy images at different diffusion strengths:

```python
from noisy_latents import get_noisy_image_and_latents

get_noisy_image_and_latents(
    image_path="path/to/image.png",
    strength=0.16,
    num_inference_steps=500,
    save_path="output.png"
)
```

---

## Quick Start

```bash
# Setup environment
conda create -n watermark python=3.10
conda activate watermark
pip install -r requirements.txt

# Download data from Google Drive
# https://drive.google.com/drive/folders/1wKitdXSulUtoGOtZvQ7wxpyFKszRRUhD?usp=sharing

# Organize images by cluster
python build_cluster.py

# (Optional) Generate captions for cluster 1
python generate_cluster1_image_captions.py
python cluster1_caption_mapping.py

# Run watermark removal
python blackbox_pipeline.py

# Evaluate results
python evaluate_blackbox_image_quality_metrics.py
```

---

## Notes

- **Image Resolutions:**
  - Main dataset: 512×512
  - StegaStamp: 400×400
- **Reproducibility:** Use fixed random seeds for consistent results
- **Metrics:** Input/output shapes are automatically matched during evaluation

## License

This code is provided for research and educational purposes only. This is an unofficial implementation based on the methodology described in the [original paper](https://arxiv.org/abs/2508.21072).

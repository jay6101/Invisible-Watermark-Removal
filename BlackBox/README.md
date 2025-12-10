# Invisible Watermark Removal — Black Box Track

This repository reimplements the Black Box track pipeline for the Invisible Watermark Removal Challenge. The goal is to remove TreeRing-style invisible watermarks from images using a cluster-specific, black-box approach.

---

### Google Drive: 
Fetch 
1. data/blackbox_images
2. data/HF_SD2.1_prompts.csv

---

## Cluster-Specific Removal Strategy

The pipeline first assigns each image to one of four clusters (based on observed watermark artifacts), then applies a tailored removal technique:

| Cluster ID | Observed Artifacts | Removal Pipeline | Key Component |
| :---: | :--- | :--- | :--- |
| **1** | No visible artifacts | Diffusion Refiner | `diffusion_pipeline.py` (strength `s=0.16`) |
| **2** | Boundary artifacts | VAE Test-Time Optimization | `vae_pipeline.py` (short TTO loop) |
| **3** | Circular Fourier patterns | VAE Test-Time Optimization | `vae_pipeline.py` |
| **4** | Square Fourier patterns | Mild Diffusion + Horizontal Translation | `diffusion_pipeline.py` (`s=0.04`) + `treering_pipeline.py` (shift `=7`) |

---

## Repository Overview

Top-level layout (see `BlackBox/` for the implementation):

- `BlackBox/`
  - `data/`
    - `blackbox_images/` — input images (e.g. `0.png` ... `299.png`)
    - `blackbox_outputs/` — per-cluster final outputs
    - `clustered_images/` — images grouped per cluster
    - `cluster_mapping.csv` — filename → cluster_id mapping
  - `models/` — model weights & configs (`config.json`, `diffusion_pytorch_model.safetensors`)
  - `results/` — computed metrics (PSNR/SSIM/LPIPS/FID/CLIP-FID)
  - `build_cluster.py` — utility to create cluster folders
  - `blackbox_pipeline.py` — main orchestration (routing + removal)
  - `diffusion_pipeline.py` — diffusion refiner and helpers
  - `vae_pipeline.py` — VAE-based removal + test-time optimization
  - `treering_pipeline.py` — horizontal shift / restore helper (cluster 4)
  - `evaluate_blackbox_image_quality_metrics.py` — metric computation
  - `generate_cluster1_image_captions.py`, `cluster1_caption_mapping.py` — optional caption helpers

---

## Environment setup

This project assumes Python 3.10 with CUDA available.

Create conda environment

```bash
conda create -n watermark python=3.10
conda activate watermark
```

Install dependencies

```bash
pip install -r requirements.txt
```

You also need access to the following Hugging Face models:

- `stabilityai/stable-diffusion-xl-refiner-1.0` (used as an image to image refiner)
- `stabilityai/sdxl-vae` (used for test time VAE optimization)

If these are not already cached you must be logged in to the Hugging Face Hub before running:

```bash
huggingface-cli login
```

For caption generation (generate_cluster1_image_captions.py) you need an OpenAI API key set as the environment variable:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## Data preparation

1. Place black box input images

Put the challenge images into:

```
data/blackbox_images/0.png
data/blackbox_images/1.png
...
data/blackbox_images/299.png
```

2. Build cluster folders

`build_cluster.py` copies each image into its cluster specific directory using the manual cluster indices from the paper.

Run:

```bash
python build_cluster.py
```

This creates:

```
data/clustered_images/cluster_1_no_artifacts/
data/clustered_images/cluster_2_boundary_artifacts/
data/clustered_images/cluster_3_circular_patterns/
data/clustered_images/cluster_4_square_patterns/
```

3. Cluster mapping CSV

`data/cluster_mapping.csv` contains at least:

```
filename,cluster_id
0.png,1
1.png,2
...
```

If you do not have this yet you can generate it manually from the same index lists used in `build_cluster.py`.

### Optional step - caption generation for cluster 1

For the diffusion refiner we optionally use short image captions.

1. Generate GPT captions

`generate_cluster1_image_captions.py` runs through `data/clustered_images/cluster_1_no_artifacts` and writes a text caption per image plus a JSON file.

Edit the settings at the top if needed:

```python
image_dir = "data/clustered_images/cluster_1_no_artifacts"
caption_dir = "data/gpt_captions_cluster1"
max_captions = 100
```

Then run:

```bash
python generate_cluster1_image_captions.py
```

2. Merge captions into mapping

`cluster1_caption_mapping.py` adds a caption column to `cluster_mapping.csv` for rows with `cluster_id == 1`.

```bash
python cluster1_caption_mapping.py
```

This creates `data/cluster_mapping_with_captions.csv` which is used by the main pipeline.
Images in clusters 2, 3 and 4 keep an empty caption string.

## Main black box pipeline

This is the core implementation of Algorithm 3 from the paper.

Script

`blackbox_pipeline.py`

Reads `data/cluster_mapping_with_captions.csv`

For each image routes to the appropriate cluster pipeline

Writes outputs into `data/blackbox_outputs/cluster_k`

Cluster logic:

- `cluster_id == 1` - uses `diffusion_pipeline.remove_watermark_diffusion` with `s = 0.16`
- `cluster_id in {2, 3}` - uses `vae_pipeline.remove_watermark_vae`
- `cluster_id == 4` - runs mild diffusion with `s = 0.04` then `treering_pipeline.translate_and_restore_left_pil` to shift by 7 pixels and restore the leftmost columns from the original image

Run:

```bash
python blackbox_pipeline.py
```

Outputs:

```
data/blackbox_outputs/cluster_1/<id>_refined.png
data/blackbox_outputs/cluster_2/<id>_refined.png
data/blackbox_outputs/cluster_3/<id>_refined.png
data/blackbox_outputs/cluster_4/<id>_refined.png
```

If an output file already exists it is skipped so the script is restart friendly.

## Diffusion refiner

`diffusion_pipeline.py` wraps `StableDiffusionXLImg2ImgPipeline` from `stabilityai/stable-diffusion-xl-refiner-1.0`.

Key function:

```python
remove_watermark_diffusion(
    image_path: str,
    save_path: Optional[str] = None,
    caption: str = " ",
    s: float = 0.16,
    T: int = 500,
) -> PIL.Image.Image
```

`s` controls the image to image strength

If `save_path` is provided the refined image is written to disk

The pipeline is created once and cached

This is used both in the main pipeline and in experiments such as forward diffusion.

## VAE based removal

`vae_pipeline.py` implements Algorithm 2 from the paper using your finetuned VAE weights.

Loads finetuned VAE once from `models/` using `AutoencoderKL.from_pretrained`

For each call:

- Encodes the watermarked image with the finetuned VAE to get an initial reconstruction
- Instantiates a fresh `stabilityai/sdxl-vae` refiner
- Runs a short test time optimization loop over the refiner parameters with loss

  MSE(D(E(x_r)), x_w) + LPIPS + 0.5 * (1 - SSIM)

- Applies color plus contrast transfer in CIELAB space to match chrominance and luminance statistics of the original watermarked image
- Saves the final tensor as an image if `save_path` is provided

Entry point:

```python
remove_watermark_vae(
    image_path: str,
    save_path: Optional[str] = None,
    steps: int = 2,
    lr: float = 1e-4,
) -> torch.Tensor
```

This is used for clusters 2 and 3.

## TreeRing translation step

`treering_pipeline.py` contains the horizontal shift defense used in the paper for TreeRing style watermarks:

```python
translate_and_restore_left_pil(
    original_img: Image.Image,
    diffused_img: Image.Image,
    shift: int = 7,
) -> Image.Image
```

Steps:

- Resize the diffused image to match the original
- Shift the diffused image by `shift` pixels to the right
- Restore the leftmost `shift` columns from the original image

This is used for `cluster_id == 4` inside `blackbox_pipeline.py`.

## Evaluation

`evaluate_blackbox_image_quality_metrics.py` computes:

- Per image and mean PSNR, SSIM, LPIPS between input and refined images
- Dataset level FID using Inception V3 features
- Dataset level CLIP FID using CLIP ViT B 32 image embeddings

It expects the folders listed in the clusters list at the top of the script, which reference `data/clustered_images/...` as inputs and `data/blackbox_outputs/...` as outputs.

Run:

```bash
python evaluate_blackbox_image_quality_metrics.py
```

Outputs:

- Per cluster CSV files in `results/blackbox_img_quality_metrics/cluster_k_metrics.csv`
- Aggregate metrics printed to the console for each cluster

If a CSV for a cluster already exists the script reuses it for per image statistics and only recomputes FID and CLIP FID.

## StegaStamp and forward diffusion experiments

### StegaStamp SD2.1 style dataset

`test_on_stegastamp100.py` applies the diffusion pipeline to a subset of StegaStamp style images to study how the strength parameter `s` affects watermark removal and image quality.

Assumes:

- Images in `data/clustered_images/beigebox_images/<idx>_hidden.png`
- Matching text prompts in `data/HF_SD2.1_prompts.csv`

Run:

```bash
python test_on_stegastamp100.py
```

Outputs go into:

```
data/cluster1_outputs/beigebox_test_output_s01/
...
data/cluster1_outputs/beigebox_test_output_s20/
```

### Forward diffusion visualizations

`noisy_latents.py` uses the SDXL refiner VAE to encode an image, apply forward diffusion noise for a chosen strength, decode back and optionally save the noisy image.

The function:

```python
get_noisy_image_and_latents(
    image_path: str,
    strength: float = 0.16,
    num_inference_steps: int = 500,
    save_path: Optional[str] = None,
) -> tuple[PIL.Image.Image, torch.Tensor]
```

You can use this to populate `data/fwd_diffusion_output/cluster_1_no_artifacts` with different strengths and show a 3 by 2 grid of noise levels in your slides.

## Reproducibility tips

- Keep a fixed random seed whenever sampling noise or generating SD images
- Use the same image resolutions as the paper
  - `512 x 512` for the main dataset
  - `400 x 400` for StegaStamp encoding
- For metrics, always make sure input and output shapes match. The evaluation script resizes the input image to the output size when needed.

## Running the full pipeline end to end

Create and activate the conda environment
```bash
conda create -n watermark python=3.10
conda activate watermark
pip install -r requirements.txt
```

Place black box input images into `data/blackbox_images`

Run clustering
```bash
python build_cluster.py
```

Optional: Generate captions and merge into mapping
```bash
python generate_cluster1_image_captions.py
python cluster1_caption_mapping.py
```

Run the black box pipeline
```bash
python blackbox_pipeline.py
```

Evaluate metrics
```bash
python evaluate_blackbox_image_quality_metrics.py
```

Optionally run StegaStamp and forward diffusion experiments
```bash
python test_on_stegastamp100.py
```
or import and call `get_noisy_image_and_latents` interactively

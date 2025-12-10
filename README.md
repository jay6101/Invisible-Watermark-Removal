# Invisible Watermark Removal Challenge — 1st Place Solution

**NeurIPS 2024 Competition**

This repository contains our first-place solution for the Invisible Watermark Removal Challenge. We tackle two main tracks: **Black Box** (cluster-specific removal) and **Beige Box** (known watermark methods).

## Competition Resources

All datasets, models, and results are available here:

**[Google Drive - Complete Resources](https://drive.google.com/drive/folders/1wKitdXSulUtoGOtZvQ7wxpyFKszRRUhD?usp=sharing)**

Includes:
- Black Box and Beige Box implementations with results
- Training and test datasets (400×400 paired data)
- Fine-tuned VAE models
- StegaStamp test dataset (100 images)
- Evaluation outputs and presentations

---

## Solution Overview

### Black Box Track

**Objective:** Remove TreeRing-style watermarks using a cluster-specific approach without knowing the exact watermark method.

**Strategy:** Images are classified into 4 clusters based on artifact patterns, with each cluster using a tailored removal technique:
- **Cluster 1** (No artifacts): Diffusion refiner
- **Cluster 2** (Boundary artifacts): VAE + test-time optimization
- **Cluster 3** (Circular patterns): VAE + test-time optimization
- **Cluster 4** (Square patterns): Diffusion + horizontal shift attack

**[See detailed Black Box documentation →](BlackBox/README.md)**

### Beige Box Track

**Objective:** Remove known watermark types (StegaStamp, Tree-Ring) using specialized techniques.

#### StegaStamp Watermark Removal
Uses fine-tuned VAE with test-time optimization and color/contrast transfer to remove StegaStamp watermarks.

**Pipeline:**
1. Fine-tune VAE on watermarked/clean image pairs
2. Apply test-time optimization with perceptual losses
3. Color/contrast transfer in CIELAB space

**[See StegaStamp documentation →](BeigeBox/StegaStamp/README.md)**

#### Tree-Ring Watermark Evaluation
Evaluates Tree-Ring watermark robustness against horizontal shift attacks.

**Approach:**
- Translate images horizontally and restore original columns
- Measure watermark detection rates on attacked images
- Compare clean, watermarked, and attacked detection metrics

**[See Tree-Ring documentation →](BeigeBox/tree-ring-watermark/README.md)**

---

## Quick Start

### Black Box
```bash
cd BlackBox
conda create -n watermark python=3.10
conda activate watermark
pip install -r requirements.txt
python blackbox_pipeline.py
```

### Beige Box - StegaStamp
```bash
cd BeigeBox/StegaStamp
pip install -r requirements.txt

# Use pre-trained model
python evaluate_vae.py \
  --data_root /path/to/data \
  --vae_model /path/to/pretrained_vae \
  --decoder_model /path/to/decoder
```

### Beige Box - Tree-Ring
```bash
cd BeigeBox/tree-ring-watermark
pip install -r requirements.txt

python run_tree_ring_watermark.py \
  --run_name horizontal_shift_attack \
  --w_channel 3 \
  --w_pattern ring \
  --attack_shift 14
```

---

## Repository Structure

```
Invisible-Watermark-Removal/
├── BlackBox/                      # Black box track implementation
│   ├── README.md                  # Detailed documentation
│   ├── blackbox_pipeline.py       # Main pipeline (Algorithm 3)
│   ├── diffusion_pipeline.py      # Diffusion-based removal
│   ├── vae_pipeline.py           # VAE-based removal
│   └── treering_pipeline.py      # Horizontal shift attack
│
├── BeigeBox/                      # Beige box track implementations
│   ├── StegaStamp/               # StegaStamp watermark removal
│   │   ├── README.md             # Detailed documentation
│   │   ├── train_vae.py          # VAE fine-tuning
│   │   ├── evaluate_vae.py       # Evaluation with decoder
│   │   └── postprocess_vae.py    # Test-time optimization
│   │
│   └── tree-ring-watermark/      # Tree-Ring evaluation
│       ├── README.md             # Detailed documentation
│       └── run_tree_ring_watermark.py  # Main evaluation script
│
└── README.md                      # This file
```

---

## Key Results

- **Black Box**: Successfully removed watermarks from 300 images using cluster-specific pipelines
- **StegaStamp**: Achieved >90% watermark removal (Hamming distance ~50/100 bits)
- **Tree-Ring**: Evaluated robustness against horizontal shift attacks

Detailed metrics and results are available in the respective subdirectory READMEs.

---

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- CUDA-capable GPU (8GB+ VRAM recommended)
- See individual `requirements.txt` files for track-specific dependencies

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{invisible-watermark-removal-2024,
  title={Invisible Watermark Removal Challenge - 1st Place Solution},
  author={Your Team},
  booktitle={NeurIPS 2024 Competition Track},
  year={2024}
}
```

---

## License

This code is provided for research purposes only.

---

## Acknowledgments

- NeurIPS 2024 Competition Organizers
- Stability AI for pre-trained models
- StegaStamp and Tree-Ring watermarking papers

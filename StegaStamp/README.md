## StegaStamp: Invisible Hyperlinks in Physical Photographs [[Project Page]](http://www.matthewtancik.com/stegastamp)

### CVPR 2020
**[Matthew Tancik](https://www.matthewtancik.com), [Ben Mildenhall](http://people.eecs.berkeley.edu/~bmild/), [Ren Ng](https://scholar.google.com/citations?hl=en&user=6H0mhLUAAAAJ)**
*University of California, Berkeley*

![](https://github.com/tancik/StegaStamp/blob/master/docs/teaser.png)


## Introduction
This repository contains code for encoding and decoding invisible watermarks in images using StegaStamp. The project explores hiding data (100-bit binary secrets) in images while maintaining perceptual similarity. This implementation uses pretrained models to encode random binary secrets into images and decode them back.

## Citation
If you find our work useful, please consider citing:
```
    @inproceedings{2019stegastamp,
        title={StegaStamp: Invisible Hyperlinks in Physical Photographs},
        author={Tancik, Matthew and Mildenhall, Ben and Ng, Ren},
        booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2020}
    }
```

## Installation

### Environment Setup
Create and activate the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate stega
```

This will install Python 3.7, TensorFlow 1.13, and all required dependencies including:
- TensorFlow 1.13.1
- NumPy
- Pillow
- OpenCV
- bchlib
- tqdm

### Download Pretrained Models
Download the pretrained StegaStamp models from Google Drive:

[Download Models](https://drive.google.com/file/d/1mwJ6YBWrEN7OpgyWZGFT0qk3D6bUjF_6/view?usp=sharing)

Extract the models to a `saved_models/` directory in your project folder.

## Encoding a Message
The script `encode_image.py` encodes a 100-bit random binary secret into an image. The encoder generates both the watermarked image and its inverse version.

### Encode a single image:

```bash
python encode_image.py \
  saved_models/stegastamp_pretrained \
  --image test_im.png \
  --save_dir out/ \
  --secret "Stega!!"
```

### Encode a directory of images:

```bash
python encode_image.py \
  saved_models/stegastamp_pretrained \
  --images_dir input_images/ \
  --save_dir out/
```

### Output Files
For each input image, the encoder creates a directory containing:
- `*_hidden.png` - Watermarked image
- `*_hidden_inv.png` - Inverse watermarked image
- `*_residual.png` - Residual applied to the original image
- `*_residual_inv.png` - Inverse residual
- `*_secret.txt` - The binary secret (100 bits)
- `*_secret_inv.txt` - The inverted binary secret

**Note:** Images are automatically resized to 400x400 pixels.

## Decoding a Message
The script `decode_image.py` decodes the binary secret from a StegaStamp watermarked image. The decoded secret can be printed to the console and optionally saved to text files.

### Decode a single image:

```bash
python decode_image.py \
  saved_models/stegastamp_pretrained \
  --image out/000000/000000_hidden.png
```

### Decode a single image and save the result:

```bash
python decode_image.py \
  saved_models/stegastamp_pretrained \
  --image out/000000/000000_hidden.png \
  --save_dir decoded_output/
```

### Decode a directory of images:

```bash
python decode_image.py \
  saved_models/stegastamp_pretrained \
  --images_dir out/
```

### Decode a directory of images and save all results:

```bash
python decode_image.py \
  saved_models/stegastamp_pretrained \
  --images_dir out/ \
  --save_dir decoded_output/
```

### Output
- The decoded binary secret (100 bits) is printed to the console
- If `--save_dir` is specified, each decoded secret is saved to a text file named `{original_filename}_decoded.txt`

## Requirements
- Python 3.7
- TensorFlow 1.13
- CUDA-compatible GPU (recommended for faster processing)

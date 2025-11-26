# src/tree_ring_watermark/attacks.py

from typing import Union
from PIL import Image
import numpy as np

ImageType = Union[Image.Image]

def pixel_restoration(
    img: ImageType,
    downscale_factor: float = 0.5,
) -> ImageType:
    """
    Simple "pixel restoration" style attack.

    Steps:
      1) Downscale the image (information is lost and pixels get mixed).
      2) Upscale back to original size with bilinear interpolation.

    This resampling can weaken structure that lives in fine pixel details.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("pixel_restoration expects a PIL.Image.Image")

    w, h = img.size
    new_w = max(1, int(w * downscale_factor))
    new_h = max(1, int(h * downscale_factor))

    # downscale then upscale
    small = img.resize((new_w, new_h), resample=Image.BILINEAR)
    restored = small.resize((w, h), resample=Image.BILINEAR)
    return restored


def horizontal_shift(
    img: ImageType,
    shift_pixels: int = 32,
    mode: str = "wrap",
) -> ImageType:
    """
    Horizontal shift attack.

    shift_pixels > 0 shifts right, shift_pixels < 0 shifts left.

    mode:
      - "wrap": pixels shifted out on one side reappear on the other.
      - "black": pixels shifted in are filled with black.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("horizontal_shift expects a PIL.Image.Image")

    arr = np.array(img)

    # handle grayscale or RGB(A)
    if mode == "wrap":
        shifted = np.roll(arr, shift_pixels, axis=1)
    elif mode == "black":
        shifted = np.zeros_like(arr)
        if shift_pixels >= 0:
            # content comes from the left
            shifted[:, shift_pixels:] = arr[:, : arr.shape[1] - shift_pixels]
        else:
            k = -shift_pixels
            shifted[:, : arr.shape[1] - k] = arr[:, k:]
    else:
        raise ValueError(f"Unknown mode '{mode}' for horizontal_shift")

    return Image.fromarray(shifted)


def apply_attacks(
    img: ImageType,
    use_pixel_restoration: bool = True,
    use_horizontal_shift: bool = True,
    downscale_factor: float = 0.5,
    shift_pixels: int = 32,
    shift_mode: str = "wrap",
) -> ImageType:
    """
    Convenience function to apply one or both attacks in sequence.
    """
    out = img
    if use_pixel_restoration:
        out = pixel_restoration(out, downscale_factor=downscale_factor)
    if use_horizontal_shift:
        out = horizontal_shift(out, shift_pixels=shift_pixels, mode=shift_mode)
    return out

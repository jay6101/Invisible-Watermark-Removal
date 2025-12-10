import numpy as np
from PIL import Image


def translate_and_restore_left_pil(
    original_img: Image.Image,
    diffused_img: Image.Image,
    shift: int = 7,
) -> Image.Image:
    """
    Cluster 4 hybrid step: horizontal shift plus left column restoration.
    """
    # Make sure sizes match
    if diffused_img.size != original_img.size:
        diffused_img = diffused_img.resize(original_img.size, Image.LANCZOS)

    orig_np = np.array(original_img)   # h, w, c
    diff_np = np.array(diffused_img)

    h, w, c = diff_np.shape
    shift = max(0, min(shift, w))

    # Shift right by required pixels
    shifted = np.zeros_like(diff_np)
    shifted[:, shift:, :] = diff_np[:, : w - shift, :]

    # Restore leftmost shifted columns from original image
    shifted[:, :shift, :] = orig_np[:, :shift, :]

    return Image.fromarray(shifted)

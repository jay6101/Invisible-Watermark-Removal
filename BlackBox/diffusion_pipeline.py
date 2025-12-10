# Diffusion pipeline

import os
from typing import Optional

import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

# Global cached pipeline loaded once
_pipeline: Optional[StableDiffusionXLImg2ImgPipeline] = None


def get_pipeline(device: Optional[str] = None) -> StableDiffusionXLImg2ImgPipeline:
    """Return a cached SDXL img2img pipeline, loading it on first use."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use float16 on GPU, float32 on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    _pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)

    return _pipeline


def remove_watermark_diffusion(
    image_path: str,
    save_path: Optional[str] = None,
    caption: str = " ",
    s: float = 0.16,
    num_steps: int = 500,
) -> Image.Image:
    """Run diffusion based watermark removal on a single image.

    Args:
        image_path: Path to input watermarked image.
        save_path: Optional path to write the processed image. If None, only returns the PIL image.
        caption: Optional text prompt describing the image.
        s: Strength parameter for img2img. Higher values apply stronger corruption.
        num_steps: Number of diffusion inference steps.
    """
    pipe = get_pipeline()

    img = Image.open(image_path).convert("RGB")
    # img = img.resize((1024, 1024), Image.LANCZOS)

    result = pipe(
        prompt=caption,
        image=img,
        strength=s,
        guidance_scale=1.0,
        num_inference_steps=num_steps,
    ).images[0]

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.save(save_path)

    return result


# # Example usage:
# remove_watermark_cluster1(
#     image_path="clustered_images/cluster_1_no_artifacts/0.png",
#     caption="It’s a cheerful illustration of a family of five sitting on a green couch outdoors — a father, a mother, and three young children, all smiling, with the mother and kids wearing matching red-striped outfits.",
#     save_path="cluster1_outputs/0_clean.png",
#     s=0.16
# )

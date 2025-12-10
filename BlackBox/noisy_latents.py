# Noisy latents utility

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from diffusers import AutoencoderKL

from diffusion_pipeline import get_pipeline


# Base paths
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
results_dir = base_dir / "results"

# Global pipeline and device
pipe = get_pipeline()
device = pipe.device

# Global fp32 vae used for forward diffusion examples
vae32 = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    subfolder="vae",
    torch_dtype=torch.float32,
).to(device)


def get_noisy_image_and_latents(
    image_path: str,
    strength: float = 0.16,
    num_inference_steps: int = 500,
    save_path: Optional[str] = None,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Run a forward diffusion step and decode noisy latents.

    Args:
        image_path: Path to input rgb image.
        strength: Fraction of the diffusion schedule to apply.
        num_inference_steps: Number of scheduler steps.
        save_path: Optional path to save the noisy image.

    Returns:
        noisy_img: Noisy image as a pil image.
        noisy_latents: Noisy latents tensor.
    """
    orig = Image.open(image_path).convert("RGB")

    image_tensor = pipe.image_processor.preprocess(orig).to(
        device, dtype=torch.float32
    )

    with torch.no_grad():
        enc = vae32.encode(image_tensor)
        latent_dist = enc.latent_dist if hasattr(enc, "latent_dist") else enc
        latents = latent_dist.sample()
        latents = latents * vae32.config.scaling_factor

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps, _ = pipe.get_timesteps(
        num_inference_steps=num_inference_steps,
        strength=strength,
        device=device,
    )

    t = timesteps[0].to(device)
    if t.ndim == 0:
        t = t[None]
    t = t.expand(latents.shape[0])

    noise = torch.randn_like(latents, dtype=torch.float32)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    with torch.no_grad():
        dec_latents = noisy_latents / vae32.config.scaling_factor
        decoded = vae32.decode(dec_latents).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)

    noisy_img = pipe.image_processor.postprocess(
        decoded.to(device), output_type="pil"
    )[0]

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        noisy_img.save(save_path)

    return noisy_img, noisy_latents


# Example usage
# if __name__ == "__main__":
#     input_img = data_dir / "clustered_images" / "cluster_1_no_artifacts" / "0.png"
#     out_img = data_dir / "fwd_diffusion_output" / "cluster_1_no_artifacts" / "noisy_s20_0.png"
#     get_noisy_image_and_latents(
#         image_path=str(input_img),
#         strength=0.20,
#         num_inference_steps=500,
#         save_path=str(out_img),
#     )

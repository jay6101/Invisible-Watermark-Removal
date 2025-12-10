import os
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
import lpips
from skimage import color
from torchvision.utils import save_image

# Config
base_dir = Path(__file__).resolve().parent
default_device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to finetuned vae weights
finetuned_vae_dir = base_dir / "models"

# Refiner vae used in the paper
refiner_vae_id = "stabilityai/sdxl-vae"

# Hyperparameters
default_test_time_steps = 2
default_test_time_lr = 1e-4

# Global cached models
_vae_finetuned = None
_lpips_model = None


def _get_finetuned_vae(device: str | None = None) -> AutoencoderKL:
    """Load finetuned vae once and cache it."""
    global _vae_finetuned

    dev = device or default_device

    if _vae_finetuned is None:
        _vae_finetuned = AutoencoderKL.from_pretrained(
            finetuned_vae_dir,
            torch_dtype=torch.float32,
        ).to(dev)
        _vae_finetuned.eval()

    return _vae_finetuned


def _get_lpips_model(device: str | None = None):
    """Load lpips model once and cache it."""
    global _lpips_model

    dev = device or default_device

    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(dev)
        _lpips_model.eval()

    return _lpips_model


# Utils
def _load_img(path: str, device: str | None = None) -> torch.Tensor:
    """Load image as tensor in [0, 1], shape [1,3,h,w]."""
    dev = device or default_device

    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(dev)
    return x


def _to_m11(x: torch.Tensor) -> torch.Tensor:
    """Map [0,1] to [-1,1]."""
    return x * 2 - 1


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Map [-1,1] to [0,1]."""
    return (x + 1) / 2


def _gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    one_d_window = _gaussian(window_size, 1.5).unsqueeze(1)
    two_d_window = one_d_window.mm(one_d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = two_d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _compute_ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute ssim between two tensors in [-1,1].
    """
    c1 = (0.01 * 2) ** 2
    c2 = (0.03 * 2) ** 2

    channel = img1.size(1)
    window = _create_window(window_size, channel).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel
    ) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel
    ) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=channel
    ) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# Pipeline core
def _test_time_optimize(
    x_r_init: torch.Tensor,
    x_w: torch.Tensor,
    steps: int,
    lr: float,
    device: str | None = None,
) -> torch.Tensor:
    """
    Test time optimization using a fresh sdxl vae for each call.

    L_total = ||D(E(x_r)) - x_w||^2 + lpips + 0.5 * (1 - ssim)
    """
    dev = device or default_device

    lpips_model = _get_lpips_model(dev)

    vae_refiner = AutoencoderKL.from_pretrained(
        refiner_vae_id,
        torch_dtype=torch.float32,
    ).to(dev)
    vae_refiner.train()

    optimizer = torch.optim.Adam(vae_refiner.parameters(), lr=lr)
    x_r = x_r_init.clone().detach()

    for _ in range(steps):
        optimizer.zero_grad()

        posterior = vae_refiner.encode(x_r).latent_dist
        z = posterior.mean
        x_recon = vae_refiner.decode(z).sample

        mse_loss = F.mse_loss(x_recon, x_w)
        lpips_loss = lpips_model(x_recon, x_w).mean()
        ssim_val = _compute_ssim(x_recon, x_w)
        ssim_loss = 0.5 * (1.0 - ssim_val)

        total_loss = mse_loss + lpips_loss + ssim_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_r = x_recon.clone()

    vae_refiner.eval()
    with torch.no_grad():
        posterior = vae_refiner.encode(x_r).latent_dist
        z = posterior.mean
        x_opt = vae_refiner.decode(z).sample

    del vae_refiner
    torch.cuda.empty_cache()

    return x_opt


def _color_contrast_transfer(x_opt: torch.Tensor, x_w: torch.Tensor) -> torch.Tensor:
    """
    Lab color plus contrast transfer step.
    """
    opt_np = _to_01(x_opt).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    w_np = _to_01(x_w).squeeze().detach().cpu().numpy().transpose(1, 2, 0)

    lab_opt = color.rgb2lab(opt_np)
    lab_w = color.rgb2lab(w_np)

    l_opt = lab_opt[:, :, 0]
    l_w = lab_w[:, :, 0]
    a_w = lab_w[:, :, 1]
    b_w = lab_w[:, :, 2]

    mu_c = l_opt.mean()
    sigma_c = l_opt.std()
    mu_w = l_w.mean()
    sigma_w = l_w.std()

    if sigma_c < 1e-6:
        sigma_c = 1e-6

    l_final = (sigma_w / sigma_c) * (l_opt - mu_c) + mu_w
    lab_final = np.stack([l_final, a_w, b_w], axis=2)

    rgb_final = color.lab2rgb(lab_final)
    x_final = torch.tensor(rgb_final).permute(2, 0, 1).unsqueeze(0).float().to(default_device)
    x_final = _to_m11(x_final)

    return x_final


# Main function
def remove_watermark_vae(
    image_path: str,
    save_path: str | None = None,
    steps: int = default_test_time_steps,
    lr: float = default_test_time_lr,
) -> torch.Tensor:
    """
    Vae based watermark removal pipeline for clusters 2 and 3.

    Args:
        image_path: path to watermarked image x_w.
        save_path: where to save the final image. if none, does not save.
        steps: number of test time optimization steps.
        lr: learning rate for test time optimization.

    Returns:
        x_final: final tensor in [-1,1], shape [1,3,h,w].
    """
    dev = default_device
    vae_finetuned = _get_finetuned_vae(dev)

    x_w_01 = _load_img(image_path, device=dev)
    x_w = _to_m11(x_w_01)

    with torch.no_grad():
        posterior = vae_finetuned.encode(x_w).latent_dist
        z = posterior.mean
        x_r_init = vae_finetuned.decode(z).sample

    x_opt = _test_time_optimize(
        x_r_init=x_r_init,
        x_w=x_w,
        steps=steps,
        lr=lr,
        device=dev,
    )

    x_final = _color_contrast_transfer(x_opt, x_w)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(_to_01(x_final), save_path)

    return x_final

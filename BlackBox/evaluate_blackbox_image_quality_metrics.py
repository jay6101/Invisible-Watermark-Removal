import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torchvision.transforms as T
from torchvision.models import inception_v3

import clip
import lpips
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Config

base_dir = Path(__file__).resolve().parent

clusters = [
    (
        "cluster_1",
        base_dir / "data" / "clustered_images" / "cluster_1_no_artifacts",
        base_dir / "results" / "blackbox_outputs" / "cluster_1",
    ),
    (
        "cluster_2",
        base_dir / "data" / "clustered_images" / "cluster_2_boundary_artifacts",
        base_dir / "results" / "blackbox_outputs" / "cluster_2",
    ),
    (
        "cluster_3",
        base_dir / "data" / "clustered_images" / "cluster_3_circular_patterns",
        base_dir / "results" / "blackbox_outputs" / "cluster_3",
    ),
    (
        "cluster_4",
        base_dir / "data" / "clustered_images" / "cluster_4_square_patterns",
        base_dir / "results" / "blackbox_outputs" / "cluster_4",
    ),
]

metrics_root = base_dir / "results" / "blackbox_img_quality_metrics"
metrics_root.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# Utility functions

def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


def load_img_pair(in_path: str | Path, out_path: str | Path, resize_to_output: bool = True):
    """Load input and output as tensors in [-1, 1] with shape [1, 3, H, W]."""
    in_path = Path(in_path)
    out_path = Path(out_path)

    xw = Image.open(in_path).convert("RGB")
    xr = Image.open(out_path).convert("RGB")

    if resize_to_output and xw.size != xr.size:
        xw = xw.resize(xr.size, Image.LANCZOS)

    to_tensor = T.ToTensor()
    xw_t = to_tensor(xw).unsqueeze(0)
    xr_t = to_tensor(xr).unsqueeze(0)

    xw_t = xw_t * 2 - 1
    xr_t = xr_t * 2 - 1

    return xw_t.to(device), xr_t.to(device)


# Per image metrics

lpips_model = lpips.LPIPS(net="alex").to(device)


def metric_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np = to_01(a).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    b_np = to_01(b).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return peak_signal_noise_ratio(a_np, b_np, data_range=1.0)


def metric_ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np = to_01(a).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    b_np = to_01(b).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(a_np, b_np, channel_axis=2, data_range=1.0)


def metric_lpips(a: torch.Tensor, b: torch.Tensor) -> float:
    return lpips_model(a, b).mean().item()


# FID and clip fid

def compute_fid(feats1: np.ndarray, feats2: np.ndarray, eps: float = 1e-6) -> float:
    """Compute fid between two feature sets shaped [n, d]."""
    mu1 = feats1.mean(axis=0)
    mu2 = feats2.mean(axis=0)
    sigma1 = np.cov(feats1, rowvar=False)
    sigma2 = np.cov(feats2, rowvar=False)

    diff = mu1 - mu2
    diff_sq = diff.dot(diff)

    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    cov_prod = sigma1 @ sigma2
    cov_prod = (cov_prod + cov_prod.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    eigvals = np.clip(eigvals, 0, None)
    covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    covmean = (covmean + covmean.T) / 2.0

    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    fid = float(diff_sq + trace_term)
    return max(fid, 0.0)


_inception_model = None
_clip_model = None
_clip_preprocess = None


def get_inception_model():
    global _inception_model
    if _inception_model is None:
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        _inception_model = model
    return _inception_model


def get_clip_model():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        _clip_model, _clip_preprocess = model, preprocess
    return _clip_model, _clip_preprocess


def get_inception_features(paths: list[str]) -> np.ndarray:
    model = get_inception_model()
    transform = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    feats = []
    with torch.no_grad():
        for p in tqdm(paths, desc="Inception features", leave=False):
            img = Image.open(p).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            f = model(x)
            feats.append(f.squeeze().cpu().numpy())
    return np.stack(feats, axis=0)


def get_clip_features(paths: list[str]) -> np.ndarray:
    model, preprocess = get_clip_model()
    feats = []
    with torch.no_grad():
        for p in tqdm(paths, desc="Clip features", leave=False):
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            f = model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.squeeze().cpu().numpy())
    return np.stack(feats, axis=0)


def run_for_cluster(cluster_name: str, input_dir: Path, output_dir: Path):
    print(f"\nCluster: {cluster_name}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    metrics_csv = metrics_root / f"{cluster_name}_metrics.csv"

    matched_in_paths: list[str] = []
    matched_out_paths: list[str] = []

    # Per image metrics
    if metrics_csv.exists():
        print(f"Found existing per image metrics: {metrics_csv}")
        df = pd.read_csv(metrics_csv)

        if "input_path" not in df.columns or "output_path" not in df.columns:
            raise ValueError(
                f"{metrics_csv} must contain 'input_path' and 'output_path' columns."
            )

        matched_in_paths = df["input_path"].tolist()
        matched_out_paths = df["output_path"].tolist()

    else:
        in_files = sorted(
            f
            for f in input_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )

        if not in_files:
            print(f"Warning no images found in {input_dir}, skipping cluster.")
            return

        records = []

        for in_path in tqdm(in_files, desc=f"Per image metrics {cluster_name}"):
            stem = in_path.stem
            cand1 = output_dir / f"{stem}.png"
            cand2 = output_dir / f"{stem}_refined.png"

            if cand1.exists():
                out_path = cand1
            elif cand2.exists():
                out_path = cand2
            else:
                print(f"Warning no output for {in_path.name}, skipping.")
                continue

            xw, xr = load_img_pair(in_path, out_path)

            psnr_val = metric_psnr(xw, xr)
            ssim_val = metric_ssim(xw, xr)
            lpips_val = metric_lpips(xw, xr)

            records.append(
                {
                    "filename": in_path.name,
                    "input_path": str(in_path),
                    "output_path": str(out_path),
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                }
            )

            matched_in_paths.append(str(in_path))
            matched_out_paths.append(str(out_path))

        if not records:
            print(f"Warning no matched pairs for {cluster_name}, skipping metrics.")
            return

        df = pd.DataFrame(records)
        df.to_csv(metrics_csv, index=False)
        print(f"Saved per image metrics to: {metrics_csv}")

    if len(matched_in_paths) == 0:
        print(f"Warning no paths available for dataset level metrics in {cluster_name}.")
        return

    print("Computing dataset level metrics fid and clip fid.")
    feats_in = get_inception_features(matched_in_paths)
    feats_out = get_inception_features(matched_out_paths)
    fid = compute_fid(feats_in, feats_out)

    clip_in = get_clip_features(matched_in_paths)
    clip_out = get_clip_features(matched_out_paths)
    clip_fid_raw = compute_fid(clip_in, clip_out)
    clip_fid = 100.0 * clip_fid_raw

    print("\nAggregate metrics")
    print(f"Cluster: {cluster_name}")
    print(f"Images evaluated: {len(df)}")
    print(f"PSNR mean: {df['psnr'].mean():.3f} dB")
    print(f"SSIM mean: {df['ssim'].mean():.4f}")
    print(f"LPIPS mean: {df['lpips'].mean():.4f}")
    print(f"FID inception: {fid:.3f}")
    print(f"CLIP FID vit B32: {clip_fid:.3f}")


def main():
    for name, in_dir, out_dir in clusters:
        run_for_cluster(name, in_dir, out_dir)


if __name__ == "__main__":
    main()

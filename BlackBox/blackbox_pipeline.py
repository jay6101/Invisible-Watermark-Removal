# black_box_pipeline.py

import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from diffusion_pipeline import remove_watermark_diffusion
from vae_pipeline import remove_watermark_vae
from treering_pipeline import translate_and_restore_left_pil


# Paths
base_dir = Path(__file__).resolve().parent
image_dir = base_dir / "data" / "blackbox_images"
mapping_csv = base_dir / "data" / "cluster_mapping_with_captions.csv"
out_dir = base_dir / "results" / "blackbox_outputs"


def main():
    # Stage 1 load mapping
    df = pd.read_csv(mapping_csv)

    if "filename" not in df.columns or "cluster_id" not in df.columns:
        raise ValueError(
            f"{mapping_csv} must have columns 'filename' and 'cluster_id'"
        )

    if "caption" not in df.columns:
        df["caption"] = ""

    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 2 cluster specific solutions
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Black Box Pipeline"):
        fname = row["filename"]
        cluster_id = int(row["cluster_id"])
        caption = str(row.get("caption", "") or "")

        x_path = image_dir / fname
        if not x_path.exists():
            print(f"[Warning] missing image {x_path}, skipping.")
            continue

        # Cluster specific output directory
        cluster_out_dir = out_dir / f"cluster_{cluster_id}"
        cluster_out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(fname).stem
        x_r_path = cluster_out_dir / f"{stem}_refined.png"

        # Skip processing if the output already exists
        if x_r_path.exists():
            print(f"Output already exists for {fname}, skipping.")
            continue

        # Algorithm
        if cluster_id == 1:
            # k = 1 aggressive diffusion with s = 0.16
            remove_watermark_diffusion(
                image_path=str(x_path),
                caption=caption,
                save_path=str(x_r_path),
                s=0.16,
                T=500,
            )

        elif cluster_id in (2, 3):
            # k in {2, 3} vae pipeline
            remove_watermark_vae(
                image_path=str(x_path),
                save_path=str(x_r_path),
                steps=2,
                lr=1e-4,
            )

        elif cluster_id == 4:
            # k = 4 hybrid mild diffusion plus translation
            # (i) mild diffusion step x_d <- diffuse(x, s = 0.04)
            xd_img = remove_watermark_diffusion(
                image_path=str(x_path),
                s=0.04,
                T=500,
            )

            # (ii) spatial translation plus left column restoration
            orig_img = Image.open(x_path).convert("RGB")
            xr_img = translate_and_restore_left_pil(
                original_img=orig_img,
                diffused_img=xd_img,
                shift=7,
            )

            os.makedirs(cluster_out_dir, exist_ok=True)
            xr_img.save(x_r_path)

        else:
            print(f"Unknown cluster_id={cluster_id} for {fname}, skipping.")
            continue

    print(f"\nDone. Refined images are in {out_dir}")


if __name__ == "__main__":
    main()

# Run diffusion on a subset of StegaStamp images

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from diffusion_pipeline import remove_watermark_diffusion


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    img_dir = data_dir / "clustered_images" / "beigebox_images"
    csv_path = data_dir / "HF_SD2.1_prompts.csv"

    df = pd.read_csv(csv_path)
    prompts = df["Prompt"].tolist()

    num_images = min(100, len(prompts))
    print(f"Found {num_images} images and prompts")

    s_values = [0.01, 0.04, 0.08, 0.12, 0.16, 0.20]

    for s in s_values:
        s_code = f"{int(round(s * 100)):02d}"

        out_dir = data_dir / "cluster1_outputs" / f"beigebox_test_output_s{s_code}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning diffusion for s={s:.2f} -> outputs in {out_dir}")

        for idx in tqdm(range(num_images), desc=f"s={s:.2f}"):
            img_name = f"{idx:06d}_hidden.png"
            img_path = img_dir / img_name

            if not img_path.exists():
                print(f"Warning: missing image {img_path}, skipping")
                continue

            caption = prompts[idx]

            out_name = f"{idx:06d}_SD{s_code}.png"
            out_path = out_dir / out_name

            remove_watermark_diffusion(
                image_path=str(img_path),
                caption=caption,
                save_path=str(out_path),
                s=s,
            )

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()

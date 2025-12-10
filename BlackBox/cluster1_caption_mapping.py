# Cluster 1 caption mapping

import os
from pathlib import Path

import pandas as pd


def main() -> None:
    # Paths
    base_dir = Path.cwd()

    mapping_csv = base_dir / "data/cluster_mapping.csv"
    captions_dir = base_dir / "data/gpt_captions_cluster1"
    out_csv = base_dir / "data/cluster_mapping_with_captions.csv"

    # Load mapping
    df = pd.read_csv(mapping_csv)

    if "filename" not in df.columns or "cluster_id" not in df.columns:
        raise ValueError("cluster_mapping.csv must have columns 'filename' and 'cluster_id'")

    # Initialize caption column
    if "caption" not in df.columns:
        df["caption"] = ""
    else:
        df["caption"] = df["caption"].fillna("")

    # Collect caption txt files
    caption_txt_files: dict[str, Path] = {}
    if captions_dir.exists():
        for path in captions_dir.glob("*.txt"):
            # key is stem, for example "0" from "0.txt"
            caption_txt_files[path.stem] = path
    else:
        print(f"Warning: captions directory does not exist: {captions_dir}")

    missing_txt_for_cluster1: list[str] = []
    used_txt_stems: set[str] = set()

    # Fill captions for cluster 1
    for idx, row in df.iterrows():
        fname = str(row["filename"]) # e.g. "0.png"
        cluster_id = int(row["cluster_id"])

        if cluster_id != 1:
            continue

        stem = Path(fname).stem # "0" from "0.png"
        txt_path = caption_txt_files.get(stem)

        if txt_path is None:
            missing_txt_for_cluster1.append(fname)
            continue

        with txt_path.open("r", encoding="utf-8") as f:
            caption = f.read().strip()

        df.at[idx, "caption"] = caption
        used_txt_stems.add(stem)

    # Caption files that do not match any cluster 1 filename
    extra_txt_files = [
        path.name
        for stem, path in caption_txt_files.items()
        if stem not in used_txt_stems
    ]

    print(f"Total rows in mapping: {len(df)}")
    print(f"Captions merged: {(df['caption'] != '').sum()}")

    if missing_txt_for_cluster1:
        print("Cluster 1 images with no caption .txt file:")
        print(missing_txt_for_cluster1)

    if extra_txt_files:
        print("Caption .txt files with no matching cluster 1 image in mapping:")
        print(extra_txt_files)

    # Save updated mapping
    df.to_csv(out_csv, index=False)
    print(f"Saved merged mapping to: {out_csv}")


if __name__ == "__main__":
    main()

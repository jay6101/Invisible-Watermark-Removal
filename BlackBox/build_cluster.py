"""
build_cluster.py

Organizes 300 blackbox images into 4 clusters based on preset ID lists.
Creates folder structure:

clustered_images/
    cluster_1_no_artifacts/
    cluster_2_boundary_artifacts/
    cluster_3_circular_patterns/
    cluster_4_square_patterns/
"""

import os
import shutil
from pathlib import Path


# Paths
base_dir = Path(__file__).resolve().parent
input_dir = base_dir / "data/blackbox_images"
output_dir = base_dir / "data/clustered_images"


# Cluster assignments
cluster_1 = [...]
cluster_2 = [...]
cluster_3 = [...]
cluster_4 = [...]

clusters = {
    "cluster_1_no_artifacts": cluster_1,
    "cluster_2_boundary_artifacts": cluster_2,
    "cluster_3_circular_patterns": cluster_3,
    "cluster_4_square_patterns": cluster_4,
}


# Build clusters
def build_clusters():
    print("Building cluster folders...")

    output_dir.mkdir(exist_ok=True)

    # Create subfolders
    for folder in clusters.keys():
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    missing = 0

    # Copy images
    for cluster_name, ids in clusters.items():
        out_dir = output_dir / cluster_name

        for img_id in ids:
            fname = f"{img_id}.png"
            src = input_dir / fname
            dst = out_dir / fname

            if src.exists():
                shutil.copy(src, dst)
            else:
                print(f"Warning: Missing file {src}")
                missing += 1

    print("\nDone.")
    print(f"Total missing files: {missing}")


# Main entry
if __name__ == "__main__":
    build_clusters()

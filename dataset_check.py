"""
dataset_check.py
================
Validates the fruit ripeness dataset structure.
- Counts images per class
- Checks image dimensions and channels
- Displays 3 sample images per class
- Reports classes below minimum threshold

Usage:
    python dataset_check.py
    python dataset_check.py --data_dir ./dataset --min_images 25 --show_samples
"""

import os
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CLASS_MAP = {
    "apple":  ["unripe", "ripe", "overripe"],
    "banana": ["unripe", "ripe", "overripe"],
    "pear":   ["unripe", "ripe", "overripe"],
}

CLASS_LABELS = [
    "apple_unripe", "apple_ripe", "apple_overripe",
    "banana_unripe", "banana_ripe", "banana_overripe",
    "pear_unripe", "pear_ripe", "pear_overripe",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_IMAGES    = 50
MIN_IMAGES       = 25


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def get_image_files(folder: Path) -> list:
    """Return all image files in a folder."""
    if not folder.exists():
        return []
    return [
        f for f in folder.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
    ]


def check_image(filepath: Path) -> dict:
    """Read and validate a single image. Returns info dict."""
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            img_rgb = img.convert("RGB")
            arr = np.array(img_rgb)
            return {
                "path": str(filepath),
                "width": arr.shape[1],
                "height": arr.shape[0],
                "channels": arr.shape[2],
                "valid": True,
                "error": None,
            }
    except Exception as e:
        return {
            "path": str(filepath),
            "width": None,
            "height": None,
            "channels": None,
            "valid": False,
            "error": str(e),
        }


def load_image_rgb(filepath: Path, size=(224, 224)) -> np.ndarray:
    """Load image and resize to display size."""
    from PIL import Image
    with Image.open(filepath) as img:
        img_resized = img.convert("RGB").resize(size)
        return np.array(img_resized)


# ──────────────────────────────────────────────
# Main validation
# ──────────────────────────────────────────────
def validate_dataset(data_dir: str, min_images: int = MIN_IMAGES) -> dict:
    """
    Walk dataset directory and collect statistics.

    Returns:
        report (dict): per-class stats
    """
    data_path = Path(data_dir)
    report = {}
    total_images = 0
    all_ok = True

    print("=" * 60)
    print("  DATASET VALIDATION REPORT")
    print(f"  Dataset root: {data_path.resolve()}")
    print("=" * 60)

    for fruit, stages in CLASS_MAP.items():
        for stage in stages:
            class_name  = f"{fruit}_{stage}"
            class_folder = data_path / fruit / stage
            images = get_image_files(class_folder)
            count  = len(images)
            total_images += count

            # Check individual images
            valid_count   = 0
            corrupt_count = 0
            dim_issues    = 0
            sample_infos  = []

            for img_path in images:
                info = check_image(img_path)
                if info["valid"]:
                    valid_count += 1
                    sample_infos.append(info)
                    # Flag unusual dimensions
                    if info["width"] < 50 or info["height"] < 50:
                        dim_issues += 1
                else:
                    corrupt_count += 1

            # Status
            if count == 0:
                status = "❌ EMPTY"
                all_ok = False
            elif count < min_images:
                status = f"⚠️  LOW ({count}/{min_images} min)"
                all_ok = False
            elif count >= TARGET_IMAGES:
                status = f"✅ GOOD ({count}/{TARGET_IMAGES} target)"
            else:
                status = f"🟡 OK   ({count}/{TARGET_IMAGES} target)"

            report[class_name] = {
                "folder":        str(class_folder),
                "count":         count,
                "valid":         valid_count,
                "corrupt":       corrupt_count,
                "dim_issues":    dim_issues,
                "status":        status,
                "sample_infos":  sample_infos[:3],
            }

            corrupt_str = f" | {corrupt_count} corrupt" if corrupt_count else ""
            print(f"  {class_name:<22} {status}{corrupt_str}")

    print("-" * 60)
    print(f"  Total images found: {total_images}")
    print(f"  Expected (target):  {TARGET_IMAGES * 9}")
    print(f"  Expected (minimum): {min_images * 9}")

    if all_ok:
        print("\n  ✅ All classes meet minimum requirements!")
    else:
        missing = [
            cls for cls, info in report.items()
            if info["count"] < min_images
        ]
        print(f"\n  ⚠️  Classes below minimum ({min_images}): {missing}")

    print("=" * 60)
    return report


def display_samples(report: dict, n_samples: int = 3):
    """Plot n_samples images per class in a grid."""
    n_classes = len(report)
    fig = plt.figure(figsize=(n_samples * 3, n_classes * 2.5))
    fig.suptitle(
        "Dataset Sample Images (3 per class)",
        fontsize=14, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(n_classes, n_samples, figure=fig,
                           hspace=0.5, wspace=0.3)

    for row_idx, (class_name, info) in enumerate(report.items()):
        samples = info["sample_infos"]
        for col_idx in range(n_samples):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if col_idx < len(samples):
                img_path = Path(samples[col_idx]["path"])
                try:
                    img_arr = load_image_rgb(img_path)
                    ax.imshow(img_arr)
                    ax.set_title(
                        f"{img_path.name[:18]}",
                        fontsize=6, pad=2
                    )
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center",
                            transform=ax.transAxes, color="red")
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=8)

            if col_idx == 0:
                ax.set_ylabel(class_name.replace("_", "\n"),
                              fontsize=7, rotation=0,
                              labelpad=50, va="center")
            ax.axis("off")

    plt.savefig("results/dataset_samples.png",
                bbox_inches="tight", dpi=100)
    plt.show()
    print("\n  📊 Sample grid saved to results/dataset_samples.png")


def generate_metadata_csv(report: dict, output_path: str = "dataset/metadata.csv"):
    """
    Regenerate metadata.csv from actual scanned images.
    Updates class_idx and records image dimensions.
    """
    import csv
    rows = []
    for class_idx, (class_name, info) in enumerate(report.items()):
        for img_info in info.get("sample_infos", []):
            fname = Path(img_info["path"]).name
            parts = class_name.split("_")
            fruit = parts[0]
            stage = "_".join(parts[1:])
            rows.append({
                "filename":   fname,
                "fruit":      fruit,
                "stage":      stage,
                "class_label": class_name,
                "class_idx":  class_idx,
                "width":      img_info.get("width", ""),
                "height":     img_info.get("height", ""),
                "channels":   img_info.get("channels", ""),
                "source":     "real_capture",
                "notes":      "",
            })

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  📋 metadata.csv updated with {len(rows)} entries → {output_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Validate the fruit ripeness dataset"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./dataset",
        help="Root directory of the dataset (default: ./dataset)"
    )
    parser.add_argument(
        "--min_images", type=int, default=MIN_IMAGES,
        help=f"Minimum images per class (default: {MIN_IMAGES})"
    )
    parser.add_argument(
        "--show_samples", action="store_true",
        help="Display a grid of 3 sample images per class"
    )
    parser.add_argument(
        "--update_csv", action="store_true",
        help="Regenerate dataset/metadata.csv from scanned images"
    )
    args = parser.parse_args()

    # Create results dir if needed
    os.makedirs("results", exist_ok=True)

    # Run validation
    report = validate_dataset(args.data_dir, args.min_images)

    if args.show_samples:
        display_samples(report)

    if args.update_csv:
        generate_metadata_csv(report)

    # Exit code: 0 if OK, 1 if issues
    any_low = any(
        info["count"] < args.min_images
        for info in report.values()
    )
    sys.exit(1 if any_low else 0)


if __name__ == "__main__":
    main()
"""
src/data_pipeline.py
====================
Data loading, preprocessing, augmentation, and tf.data pipeline.

Key features:
- Supports real capture images AND synthetic/augmented fallback
- Stratified train/val/test split (70/15/15)
- Augmentation via tf.data (reproducible with seed)
- Returns tf.data.Dataset objects ready for model.fit()

Usage:
    from src.data_pipeline import build_datasets
    train_ds, val_ds, test_ds = build_datasets(data_dir="./dataset", seed=42)
"""

import os
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from src.utils import (
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, DEFAULT_SEED,
    DATASET_TREE, get_class_folder_map, list_image_files,
    set_seeds, get_logger, ensure_dirs
)

logger = get_logger("data_pipeline")


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

AUTOTUNE = tf.data.AUTOTUNE

# Augmentation parameters (as specified in project brief)
AUG_PARAMS = {
    "horizontal_flip":    True,       # 50% chance
    "rotation_range":     0.0833,     # ±15° = 15/180
    "brightness_delta":   0.2,        # factor 0.8–1.2
    "zoom_range":         0.1,        # 0.9–1.1
    "translation_frac":   0.10,       # ±10% of image size
}


# ──────────────────────────────────────────────
# Split helpers
# ──────────────────────────────────────────────
def stratified_split(
    data_dir: str,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Split dataset into (train, val, test) with stratification per class.

    Returns:
        train_samples, val_samples, test_samples
        Each element: (image_path_str, class_idx)
    """
    random.seed(seed)
    folder_map = get_class_folder_map(data_dir)

    train_samples, val_samples, test_samples = [], [], []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        folder = folder_map[class_name]
        files  = [str(f) for f in list_image_files(folder)]

        if not files:
            logger.warning(f"No images found for class '{class_name}' in {folder}")
            continue

        random.shuffle(files)
        n        = len(files)
        n_train  = max(1, math.floor(n * TRAIN_RATIO))
        n_val    = max(1, math.floor(n * VAL_RATIO))
        # rest goes to test
        n_test   = n - n_train - n_val

        if n_test < 1:
            # Adjust: take at least 1 test sample
            if n_val > 1:
                n_val  -= 1
                n_test  = 1
            elif n_train > 1:
                n_train -= 1
                n_test   = 1

        train_files = files[:n_train]
        val_files   = files[n_train:n_train + n_val]
        test_files  = files[n_train + n_val:]

        train_samples += [(fp, class_idx) for fp in train_files]
        val_samples   += [(fp, class_idx) for fp in val_files]
        test_samples  += [(fp, class_idx) for fp in test_files]

        logger.info(
            f"  {class_name:<22}: {len(train_files):>3} train | "
            f"{len(val_files):>3} val | {len(test_files):>3} test"
        )

    return train_samples, val_samples, test_samples


def save_split_csv(
    train_samples: list, val_samples: list, test_samples: list,
    output_dir: str = "results",
) -> None:
    """Save split information to CSV for reference."""
    ensure_dirs(output_dir)
    rows = []
    for split_name, samples in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ]:
        for path, idx in samples:
            rows.append({
                "split": split_name,
                "path": path,
                "class_idx": idx,
                "class_name": CLASS_NAMES[idx],
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/data_split.csv", index=False)
    logger.info(f"Split CSV saved to {output_dir}/data_split.csv")


# ──────────────────────────────────────────────
# tf.data loading helpers
# ──────────────────────────────────────────────
def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> Tuple:
    """
    Load image from path, resize, normalize to [0,1].
    """
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label_oh = tf.one_hot(label, NUM_CLASSES)
    return image, label_oh


def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple:
    """
    Apply training augmentations:
    - Random horizontal flip (50%)
    - Random rotation ±15°
    - Random brightness ±20%
    - Random zoom 0.9–1.1
    - Random translation ±10%
    """
    # Horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Brightness
    image = tf.image.random_brightness(
        image, max_delta=AUG_PARAMS["brightness_delta"]
    )
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Rotation + zoom + translation via combined affine transform
    # Using keras preprocessing layers for reproducibility
    image = tf.keras.preprocessing.image.apply_affine_transform(
        image.numpy(),
        theta=random.uniform(-15, 15),
        zx=random.uniform(0.9, 1.1),
        zy=random.uniform(0.9, 1.1),
        tx=random.uniform(-0.1, 0.1) * IMAGE_SIZE[1],
        ty=random.uniform(-0.1, 0.1) * IMAGE_SIZE[0],
        row_axis=0, col_axis=1, channel_axis=2,
        fill_mode="nearest",
    )
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def _augment_tf(image: tf.Tensor, label: tf.Tensor) -> Tuple:
    """
    Pure tf.ops augmentation (compatible with graph mode).
    """
    # Horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Brightness
    image = tf.image.random_brightness(
        image, max_delta=AUG_PARAMS["brightness_delta"]
    )
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Rotation via keras layer
    image = tf.expand_dims(image, 0)
    image = tf.keras.layers.RandomRotation(
        factor=AUG_PARAMS["rotation_range"],
        fill_mode="nearest"
    )(image, training=True)
    image = tf.squeeze(image, 0)

    # Zoom
    image = tf.expand_dims(image, 0)
    image = tf.keras.layers.RandomZoom(
        height_factor=(-AUG_PARAMS["zoom_range"], AUG_PARAMS["zoom_range"]),
        fill_mode="nearest"
    )(image, training=True)
    image = tf.squeeze(image, 0)

    # Translation
    image = tf.expand_dims(image, 0)
    image = tf.keras.layers.RandomTranslation(
        height_factor=AUG_PARAMS["translation_frac"],
        width_factor=AUG_PARAMS["translation_frac"],
        fill_mode="nearest"
    )(image, training=True)
    image = tf.squeeze(image, 0)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ──────────────────────────────────────────────
# Dataset builders
# ──────────────────────────────────────────────
def build_tf_dataset(
    samples: List[Tuple],
    batch_size: int = 16,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = DEFAULT_SEED,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from (path, class_idx) pairs.

    Args:
        samples:    list of (image_path_str, class_idx)
        batch_size: batch size
        augment:    apply training augmentation
        shuffle:    shuffle dataset
        seed:       random seed

    Returns:
        Batched, prefetched tf.data.Dataset
    """
    if not samples:
        raise ValueError("No samples provided to build_tf_dataset.")

    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    paths_t  = tf.constant(paths,  dtype=tf.string)
    labels_t = tf.constant(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(samples),
            seed=seed,
            reshuffle_each_iteration=True
        )

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(_augment_tf, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_datasets(
    data_dir: str = "./dataset",
    batch_size: int = 16,
    seed: int = DEFAULT_SEED,
    save_split: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Main entry point: split + build train/val/test datasets.

    Args:
        data_dir:   root dataset directory
        batch_size: batch size for all splits
        seed:       random seed
        save_split: write data_split.csv to results/

    Returns:
        (train_ds, val_ds, test_ds) – tf.data.Dataset objects
    """
    set_seeds(seed)

    logger.info(f"Building datasets from: {data_dir}")
    train_samples, val_samples, test_samples = stratified_split(
        data_dir, seed=seed
    )

    logger.info(
        f"Splits → train: {len(train_samples)} | "
        f"val: {len(val_samples)} | test: {len(test_samples)}"
    )

    if save_split:
        save_split_csv(train_samples, val_samples, test_samples)

    train_ds = build_tf_dataset(
        train_samples, batch_size=batch_size,
        augment=True, shuffle=True, seed=seed
    )
    val_ds = build_tf_dataset(
        val_samples, batch_size=batch_size,
        augment=False, shuffle=False
    )
    test_ds = build_tf_dataset(
        test_samples, batch_size=batch_size,
        augment=False, shuffle=False
    )

    return train_ds, val_ds, test_ds


# ──────────────────────────────────────────────
# Synthetic data generator (fallback)
# ──────────────────────────────────────────────
def generate_synthetic_dataset(
    output_dir: str = "./dataset",
    n_per_class: int = 25,
    seed: int = DEFAULT_SEED,
) -> None:
    """
    Generate synthetic placeholder images when real capture is unavailable.

    Each class gets a distinct hue range so the model has something to learn.
    This is a FALLBACK – replace with real images as soon as possible.

    Saves PNG images to dataset/{fruit}/{stage}/ folders.
    """
    from PIL import Image, ImageDraw, ImageFont

    set_seeds(seed)
    np.random.seed(seed)

    # Approximate hue (H in HSV) per class
    class_hue = {
        "apple_unripe":     80,   # green
        "apple_ripe":       0,    # red
        "apple_overripe":   20,   # brownish
        "banana_unripe":    70,   # yellow-green
        "banana_ripe":      55,   # yellow
        "banana_overripe":  30,   # dark yellow/brown
        "pear_unripe":      90,   # green
        "pear_ripe":        65,   # yellow-green
        "pear_overripe":    40,   # brownish yellow
    }

    folder_map = get_class_folder_map(output_dir)

    for class_name, folder in folder_map.items():
        ensure_dirs(folder)
        hue = class_hue.get(class_name, 0)

        for i in range(1, n_per_class + 1):
            # Create synthetic image: colored background + noise
            h  = hue + np.random.randint(-10, 10)
            s  = 180 + np.random.randint(-30, 30)
            v  = 160 + np.random.randint(-20, 20)

            # HSV → RGB (simple approximation)
            h_norm = (h % 180) / 60.0
            hi     = int(h_norm)
            f      = h_norm - hi
            p      = int(v * (1 - s / 255))
            q      = int(v * (1 - f * s / 255))
            t      = int(v * (1 - (1 - f) * s / 255))
            rgb_map = [
                (v, t, p), (q, v, p), (p, v, t),
                (p, q, v), (t, p, v), (v, p, q),
            ]
            r, g, b = rgb_map[hi % 6]

            # Create image with noise
            arr = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + r, 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] + g, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + b, 0, 255)

            # Draw ellipse to simulate fruit shape
            img = Image.fromarray(arr, "RGB")
            draw = ImageDraw.Draw(img)
            cx, cy = 112, 112
            rx = 80 + np.random.randint(-10, 10)
            ry = 90 + np.random.randint(-10, 10)
            draw.ellipse(
                [(cx - rx, cy - ry), (cx + rx, cy + ry)],
                fill=(r, g, b)
            )

            # Add label text
            try:
                draw.text((5, 5), class_name[:12], fill=(255, 255, 255))
            except Exception:
                pass

            fname = f"{class_name}_{i:03d}.jpg"
            img.save(str(Path(folder) / fname), "JPEG", quality=90)

        logger.info(f"  Synthetic: {n_per_class} images → {folder}")

    logger.warning(
        "⚠️  SYNTHETIC DATA GENERATED — replace with real images for final submission!"
    )


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
def visualize_batch(
    dataset: tf.data.Dataset,
    n_images: int = 9,
    title: str = "Sample Batch",
    save_path: Optional[str] = None,
) -> None:
    """Display a grid of images from a dataset batch."""
    images, labels = next(iter(dataset))
    images = images.numpy()
    labels = labels.numpy()

    cols  = min(n_images, 3)
    rows  = math.ceil(n_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i in range(n_images):
        ax = axes[i]
        ax.imshow(images[i])
        class_idx = np.argmax(labels[i])
        ax.set_title(CLASS_NAMES[class_idx], fontsize=8)
        ax.axis("off")

    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        ensure_dirs(str(Path(save_path).parent))
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Batch visualization saved to {save_path}")
    plt.show()


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data first")
    args = parser.parse_args()

    if args.synthetic:
        logger.info("Generating synthetic dataset...")
        generate_synthetic_dataset(output_dir=args.data_dir)

    logger.info("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(args.data_dir)

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        for images, labels in ds.take(1):
            logger.info(
                f"{name}: batch shape {images.shape}, "
                f"labels shape {labels.shape}"
            )

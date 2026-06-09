"""
src/utils.py
============
Shared utilities for the fruit ripeness classification project.
- Seed fixing
- Path helpers
- Logging helpers
- Class label definitions
- Class-order conversion (alphabetical <-> canonical)
- Metrics helpers
"""

import os
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CLASS_NAMES: List[str] = [
    "apple_unripe", "apple_ripe", "apple_overripe",
    "banana_unripe", "banana_ripe", "banana_overripe",
    "pear_unripe", "pear_ripe", "pear_overripe",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Folder structure: fruit/stage
DATASET_TREE = {
    "apple":  ["unripe", "ripe", "overripe"],
    "banana": ["unripe", "ripe", "overripe"],
    "pear":   ["unripe", "ripe", "overripe"],
}

IMAGE_SIZE    = (224, 224)
NUM_CHANNELS  = 3
DEFAULT_SEED  = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ──────────────────────────────────────────────
# Class-order conversion
# ──────────────────────────────────────────────
# The baseline path (src.data_pipeline) labels classes in the canonical
# CLASS_NAMES order above (grouped fruit then ripeness). The MobileNetV2 models
# were trained with tf.keras.utils.image_dataset_from_directory, which orders
# classes ALPHABETICALLY by folder name. The two orderings differ, and a
# trained model's output indices are fixed to the order it was trained with.
# To report every model's results in the single canonical order, MobileNetV2
# outputs (and their integer labels) are permuted from alphabetical to canonical
# using the helpers below. No retraining is involved; this is a pure relabel.
ALPHABETICAL_CLASS_NAMES: List[str] = sorted(CLASS_NAMES)

# alphabetical index j  ->  canonical index
ALPHA_TO_CANONICAL: List[int] = [CLASS_NAMES.index(c) for c in ALPHABETICAL_CLASS_NAMES]


def alpha_probs_to_canonical(probs: np.ndarray) -> np.ndarray:
    """
    Reorder a (N, C) probability/score array whose columns are in alphabetical
    class order into the canonical CLASS_NAMES column order.
    """
    probs = np.asarray(probs)
    out = np.empty_like(probs)
    for alpha_idx, canon_idx in enumerate(ALPHA_TO_CANONICAL):
        out[:, canon_idx] = probs[:, alpha_idx]
    return out


def alpha_labels_to_canonical(labels: np.ndarray) -> np.ndarray:
    """
    Map integer labels expressed in alphabetical class order to canonical order.
    """
    lut = np.asarray(ALPHA_TO_CANONICAL)
    return lut[np.asarray(labels)]


# ──────────────────────────────────────────────
# Saved-model filenames (single source of truth)
# ──────────────────────────────────────────────
# These match the artefacts produced by the training notebooks/scripts.
SAVED_MODELS: Dict[str, str] = {
    "baseline":         "baseline_best.h5",
    "mobilenet_phase1": "mobilenet_phase1_best.h5",
    "mobilenet_phase2": "mobilenet_phase2_best.h5",
}


# ──────────────────────────────────────────────
# Seed fixing
# ──────────────────────────────────────────────
def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Fix all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    print(f"[utils] Seeds fixed to {seed}")


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(Path(log_file).parent, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────
def ensure_dirs(*paths: str) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def list_image_files(folder: str) -> List[Path]:
    """Return all image files in a folder (non-recursive)."""
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    return sorted([
        f for f in folder_path.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
    ])


def get_class_folder_map(data_dir: str) -> Dict[str, str]:
    """
    Build mapping: class_name -> folder_path
    e.g., "apple_unripe" -> "dataset/apple/unripe"
    """
    mapping = {}
    for fruit, stages in DATASET_TREE.items():
        for stage in stages:
            class_name = f"{fruit}_{stage}"
            mapping[class_name] = str(Path(data_dir) / fruit / stage)
    return mapping


def count_images_per_class(data_dir: str) -> Dict[str, int]:
    """Count images in each class folder."""
    folder_map = get_class_folder_map(data_dir)
    return {
        cls: len(list_image_files(folder))
        for cls, folder in folder_map.items()
    }


# ──────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────
def format_metrics_table(metrics_dict: dict) -> str:
    """Format metrics dict as an ASCII table string."""
    lines = ["=" * 55]
    lines.append(f"{'Metric':<30} {'Value':>10}")
    lines.append("-" * 55)
    for key, val in metrics_dict.items():
        if isinstance(val, float):
            lines.append(f"  {key:<28} {val:>10.4f}")
        else:
            lines.append(f"  {key:<28} {str(val):>10}")
    lines.append("=" * 55)
    return "\n".join(lines)


def save_model_summary(model, output_path: str) -> None:
    """Save model.summary() to a text file."""
    ensure_dirs(str(Path(output_path).parent))
    with open(output_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"[utils] Model summary saved to {output_path}")


# ──────────────────────────────────────────────
# Training history helpers
# ──────────────────────────────────────────────
def save_history_csv(history, output_path: str) -> None:
    """Save Keras training history to CSV."""
    import pandas as pd
    ensure_dirs(str(Path(output_path).parent))
    df = pd.DataFrame(history.history)
    df.index.name = "epoch"
    df.to_csv(output_path)
    print(f"[utils] Training history saved to {output_path}")


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    set_seeds(42)
    logger = get_logger("utils_test")
    logger.info(f"NUM_CLASSES = {NUM_CLASSES}")
    logger.info(f"CLASS_NAMES (canonical)    = {CLASS_NAMES}")
    logger.info(f"ALPHABETICAL_CLASS_NAMES   = {ALPHABETICAL_CLASS_NAMES}")
    logger.info(f"ALPHA_TO_CANONICAL         = {ALPHA_TO_CANONICAL}")
    # Sanity: permuting an identity-scores matrix maps alpha->canonical correctly
    probs = np.eye(NUM_CLASSES)
    remapped = alpha_probs_to_canonical(probs)
    for j, name in enumerate(ALPHABETICAL_CLASS_NAMES):
        assert remapped[j].argmax() == CLASS_NAMES.index(name)
    logger.info("Class-order permutation self-test passed.")
    print(format_metrics_table({"accuracy": 0.9243, "f1_macro": 0.9101}))

"""
src/train.py
============
Training script for fruit ripeness classification.

Supports:
    - Baseline CNN
    - MobileNetV2 Phase 1 (head-only training)
    - MobileNetV2 Phase 2 (fine-tuning)

Usage:
    python src/train.py --model baseline --data_dir ./dataset --batch_size 16 --epochs 50 --seed 42
    python src/train.py --model mobilenet --phase phase1 --data_dir ./dataset --batch_size 16 --epochs 20
    python src/train.py --model mobilenet --phase finetune --data_dir ./dataset --batch_size 16 --epochs 20
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seeds, get_logger, ensure_dirs, save_history_csv
from src.data_pipeline import build_datasets
from src.models import get_model

logger = get_logger("train")


# ──────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────
def build_callbacks(model_save_path: str, patience: int = 5):
    """
    Standard callback stack:
    - ModelCheckpoint (save best)
    - EarlyStopping (patience=5 on val_loss)
    - ReduceLROnPlateau
    - TensorBoard (optional)
    """
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return callbacks


# ──────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────
def plot_training_curves(history, model_name: str, output_dir: str = "results"):
    """Plot and save loss + accuracy training curves."""
    ensure_dirs(output_dir)
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(hist["loss"],     label="train loss")
    axes[0].plot(hist["val_loss"], label="val loss")
    axes[0].set_title(f"{model_name} – Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(hist["accuracy"],     label="train acc")
    axes[1].plot(hist["val_accuracy"], label="val acc")
    axes[1].set_title(f"{model_name} – Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    out_path = f"{output_dir}/training_curves_{model_name}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {out_path}")


def append_results_csv(
    history,
    model_name: str,
    output_path: str = "results/results.csv",
) -> None:
    """
    Append best epoch metrics to results/results.csv.
    Columns: model, epoch, val_loss, val_acc, precision_macro, recall_macro, f1_macro
    """
    ensure_dirs(str(Path(output_path).parent))
    hist = history.history

    best_epoch = int(np.argmin(hist["val_loss"]))
    row = {
        "model":            model_name,
        "epoch":            best_epoch + 1,
        "val_loss":         round(hist["val_loss"][best_epoch], 4),
        "val_acc":          round(hist.get("val_accuracy", [0])[best_epoch], 4),
        "precision_macro":  "N/A",   # filled by evaluate.py
        "recall_macro":     "N/A",
        "f1_macro":         "N/A",
    }

    df_new = pd.DataFrame([row])

    if Path(output_path).exists():
        df_existing = pd.read_csv(output_path)
        # Remove duplicate model entry if exists
        df_existing = df_existing[df_existing["model"] != model_name]
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(output_path, index=False)
    logger.info(f"Results updated in {output_path}")


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────
def train(args):
    """Full training run based on CLI args."""
    set_seeds(args.seed)

    ensure_dirs("saved_models", "results")

    # ── Determine model save path ──────────────
    if args.model == "baseline":
        model_name      = "baseline"
        model_save_path = "saved_models/baseline_best.h5"
        patience        = 5
        lr              = 1e-3

    elif args.model == "mobilenet" and args.phase == "phase1":
        model_name      = "mobilenet_phase1"
        model_save_path = "saved_models/mobilenet_phase1.h5"
        patience        = 5
        lr              = 1e-3

    elif args.model == "mobilenet" and args.phase == "finetune":
        model_name      = "mobilenet_finetuned"
        model_save_path = "saved_models/mobilenet_finetuned.h5"
        patience        = 5
        lr              = 1e-5

    else:
        raise ValueError(
            f"Unknown combination: model={args.model} phase={args.phase}"
        )

    logger.info(f"{'='*50}")
    logger.info(f"  Model:      {model_name}")
    logger.info(f"  Data dir:   {args.data_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max epochs: {args.epochs}")
    logger.info(f"  Seed:       {args.seed}")
    logger.info(f"  Save path:  {model_save_path}")
    logger.info(f"{'='*50}")

    # ── Load data ─────────────────────────────
    logger.info("Loading datasets...")
    train_ds, val_ds, test_ds = build_datasets(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # ── Build model ───────────────────────────
    logger.info("Building model...")
    if args.model == "baseline":
        model = get_model("baseline", learning_rate=lr)

    elif args.model == "mobilenet" and args.phase == "phase1":
        model = get_model("mobilenet", phase="phase1", learning_rate=lr)

    elif args.model == "mobilenet" and args.phase == "finetune":
        phase1_path = args.phase1_path or "saved_models/mobilenet_phase1.h5"
        if not Path(phase1_path).exists():
            raise FileNotFoundError(
                f"Phase 1 model not found at {phase1_path}. "
                "Run phase1 training first."
            )
        model = get_model(
            "mobilenet", phase="finetune",
            phase1_model_path=phase1_path,
            learning_rate=lr,
        )

    # ── Callbacks ─────────────────────────────
    callbacks = build_callbacks(model_save_path, patience=patience)

    # ── Train ─────────────────────────────────
    logger.info("Starting training...")
    start_time = datetime.now()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = datetime.now() - start_time
    logger.info(f"Training completed in {elapsed}")

    # ── Save artifacts ────────────────────────
    # Training curves
    plot_training_curves(history, model_name)

    # History CSV
    save_history_csv(
        history,
        output_path=f"results/history_{model_name}.csv"
    )

    # Results CSV (partial – precision/recall/f1 filled by evaluate.py)
    append_results_csv(history, model_name)

    # Best epoch log
    best_epoch = int(np.argmin(history.history["val_loss"]))
    best_val_loss = history.history["val_loss"][best_epoch]
    best_val_acc  = history.history.get("val_accuracy", [0])[best_epoch]
    logger.info(
        f"Best epoch: {best_epoch+1} | "
        f"val_loss: {best_val_loss:.4f} | "
        f"val_acc: {best_val_acc:.4f}"
    )
    logger.info(f"Model saved to: {model_save_path}")

    return model, history


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train fruit ripeness classifier"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["baseline", "mobilenet"],
        help="Model architecture to train"
    )
    parser.add_argument(
        "--phase", type=str, default=None,
        choices=["phase1", "finetune"],
        help="Training phase (required for mobilenet)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./dataset",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--phase1_path", type=str,
        default="saved_models/mobilenet_phase1.h5",
        help="Path to Phase 1 model (for finetune phase)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

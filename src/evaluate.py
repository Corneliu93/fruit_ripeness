"""
src/evaluate.py
===============
Full evaluation pipeline:
- Accuracy, Precision, Recall, F1 (per class + macro)
- Confusion matrix (saved as PNG)
- ROC curves (one-vs-rest)
- Grad-CAM visualizations (3 examples per class)
- Updates results/results.csv

Usage:
    python src/evaluate.py --model_path saved_models/mobilenet_finetuned.h5 \
                            --data_dir ./dataset/test --output results/
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import CLASS_NAMES, NUM_CLASSES, ensure_dirs, get_logger

logger = get_logger("evaluate")


# ──────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────
def get_predictions(model, dataset):
    """
    Run inference on a tf.data dataset.
    Returns: y_true (int), y_pred (int), y_scores (float[N, C])
    """
    y_true   = []
    y_scores = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_scores.append(preds)
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_scores = np.vstack(y_scores)
    y_pred   = np.argmax(y_scores, axis=1)
    return np.array(y_true), np.array(y_pred), y_scores


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_scores):
    """
    Compute classification metrics.
    Returns dict with per-class and macro metrics.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report
    )

    metrics = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision_macro":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":         f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class   = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, cls in enumerate(CLASS_NAMES):
        if i < len(prec_per_class):
            metrics[f"precision_{cls}"] = prec_per_class[i]
            metrics[f"recall_{cls}"]    = rec_per_class[i]
            metrics[f"f1_{cls}"]        = f1_per_class[i]

    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
    logger.info(f"\nClassification Report:\n{report}")

    return metrics, report


def update_results_csv(metrics: dict, model_name: str, output_dir: str = "results"):
    """Update results.csv with evaluation metrics."""
    ensure_dirs(output_dir)
    csv_path = f"{output_dir}/results.csv"

    new_row = {
        "model":           model_name,
        "epoch":           "eval",
        "val_loss":        "N/A",
        "val_acc":         round(metrics["accuracy"], 4),
        "precision_macro": round(metrics["precision_macro"], 4),
        "recall_macro":    round(metrics["recall_macro"], 4),
        "f1_macro":        round(metrics["f1_macro"], 4),
    }

    df_new = pd.DataFrame([new_row])

    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        # Update the row for this model (eval row)
        df = df[~((df["model"] == model_name) & (df["epoch"] == "eval"))]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)
    logger.info(f"Results CSV updated at {csv_path}")


# ──────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str, output_dir: str):
    """Plot and save normalized confusion matrix."""
    from sklearn.metrics import confusion_matrix

    ensure_dirs(output_dir)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    short_names = [c.replace("_", "\n") for c in CLASS_NAMES]

    for ax, data, title, fmt in [
        (axes[0], cm,      "Count",      "d"),
        (axes[1], cm_norm, "Normalized", ".2f"),
    ]:
        sns.heatmap(
            data,
            annot=True, fmt=fmt,
            xticklabels=short_names,
            yticklabels=short_names,
            cmap="Blues", ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix ({title})\n{model_name}", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    plt.tight_layout()
    out_path = f"{output_dir}/confusion_matrix_{model_name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {out_path}")


# ──────────────────────────────────────────────
# ROC curves
# ──────────────────────────────────────────────
def plot_roc_curves(y_true, y_scores, model_name: str, output_dir: str):
    """Plot one-vs-rest ROC curves for all classes."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    ensure_dirs(output_dir)
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        if i < y_scores.shape[1]:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{cls} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (One-vs-Rest) – {model_name}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{output_dir}/roc_curves_{model_name}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved to {out_path}")


# ──────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────
def compute_gradcam(model, image_array, class_idx, last_conv_layer_name=None):
    """
    Compute Grad-CAM heatmap for a single image.

    Args:
        model:                Keras model
        image_array:          numpy array, shape (H, W, C), values in [0, 1]
        class_idx:            target class index
        last_conv_layer_name: name of last conv layer (auto-detected if None)

    Returns:
        heatmap: numpy array (H, W), values in [0, 1]
    """
    import tensorflow as tf

    # Auto-detect last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if hasattr(layer, "filters") or "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            # For MobileNetV2, use the last conv inside the base
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.Model):
                    for inner in reversed(layer.layers):
                        if "conv" in inner.name.lower():
                            last_conv_layer_name = inner.name
                            break
                if last_conv_layer_name:
                    break

    if last_conv_layer_name is None:
        raise ValueError("Could not find last conv layer automatically.")

    # Build grad model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(
            np.expand_dims(image_array, 0), tf.float32
        )
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_overlay(
    image_array, heatmap,
    true_label, pred_label,
    example_id: str,
    output_dir: str,
):
    """Save Grad-CAM heatmap overlaid on original image."""
    import cv2

    ensure_dirs(output_dir)

    # Resize heatmap to image size
    h, w = image_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_uint8 = np.uint8(255 * image_array)
    overlay   = np.uint8(0.6 * img_uint8 + 0.4 * heatmap_rgb)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_uint8);      axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap_rgb);    axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay);        axes[2].set_title("Overlay");  axes[2].axis("off")

    fig.suptitle(
        f"True: {true_label} | Pred: {pred_label}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    out_path = f"{output_dir}/gradcam_{example_id}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    return out_path


def generate_gradcam_examples(
    model,
    dataset,
    y_true,
    y_pred,
    output_dir: str,
    n_per_class: int = 3,
):
    """Generate Grad-CAM for n_per_class examples per class."""
    # Collect all images and labels
    all_images = []
    for images, _ in dataset:
        all_images.extend(images.numpy())

    all_images = np.array(all_images[:len(y_true)])

    logger.info("Generating Grad-CAM visualizations...")
    saved = []
    class_counts = {i: 0 for i in range(NUM_CLASSES)}

    for idx in range(len(y_true)):
        cls = int(y_true[idx])
        if class_counts[cls] >= n_per_class:
            continue

        image   = all_images[idx]
        true_cls = CLASS_NAMES[cls]
        pred_cls = CLASS_NAMES[int(y_pred[idx])]

        try:
            heatmap = compute_gradcam(model, image, cls)
            example_id = f"{true_cls}_{class_counts[cls]+1:02d}"
            path = save_gradcam_overlay(
                image, heatmap,
                true_label=true_cls,
                pred_label=pred_cls,
                example_id=example_id,
                output_dir=output_dir,
            )
            saved.append(path)
            class_counts[cls] += 1
            logger.info(f"  Grad-CAM saved: {path}")
        except Exception as e:
            logger.warning(f"  Grad-CAM failed for index {idx}: {e}")

        if all(v >= n_per_class for v in class_counts.values()):
            break

    logger.info(f"Grad-CAM complete: {len(saved)} visualizations saved")


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────
def evaluate(args):
    import tensorflow as tf
    from src.data_pipeline import build_datasets, build_tf_dataset, stratified_split

    ensure_dirs(args.output)

    # ── Load model ────────────────────────────
    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    model_name = Path(args.model_path).stem

    # ── Load test data ────────────────────────
    logger.info(f"Loading test data from {args.data_dir}")
    # If data_dir ends with /test, load directly; else use split
    data_path = Path(args.data_dir)

    # Build full dataset splits and use test split
    parent_dir = str(data_path.parent) if data_path.name == "test" else str(data_path)
    _, _, test_samples = stratified_split(parent_dir, seed=args.seed)

    if not test_samples:
        logger.error("No test samples found. Check data_dir.")
        return

    test_ds = build_tf_dataset(
        test_samples, batch_size=args.batch_size,
        augment=False, shuffle=False
    )

    # ── Get predictions ───────────────────────
    logger.info("Running inference...")
    y_true, y_pred, y_scores = get_predictions(model, test_ds)

    # ── Compute metrics ───────────────────────
    metrics, report = compute_metrics(y_true, y_pred, y_scores)

    # Print summary
    print("\n" + "=" * 50)
    print(f"  Evaluation Results: {model_name}")
    print("=" * 50)
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    print("=" * 50)

    # Save classification report
    with open(f"{args.output}/classification_report_{model_name}.txt", "w") as f:
        f.write(report)

    # ── Update results.csv ────────────────────
    update_results_csv(metrics, model_name, args.output)

    # ── Confusion matrix ──────────────────────
    plot_confusion_matrix(y_true, y_pred, model_name, args.output)

    # ── ROC curves ────────────────────────────
    plot_roc_curves(y_true, y_scores, model_name, args.output)

    # ── Grad-CAM ──────────────────────────────
    if not args.skip_gradcam:
        generate_gradcam_examples(
            model, test_ds, y_true, y_pred,
            output_dir=args.output,
            n_per_class=3,
        )

    logger.info(f"Evaluation complete. Results in {args.output}/")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained fruit ripeness model"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="saved_models/mobilenet_finetuned.h5",
        help="Path to saved Keras model (.h5)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./dataset",
        help="Root dataset directory"
    )
    parser.add_argument(
        "--output", type=str, default="results/",
        help="Output directory for evaluation artifacts"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--skip_gradcam", action="store_true",
        help="Skip Grad-CAM generation (faster)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
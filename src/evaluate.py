"""
src/evaluate.py
===============
Full evaluation pipeline for the fruit ripeness models.

Produces:
    - Accuracy, Precision, Recall, F1 (per class + macro)
    - Confusion matrix (count + normalised)
    - One-vs-rest ROC curves with AUC
    - Grad-CAM overlays (transfer-learning model)
    - Updates results/results.csv

Two model families are supported via --model_kind:
    baseline   : custom CNN. Trained in canonical CLASS_NAMES order on [0, 1]
                 inputs through src.data_pipeline.build_tf_dataset.
    mobilenet  : transfer-learning model. Trained via image_dataset_from_directory
                 (alphabetical class order) on [-1, 1] inputs. Its predictions and
                 labels are permuted back to the canonical CLASS_NAMES order, so all
                 reported figures share one class ordering. No retraining occurs.

Usage:
    python src/evaluate.py --model_kind mobilenet \
        --model_path saved_models/mobilenet_phase2_best.h5 \
        --splits_dir dataset_splits --output results/

    python src/evaluate.py --model_kind baseline \
        --model_path saved_models/baseline_best.h5 \
        --data_dir ./dataset --output results/
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

from src.utils import (
    CLASS_NAMES, ALPHABETICAL_CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE,
    ensure_dirs, get_logger,
    alpha_probs_to_canonical, alpha_labels_to_canonical,
)

logger = get_logger("evaluate")


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def get_predictions(model, dataset):
    """
    Run inference on a tf.data dataset.
    Returns y_true (int labels in the dataset's own order) and y_scores (float[N, C]).
    """
    y_true, y_scores = [], []
    for images, labels in dataset:
        y_scores.append(model.predict(images, verbose=0))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_scores = np.vstack(y_scores)
    return np.array(y_true), y_scores


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Compute classification metrics in canonical CLASS_NAMES order."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, classification_report
    )
    metrics = {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    prec = precision_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    for i, cls in enumerate(CLASS_NAMES):
        metrics[f"precision_{cls}"] = float(prec[i])
        metrics[f"recall_{cls}"]    = float(rec[i])
        metrics[f"f1_{cls}"]        = float(f1[i])

    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    logger.info(f"\nClassification Report:\n{report}")
    return metrics, report


def update_results_csv(metrics: dict, model_name: str, output_dir: str = "results"):
    """
    Update results/results.csv eval row for this model. The eval row stores the
    test accuracy (in the val_acc column, matching the train.py schema) plus the
    macro precision/recall/F1.
    """
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
        df = df[~((df["model"] == model_name) & (df["epoch"].astype(str) == "eval"))]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    logger.info(f"Results CSV updated at {csv_path}")


# ──────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str, output_dir: str, show: bool = False):
    """Plot and save count + normalised confusion matrices (canonical order)."""
    from sklearn.metrics import confusion_matrix
    ensure_dirs(output_dir)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    short = [c.replace("_", "\n") for c in CLASS_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, title, fmt in [
        (axes[0], cm,      "Count",      "d"),
        (axes[1], cm_norm, "Normalized", ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, xticklabels=short, yticklabels=short,
                    cmap="Blues", ax=ax, linewidths=0.5)
        ax.set_title(f"Confusion Matrix ({title}) - {model_name}", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    out = f"{output_dir}/confusion_matrix_{model_name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    logger.info(f"Confusion matrix saved to {out}")
    if show:
        plt.show()
    else:
        plt.close()


# ──────────────────────────────────────────────
# ROC curves
# ──────────────────────────────────────────────
def plot_roc_curves(y_true, y_scores, model_name: str, output_dir: str, show: bool = False):
    """Plot one-vs-rest ROC curves (canonical order). Returns macro AUC."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    ensure_dirs(output_dir)
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    aucs = []
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        a = auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{cls} (AUC={a:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (One-vs-Rest) - {model_name} (macro AUC={macro_auc:.4f})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{output_dir}/roc_curves_{model_name}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    logger.info(f"ROC curves saved to {out} (macro AUC={macro_auc:.4f})")
    if show:
        plt.show()
    else:
        plt.close()
    return macro_auc


# ──────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────
def build_gradcam_model(model):
    """
    Build a Grad-CAM model for a Sequential([MobileNetV2_base, GAP, Dropout, Dense]).

    With include_top=False and no pooling, the base output IS the final
    convolutional feature map (out_relu, 7x7x1280). We expose it and re-apply the
    head so the single functional graph connects input -> (feature_map, predictions).
    This avoids the 'graph disconnected' error that arises from calling
    model.get_layer() on a layer nested inside the base sub-model.
    """
    import tensorflow as tf
    base = model.layers[0]
    if not isinstance(base, tf.keras.Model):
        raise ValueError("Expected the MobileNetV2 base as the first layer of the model.")
    feature_map = base.output
    x = feature_map
    for layer in model.layers[1:]:
        x = layer(x)
    return tf.keras.Model(base.input, [feature_map, x])


def compute_gradcam(grad_model, img_batch, pred_index=None):
    """Compute a Grad-CAM heatmap for one preprocessed image batch (shape (1,H,W,3))."""
    import tensorflow as tf
    with tf.GradientTape() as tape:
        feature_map, preds = grad_model(img_batch, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, feature_map)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    fmap = feature_map[0]
    heatmap = tf.squeeze(fmap @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index, float(preds[0][pred_index])


def overlay_heatmap(raw01, heatmap, alpha: float = 0.4):
    """Overlay a heatmap on a [0,1] RGB image using a matplotlib colormap (no cv2)."""
    import tensorflow as tf
    jet = plt.get_cmap("jet")
    hm = tf.image.resize(heatmap[..., np.newaxis], raw01.shape[:2]).numpy().squeeze()
    colored = jet(hm)[..., :3]
    return np.clip((1.0 - alpha) * raw01 + alpha * colored, 0, 1)


def generate_gradcam_grid(model, splits_dir, output_dir, model_name,
                          n_per_class: int = 1, show: bool = False):
    """
    Generate a Grad-CAM grid for the transfer-learning model: one row per class
    (canonical order), n_per_class example(s) each, shown as original | overlay.
    Predicted-class names use the model's alphabetical index space.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.utils import load_img, img_to_array
    ensure_dirs(output_dir)

    grad_model = build_gradcam_model(model)
    test_dir = Path(splits_dir) / "test"

    rows = []
    for cls in CLASS_NAMES:  # canonical order
        cdir = test_dir / cls
        imgs = sorted([p for p in cdir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")]) if cdir.exists() else []
        rows.append((cls, imgs[:n_per_class]))

    ncols = max(1, n_per_class) * 2  # (original + overlay) per example
    fig, axes = plt.subplots(NUM_CLASSES, ncols, figsize=(ncols * 2.0, NUM_CLASSES * 2.2))
    axes = np.atleast_2d(axes)

    for r, (cls, imgs) in enumerate(rows):
        for k in range(max(1, n_per_class)):
            ax_o = axes[r, k * 2]
            ax_h = axes[r, k * 2 + 1]
            if k < len(imgs):
                raw = img_to_array(load_img(imgs[k], target_size=IMAGE_SIZE))
                inp = preprocess_input(np.expand_dims(raw.copy(), 0).astype("float32"))
                heatmap, pred_idx, conf = compute_gradcam(grad_model, inp)
                pred_name = ALPHABETICAL_CLASS_NAMES[pred_idx]
                overlay = overlay_heatmap(raw / 255.0, heatmap)
                mark = "OK" if pred_name == cls else "X"
                ax_o.imshow(raw.astype("uint8")); ax_o.set_title(cls, fontsize=7)
                ax_h.imshow(overlay); ax_h.set_title(f"[{mark}] {pred_name} ({conf:.2f})", fontsize=7)
            ax_o.axis("off"); ax_h.axis("off")

    fig.suptitle(f"Grad-CAM - {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = f"{output_dir}/gradcam_{model_name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    logger.info(f"Grad-CAM grid saved to {out}")
    if show:
        plt.show()
    else:
        plt.close()


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────
def evaluate(args):
    import tensorflow as tf

    ensure_dirs(args.output)
    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    model_name = args.model_name or Path(args.model_path).stem.replace("_best", "")

    if args.model_kind == "mobilenet":
        from src.data_pipeline import build_mobilenet_test_dataset
        test_ds, class_names_alpha = build_mobilenet_test_dataset(
            args.splits_dir, batch_size=args.batch_size
        )
        if class_names_alpha != ALPHABETICAL_CLASS_NAMES:
            logger.warning(
                f"Loaded class order {class_names_alpha} differs from expected alphabetical order."
            )
        y_true_alpha, y_scores_alpha = get_predictions(model, test_ds)
        # Permute alphabetical -> canonical so all reporting shares one order
        y_scores = alpha_probs_to_canonical(y_scores_alpha)
        y_true = alpha_labels_to_canonical(y_true_alpha)
    else:  # baseline
        from src.data_pipeline import stratified_split, build_tf_dataset
        _, _, test_samples = stratified_split(args.data_dir, seed=args.seed)
        if not test_samples:
            logger.error("No test samples found. Check --data_dir.")
            return
        test_ds = build_tf_dataset(test_samples, batch_size=args.batch_size,
                                   augment=False, shuffle=False)
        y_true, y_scores = get_predictions(model, test_ds)

    y_pred = np.argmax(y_scores, axis=1)
    metrics, report = compute_metrics(y_true, y_pred)
    macro_auc = plot_roc_curves(y_true, y_scores, model_name, args.output, show=False)
    metrics["macro_auc"] = macro_auc

    print("\n" + "=" * 52)
    print(f"  Evaluation Results: {model_name}")
    print("=" * 52)
    print(f"  Test accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    print(f"  Macro AUC:          {macro_auc:.4f}")
    print("=" * 52)

    with open(f"{args.output}/classification_report_{model_name}.txt", "w") as f:
        f.write(report)

    update_results_csv(metrics, model_name, args.output)
    plot_confusion_matrix(y_true, y_pred, model_name, args.output, show=False)

    if args.model_kind == "mobilenet" and not args.skip_gradcam:
        generate_gradcam_grid(model, args.splits_dir, args.output, model_name,
                              n_per_class=args.gradcam_per_class, show=False)

    logger.info(f"Evaluation complete. Results in {args.output}/")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained fruit ripeness model")
    p.add_argument("--model_kind", choices=["baseline", "mobilenet"], default="mobilenet",
                   help="Which model family is being evaluated")
    p.add_argument("--model_path", type=str, default="saved_models/mobilenet_phase2_best.h5",
                   help="Path to saved Keras model (.h5)")
    p.add_argument("--model_name", type=str, default=None,
                   help="Override the model name used in output filenames")
    p.add_argument("--splits_dir", type=str, default="dataset_splits",
                   help="dataset_splits root (used by --model_kind mobilenet)")
    p.add_argument("--data_dir", type=str, default="./dataset",
                   help="Original dataset root (used by --model_kind baseline)")
    p.add_argument("--output", type=str, default="results/", help="Output directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradcam_per_class", type=int, default=1,
                   help="Grad-CAM examples per class in the grid (mobilenet)")
    p.add_argument("--skip_gradcam", action="store_true", help="Skip Grad-CAM generation")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
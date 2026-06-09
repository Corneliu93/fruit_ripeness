"""
src/models.py
=============
Model definitions for fruit ripeness classification.

Models:
    1. BaselineCNN     - custom lightweight CNN (trained from scratch)
    2. MobileNetV2     - transfer learning with 2-phase training strategy

Usage:
    from src.models import build_baseline_cnn, build_mobilenetv2_phase1

NOTE ON PREPROCESSING:
    The MobileNetV2 builders below do NOT embed mobilenet_v2.preprocess_input
    inside the model. Preprocessing to the [-1, 1] range is applied in the data
    pipeline (see src.data_pipeline.build_mobilenet_datasets). This matches the
    trained checkpoints (mobilenet_phase1_best.h5 / mobilenet_phase2_best.h5).
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# Lazy TF import - allows importing this module without GPU
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.applications import MobileNetV2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.utils import NUM_CLASSES, IMAGE_SIZE, save_model_summary


# ──────────────────────────────────────────────
# 1. Baseline CNN
# ──────────────────────────────────────────────
def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
    save_summary: bool = True,
) -> "tf.keras.Model":
    """
    Baseline CNN architecture:

        Conv(32) -> BN -> ReLU -> MaxPool
        Conv(64) -> BN -> ReLU -> MaxPool
        Conv(128) -> BN -> ReLU -> MaxPool
        Conv(256) -> BN -> ReLU -> GlobalAvgPool
        Dense(256) -> Dropout -> Dense(num_classes, softmax)
    """
    reg = regularizers.l2(l2_reg)

    inputs = keras.Input(shape=input_shape, name="input_layer")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Head
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=reg, name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="output")(x)

    model = keras.Model(inputs, outputs, name="BaselineCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if save_summary:
        os.makedirs("saved_models", exist_ok=True)
        save_model_summary(model, "saved_models/model_summary_baseline.txt")

    return model


# ──────────────────────────────────────────────
# 2. MobileNetV2 - Phase 1 (frozen base, feature extraction)
# ──────────────────────────────────────────────
def build_mobilenetv2_phase1(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    save_summary: bool = True,
) -> "tf.keras.Model":
    """
    MobileNetV2 transfer learning - Phase 1 (feature extraction).

    Architecture (matches the trained checkpoint mobilenet_phase1_best.h5):

        MobileNetV2(include_top=False, weights="imagenet", frozen)
          -> GlobalAveragePooling2D
          -> Dropout(0.2)
          -> Dense(num_classes, softmax)

    The convolutional base is fully frozen; only the head is trained.
    Pre-processing to [-1, 1] is performed in the data pipeline, not here.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze entire base

    model = keras.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(name="gap"),
            layers.Dropout(dropout_rate, name="dropout"),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="mobilenetv2_phase1",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if save_summary:
        os.makedirs("saved_models", exist_ok=True)
        save_model_summary(model, "saved_models/model_summary_mobilenet_phase1.txt")

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total     = int(sum(np.prod(w.shape) for w in model.weights))
    print(f"[models] Phase 1 - Trainable params: {trainable:,} / {total:,}")

    return model


# ──────────────────────────────────────────────
# 3. MobileNetV2 - Phase 2 (fine-tuning)
# ──────────────────────────────────────────────
def build_mobilenetv2_finetune(
    phase1_model_path: str,
    fine_tune_at: int = 100,
    learning_rate: float = 1e-5,
    save_summary: bool = True,
) -> "tf.keras.Model":
    """
    MobileNetV2 fine-tuning - Phase 2 (matches mobilenet_phase2_best.h5).

    Strategy:
        - Load the Phase 1 model.
        - Unfreeze base layers from index `fine_tune_at` (default 100) onward,
          following the TensorFlow transfer-learning guidance; layers below stay
          frozen to preserve generic low-level features.
        - Keep EVERY BatchNormalization layer frozen (inference mode), so the
          ImageNet population statistics are not corrupted on a small dataset.
        - Recompile with a low learning rate (default 1e-5).

    Args:
        phase1_model_path: path to the saved Phase 1 model.
        fine_tune_at:      first base-layer index to unfreeze.
        learning_rate:     Adam LR for fine-tuning.
    """
    model = keras.models.load_model(phase1_model_path, compile=False)
    print(f"[models] Loaded Phase 1 model from {phase1_model_path}")

    base_model = model.layers[0]
    if not isinstance(base_model, keras.Model) or len(base_model.layers) < 100:
        raise ValueError(
            "Expected the MobileNetV2 base as the first layer of the model."
        )

    base_model.trainable = True

    # Freeze everything before the fine-tune cutoff
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Keep every BatchNormalization layer frozen regardless of position
    bn_frozen = 0
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
            bn_frozen += 1

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    frozen   = sum(1 for l in base_model.layers if not l.trainable)
    print(f"[models] Phase 2 - cutoff={fine_tune_at} | "
          f"base layers: {frozen} frozen, {unfrozen} unfrozen | "
          f"BatchNorm frozen: {bn_frozen}")

    if save_summary:
        os.makedirs("saved_models", exist_ok=True)
        save_model_summary(
            model, "saved_models/model_summary_mobilenet_phase2.txt"
        )

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total     = int(sum(np.prod(w.shape) for w in model.weights))
    print(f"[models] Phase 2 - Trainable params: {trainable:,} / {total:,}")

    return model


# ──────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────
def get_model(
    model_name: str,
    phase: Optional[str] = None,
    phase1_model_path: Optional[str] = None,
    **kwargs,
) -> "tf.keras.Model":
    """
    Model factory.

    Args:
        model_name:        "baseline" | "mobilenet"
        phase:             for mobilenet: "phase1" | "finetune"
        phase1_model_path: required when phase="finetune"
    """
    if model_name == "baseline":
        return build_baseline_cnn(**kwargs)

    elif model_name == "mobilenet":
        if phase == "phase1":
            return build_mobilenetv2_phase1(**kwargs)
        elif phase == "finetune":
            if phase1_model_path is None:
                raise ValueError("phase1_model_path required for finetune phase")
            return build_mobilenetv2_finetune(phase1_model_path, **kwargs)
        else:
            raise ValueError(f"Unknown phase '{phase}'. Use 'phase1' or 'finetune'.")

    else:
        raise ValueError(f"Unknown model '{model_name}'. Use 'baseline' or 'mobilenet'.")


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing BaselineCNN ---")
    m = build_baseline_cnn(save_summary=False)
    m.summary()

    print("\n--- Testing MobileNetV2 Phase 1 ---")
    m2 = build_mobilenetv2_phase1(save_summary=False)
    m2.summary()

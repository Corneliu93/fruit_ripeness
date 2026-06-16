"""
web_app/app.py
==============
Flask demo for fruit ripeness classification.

Usage:
    python web_app/app.py
    # Open http://127.0.0.1:5000

Features:
    - Upload image via browser
    - Real-time prediction with confidence scores
"""

import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]    = Path(__file__).parent / "static" / "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# Locate the model: prefer the deployed copy in web_app/model/, then fall back
# to the project's saved_models/ so the app runs without an extra copy step.
_MODEL_CANDIDATES = [
    Path(__file__).parent / "model" / "mobilenet_phase2_best.h5",
    Path(__file__).parent.parent / "saved_models" / "mobilenet_phase2_best.h5",
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if p.exists()), _MODEL_CANDIDATES[0])

# Class order MUST match the model's output order. The model was trained with
# image_dataset_from_directory, which orders classes ALPHABETICALLY, so output
# index i corresponds to CLASS_NAMES[i] below (this is the alphabetical order
# from src.utils, not the canonical fruit-then-ripeness order).
CLASS_NAMES = [
    "apple_overripe", "apple_ripe", "apple_unripe",
    "banana_overripe", "banana_ripe", "banana_unripe",
    "pear_overripe", "pear_ripe", "pear_unripe",
]
IMAGE_SIZE = (224, 224)

# Global model (loaded once)
model = None


def load_model():
    """Load model once at startup."""
    global model
    if not MODEL_PATH.exists():
        print(f"⚠️  Model not found. Searched: "
              f"{', '.join(str(p) for p in _MODEL_CANDIDATES)}")
        print("   Copy mobilenet_phase2_best.h5 to web_app/model/ or saved_models/.")
        return False
    import tensorflow as tf
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"✅ Model loaded from {MODEL_PATH}")
    return True


def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for inference.

    The model has no embedded preprocessing and was trained on inputs scaled to
    [-1, 1] by mobilenet_v2.preprocess_input, so the same scaling is applied here.
    Bilinear resizing matches the interpolation used by image_dataset_from_directory.
    """
    from PIL import Image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)          # range [0, 255]
    arr = preprocess_input(arr)                     # -> range [-1, 1]
    return np.expand_dims(arr, 0)                   # (1, H, W, C)


def predict_image(image_path: str) -> dict:
    """Run inference and return top predictions."""
    if model is None:
        return {"error": "Model not loaded"}

    image_batch = preprocess_image(image_path)
    probs = model.predict(image_batch, verbose=0)[0]

    top3_idx  = np.argsort(probs)[::-1][:3]
    top3      = [
        {
            "class":      CLASS_NAMES[i],
            "confidence": float(round(probs[i] * 100, 2)),
        }
        for i in top3_idx
    ]

    return {
        "top_class":    CLASS_NAMES[top3_idx[0]],
        "confidence":   float(round(probs[top3_idx[0]] * 100, 2)),
        "top3":         top3,
        "all_probs":    {CLASS_NAMES[i]: float(round(probs[i] * 100, 2))
                         for i in range(len(CLASS_NAMES))},
    }


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"
        }), 400

    filename   = secure_filename(file.filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filepath   = app.config["UPLOAD_FOLDER"] / filename
    file.save(str(filepath))

    result = predict_image(str(filepath))
    result["image_url"] = f"/static/uploads/{filename}"
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    })


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    load_model()
    print("\n🍎 Fruit Ripeness Classifier – Flask Demo")
    print("   http://127.0.0.1:5000\n")
    # debug=False so the model is loaded once; the reloader would load it twice.
    app.run(host="127.0.0.1", port=5000, debug=False)
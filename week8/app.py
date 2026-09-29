"""
app.py

Week 8 - AI/ML Foundation
Mobile-friendly deployment: a simple Flask web app that serves the
trained Week 6 plant disease CNN.

The model is NOT retrained here. It is loaded from:
    week6/models/plant_disease_cnn_week6.keras
(the same model Week 7 used for per-class evaluation).
"""

import io
import logging
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

# Keep TensorFlow start-up logs quiet (set before importing TF).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

WEEK8_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEEK8_DIR.parent

# Same model used by Week 7. Can be overridden with MODEL_PATH.
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        REPO_ROOT / "week6" / "models" / "plant_disease_cnn_week6.keras",
    )
)

# Same value as IMAGE_SIZE in week6/src/dataset_utils.py
IMAGE_SIZE = (128, 128)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_MB = 5

# The 38 PlantVillage classes, in the exact order used for training.
# Weeks 4-7 built this list with sorted(set(labels)) from the folder
# names, so class index i in the model output = CLASS_NAMES[i].
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("week8")

# ---------------------------------------------------------
# MODEL (loaded lazily, once per worker)
# ---------------------------------------------------------

_model = None


def get_model():
    """Load the trained model on first use and cache it."""

    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        from tensorflow.keras.models import load_model

        log.info("Loading model from %s", MODEL_PATH)
        _model = load_model(MODEL_PATH, compile=False)

        output_size = _model.output_shape[-1]
        if output_size != len(CLASS_NAMES):
            _model = None
            raise ValueError(
                f"Model outputs {output_size} classes, expected {len(CLASS_NAMES)}"
            )

    return _model


# ---------------------------------------------------------
# PREPROCESSING + PREDICTION
# ---------------------------------------------------------

def preprocess_image(file_bytes):
    """
    Match the training pipeline (week6/src/data_loader.py):
    RGB (3 channels) -> tf.image.resize to 128x128 -> float32 -> /255.

    tf.image.resize is used (not PIL's resize) on purpose: PIL applies
    anti-aliasing when shrinking, TensorFlow's default does not, and the
    model was trained on TensorFlow-resized images.
    """

    import tensorflow as tf

    image = Image.open(io.BytesIO(file_bytes))
    image.verify()  # detects truncated/corrupted files

    # verify() invalidates the object, so reopen for real use
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)

    resized = tf.image.resize(array, IMAGE_SIZE)
    resized = tf.cast(resized, tf.float32) / 255.0

    return np.expand_dims(resized.numpy(), axis=0)


def predict(batch):
    """Run the model on a preprocessed (1, 128, 128, 3) batch."""

    model = get_model()

    probabilities = model.predict(batch, verbose=0)[0]
    top = np.argsort(probabilities)[::-1][:3]

    return [
        {
            "class_name": CLASS_NAMES[i],
            "label": CLASS_NAMES[i].replace("___", " - ").replace("_", " "),
            "confidence": round(float(probabilities[i]) * 100, 2),
        }
        for i in top
    ]


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB)


@app.route("/health")
def health():
    """Lightweight check for Render (does not load the model)."""
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict_route():

    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify(error="Please select an image first."), 400

    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error="Invalid file type. Please upload a JPG or PNG image."), 400

    file_bytes = file.read()

    try:
        batch = preprocess_image(file_bytes)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return jsonify(error="This file could not be read as an image. It may be corrupted."), 400
    except Exception:
        log.exception("Image preprocessing failed")
        return jsonify(error="Prediction failed. Please try another image."), 500

    try:
        results = predict(batch)
    except (FileNotFoundError, ValueError):
        log.exception("Model could not be loaded")
        return jsonify(error="The model is currently unavailable. Please try again later."), 503
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="Prediction failed. Please try another image."), 500

    return jsonify(prediction=results[0], top3=results)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Image is too large. Maximum size is {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

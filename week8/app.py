"""
app.py

Week 8 - AI/ML Foundation
Realistic Plant Disease Image Analysis - Flask entry point.

This route does NOT run "image -> CNN -> softmax -> prediction" as a
single blind step. It runs the full validation pipeline in
prediction_service.py and only shows a disease name when there is
real evidence for it. See README.md for the full explanation.
"""

import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

from flask import Flask, jsonify, render_template, request

import config
from model_loader import ModelUnavailableError, model_status
from prediction_service import InvalidUploadError, analyze_upload

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("week8")


@app.route("/")
def index():
    supported_classes = [
        {"raw": name, "label": name.replace("___", " - ").replace("_", " ")}
        for name in config.CLASS_NAMES
    ]
    return render_template(
        "index.html",
        max_mb=config.MAX_UPLOAD_MB,
        supported_classes=supported_classes,
        num_classes=config.NUM_CLASSES,
    )


@app.route("/health")
def health():
    """Lightweight check for uptime monitors / Render. Never loads the model."""
    return jsonify(status="ok", **model_status())


@app.route("/predict", methods=["POST"])
def predict_route():

    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify(error="Please select an image first."), 400

    file_bytes = file.read()

    try:
        result = analyze_upload(file.filename, file_bytes)
    except InvalidUploadError as exc:
        return jsonify(error=str(exc)), 400
    except ModelUnavailableError:
        log.exception("Model is unavailable")
        return jsonify(
            error="The prediction model is currently unavailable on this "
            "server. Please try again later."
        ), 503
    except Exception:  # noqa: BLE001 - never leak internals to the client
        log.exception("Unexpected error while analyzing the upload")
        return jsonify(
            error="Something went wrong while analyzing this image. "
            "Please try a different image."
        ), 500

    return jsonify(**result)


@app.errorhandler(413)
def too_large(_):
    return jsonify(
        error=f"Image is too large. Maximum size is {config.MAX_UPLOAD_MB} MB."
    ), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

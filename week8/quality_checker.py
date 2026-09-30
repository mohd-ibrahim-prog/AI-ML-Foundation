"""
quality_checker.py

Purpose:
    Second stage of the pipeline: is the image good enough quality to
    even attempt analysis? Checks dimensions, blur, brightness and
    contrast using simple, explainable, classic image-processing
    metrics - no machine learning model is needed for this stage.

    Blur is estimated with "variance of the Laplacian", a standard,
    well-known technique: a sharp image has a lot of high-frequency
    edge energy, so the Laplacian (a simple edge filter) has high
    variance; a blurry image's edges are smoothed away, so the
    variance is low. It is implemented here directly with NumPy (no
    OpenCV/SciPy dependency) to keep the project lightweight.

Week:
    AI/ML Foundation - Week 8
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

import config


@dataclass
class QualityResult:
    ok: bool
    reason: str = ""
    metrics: dict = field(default_factory=dict)


def _to_grayscale_array(image, max_side=400):
    """Resize (for a consistent, fast blur score) and convert to grayscale."""

    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.BILINEAR,
        )

    gray = np.asarray(image.convert("L"), dtype=np.float64)
    return gray


def laplacian_variance(gray):
    """
    Variance of the Laplacian, computed with a plain NumPy convolution
    against the standard 4-connected Laplacian kernel:

        0  1  0
        1 -4  1
        0  1  0

    Implemented as shifted-array addition instead of calling an image
    library's convolution function, so no extra dependency is needed.
    """

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    center = gray[1:-1, 1:-1]
    up = gray[:-2, 1:-1]
    down = gray[2:, 1:-1]
    left = gray[1:-1, :-2]
    right = gray[1:-1, 2:]

    laplacian = up + down + left + right - 4 * center
    return float(np.var(laplacian))


def check_quality(image):
    """
    Run all quality checks on a decoded PIL image.

    Returns a QualityResult. `metrics` always contains the computed
    values, even when the check fails, so the caller can log/display
    them if useful.
    """

    width, height = image.size
    gray = _to_grayscale_array(image)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = laplacian_variance(gray)

    metrics = {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
    }

    if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT:
        return QualityResult(
            False,
            f"Image is too small ({width}x{height}px). Please upload an "
            f"image at least {config.MIN_IMAGE_WIDTH}x{config.MIN_IMAGE_HEIGHT}px.",
            metrics,
        )

    if brightness < config.BRIGHTNESS_MIN:
        return QualityResult(
            False,
            "Image is too dark. Please retake the photo with better lighting.",
            metrics,
        )

    if brightness > config.BRIGHTNESS_MAX:
        return QualityResult(
            False,
            "Image is too bright / overexposed. Please retake the photo.",
            metrics,
        )

    if contrast < config.CONTRAST_MIN:
        return QualityResult(
            False,
            "Image has very little detail (too flat/uniform). Please "
            "upload a clearer photo of the leaf.",
            metrics,
        )

    if blur_score < config.BLUR_THRESHOLD:
        return QualityResult(
            False,
            "Image appears too blurry. Please upload a sharper, "
            "in-focus close-up of the leaf.",
            metrics,
        )

    return QualityResult(True, metrics=metrics)

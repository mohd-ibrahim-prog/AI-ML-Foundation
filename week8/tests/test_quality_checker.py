"""
Tests for quality_checker.py.

These use synthetic images (no real photos are bundled with the
project), built to isolate one quality dimension at a time: dimension,
brightness, contrast, blur. See README.md for why synthetic images are
used and what that does and doesn't prove.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quality_checker as qc


def _flat(rgb, size=(300, 300), noise=5, seed=1):
    rng = np.random.default_rng(seed)
    arr = np.clip(
        np.array(rgb, dtype=np.int16) + rng.integers(-noise, noise, size=(*size, 3)),
        0, 255,
    ).astype("uint8")
    return Image.fromarray(arr)


def _textured(rgb, size=(300, 300), color_spread=35, small=25, fine_noise_std=18, seed=7):
    rng = np.random.default_rng(seed)
    coarse = np.clip(
        np.array(rgb, dtype=np.int16) + rng.integers(-color_spread, color_spread, size=(small, small, 3)),
        0, 255,
    ).astype("uint8")
    up = np.asarray(Image.fromarray(coarse).resize(size, Image.BICUBIC), dtype=np.int16)
    fine = rng.normal(0, fine_noise_std, size=(*size, 3))
    return Image.fromarray(np.clip(up + fine, 0, 255).astype("uint8"))


def test_too_small_is_rejected():
    img = _textured((60, 150, 50), size=(40, 40), small=8)
    result = qc.check_quality(img)
    assert result.ok is False
    assert "small" in result.reason.lower()


def test_too_dark_is_rejected():
    img = _flat((8, 8, 8), noise=3)
    result = qc.check_quality(img)
    assert result.ok is False
    assert "dark" in result.reason.lower()


def test_too_bright_is_rejected():
    img = _flat((250, 250, 250), noise=3)
    result = qc.check_quality(img)
    assert result.ok is False
    assert "bright" in result.reason.lower()


def test_flat_low_contrast_is_rejected():
    img = _flat((128, 128, 128), noise=1)
    result = qc.check_quality(img)
    assert result.ok is False


def test_blurry_image_is_rejected():
    sharp = _textured((60, 150, 50))
    blurry = sharp.filter(ImageFilter.GaussianBlur(6))
    result = qc.check_quality(blurry)
    # Heavy blur flattens both edge energy and local contrast, so either
    # the blur check or the contrast check may be the one that trips -
    # both are legitimate, correct reasons to reject this image.
    assert result.ok is False
    assert "blur" in result.reason.lower() or "flat" in result.reason.lower()
    assert result.metrics["blur_score"] < qc.laplacian_variance(qc._to_grayscale_array(sharp))


def test_good_quality_image_passes():
    img = _textured((60, 150, 50))
    result = qc.check_quality(img)
    assert result.ok is True
    assert result.metrics["width"] == 300


def test_laplacian_variance_is_higher_for_sharper_images():
    sharp = _textured((60, 150, 50))
    blurry = sharp.filter(ImageFilter.GaussianBlur(6))

    sharp_score = qc.laplacian_variance(qc._to_grayscale_array(sharp))
    blurry_score = qc.laplacian_variance(qc._to_grayscale_array(blurry))

    assert sharp_score > blurry_score

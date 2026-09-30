"""
Integration tests for the full pipeline (prediction_service.py),
covering the required scenarios from the project brief:

  - a supported, plant-colored image -> "clear"
  - a car / laptop / person (skin-toned) image -> "non_plant"
  - a blurry image -> "poor_quality"
  - a very dark image -> "poor_quality"
  - a genuinely ambiguous, mixed-content image -> "uncertain"

These tests require the real trained Week 6 model to be present
(see config.MODEL_PATH / the MODEL_PATH environment variable) and
TensorFlow installed - they are SKIPPED automatically if the model
file cannot be found, so the rest of the test suite can still run
without it.

As with the other test files: the "plant" and "non_plant" images
here are synthetic color patches, not real photographs. They confirm
the pipeline's *decision logic* works correctly, not that the model
is accurate on real leaves - Week 6/7's own reports are the source of
truth for real accuracy.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason=f"Trained model not found at {config.MODEL_PATH}; skipping pipeline tests.",
)


def _synthetic_photo(rgb, size=(300, 300), color_spread=35, small=25, fine_noise_std=18, seed=7):
    rng = np.random.default_rng(seed)
    coarse = np.clip(
        np.array(rgb, dtype=np.int16) + rng.integers(-color_spread, color_spread, size=(small, small, 3)),
        0, 255,
    ).astype("uint8")
    up = np.asarray(Image.fromarray(coarse).resize(size, Image.BICUBIC), dtype=np.int16)
    fine = rng.normal(0, fine_noise_std, size=(*size, 3))
    return Image.fromarray(np.clip(up + fine, 0, 255).astype("uint8"))


def _to_bytes(image, fmt="JPEG"):
    buf = io.BytesIO()
    image.save(buf, fmt)
    return buf.getvalue()


def _split_photo(rgb1, rgb2, size=(300, 300), seed=3):
    rng = np.random.default_rng(seed)
    small = 25
    half = small // 2
    coarse = np.zeros((small, small, 3), dtype=np.int16)
    coarse[:half, :, :] = rgb1
    coarse[half:, :, :] = rgb2
    coarse = np.clip(coarse + rng.integers(-20, 20, size=(small, small, 3)), 0, 255).astype("uint8")
    up = np.asarray(Image.fromarray(coarse).resize(size, Image.BICUBIC), dtype=np.int16)
    fine = rng.normal(0, 18, size=(*size, 3))
    return Image.fromarray(np.clip(up + fine, 0, 255).astype("uint8"))


def test_supported_plant_image_gives_clear_result():
    import prediction_service as ps

    img = _synthetic_photo((60, 150, 50))  # healthy-leaf-colored
    result = ps.analyze_upload("leaf.jpg", _to_bytes(img))

    assert result["status"] == ps.STATUS_CLEAR
    assert result["prediction"] is not None
    assert 0 <= result["prediction"]["confidence"] <= 100
    assert len(result["top3"]) == config.TOP_K_PREDICTIONS


def test_car_image_is_rejected_as_non_plant():
    import prediction_service as ps

    img = _synthetic_photo((95, 98, 105))  # car-gray colored
    result = ps.analyze_upload("car.jpg", _to_bytes(img))

    assert result["status"] == ps.STATUS_NON_PLANT
    assert result["prediction"] is None
    assert result["top3"] is None


def test_laptop_image_is_rejected_as_non_plant():
    import prediction_service as ps

    img = _synthetic_photo((175, 177, 180))
    result = ps.analyze_upload("laptop.jpg", _to_bytes(img))

    assert result["status"] == ps.STATUS_NON_PLANT
    assert result["prediction"] is None


def test_skin_toned_image_is_rejected_as_non_plant():
    import prediction_service as ps

    img = _synthetic_photo((205, 155, 125))
    result = ps.analyze_upload("person.jpg", _to_bytes(img))

    assert result["status"] == ps.STATUS_NON_PLANT
    assert result["prediction"] is None


def test_blurry_image_is_rejected_as_poor_quality():
    import prediction_service as ps

    img = _synthetic_photo((60, 150, 50)).filter(ImageFilter.GaussianBlur(6))
    result = ps.analyze_upload("blurry.jpg", _to_bytes(img))

    assert result["status"] == ps.STATUS_POOR_QUALITY
    assert result["prediction"] is None


def test_dark_image_is_rejected_as_poor_quality():
    import prediction_service as ps

    img = _synthetic_photo((15, 25, 12), color_spread=8, fine_noise_std=5)
    result = ps.analyze_upload("dark.jpg", _to_bytes(img))

    assert result["status"] == ps.STATUS_POOR_QUALITY
    assert result["prediction"] is None


def test_tiny_image_is_rejected_as_poor_quality():
    import prediction_service as ps

    img = _synthetic_photo((60, 150, 50), size=(30, 30), small=8)
    result = ps.analyze_upload("tiny.jpg", _to_bytes(img))

    assert result["status"] == ps.STATUS_POOR_QUALITY
    assert result["prediction"] is None


def test_ambiguous_mixed_content_is_uncertain_or_out_of_domain():
    """
    A genuinely divided image (half one plant color, half another) should
    NOT produce a single confident disease name - it should land as
    "uncertain" (or, less commonly, "out_of_domain"), never "clear".
    """
    import prediction_service as ps

    img = _split_photo((60, 150, 50), (150, 120, 40))
    result = ps.analyze_upload("split.jpg", _to_bytes(img))

    assert result["status"] in (ps.STATUS_UNCERTAIN, ps.STATUS_OUT_OF_DOMAIN)
    assert result["prediction"] is None


def test_rejected_results_never_include_top3():
    """No matter which rejection path is hit, the disease list must be hidden."""
    import prediction_service as ps

    non_plant = ps.analyze_upload("car.jpg", _to_bytes(_synthetic_photo((95, 98, 105))))
    poor_quality = ps.analyze_upload(
        "dark.jpg", _to_bytes(_synthetic_photo((15, 25, 12), color_spread=8, fine_noise_std=5))
    )

    for result in (non_plant, poor_quality):
        assert result["top3"] is None
        assert result["prediction"] is None

"""
Tests for plant_detector.py.

IMPORTANT HONESTY NOTE: these use synthetic, color-only test images
(smooth color patches with added texture/noise), NOT real photographs
of cars, laptops, people, or leaves. They prove that the vegetation-
color heuristic behaves consistently and sensibly for clearly
green/yellow-brown vs clearly gray/blue/skin-toned input - they do
NOT prove it will correctly classify arbitrary real-world photos.
See plant_detector.py's module docstring for the documented
limitations of this approach.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import plant_detector as pd


def _synthetic_photo(rgb, size=(300, 300), color_spread=35, small=25, fine_noise_std=18, seed=7):
    rng = np.random.default_rng(seed)
    coarse = np.clip(
        np.array(rgb, dtype=np.int16) + rng.integers(-color_spread, color_spread, size=(small, small, 3)),
        0, 255,
    ).astype("uint8")
    up = np.asarray(Image.fromarray(coarse).resize(size, Image.BICUBIC), dtype=np.int16)
    fine = rng.normal(0, fine_noise_std, size=(*size, 3))
    return Image.fromarray(np.clip(up + fine, 0, 255).astype("uint8"))


NON_PLANT_COLORS = {
    "car_gray": (95, 98, 105),
    "laptop_silver": (175, 177, 180),
    "skin_light": (220, 180, 150),
    "skin_dark": (120, 85, 65),
    "sky_blue": (150, 190, 230),
}

PLANT_COLORS = {
    "healthy_green": (60, 150, 50),
    "yellow_green": (140, 175, 45),
    "diseased_brown_yellow": (150, 120, 40),
}


def test_non_plant_colors_score_below_threshold_on_average():
    for name, rgb in NON_PLANT_COLORS.items():
        ratios = [pd.compute_vegetation_ratio(_synthetic_photo(rgb, seed=s)) for s in range(6)]
        mean_ratio = sum(ratios) / len(ratios)
        assert mean_ratio < config.VEGETATION_MIN_RATIO, (
            f"{name}: mean vegetation ratio {mean_ratio:.3f} was not below the threshold"
        )


def test_plant_colors_score_above_threshold_on_average():
    for name, rgb in PLANT_COLORS.items():
        ratios = [pd.compute_vegetation_ratio(_synthetic_photo(rgb, seed=s)) for s in range(6)]
        mean_ratio = sum(ratios) / len(ratios)
        assert mean_ratio > config.VEGETATION_MIN_RATIO, (
            f"{name}: mean vegetation ratio {mean_ratio:.3f} was not above the threshold"
        )


def test_check_plant_presence_rejects_car():
    img = _synthetic_photo(NON_PLANT_COLORS["car_gray"])
    result = pd.check_plant_presence(img)
    assert result.ok is False
    assert "no plant detected" in result.reason.lower()


def test_check_plant_presence_accepts_healthy_leaf():
    img = _synthetic_photo(PLANT_COLORS["healthy_green"])
    result = pd.check_plant_presence(img)
    assert result.ok is True

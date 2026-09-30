"""
plant_detector.py

Purpose:
    Third stage of the pipeline: does this image plausibly contain a
    plant/leaf at all, BEFORE the disease CNN ever sees it?

    This is deliberately independent of the CNN's own softmax output,
    because a 38-class disease classifier has no way to say "this is
    not a plant" - it can only pick among the 38 diseases it knows,
    however unrelated the image actually is. Using its own confidence
    to judge "plant-ness" would be circular and unreliable, which is
    the exact problem this rebuild is fixing.

APPROACH (and its honest limitations):
    This project runs on a CPU-only student laptop and prefers not to
    add a large extra pretrained vision model (see config.py's
    performance notes). Instead, it uses a lightweight, classic
    color-based heuristic operating directly on RGB channel
    relationships: a pixel counts as "vegetation-colored" if either
      - its green channel clearly dominates red and blue (healthy
        foliage), or
      - its red and green channels are both clearly above blue and
        close to each other (yellow/brown chlorotic or senescent
        leaf tones, without also matching typical skin tones, where
        red is usually well above green).
    The fraction of pixels meeting either condition is the
    "vegetation ratio" for the image.

    This is a real, explainable, testable signal - not a lookup
    table, not a filename check - but it is a heuristic, not object
    detection, and it has known, documented failure modes:
      - A green car, green wall, green fabric, or green packaging can
        be misread as "plant-colored".
      - A fully brown/dead leaf, or a leaf lit by very warm indoor
        light, may score lower than a similar leaf in daylight.
      - It says nothing about shape, texture, or veins - only the
        color distribution of pixels.
    Thresholds were calibrated against synthetic car/laptop/skin-tone
    and leaf-colored test images (see tests/test_plant_detector.py),
    not against real photographs, so results on real-world images may
    differ; VEGETATION_MIN_RATIO in config.py is deliberately easy to
    re-tune. It is used as ONE gate in the pipeline, not as a claim of
    general object recognition.

Week:
    AI/ML Foundation - Week 8
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

import config


@dataclass
class PlantCheckResult:
    ok: bool
    vegetation_ratio: float
    reason: str = ""


def _vegetation_mask(rgb_array):
    """
    rgb_array: (H, W, 3) array, values on the 0-255 scale (any numeric dtype).
    """

    rgb = rgb_array.astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    green_dominant = (green > red + config.GREEN_MARGIN) & (
        green > blue + config.GREEN_MARGIN
    )

    yellow_brown = (
        (green > blue + config.YELLOW_MARGIN)
        & (red > blue + config.YELLOW_MARGIN)
        & (np.abs(red - green) < config.YELLOW_RG_MAX_DIFF)
    )

    return green_dominant | yellow_brown


def _to_rgb_array(image, max_side=200):
    """Downscale for speed, then return a plain RGB numpy array."""

    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.BILINEAR,
        )

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compute_vegetation_ratio(image):
    """Fraction of pixels (0.0-1.0) that fall inside the vegetation color window."""

    rgb = _to_rgb_array(image)
    mask = _vegetation_mask(rgb)
    return float(np.mean(mask))


def check_plant_presence(image):
    """
    Decide whether the image plausibly contains a plant/leaf.

    Returns a PlantCheckResult with the raw vegetation_ratio always
    included, so the caller can display or log it as supporting
    evidence.
    """

    ratio = compute_vegetation_ratio(image)

    if ratio < config.VEGETATION_MIN_RATIO:
        return PlantCheckResult(
            ok=False,
            vegetation_ratio=round(ratio, 4),
            reason=(
                "No plant detected. This image does not appear to contain "
                "enough leaf/plant-colored content for analysis."
            ),
        )

    return PlantCheckResult(ok=True, vegetation_ratio=round(ratio, 4))

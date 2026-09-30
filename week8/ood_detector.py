"""
ood_detector.py

Purpose:
    Fifth stage of the pipeline: given the CNN's own output, decide
    whether there is genuinely enough evidence to trust it, using
    several independent signals rather than trusting raw softmax
    alone. A softmax layer always sums to 100% across the 38 known
    classes, even for an image the model has never really learned
    about - so "high confidence" by itself proves nothing.

SIGNALS USED (all computed from real model output, nothing invented):
    1. Top-1 softmax probability
    2. Margin between the top-1 and top-2 probabilities (a model
       that is genuinely sure of one class usually beats the runner-up
       by a wide margin; a model just picking the "least wrong" option
       among 38 unrelated classes usually does not)
    3. Normalized entropy of the full probability distribution (how
       spread out the model's "attention" is across all 38 classes)
    4. Test-time augmentation (TTA) consistency: the same image is
       classified 4 times - as-is, horizontally flipped, and with two
       brightness adjustments - and we check how often the model
       lands on the same top class. A real, well-recognized pattern
       tends to survive small, meaningless changes to the photo; a
       shaky, out-of-domain guess often does not.

Week:
    AI/ML Foundation - Week 8
"""

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

import config


@dataclass
class OODResult:
    top1_index: int
    top1_confidence: float
    top2_confidence: float
    margin: float
    entropy_normalized: float
    consistency_ratio: float
    tta_top1_indices: list = field(default_factory=list)
    all_probabilities: np.ndarray = None


def _resize_normalize(rgb_uint8):
    """Exactly mirrors week6/src/data_loader.py's load_image preprocessing."""

    tensor = tf.image.resize(rgb_uint8, config.IMAGE_SIZE)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    return tensor.numpy()


def _make_tta_variants(rgb_uint8):
    """
    Build a small set of meaning-preserving variants of the same photo.
    None of these should change what the leaf actually is, so a model
    that "really" recognizes the image should give a similar answer
    for all of them.
    """

    original = rgb_uint8
    flipped = np.fliplr(rgb_uint8)

    darker = np.clip(rgb_uint8.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    brighter = np.clip(rgb_uint8.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)

    return [original, flipped, darker, brighter]


def _entropy_normalized(probabilities):
    eps = 1e-12
    entropy = -np.sum(probabilities * np.log(probabilities + eps))
    max_entropy = np.log(config.NUM_CLASSES)
    return float(entropy / max_entropy)


def analyze(model, rgb_uint8):
    """
    Run the CNN on the original image and 3 TTA variants, and compute
    all evidence signals from the results.

    Args:
        model: the loaded Keras model
        rgb_uint8: HxWx3 uint8 numpy array (full-resolution RGB image,
                   NOT yet resized/normalized)

    Returns:
        OODResult
    """

    variants = _make_tta_variants(rgb_uint8)
    batch = np.stack([_resize_normalize(v) for v in variants], axis=0)

    predictions = model.predict(batch, verbose=0)

    original_probs = predictions[0]
    sorted_indices = np.argsort(original_probs)[::-1]

    top1_index = int(sorted_indices[0])
    top1_confidence = float(original_probs[top1_index])
    top2_confidence = float(original_probs[sorted_indices[1]])
    margin = top1_confidence - top2_confidence
    entropy_normalized = _entropy_normalized(original_probs)

    tta_top1_indices = [int(np.argmax(p)) for p in predictions]
    agreement = sum(1 for idx in tta_top1_indices if idx == top1_index)
    consistency_ratio = agreement / len(tta_top1_indices)

    return OODResult(
        top1_index=top1_index,
        top1_confidence=top1_confidence,
        top2_confidence=top2_confidence,
        margin=margin,
        entropy_normalized=entropy_normalized,
        consistency_ratio=consistency_ratio,
        tta_top1_indices=tta_top1_indices,
        all_probabilities=original_probs,
    )


def passes_domain_check(result: OODResult):
    """
    True if the TTA-consistency / entropy signals suggest the image is
    within the model's learned domain (separate from the raw-confidence
    check, which prediction_service applies afterwards).
    """

    return (
        result.consistency_ratio >= config.MIN_CONSISTENCY_RATIO
        and result.entropy_normalized <= config.MAX_NORMALIZED_ENTROPY
    )


def passes_confidence_check(result: OODResult):
    """True if the raw confidence/margin are high enough for a clear result."""

    return (
        result.top1_confidence >= config.MIN_ACCEPTED_CONFIDENCE
        and result.margin >= config.MIN_PROBABILITY_MARGIN
    )
